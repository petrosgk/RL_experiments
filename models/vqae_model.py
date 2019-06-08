import keras
import keras.layers as L
import tensorflow as tf
import options as opt


class VQ(L.Layer):
  def __init__(self, name):
    super(VQ, self).__init__(name=name)

  def call(self, inputs, **kwargs):
    one_hot = tf.one_hot(tf.argmax(inputs, axis=-1), depth=tf.shape(inputs)[-1])
    return inputs + tf.stop_gradient(one_hot - inputs)

  def compute_output_shape(self, input_shape):
    return input_shape


def conv_bn_relu(inputs, filters, kernel_size, strides, padding, name, transpose):
  if transpose:
    conv = L.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, name=name)(inputs)
  else:
    conv = L.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, name=name)(inputs)
  bn = L.BatchNormalization(name=name + '_bn')(conv)
  ac = L.Activation('relu', name=name + '_ac')(bn)
  return ac


def conv_block(inputs, filters, kernel_size, strides, name, transpose=False):
    strided_conv = conv_bn_relu(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                padding='valid', strides=strides, name=name + '_strided_conv',
                                transpose=transpose)
    return strided_conv


def encoder(inputs, vector_length):
  # shape: [batch_size, 36, 36, channels]

  conv_0 = conv_block(inputs=inputs, filters=64, kernel_size=4, strides=2, name='enc_block_0')
  # shape: [batch_size, 17, 17, filters]
  conv_1 = conv_block(inputs=conv_0, filters=64, kernel_size=3, strides=2, name='enc_block_1')
  # shape: [batch_size, 8, 8, filters]

  flatten = L.Flatten(name='enc_head_flatten')(conv_1)
  # [batch_size, 8 * 8 * filters]
  outputs = L.Dense(units=vector_length, name='enc_head_out')(flatten)
  # shape: [batch_size, vector_length]

  return outputs


def decoder(inputs, channels):
  # shape: [batch_size, vector_length]

  flattened_inputs = L.Dense(units=4096, name='dec_inputs')(inputs)
  # shape: [batch_size, vector_length]
  reshaped_inputs = L.Reshape(target_shape=(8, 8, 64), name='dec_reshaped_inputs')(flattened_inputs)
  # shape: [batch_size, 8, 8, 64)]

  conv_0 = conv_block(inputs=reshaped_inputs, filters=64, kernel_size=3, strides=2, transpose=True, name='dec_block_0')
  # shape: [batch_size, 17, 17, filters]
  conv_1 = conv_block(inputs=conv_0, filters=64, kernel_size=4, strides=2, transpose=True, name='dec_block_1')
  # shape: [batch_size, 36, 36, filters]

  head = L.Conv2D(filters=channels, kernel_size=1, activation='sigmoid', name='dec_head')(conv_1)
  # shape: [batch_size, 36, 36, channels]

  return head


def normalize(inputs):
  inputs = tf.cast(inputs, dtype='float32')
  inputs = inputs / 255.0
  return inputs


def L1_loss(y_true, y_pred):
  y_true = normalize(y_true)
  return tf.reduce_mean(tf.abs(y_true - y_pred))


def L2_loss(y_true, y_pred):
  y_true = normalize(y_true)
  return tf.reduce_mean(tf.square(y_true - y_pred))


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def SNR(y_true, y_pred):
  y_true = normalize(y_true)
  mean_signal = tf.math.reduce_mean(y_true)
  noise = tf.abs(y_true - y_pred)
  noise_std = tf.math.reduce_std(noise)
  snr = 20 * log10(mean_signal / noise_std)
  return snr


def model(channels, state_vector_length):
  state_input = L.Input(shape=(36, 36, channels), dtype='uint8', name='state_input')
  model_input = L.Lambda(normalize, name='model_input')(state_input)

  encoded_state = encoder(inputs=model_input, vector_length=state_vector_length)
  quantized_state = VQ(name='VQ')(encoded_state)
  decoded_state = decoder(inputs=quantized_state, channels=channels)

  model = keras.models.Model(inputs=state_input, outputs=[decoded_state, quantized_state])
  model.compile(optimizer=keras.optimizers.Adam(lr=opt.learning_rate),
                loss={'dec_head': L2_loss}, metrics={'dec_head': SNR})
  print(model.summary())
  return model