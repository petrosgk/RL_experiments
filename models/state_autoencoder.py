import keras
import keras.layers as L
import tensorflow as tf
import tensorflow_probability as tfp
import options as opt
import scipy.interpolate as interp
import numpy as np


class AnnealedQuantization(L.Layer):
  def __init__(self, temperature_per_step, name):
    assert isinstance(temperature_per_step, list)
    # Collect temperature per step and steps into separate lists.
    self.steps = []
    self.temperatures = []
    for entry in temperature_per_step:
      assert isinstance(entry, list)
      self.steps.append(entry[0])
      self.temperatures.append(entry[1])
    assert len(self.steps) == len(self.temperatures)
    self.max_annealing_steps = self.steps[-1]
    # Interpolate.
    interpolation_function = interp.interp1d(x=self.steps,
                                             y=self.temperatures,
                                             bounds_error=False,
                                             fill_value=(self.temperatures[0],
                                                         self.temperatures[-1]))
    self.interpolated_temperatures = interpolation_function(np.arange(self.max_annealing_steps))
    self.minimum_temperature = self.interpolated_temperatures[-1]
    super(AnnealedQuantization, self).__init__(name=name)

  def build(self, input_shape):
    self.interpolated_temperatures = tf.constant(self.interpolated_temperatures,
                                                 dtype='float32',
                                                 name='interpolated_temperatures')
    self.step_index = self.add_weight(name='step_index',
                                      shape=(),
                                      initializer='zeros',
                                      trainable=False)
    super(AnnealedQuantization, self).build(input_shape)

  @staticmethod
  def softmax(inputs, softmax_temperature):
    return tf.nn.softmax(inputs / softmax_temperature, axis=-1)

  @staticmethod
  def gumbel_softmax(inputs, softmax_temperature):
    probs = tf.nn.softmax(inputs)
    return tfp.distributions.RelaxedOneHotCategorical(temperature=softmax_temperature,
                                                      probs=probs,
                                                      allow_nan_stats=False).sample()

  def straight_through(self, inputs, softmax_temperature):
    soft_one_hot = self.softmax(inputs, softmax_temperature=softmax_temperature)
    hard_one_hot = tf.one_hot(tf.argmax(soft_one_hot, axis=-1), depth=tf.shape(inputs)[-1])
    return soft_one_hot + tf.stop_gradient(hard_one_hot - soft_one_hot)

  def straight_through_gumbel(self, inputs, softmax_temperature):
    soft_one_hot = self.gumbel_softmax(inputs, softmax_temperature=softmax_temperature)
    hard_one_hot = tf.one_hot(tf.argmax(soft_one_hot, axis=-1), depth=tf.shape(inputs)[-1])
    return soft_one_hot + tf.stop_gradient(hard_one_hot - soft_one_hot)

  def call(self, inputs, **kwargs):
    step_index = tf.cond(tf.greater_equal(self.step_index, self.max_annealing_steps),
                         true_fn=lambda: tf.cast(self.max_annealing_steps, dtype='float32'),
                         false_fn=lambda: tf.assign_add(self.step_index, 1))
    step_index = step_index - 1
    step_index = tf.cast(step_index, dtype='int32')
    temperature = self.interpolated_temperatures[step_index]
    quantized_input = self.straight_through(inputs, softmax_temperature=temperature)
    return [quantized_input, temperature]

  def compute_output_shape(self, input_shape):
    return [input_shape, ()]


def conv_bn_relu(inputs, filters, kernel_size, strides, padding, name, transpose):
  if transpose:
    conv = L.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                             padding=padding, strides=strides, name=name)(inputs)
  else:
    conv = L.Conv2D(filters=filters, kernel_size=kernel_size,
                    padding=padding, strides=strides, name=name)(inputs)
  bn = L.BatchNormalization(name=name + '_bn')(conv)
  ac = L.Activation('relu', name=name + '_ac')(bn)
  return ac


def conv_block(inputs, filters, kernel_size, strides, name, transpose=False):
    strided_conv = conv_bn_relu(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                padding='valid', strides=strides, name=name + '_strided_conv',
                                transpose=transpose)
    return strided_conv


def encoder(inputs):
  # shape: [batch_size, 84, 84, channels * sequence_length]

  conv_0 = conv_block(inputs=inputs, filters=64, kernel_size=8, strides=4, name='enc_block_0')
  # shape: [batch_size, 20, 20, 64]
  conv_1 = conv_block(inputs=conv_0, filters=64, kernel_size=4, strides=2, name='enc_block_1')
  # shape: [batch_size, 9, 9, 64]
  conv_2 = conv_block(inputs=conv_1, filters=64, kernel_size=3, strides=1, name='enc_block_2')
  # shape: [batch_size, 7, 7, 64]

  return conv_2


def decoder(inputs, channels, sequence_length):
  # shape: [batch_size, 7, 7, features]

  conv_0 = conv_block(inputs=inputs, filters=256, kernel_size=3, strides=1,
                      transpose=True, name='dec_block_0')
  # shape: [batch_size, 9, 9, 64]
  conv_1 = conv_block(inputs=conv_0, filters=256, kernel_size=4, strides=2,
                      transpose=True, name='dec_block_1')
  # shape: [batch_size, 20, 20, 64]
  conv_2 = conv_block(inputs=conv_1, filters=256, kernel_size=8, strides=4,
                      transpose=True, name='dec_block_2')
  # shape: [batch_size, 84, 84, 64]

  head = L.Conv2D(filters=channels * sequence_length, kernel_size=1,
                  activation='sigmoid', name='dec_head')(conv_2)
  # shape: [batch_size, 84, 84, channels * sequence_length]

  return head


def dense_bn_relu(inputs, units, name):
  dense = L.Dense(units=units, name=name)(inputs)
  bn = L.BatchNormalization(name=name + '_bn')(dense)
  ac = L.Activation('relu', name=name + '_ac')(bn)
  return ac


def vectorize(inputs, vector_length, name):
  flattened_inputs = L.Flatten(name='vectorize_flatten')(inputs)
  vectorized_inputs = dense_bn_relu(inputs=flattened_inputs, units=vector_length, name=name)
  return vectorized_inputs


def devectorize(inputs, matrix_features, name):
  expanded_inputs = dense_bn_relu(inputs=inputs, units=7 * 7 * matrix_features, name=name)
  devectorized_inputs = L.Reshape(target_shape=(7, 7, matrix_features),
                                  name='devectorize_reshape')(expanded_inputs)
  return devectorized_inputs


def model(sequence_length, channels,
          state_vector_length, state_matrix_features,
          temperature_per_step):
  state_input = L.Input(shape=(84, 84, channels * sequence_length),
                        name='state_input')

  encoded_state = encoder(inputs=state_input)
  encoded_state_vector = vectorize(inputs=encoded_state, vector_length=state_vector_length,
                                   name='encoded_state_vector')
  quantized_encoding, temperature = AnnealedQuantization(temperature_per_step=temperature_per_step,
                                                         name='VQ')(encoded_state_vector)
  encoded_state_matrix = devectorize(inputs=quantized_encoding,
                                     matrix_features=state_matrix_features,
                                     name='quantized_state_matrix')
  decoded_state = decoder(inputs=encoded_state_matrix, channels=channels,
                          sequence_length=sequence_length)

  model = keras.models.Model(inputs=state_input,
                             outputs=[decoded_state, quantized_encoding, temperature])
  model.compile(optimizer=keras.optimizers.RMSprop(lr=opt.learning_rate), loss={'dec_head': 'mse'})
  print(model.summary())
  return model