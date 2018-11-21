import keras
import keras.layers as L
import tensorflow as tf
import tensorflow_probability as tfp
import options as opt


def conv_max_pool(inputs, filters, kernel_size, name):
  conv_0 = L.TimeDistributed(
    L.Conv2D(filters=filters, kernel_size=kernel_size,
             padding='same', activation='selu'),
    name=name + '_0')(inputs)
  conv_1 = L.TimeDistributed(
    L.Conv2D(filters=filters, kernel_size=kernel_size,
             padding='same', activation='selu'),
    name=name + '_1')(conv_0)
  max_pool = L.TimeDistributed(L.MaxPooling2D(pool_size=2), name=name + '_max_pool')(conv_1)
  return max_pool


def conv_upsample(inputs, filters, kernel_size, name):
  upsample = L.TimeDistributed(L.UpSampling2D(size=2), name=name + '_upsample')(inputs)
  conv_0 = L.TimeDistributed(
    L.Conv2D(filters=filters, kernel_size=kernel_size,
            padding='same', activation='selu'),
    name=name + '_0')(upsample)
  conv_1 = L.TimeDistributed(
    L.Conv2D(filters=filters, kernel_size=kernel_size,
             padding='same', activation='selu'),
    name=name + '_1')(conv_0)
  return conv_1


def encoder(inputs, bottleneck_size):
  # shape: [batch_size, sequence_length, 128, 128, channels]

  conv_max_pool_0 = conv_max_pool(inputs=inputs, filters=32, kernel_size=3, name='enc_conv_0')
  # shape: [batch_size, sequence_length, 64, 64, 32]
  conv_max_pool_1 = conv_max_pool(inputs=conv_max_pool_0, filters=32, kernel_size=3, name='enc_conv_1')
  # shape: [batch_size, sequence_length, 32, 32, 32]
  conv_max_pool_2 = conv_max_pool(inputs=conv_max_pool_1, filters=32, kernel_size=3, name='enc_conv_2')
  # shape: [batch_size, sequence_length, 16, 16, 32]
  conv_max_pool_3 = conv_max_pool(inputs=conv_max_pool_2, filters=32, kernel_size=3, name='enc_conv_3')
  # shape: [batch_size, sequence_length, 8, 8, 32]

  flatten = L.TimeDistributed(L.Flatten(), name='enc_flatten')(conv_max_pool_3)
  # shape: [batch_size, sequence_length, 8 * 8 * 32]
  head = L.CuDNNLSTM(units=8 * 8 * bottleneck_size,
                     return_sequences=True, stateful=False,
                     name='enc_head')(flatten)
  # shape: [batch_size, sequence_length, 8 * 8 * bottleneck_size]

  return head


def quantizer(inputs, softmax_temperature):
  def _annealed_softmax(inputs, softmax_temperature):
    return tf.nn.softmax(inputs / softmax_temperature, axis=-1)

  def _gumbel_softmax(inputs, softmax_temperature):
    probs = _annealed_softmax(inputs, softmax_temperature=1.0)
    return tfp.distributions.RelaxedOneHotCategorical(temperature=softmax_temperature,
                                                      probs=probs,
                                                      allow_nan_stats=False).sample()

  def _hard_one_hot(inputs):
    return tf.one_hot(tf.argmax(inputs, axis=-1), depth=tf.shape(inputs)[-1])

  def _straight_through(inputs, softmax_temperature):
    soft_one_hot = _annealed_softmax(inputs, softmax_temperature=softmax_temperature)
    hard_one_hot = _hard_one_hot(soft_one_hot)
    return soft_one_hot + tf.stop_gradient(hard_one_hot - soft_one_hot)

  def _straight_through_gumbel(inputs, softmax_temperature):
    # Get relaxed categorical distribution
    soft_one_hot = _gumbel_softmax(inputs, softmax_temperature=softmax_temperature)
    # Convert to hard 1-hot categorical distribution
    hard_one_hot = _hard_one_hot(soft_one_hot)
    # Hard 1-hot categorical in forward pass, relaxed categorical in backward pass
    return soft_one_hot + tf.stop_gradient(hard_one_hot - soft_one_hot)

  return L.TimeDistributed(
    L.Lambda(_straight_through,
             arguments={'softmax_temperature': softmax_temperature}),
    name='quantizer')(inputs)


def decoder(inputs, channels, bottleneck_size):
  # shape: [batch_size, sequence_length, 8 * 8 * bottleneck_size]

  rnn = L.CuDNNLSTM(units=8 * 8 * bottleneck_size,
                    return_sequences=True, stateful=False,
                    name='dec_rnn')(inputs)
  # shape: [batch_size, sequence_length, 8 * 8 * bottleneck_size]
  reshape = L.TimeDistributed(L.Reshape(target_shape=(8, 8, bottleneck_size)),
                              name='dec_reshape_1')(rnn)
  # shape: [batch_size, sequence_length, 8, 8, bottleneck_size]

  conv_upsample_0 = conv_upsample(inputs=reshape, filters=32, kernel_size=3, name='dec_conv_0')
  # shape: [batch_size, sequence_length, 16, 16, 32]
  conv_upsample_1 = conv_upsample(inputs=conv_upsample_0, filters=32, kernel_size=3, name='dec_conv_1')
  # shape: [batch_size, sequence_length, 32, 32, 32]
  conv_upsample_2 = conv_upsample(inputs=conv_upsample_1, filters=32, kernel_size=3, name='dec_conv_2')
  # shape: [batch_size, sequence_length, 64, 64, 32]
  conv_upsample_3 = conv_upsample(inputs=conv_upsample_2, filters=32, kernel_size=3, name='dec_conv_3')
  # shape: [batch_size, sequence_length, 128, 128, 32]

  head = L.TimeDistributed(
    L.Conv2D(filters=channels, kernel_size=3,
             padding='same', activation='sigmoid'),
    name='dec_head')(conv_upsample_3)
  # shape: [batch_size, sequence_length, 128, 128, channels]

  return head



def model(sequence_length, channels, bottleneck_size, softmax_temperature=1.0):
  state_input = L.Input(shape=(None, 128, 128, channels),
                        name='state_input')

  encoded_state = encoder(inputs=state_input, bottleneck_size=bottleneck_size)
  quantized_encoding = quantizer(inputs=encoded_state, softmax_temperature=softmax_temperature)
  decoded_state = decoder(inputs=quantized_encoding, channels=channels,
                          bottleneck_size=bottleneck_size)

  model = keras.models.Model(inputs=state_input, outputs=[decoded_state, quantized_encoding])
  model.compile(optimizer=keras.optimizers.RMSprop(lr=opt.learning_rate), loss={'dec_head': 'mse'})
  print(model.summary())
  return model