import keras


def conv_bn_relu(x, filters, kernel_size, strides, recurrent=False, batchnorm=False):
  if recurrent:
   x = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides))(x)
   if batchnorm:
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
   x = keras.layers.TimeDistributed(keras.layers.Activation('relu'))(x)
  else:
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    if batchnorm:
      x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
  return x


def DQN_Model(window_length, num_actions):
  input_shape = (84, 84, window_length)

  inputs = keras.layers.Input(shape=input_shape)
  # shape: [batch_size, 84, 84, window_length * channels]
  x = conv_bn_relu(inputs, filters=32, kernel_size=8, strides=4)
  # shape: [batch_size, 20, 20, 32]
  x = conv_bn_relu(x, filters=64, kernel_size=4, strides=2)
  # shape: [batch_size, 9, 9, 64]
  x = conv_bn_relu(x, filters=64, kernel_size=3, strides=1)
  # shape: [batch_size, 7, 7, 64]
  x = keras.layers.Flatten()(x)
  # shape: [batch_size, 3136]
  x = keras.layers.Dense(units=512, activation='relu')(x)
  # shape: [batch_size, 512]
  x = keras.layers.Dense(num_actions, activation='linear')(x)
  # shape: [batch_size, num_actions]
  model = keras.models.Model(inputs=inputs, outputs=x)
  print(model.summary())

  return model


def DQN_Dense_Model(window_length, state_size, num_actions):
  input_shape = (state_size * window_length)

  inputs = keras.layers.Input(shape=(input_shape,))
  # shape: [batch_size, window_length * state_size]
  x = keras.layers.Dense(units=64, activation='relu')(inputs)
  # shape: [batch_size, 64]
  x = keras.layers.Dense(num_actions, activation='linear')(x)
  # shape: [batch_size, num_actions]
  model = keras.models.Model(inputs=inputs, outputs=x)
  print(model.summary())

  return model


def DRQN_Model(window_length, num_actions):
  input_shape = (window_length, 84, 84, 1)

  inputs = keras.layers.Input(shape=input_shape)
  # shape: [batch_size, window_length, 84, 84, channels]
  x = conv_bn_relu(inputs, filters=32, kernel_size=8, strides=4, recurrent=True)
  # shape: [batch_size, window_length, 20, 20, 32]
  x = conv_bn_relu(x, filters=64, kernel_size=4, strides=2, recurrent=True)
  # shape: [batch_size, window_length, 9, 9, 64]
  x = conv_bn_relu(x, filters=64, kernel_size=3, strides=1, recurrent=True)
  # shape: [batch_size, window_length, 7, 7, 64]
  x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
  # shape: [batch_size, window_length, 3136]
  x = keras.layers.CuDNNLSTM(units=512)(x)
  # shape: [batch_size, 512]
  x = keras.layers.Dense(num_actions, activation='linear')(x)
  # shape: [batch_size, num_actions]
  model = keras.models.Model(inputs=inputs, outputs=x)
  print(model.summary())

  return model


def DRQN_Dense_Model(window_length, state_size, num_actions):
  input_shape = (window_length, state_size)

  inputs = keras.layers.Input(shape=input_shape)
  # shape: [batch_size, window_length, state_size]
  x = keras.layers.CuDNNLSTM(units=64)(inputs)
  # shape: [batch_size, 128]
  x = keras.layers.Dense(num_actions, activation='linear')(x)
  # shape: [batch_size, num_actions]
  model = keras.models.Model(inputs=inputs, outputs=x)
  print(model.summary())

  return model


def DDPG_Model(window_length, num_actions):
  observation_shape = (36, 36, window_length)

  # Build actor and critic networks
  inputs = keras.layers.Input(shape=observation_shape)
  # shape: [batch_size, 36, 36, channels * window_length]
  x = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(inputs)
  # shape: [batch_size, 17, 17, 32]
  x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')(x)
  # shape: [batch_size, 8, 8, 32]
  x = keras.layers.Flatten()(x)
  # shape: [batch_size, 2048]
  x = keras.layers.Dense(256, activation='relu')(x)
  x = keras.layers.Dense(256, activation='relu')(x)
  x = keras.layers.Dense(num_actions, activation='tanh',
                         kernel_initializer=keras.initializers.RandomUniform(-3e-4, 3e-4))(x)
  actor = keras.models.Model(inputs=inputs, outputs=x)
  print(actor.summary())

  ## critic network has 2 inputs, one action input and one observation input.
  action_input = keras.layers.Input(shape=(num_actions,), name='action_input')
  # shape: [batch_size, num_actions]
  observation_input = keras.layers.Input(shape=observation_shape, name='observation_input')
  # shape: [batch_size, 36, 36, channels * window_length]

  x = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(observation_input)
  # shape: [batch_size, 17, 17, 32]
  x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')(x)
  # shape: [batch_size, 8, 8, 32]
  x = keras.layers.Flatten()(x)
  # shape: [batch_size, 2048]
  x = keras.layers.Dense(256, activation='relu')(x)
  x = keras.layers.Concatenate()([x, action_input])  # Actions are not included until the 2nd dense layer
  x = keras.layers.Dense(256, activation='relu')(x)
  x = keras.layers.Dense(1, activation='linear',
                         kernel_initializer=keras.initializers.RandomUniform(-3e-4, 3e-4))(x)
  critic = keras.models.Model(inputs=[action_input, observation_input], outputs=x)
  print(critic.summary())

  return actor, critic, action_input
