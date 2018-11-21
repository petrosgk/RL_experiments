import keras


def DQN_32(window_length, grayscale, width, height, num_actions):
  assert width == 84 and height == 84, \
    'Model accepts 84x84 input size, got {}x{}'.format(width, height)
  if grayscale:
    channels = 1
  else:
    channels = 3
  input_shape = (84, 84, window_length * channels)

  inputs = keras.layers.Input(shape=input_shape)
  # shape: [batch_size, 84, 84, window_length * channels]
  x = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(inputs)
  # shape: [batch_size, 20, 20, 64]
  x = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(x)
  # shape: [batch_size, 9, 9, 64]
  x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
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