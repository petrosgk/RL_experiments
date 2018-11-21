import numpy as np
import models.state_autoencoder as state_autoencoder


# def CreateInputData(window_length, channels):
#     state = np.random.uniform(size=(32, 32, window_length * channels)).astype(np.float32)
#
#   # action_space = np.eye(num_actions)
#   # actions = []
#   # for _ in range(sequence_length):
#   #   action = action_space[np.random.randint(num_actions)]
#   #   actions.append(action)
#   # actions = np.concatenate(actions, axis=-1)
#
#   return states


def TestModel(sequence_length, channels, state_vector_length,
              state_matrix_features, temperature_per_step):
  # Create single batch of input data
  state = np.random.uniform(size=(84, 84, sequence_length * channels)).astype(np.float32)
  state = np.expand_dims(state, axis=0)
  # Create model
  model = state_autoencoder.model(channels=channels, sequence_length=sequence_length,
                                  state_vector_length=state_vector_length,
                                  state_matrix_features=state_matrix_features,
                                  temperature_per_step=temperature_per_step)
  # Test model
  loss = model.train_on_batch(x=state, y=state)
  return loss[0]


sequence_length = 4
channels = 3
state_vector_length = 128
state_matrix_features = 32
temperature_per_step = [[0, 1.0], [50000, 0.1]]

loss = TestModel(sequence_length=sequence_length, channels=channels,
                 state_vector_length=state_vector_length,
                 state_matrix_features=state_matrix_features,
                 temperature_per_step=temperature_per_step)
assert not np.isnan(loss)
