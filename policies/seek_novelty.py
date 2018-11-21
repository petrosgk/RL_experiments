from rl.policy import Policy
import numpy as np
import options as opt
from collections import deque


class SeekNovelStatesPolicy(Policy):
  def __init__(self, num_actions, autoencoder, eps=1.0, plot_class=None):
    super(SeekNovelStatesPolicy, self).__init__()
    self.num_actions = num_actions
    self.autoencoder = autoencoder
    self.autoencoder_state_buffer = deque()
    # self.autoencoder_loss_buffer = deque(maxlen=opt.losses_buffer_size)
    # self.autoencoder_loss_buffer.append(1.0)
    self.state_visit_counts = np.zeros(shape=opt.state_vector_length)
    self.eps = eps
    self.last_discrete_state = 0
    self.last_action = 0
    self.steps = 0

    self.plot_class = plot_class

  @staticmethod
  def preprocess_state(inputs):
    inputs = np.array(inputs, dtype=np.uint8)
    if opt.grayscale:
      inputs = np.expand_dims(inputs, axis=-1)
    sequence = []
    for frame in range(inputs.shape[0]):
      sequence.append(inputs[frame])
    sequence = np.concatenate(sequence, axis=-1)
    return sequence

  def train_autoencoder(self):
    states = np.expand_dims(self.autoencoder_state_buffer, axis=0)
    states = np.concatenate(states, axis=0)
    states = (states / 255.0).astype(np.float32)
    history = self.autoencoder.fit(x=states, y=states,
                                   epochs=opt.epochs, batch_size=opt.batch_size, verbose=1)
    loss = np.mean(history.history['loss'])
    self.trained_once = True
    if self.plot_class:
      self.plot_class.plot_autoencoder_train_loss_buffer.append(loss)

  def get_autoencoder_outputs(self, inputs):
    inputs = np.expand_dims(inputs, axis=0)
    inputs = (inputs / 255.0).astype(np.float32)
    decoded_sequence, quantized_sequence, temperature = self.autoencoder.predict_on_batch(inputs)
    autoencoder_test_loss = np.mean(np.square(inputs - decoded_sequence))
    # print('\n', np.argmax(quantized_sequence))
    decoded_sequence = np.squeeze(decoded_sequence, axis=0)
    quantized_sequence = np.squeeze(quantized_sequence, axis=0)
    return decoded_sequence, quantized_sequence, temperature, autoencoder_test_loss

  def seek_novel_states(self, state):
    # Get reconstruction error on current state
    decoded_sequence, quantized_encoding, temperature, autoencoder_test_loss = self.get_autoencoder_outputs(state)
    max_ind = np.argmax(quantized_encoding)
    self.state_visit_counts[max_ind] += 1

    if max_ind == self.last_discrete_state:
      action = np.random.randint(self.num_actions)  # Choose a new action
    else:
      action = self.last_action  # Keep doing last action
    self.last_discrete_state = max_ind

    # if autoencoder_test_loss < np.mean(self.autoencoder_loss_buffer):
    #   action = np.random.randint(self.num_actions)  # Choose a new action
    # else:
    #   action = self.last_action  # Keep doing last action
    # # Append to memory
    # self.autoencoder_loss_buffer.append(autoencoder_test_loss)

    if self.plot_class:
      self.plot_class.plot_autoencoder_temperature_buffer.append(temperature)
      self.plot_class.plot_autoencoder_test_loss_buffer.append(autoencoder_test_loss)
      self.plot_class.state = decoded_sequence
      self.plot_class.state_visit_counts = self.state_visit_counts

    return action

  @staticmethod
  def check_zero_state(inputs):
    isZero = False
    for state in inputs:
      if np.array_equal(state, np.zeros_like(state)):
        isZero = True
        break
    return isZero

  def select_action(self, q_values, state, terminal):
    state = self.preprocess_state(state)
    self.autoencoder_state_buffer.append(state)
    if terminal:
      self.train_autoencoder()
      self.autoencoder_state_buffer.clear()
    if np.random.uniform() < self.eps:
      if self.steps > opt.steps_warmup:
        action = self.seek_novel_states(state)
      else:
        action = np.random.randint(self.num_actions)
    else:
      action = np.argmax(q_values)
    self.last_action = action
    self.steps += 1
    return action

  def get_config(self):
    config = super(SeekNovelStatesPolicy, self).get_config()
    config['eps'] = self.eps
    return config