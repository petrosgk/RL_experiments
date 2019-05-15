import os
from rl.policy import Policy
import numpy as np
import options as opt
from collections import deque


class SeekNovelStatesPolicy(Policy):
  def __init__(self, num_actions, autoencoder, mode, eps=1.0, plot_class=None):
    super(SeekNovelStatesPolicy, self).__init__()
    self.num_actions = num_actions
    self.autoencoder = autoencoder
    self.autoencoder_frame_buffer = deque(maxlen=opt.frame_buffer_size)
    self.state_visit_counts = np.zeros(shape=opt.state_vector_length)
    self.actions_buffer = deque()
    self.eps = eps
    self.episode_step = 0
    self.step = 0

    self.actions_to_replay = None
    self.last_discrete_state = None
    self.last_action = None

    self.mode = mode
    if mode == 'testing':
      self.autoencoder.load_weights(os.path.join('weights', 'state_autoencoder.hdf5'))

    self.plot_class = plot_class

    assert opt.frame_buffer_size % opt.train_sequence_length == 0, (opt.frame_buffer_size, opt.train_sequence_length)
    self.num_sequences = int(opt.frame_buffer_size / opt.train_sequence_length)

  @staticmethod
  def preprocess_state(inputs):
    inputs = np.array(inputs, dtype=np.uint8)
    if opt.grayscale:
      inputs = np.expand_dims(inputs, axis=-1)
    return inputs

  def train_autoencoder(self):
    frames = np.asarray(self.autoencoder_frame_buffer)
    states = np.split(frames, self.num_sequences)
    states = np.asarray(states)
    train_history = self.autoencoder.fit(states, states, batch_size=opt.batch_size, epochs=opt.epochs)
    self.autoencoder.save_weights(os.path.join('weights', 'state_autoencoder.hdf5'))
    if self.plot_class:
      losses = train_history.history['loss']
      for loss in losses:
        self.plot_class.plot_autoencoder_train_loss_buffer.append(loss)

  def get_autoencoder_outputs(self, state):
    state = np.expand_dims(state, axis=0)  # Introduce batch dimension
    decoded_sequence, quantized_sequence, temperature = self.autoencoder.predict_on_batch(state)
    decoded_sequence = np.squeeze(decoded_sequence, axis=0)  # Remove batch dimension
    quantized_sequence = np.squeeze(quantized_sequence, axis=0)  # Remove batch dimension
    return decoded_sequence, quantized_sequence, temperature

  @staticmethod
  def L2_loss(decoded_sequence, state):
    state = (state / 255.0).astype(np.float32)
    loss = np.mean(np.square(decoded_sequence - state))
    return loss

  def seek_novel_states(self, state):
    decoded_state, quantized_state, temperature = self.get_autoencoder_outputs(state)
    loss = self.L2_loss(decoded_state, state)
    if self.plot_class:
      self.plot_class.plot_autoencoder_test_loss_buffer.append(loss)

    state = state[-1]
    decoded_state = decoded_state[-1]
    quantized_state = quantized_state[-1]

    max_ind = np.argmax(quantized_state)
    self.state_visit_counts[max_ind] += 1

    # if max_ind == self.last_discrete_state:
    #   action = np.random.randint(self.num_actions)  # Choose a new action
    # else:
    #   action = self.last_action  # Keep doing last action
    # self.last_discrete_state = max_ind

    action = np.random.randint(self.num_actions)  # Choose a new action

    # Normalize loss
    # autoencoder_test_loss = self.normalization.normalize_mean(autoencoder_test_loss)

    # if autoencoder_test_loss > np.mean(self.autoencoder_loss_buffer):
    #   print('\nStep %d. Novel state encountered! Replaying next episode!' % self.episode_step)
    #   # On next episode, we will replay this action sequence
    #   self.actions_to_replay = list(self.actions_buffer)
    #   action = np.random.randint(self.num_actions)
    # else:
    #   if self.actions_to_replay is not None and self.episode_step < len(self.actions_to_replay):
    #     action = self.actions_to_replay[self.episode_step]
    #   else:
    #     action = np.random.randint(self.num_actions)

    if self.plot_class:
      self.plot_class.plot_autoencoder_temperature_buffer.append(temperature)
      decoded_state = (decoded_state * 255.0).astype('uint8')
      self.plot_class.state_buffer.append(state)
      self.plot_class.reconstructed_state_buffer.append(decoded_state)
      self.plot_class.state_index_buffer.append(max_ind)
      self.plot_class.state_visit_counts[max_ind] += 1
      self.plot_class.update_state_graph(max_ind)

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
    for frame in state:
      self.autoencoder_frame_buffer.append(frame)
    if 0 < self.step <= opt.steps_warmup:
      if (self.step * opt.dqn_window_length) % opt.frame_buffer_size == 0:
        self.train_autoencoder()
    if np.random.uniform() < self.eps:
      if self.step > opt.steps_warmup:
        action = self.seek_novel_states(state)
      else:
        action = np.random.randint(self.num_actions)
    else:
      action = np.argmax(q_values)
    self.last_action = action
    self.actions_buffer.append(action)  # Save action to buffer
    self.episode_step += 1
    self.step += 1
    if terminal:
      self.actions_buffer.clear()
      self.episode_step = 0
    return action

  def get_config(self):
    config = super(SeekNovelStatesPolicy, self).get_config()
    config['eps'] = self.eps
    return config