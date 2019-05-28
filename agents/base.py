from rl.core import Processor
import numpy as np
import options as opt
from PIL import Image


class BaseAgent(object):
  def fit(self, env, num_steps, weights_path=None, visualize=False):
    raise NotImplementedError

  def test(self, env, num_episodes, visualize=False):
    pass

  def save(self, out_dir):
    pass

  def load(self, out_dir):
    pass


class MyProcessor(Processor):
  def __init__(self, autoencoder=None, autoencoder_mode='training', plot_class=None, action_space='discrete'):
    super(MyProcessor, self).__init__()
    self.autoencoder = autoencoder
    self.autoencoder_mode = autoencoder_mode
    self.plot_class = plot_class
    self.action_space = action_space
    self.step = 0

    self.quantized_observation = None
    if self.autoencoder is not None and self.autoencoder_mode == 'testing':
      self.state_visit_counts = np.zeros(shape=opt.state_vector_length, dtype=np.float32)

  def process_observation(self, observation):
    observation = observation.astype(np.uint8)
    if self.autoencoder is not None:
      autoenc_inputs = Image.fromarray(observation)
      autoenc_inputs = autoenc_inputs.resize((36, 36))
      autoenc_inputs = np.array(autoenc_inputs)
      if self.autoencoder_mode == 'training':
        self.autoencoder.frame_buffer.append(autoenc_inputs)
        if (self.step > 0) and (self.step % opt.frame_buffer_size == 0):
          self.autoencoder.train()
      else:
        self.autoencoder.frame_buffer.append(autoenc_inputs)
        self.quantized_observation = self.autoencoder.get_outputs()
        if self.quantized_observation is not None:
          self.state_visit_counts[np.argmax(self.quantized_observation)] += 1
    self.step += 1
    if opt.use_quantized_observations and self.autoencoder_mode == 'testing':
      if self.quantized_observation is not None:
        return np.argmax(self.quantized_observation)
      else:
        return 0
    else:
      return observation

  def process_step(self, observation, reward, done, info, isTraining=True):
    observation = self.process_observation(observation)
    if self.plot_class:
      if not isTraining:
        self.plot_class.episode_reward_buffer.append(reward)
      if self.autoencoder_mode == 'training':
        if (self.step > 0) and (self.step % 5000 == 0):
          self.plot_class.plot()
      else:
        if done:
          self.plot_class.plot(isTraining=isTraining)
    if opt.use_intrinsic_rewards and isTraining:
      if (self.autoencoder is not None) and (self.autoencoder_mode == 'testing'):
        if self.quantized_observation is not None:
          intrinsic_reward = opt.beta * (1 / np.sqrt(self.state_visit_counts[np.argmax(self.quantized_observation)]))
          intrinsic_reward = max(intrinsic_reward, 1e-5)
          # print(self.state_visit_counts[np.argmax(self.quantized_observation)], intrinsic_reward)
        else:
          intrinsic_reward = 0.0
        reward += intrinsic_reward
    reward = np.clip(reward, -1.0, 1.0)
    return observation, reward, done, info

  def process_state_batch(self, batch):
    if opt.recurrent:
      if opt.grayscale:
        batch = np.expand_dims(batch, axis=-1)
    else:
      if not opt.grayscale:
        batch_samples = []
        for batch_sample in range(batch.shape[0]):
          sequence = []
          for frame in range(batch.shape[1]):
            sequence.append(batch[batch_sample][frame])
          sequence = np.concatenate(sequence, axis=-1)
          sequence = np.expand_dims(sequence, axis=0)
          batch_samples.append(sequence)
        batch = np.concatenate(batch_samples, axis=0)
      else:
        batch = np.transpose(batch, [0, 2, 3, 1])
    batch = batch.astype(np.float32) / 255.
    return batch

  def process_action(self, action):
    if self.action_space == 'continuous':
      action = list(action)
    return action