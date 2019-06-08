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
  def __init__(self, autoencoder=None, plot_class=None, action_space='discrete'):
    super(MyProcessor, self).__init__()
    self.autoencoder = autoencoder
    self.plot_class = plot_class
    self.action_space = action_space
    self.step = 0

    self.quantized_observation = None
    self.state_visit_counts = np.zeros(shape=opt.state_vector_length, dtype=np.float32)

  def process_observation(self, observation):
    if self.autoencoder is not None:
      inputs = Image.fromarray(observation)
      inputs = inputs.resize((36, 36))
      if opt.grayscale:
        inputs = inputs.convert('L')
      inputs = np.array(inputs).astype(np.uint8)
      self.autoencoder.frame_buffer.append(inputs)
      if self.step % opt.train_interval == 0:
        self.autoencoder.train()
      self.quantized_observation = self.autoencoder.get_outputs(inputs)
      if self.quantized_observation is not None:
        self.state_visit_counts[np.argmax(self.quantized_observation)] += 1
      if opt.use_quantized_observations:
        if self.quantized_observation is not None:
          observation = np.argmax(self.quantized_observation)
        else:
          observation = 0
      if self.step % 10000 == 0:
        self.autoencoder.save_weights()
    self.step += 1
    return observation

  def process_step(self, observation, reward, done, info, isTraining=True):
    if self.plot_class:
      if not isTraining:
        self.plot_class.episode_reward_buffer.append(reward)
      if done:
        self.plot_class.plot(isTraining=isTraining)
    if self.autoencoder is not None:
      if opt.use_intrinsic_rewards and isTraining:
        if self.quantized_observation is not None:
          intrinsic_reward = opt.beta * (1 / np.sqrt(self.state_visit_counts[np.argmax(self.quantized_observation)]))
          # print(self.state_visit_counts[np.argmax(self.quantized_observation)], intrinsic_reward)
        else:
          intrinsic_reward = 0.0
        reward += intrinsic_reward
    if isTraining:
      reward = np.clip(reward, -1.0, 1.0)
    return observation, reward, done, info

  def process_state_batch(self, batch):
    if opt.recurrent:
      batch = np.expand_dims(batch, axis=-1)
    else:
      batch = np.transpose(batch, [0, 2, 3, 1])
    batch = batch.astype(np.float32) / 255.
    return batch

  def process_action(self, action):
    if self.action_space == 'continuous':
      action = list(action)
    return action