from rl.core import Processor
import numpy as np
import options as opt


class BaseAgent(object):
  def __init__(self, processor, policy, num_actions):
    self.processor = processor
    self.policy = policy
    self.num_actions = num_actions

  def fit(self, env, num_steps, weights_path, visualize):
    raise NotImplementedError

  def test(self, env, num_episodes, visualize):
    raise NotImplementedError

  def save(self, out_dir):
    raise NotImplementedError

  def load(self, out_dir):
    raise NotImplementedError


class MyProcessor(Processor):
  def __init__(self, plot_class=None):
    super(MyProcessor, self).__init__()
    self.plot_class = plot_class
    self.step = 0

  def process_step(self, observation, reward, done, info):
    self.step += 1
    observation = observation.astype(np.uint8)
    if self.plot_class:
      self.plot_class.episode_reward_buffer.append(reward)
      if done:
        self.plot_class.reward_buffer.append(np.sum(self.plot_class.episode_reward_buffer))
        self.plot_class.episode_reward_buffer.clear()
      if self.step % opt.plot_frequence == 0:
        self.plot_class.plot()
    reward = np.clip(reward, -1.0, 1.0)
    return observation, reward, done, info

  def process_state_batch(self, batch):
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
    batch = batch.astype(np.float32) / 255.0
    return batch
