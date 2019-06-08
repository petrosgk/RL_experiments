from agents.base import MyProcessor
from PIL import Image
import numpy as np
import options as opt


class MalmoProcessor(MyProcessor):
  def __init__(self, autoencoder=None, plot_class=None, action_space='discrete'):
    super(MalmoProcessor, self).__init__(autoencoder=autoencoder,
                                         plot_class=plot_class,
                                         action_space=action_space)

  def process_observation(self, observation):
    img, obs = observation
    img = super(MalmoProcessor, self).process_observation(img)
    img = Image.fromarray(img)
    img = img.resize((84, 84))
    img = img.convert('L')
    img = np.array(img).astype(np.uint8)
    if self.autoencoder is not None and opt.plot_replay:
      self.plot_class.state_coordinates_buffer.append([obs['XPos'], obs['ZPos'], obs['Yaw']])
    return img

  def process_step(self, observation, reward, done, info, isTraining=True):
    observation = self.process_observation(observation)
    observation, reward, done, info = super(MalmoProcessor, self).process_step(observation=observation,
                                                                               reward=reward,
                                                                               done=done,
                                                                               info=info,
                                                                               isTraining=isTraining)
    return observation, reward, done, info
