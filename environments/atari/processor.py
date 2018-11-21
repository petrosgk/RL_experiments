from agents.base import MyProcessor
import numpy as np
import options as opt
from PIL import Image


class AtariProcessor(MyProcessor):
  def __init__(self, plot_class=None):
    super(AtariProcessor, self).__init__(plot_class=plot_class)

  def process_observation(self, observation):
    img = Image.fromarray(observation)
    img = img.resize((opt.height, opt.width))
    if opt.grayscale:
      img = img.convert('L')
    observation = np.array(img)
    return observation

  def process_step(self, observation, reward, done, info):
    observation = self.process_observation(observation)
    observation, reward, done, info = super(AtariProcessor, self).process_step(observation=observation,
                                                                               reward=reward,
                                                                               done=done,
                                                                               info=info)
    return observation, reward, done, info
