from agents.base import MyProcessor
import numpy as np
from PIL import Image


class AtariProcessor(MyProcessor):
  def __init__(self, autoencoder=None, plot_class=None):
    super(AtariProcessor, self).__init__(autoencoder=autoencoder,
                                         plot_class=plot_class)

  def process_observation(self, observation):
    observation = super(AtariProcessor, self).process_observation(observation)
    img = Image.fromarray(observation)
    img = img.resize((84, 84))
    img = img.convert('L')
    observation = np.array(img).astype(np.uint8)
    return observation


  def process_step(self, observation, reward, done, info, isTraining=True):
    observation = self.process_observation(observation)
    observation, reward, done, info = super(AtariProcessor, self).process_step(observation=observation,
                                                                               reward=reward,
                                                                               done=done,
                                                                               info=info,
                                                                               isTraining=isTraining)
    return observation, reward, done, info
