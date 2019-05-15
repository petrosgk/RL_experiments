from agents.base import MyProcessor
import numpy as np
import options as opt
from PIL import Image


class AtariProcessor(MyProcessor):
  def __init__(self, autoencoder=None, autoencoder_mode='training', plot_class=None):
    super(AtariProcessor, self).__init__(autoencoder=autoencoder,
                                         autoencoder_mode=autoencoder_mode,
                                         plot_class=plot_class)

  def process_observation(self, observation):
    img = Image.fromarray(observation)
    img = img.resize((opt.height, opt.width))
    if opt.grayscale:
      img = img.convert('L')
    observation = np.array(img)
    observation = super(AtariProcessor, self).process_observation(observation)
    return observation
