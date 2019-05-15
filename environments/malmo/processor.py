from agents.base import MyProcessor
import options as opt


class MalmoProcessor(MyProcessor):
  def __init__(self, autoencoder=None, autoencoder_mode='training', plot_class=None, action_space='discrete'):
    super(MalmoProcessor, self).__init__(autoencoder=autoencoder,
                                         autoencoder_mode=autoencoder_mode,
                                         plot_class=plot_class,
                                         action_space=action_space)

  def process_observation(self, observation):
    img, obs = observation
    state = super(MalmoProcessor, self).process_observation(img)
    if self.autoencoder is not None and self.autoencoder_mode == 'testing' and opt.plot_replay:
      self.plot_class.state_coordinates_buffer.append([obs['XPos'], obs['ZPos'], obs['Yaw']])
    return state
