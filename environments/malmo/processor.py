from agents.base import MyProcessor


class MalmoProcessor(MyProcessor):
  def __init__(self, abs_max_reward, plot_class=None):
    super(MalmoProcessor, self).__init__(plot_class=plot_class)
    self.abs_max_reward = abs_max_reward

  def process_step(self, observation, reward, done, info):
    reward = reward / self.abs_max_reward
    observation, reward, done, info = super(MalmoProcessor, self).process_step(observation=observation,
                                                                               reward=reward,
                                                                               done=done,
                                                                               info=info)
    return observation, reward, done, info