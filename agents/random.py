import numpy as np
from rl.core import Agent
from agents.base import BaseAgent
import options as opt


class Random(Agent):
  def __init__(self, num_actions, processor=None):
    super(Random, self).__init__(processor=processor)
    self.num_actions = num_actions
    self.compiled = False

  def forward(self, observation, terminal=False):
    return np.random.randint(self.num_actions)

  def backward(self, reward, terminal):
    pass

  def compile(self, optimizer=None, metrics=[]):
    self.compiled = True


class RandomAgent(BaseAgent):
  def __init__(self, num_actions, processor):
    self.agent = Random(num_actions=num_actions, processor=processor)
    self.agent.compile()

  def fit(self, env, num_steps, weights_path=None, visualize=False):
    self.agent.fit(env=env,
                   nb_steps=num_steps,
                   visualize=visualize,
                   action_repetition=opt.random_action_repetition,
                   test_action_repetition=opt.random_action_repetition,
                   test_nb_episodes=opt.test_nb_episodes,
                   test_interval=opt.test_interval,
                   test_visualize=False,
                   log_interval=opt.log_interval,
                   verbose=2)

  def test(self, env, num_episodes, visualize=False):
    self.agent.test(env=env,
                    nb_episodes=num_episodes,
                    visualize=visualize)