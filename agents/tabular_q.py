import numpy as np
from collections import deque
from rl.core import Agent
from agents.base import BaseAgent
import options as opt


class TabularQ(Agent):
  def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, nb_steps_warmup=10,
               enable_double_dqn=False, policy=None, test_policy=None, processor=None):
    super(TabularQ, self).__init__(processor=processor)
    self.alpha = alpha
    self.gamma = gamma
    self.nb_steps_warmup = nb_steps_warmup
    self.policy = policy
    self.test_policy = test_policy
    self.enable_double_dqn = enable_double_dqn
    self.compiled=False

    self.state_buffer = deque(maxlen=2)
    self.action_buffer = deque(maxlen=2)
    self.q_table = np.zeros(shape=(num_states, num_actions), dtype=np.float32)

  def compile(self, optimizer=None, metrics=[]):
    self.compiled=True

  def forward(self, observation):
    state = observation
    q_values = self.q_table[state]
    if self.training:
      action = self.policy.select_action(q_values=q_values)
    else:
      action = self.test_policy.select_action(q_values=q_values)
    self.state_buffer.append(state)
    self.action_buffer.append(action)
    return action

  def backward(self, reward, terminal):
    metrics = [np.nan for _ in self.metrics_names]
    if not self.training:
      return metrics
    if self.step > self.nb_steps_warmup:
      old_q_value = self.q_table[self.state_buffer[0], self.action_buffer[0]]
      next_max_q_value = np.max(self.q_table[self.state_buffer[1]])
      new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max_q_value)
      self.q_table[self.state_buffer[0], self.action_buffer[0]] = new_q_value
      metrics = self.policy.metrics
      if self.processor is not None:
        metrics += self.processor.metrics
      metrics += [np.mean(self.q_table)]
    return metrics

  @property
  def metrics_names(self):
    names = self.policy.metrics_names[:]
    if self.processor is not None:
      names += self.processor.metrics_names[:]
    names += ['mean_q']
    return names

  @property
  def policy(self):
      return self.__policy

  @policy.setter
  def policy(self, policy):
      self.__policy = policy
      self.__policy._set_agent(self)

  @property
  def test_policy(self):
      return self.__test_policy

  @test_policy.setter
  def test_policy(self, policy):
      self.__test_policy = policy
      self.__test_policy._set_agent(self)


class TabularQAgent(BaseAgent):
  def __init__(self, num_states, num_actions, policy, test_policy, processor):
    self.agent = TabularQ(num_states=num_states,
                          num_actions=num_actions,
                          policy=policy,
                          test_policy=test_policy,
                          processor=processor,
                          alpha=opt.tabular_q_learning_rate,
                          gamma=opt.tabular_q_gamma)
    self.agent.compile()

  def fit(self, env, num_steps, weights_path=None, visualize=False):
    self.agent.fit(env=env,
                   nb_steps=num_steps,
                   visualize=visualize,
                   test_nb_episodes=opt.test_nb_episodes,
                   test_interval=opt.test_interval,
                   test_visualize=False,
                   log_interval=opt.log_interval,
                   verbose=2)

  def test(self, env, num_episodes, visualize=False):
    self.agent.test(env=env,
                    nb_episodes=num_episodes,
                    visualize=visualize)