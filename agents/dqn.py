import options as opt
import keras
from agents.base import BaseAgent
from rl.callbacks import ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent


class DQN(BaseAgent):
  def __init__(self, model, processor, policy, test_policy, num_actions):
    # Replay memory
    memory = SequentialMemory(limit=opt.dqn_replay_memory_size,
                              window_length=opt.dqn_window_length)
    self.agent = DQNAgent(model=model,
                          nb_actions=num_actions,
                          policy=policy,
                          test_policy=test_policy,
                          memory=memory,
                          processor=processor,
                          batch_size=opt.dqn_batch_size,
                          nb_steps_warmup=opt.dqn_nb_steps_warmup,
                          gamma=opt.dqn_gamma,
                          target_model_update=opt.dqn_target_model_update,
                          enable_double_dqn=opt.enable_double_dqn,
                          enable_dueling_network=opt.enable_dueling_network,
                          train_interval=opt.dqn_train_interval,
                          delta_clip=opt.dqn_delta_clip)
    self.agent.compile(optimizer=keras.optimizers.Adam(lr=opt.dqn_learning_rate), metrics=['mae'])

  def fit(self, env, num_steps, weights_path=None, visualize=False):
    callbacks = []
    if weights_path is not None:
      callbacks += [ModelIntervalCheckpoint(weights_path, interval=50000, verbose=1)]
    self.agent.fit(env=env,
                   nb_steps=num_steps,
                   action_repetition=opt.dqn_action_repetition,
                   callbacks=callbacks,
                   log_interval=opt.log_interval,
                   test_interval=opt.test_interval,
                   test_nb_episodes=opt.test_nb_episodes,
                   test_action_repetition=opt.dqn_test_action_repetition,
                   visualize=visualize,
                   test_visualize=False,
                   verbose=2)

  def test(self, env, num_episodes, visualize=False):
    self.agent.test(env=env,
                    nb_episodes=num_episodes,
                    action_repetition=opt.dqn_action_repetition,
                    verbose=2,
                    visualize=visualize)

  def save(self, out_dir):
    self.agent.save_weights(out_dir, overwrite=True)

  def load(self, out_dir):
    self.agent.load_weights(out_dir)