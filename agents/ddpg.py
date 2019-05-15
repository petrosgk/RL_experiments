import options as opt
import keras
from agents.base import BaseAgent
from rl.callbacks import ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.agents.ddpg import DDPGAgent


class DDPG(BaseAgent):
  def __init__(self, actor, critic, critic_action_input, processor, random_process, num_actions):
    # Replay memory
    memory = SequentialMemory(limit=opt.ddpg_replay_memory_size,
                              window_length=opt.ddpg_window_length)
    self.agent = DDPGAgent(actor=actor,
                           critic=critic,
                           critic_action_input=critic_action_input,
                           memory=memory,
                           nb_actions=num_actions,
                           processor=processor,
                           batch_size=opt.ddpg_batch_size,
                           nb_steps_warmup_actor=opt.ddpg_nb_steps_warmup_actor,
                           nb_steps_warmup_critic=opt.ddpg_nb_steps_warmup_critic,
                           target_model_update=opt.ddpg_target_model_update,
                           random_process=random_process,
                           train_interval=opt.ddpg_train_interval)
    self.agent.compile([keras.optimizers.Adam(lr=opt.ddpg_learning_rate_actor),
                        keras.optimizers.Adam(lr=opt.ddpg_learning_rate_critic)],
                       metrics=['mae'])

  def fit(self, env, num_steps, weights_path=None, visualize=False):
    callbacks = []
    if weights_path is not None:
      callbacks += [ModelIntervalCheckpoint(weights_path, interval=50000, verbose=1)]
    self.agent.fit(env=env,
                   nb_steps=num_steps,
                   action_repetition=opt.ddpg_action_repetition,
                   callbacks=callbacks,
                   log_interval=opt.log_interval,
                   test_interval=opt.test_interval,
                   test_nb_episodes=opt.test_nb_episodes,
                   test_action_repetition=opt.ddpg_action_repetition,
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
