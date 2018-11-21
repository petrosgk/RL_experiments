import os
from argparse import ArgumentParser
import options as opt
from agents.dqn import DQN
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from policies.seek_novelty import SeekNovelStatesPolicy
from models.state_autoencoder import model as state_model
from environments.atari.processor import AtariProcessor
import gym
import tensorflow as tf
import keras.backend as K
import utility.plot as plot

arg_parser = ArgumentParser('Atari experiment')

arg_parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of steps to train for')
arg_parser.add_argument('--exploration-policy', type=str, default='e-greedy',
                        choices=['e-greedy', 'seek-novel'])
arg_parser.add_argument('--env-name', type=str, default='Breakout-v0')
arg_parser.add_argument('--mode', default='training',
                        choices=['training', 'testing'],
                        help='Training or testing mode')
args = arg_parser.parse_args()

# Setup tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(session=sess)

env = gym.make(args.env_name)

# Setup plotting
if args.exploration_policy == 'seek-novel':
  plot_vqvae = True
else:
  plot_vqvae = False
if opt.plot_frequence > 0:
  plot_class = plot.Plot(plot_vqvae=plot_vqvae)
else:
  plot_class = None

# Setup processor
processor = AtariProcessor(plot_class=plot_class)

# Setup exploration policy
if args.exploration_policy == 'seek-novel':
  autoencoder = state_model(channels=1 if opt.grayscale else 3,
                            sequence_length=opt.dqn_window_length,
                            state_vector_length=opt.state_vector_length,
                            state_matrix_features=opt.state_matrix_features,
                            temperature_per_step=opt.temperature_per_step)
  policy = LinearAnnealedPolicy(SeekNovelStatesPolicy(num_actions=env.action_space.n,
                                                      autoencoder=autoencoder,
                                                      plot_class=plot_class),
                                attr='eps', value_max=opt.eps_value_max,
                                value_min=opt.eps_value_min,
                                value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)
else:
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                attr='eps', value_max=opt.eps_value_max,
                                value_min=opt.eps_value_min,
                                value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)

# Setup agent
agent = DQN(num_actions=env.action_space.n,
            policy=policy, test_policy=policy, processor=processor)

print(args.env_name + ' initialized.')

# Setup weights path
path = os.path.join('weights', 'Atari', '{}'.format(args.env_name))
if not os.path.exists(path):
  os.makedirs(path)
weights_path = os.path.join(path, 'weights.hdf5')

# Start training or testing agent
if args.mode == 'training':
  agent.fit(env=env, num_steps=args.steps, weights_path=weights_path, visualize=opt.visualize)
  agent.save(weights_path)
else:
  agent.load(weights_path)
  agent.test(env=env, num_episodes=10, visualize=True)