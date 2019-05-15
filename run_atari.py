import os
from argparse import ArgumentParser
import options as opt
from agents.random import RandomAgent
from models.deep_rl import DRQN_Model, DQN_Model
from agents.dqn import DQN
from agents.tabular_q import TabularQAgent
from environments.atari.processor import AtariProcessor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from models.autoencoder import Autoencoder
import tensorflow as tf
import keras.backend as K
import utility.plot as plot
import warnings
import gym


def parse_clients_args(args_clients):
  """
  Return an array of tuples (ip, port) extracted from ip:port string
  :param args_clients:
  :return:
  """
  return [str.split(str(client), ':') for client in args_clients]


arg_parser = ArgumentParser('Atari experiment')
arg_parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of steps to train for')
arg_parser.add_argument('--agent_mode', choices=['training', 'testing'], default='training',
                        help='Agent training or testing mode')
arg_parser.add_argument('--autoencoder_mode', choices=['none', 'training', 'testing'], default='none',
                        help='Do not use autoencoder or use autoencoder in training or testing mode')
arg_parser.add_argument('--env_name', type=str, default='MontezumaRevenge-v0')
args = arg_parser.parse_args()

steps = args.steps
agent_mode = args.agent_mode
autoencoder_mode = args.autoencoder_mode if agent_mode == 'training' else 'testing'

env = gym.make(args.env_name)

# Setup tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(session=sess)

warnings.simplefilter('ignore')

# Setup plotting
if opt.plot:
  plot_class = plot.Plot(autoencoder_mode=autoencoder_mode)
else:
  plot_class = None

vqae = None
if autoencoder_mode != 'none':
  # Initialize VQAE
  vqae = Autoencoder(mode=autoencoder_mode, plot_class=plot_class)

# Initialize processor
processor = AtariProcessor(autoencoder=vqae,
                           autoencoder_mode=autoencoder_mode,
                           plot_class=plot_class)

if autoencoder_mode == 'training':
  # Use Random agent to train the VQAE
  agent = RandomAgent(num_actions=env.action_space.n, processor=processor)
else:
  # Setup exploration policy
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                attr='eps', value_max=opt.eps_value_max,
                                value_min=opt.eps_value_min,
                                value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)
  if opt.use_quantized_observations:
    agent = TabularQAgent(num_states=opt.state_vector_length,
                          num_actions=env.action_space.n,
                          policy=policy,
                          test_policy=policy,
                          processor=processor)
  else:
    # Setup DQN agent
    if opt.recurrent:
      model = DRQN_Model(window_length=opt.dqn_window_length,
                         grayscale=opt.grayscale,
                         width=opt.width,
                         height=opt.height,
                         num_actions=env.action_space.n)
    else:
      model = DQN_Model(window_length=opt.dqn_window_length,
                        grayscale=opt.grayscale,
                        width=opt.width,
                        height=opt.height,
                        num_actions=env.action_space.n)
    # Setup DQN agent
    agent = DQN(model=model, num_actions=env.action_space.n,
                policy=policy, test_policy=policy, processor=processor)

print(args.env_name + ' initialized.')

# Setup weights path
path = os.path.join('weights', 'Atari', '{}'.format(args.env_name))
if not os.path.exists(path):
  os.makedirs(path)
weights_path = os.path.join(path, 'weights.hdf5')

# Start training or testing agent
if agent_mode == 'training':
    agent.fit(env=env, num_steps=steps, weights_path=weights_path, visualize=False)
    agent.save(weights_path)
else:
    agent.load(weights_path)
    agent.test(env=env, num_episodes=10, visualize=True)
