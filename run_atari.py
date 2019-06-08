import os
from argparse import ArgumentParser
import options as opt
from models.deep_rl import DRQN_Model, DQN_Model
from agents.dqn import DQN
from agents.tabular_q import TabularQAgent
from agents.random import RandomAgent
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
arg_parser.add_argument('--agent_type', choices=['DDQN', 'Random'], default='DDQN',
                        help='Set agent type (Double-DQN or Random agent).')
arg_parser.add_argument('--use_vqae', action='store_true', default=False, help='Use VQ-AE.')
arg_parser.add_argument('--env_name', type=str, default='MontezumaRevenge-v0')
arg_parser.add_argument('--visualize', action='store_true', default=False, help='Visualize')
args = arg_parser.parse_args()

env = gym.make(args.env_name)

# Setup tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(session=sess)

warnings.simplefilter('ignore')

# Setup plotting
if opt.plot:
  plot_class = plot.Plot(autoencoder=args.use_vqae)
else:
  plot_class = None

vqae = None
if args.use_vqae:
  # Initialize VQAE
  vqae = Autoencoder(plot_class=plot_class)

# Initialize processor
processor = AtariProcessor(autoencoder=vqae,
                           plot_class=plot_class)

if args.agent_type == 'DDQN':
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
                         num_actions=env.action_space.n)
    else:
      model = DQN_Model(window_length=opt.dqn_window_length,
                        num_actions=env.action_space.n)
    # Setup DQN agent
    agent = DQN(model=model, num_actions=env.action_space.n,
                policy=policy, test_policy=policy, processor=processor)
else:
  agent = RandomAgent(num_actions=env.action_space.n, processor=processor)

print(args.env_name + ' initialized.')

# Setup weights path
path = os.path.join('weights', 'Atari', '{}'.format(args.env_name))
if not os.path.exists(path):
  os.makedirs(path)
weights_path = os.path.join(path, 'weights.hdf5')

# Run the agent
agent.fit(env=env, num_steps=args.steps, weights_path=weights_path, visualize=args.visualize)
agent.save(weights_path)
