import os
from argparse import ArgumentParser
import options as opt
from environments.malmo.missions.rooms import Rooms, RoomsEnvironment, RoomsStateBuilder
from agents.random import RandomAgent
from models.deep_rl import DRQN_Model, DQN_Model, DDPG_Model
from agents.dqn import DQN
from agents.ddpg import DDPG
from agents.tabular_q import TabularQAgent
from environments.malmo.processor import MalmoProcessor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.random import GaussianWhiteNoiseProcess
from models.autoencoder import Autoencoder
import tensorflow as tf
import keras.backend as K
import utility.plot as plot
import warnings


def parse_clients_args(args_clients):
  """
  Return an array of tuples (ip, port) extracted from ip:port string
  :param args_clients:
  :return:
  """
  return [str.split(str(client), ':') for client in args_clients]


arg_parser = ArgumentParser('Malmo experiment')
arg_parser.add_argument('--ms_per_tick', type=int, default=50,
                        help='Malmo running speed')
arg_parser.add_argument('--clients', default='clients.txt',
                        help='.txt file with client(s) IP addresses')
arg_parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of steps to train for')
arg_parser.add_argument('--agent_type', choices=['DDQN, DDPG, Random'], default='DDQN',
                        help='Set agent type (Double-DQN, DDPG, Random).')
arg_parser.add_argument('--use_vqae', action='store_true', default=False, help='Use VQ-AE.')
args = arg_parser.parse_args()

mission = Rooms(args.ms_per_tick)
mission_agent_names = mission.agent_names
mission_name = mission.mission_name
mission_xml = mission.mission_xml
if args.agent_type == 'DDQN' or args.agent_type == 'Random':
  action_space = 'discrete'
else:
  action_space = 'continuous'

with open(args.clients, 'r') as f:
  clients = f.read().splitlines()
print('Clients: {}'.format(clients))
assert len(clients) >= len(mission_agent_names), '1 Malmo client for each agent must be specified in clients.txt'
clients = parse_clients_args(clients)

# Setup tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(session=sess)

warnings.simplefilter('ignore')

for e, name in enumerate(mission_agent_names):
    # Setup plotting
    if opt.plot:
      plot_class = plot.Plot(autoencoder=args.use_vqae)
    else:
      plot_class = None

    # Setup recording
    recording_dir = os.path.join('records', '{}'.format(mission_name))
    if not os.path.exists(recording_dir):
      os.makedirs(recording_dir)
    recording_path = os.path.join(recording_dir, '{}.tgz'.format(name))

    # Setup environment and environment state builder
    state_builder = RoomsStateBuilder(width=opt.width, height=opt.height, grayscale=opt.grayscale)
    env = RoomsEnvironment(action_space=action_space, mission_name=mission_name, mission_xml=mission_xml,
                           remotes=clients, state_builder=state_builder, role=e, recording_path=None)

    vqae = None
    if args.use_vqae:
      # Initialize VQAE
      vqae = Autoencoder(plot_class=plot_class)

    # Initialize processor
    processor = MalmoProcessor(autoencoder=vqae,
                               plot_class=plot_class,
                               action_space=action_space)

    if args.agent_type == 'Random':
      # Use Random agent to train the VQAE
      agent = RandomAgent(num_actions=env.available_actions, processor=processor)
    elif args.agent_type == 'DDQN':
      # Setup exploration policy
      policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                    attr='eps', value_max=opt.eps_value_max,
                                    value_min=opt.eps_value_min,
                                    value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)
      if opt.use_quantized_observations:
        agent = TabularQAgent(num_states=opt.state_vector_length,
                              num_actions=env.available_actions,
                              policy=policy,
                              test_policy=policy,
                              processor=processor)
      else:
        # Setup DQN agent
        if opt.recurrent:
          model = DRQN_Model(window_length=opt.dqn_window_length,
                             num_actions=env.available_actions)
        else:
          model = DQN_Model(window_length=opt.dqn_window_length,
                            num_actions=env.available_actions)
        # Setup DQN agent
        agent = DQN(model=model, num_actions=env.available_actions,
                    policy=policy, test_policy=policy, processor=processor)
    else:
      assert not opt.recurrent
      # Setup random process for exploration
      random_process = [GaussianWhiteNoiseProcess(sigma=0.0, mu=1.0),
                        GaussianWhiteNoiseProcess(sigma=1.0, mu=0.0)]
      # Setup DDPG agent model
      actor, critic, action_input = DDPG_Model(window_length=opt.ddpg_window_length,
                                               num_actions=env.available_actions)
      # Setup DDPG agent
      agent = DDPG(actor=actor, critic=critic, critic_action_input=action_input,
                   num_actions=env.available_actions, processor=processor,
                   random_process=random_process)

    print(mission_name + ' initialized.')

    # Setup weights path
    path = os.path.join('weights', 'Malmo', '{}'.format(mission_name))
    if not os.path.exists(path):
      os.makedirs(path)
    weights_path = os.path.join(path, '{}.hdf5'.format(name))

    # Run the agent
    agent.fit(env=env, num_steps=args.steps, weights_path=weights_path, visualize=False)
    agent.save(weights_path)
