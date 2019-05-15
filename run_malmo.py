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
arg_parser.add_argument('--ms-per-tick', type=int, default=50,
                        help='Malmo running speed')
arg_parser.add_argument('--clients', default='clients.txt',
                        help='.txt file with client(s) IP addresses')
arg_parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of steps to train for')
arg_parser.add_argument('--action-space', default='discrete',
                        help='Action space to use (discrete, continuous)')
arg_parser.add_argument('--agent_mode', choices=['training', 'testing'], default='training',
                        help='Agent training or testing mode')
arg_parser.add_argument('--autoencoder_mode', choices=['none', 'training', 'testing'], default='none',
                        help='Do not use autoencoder or use autoencoder in training or testing mode')
args = arg_parser.parse_args()

ms_per_tick = args.ms_per_tick
clients = args.clients
steps = args.steps
action_space = args.action_space
agent_mode = args.agent_mode
autoencoder_mode = args.autoencoder_mode if agent_mode == 'training' else 'testing'

mission = Rooms(ms_per_tick)
mission_agent_names = mission.agent_names
mission_name = mission.mission_name
mission_xml = mission.mission_xml

with open(clients, 'r') as f:
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
      plot_class = plot.Plot(autoencoder_mode=autoencoder_mode)
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
    if autoencoder_mode != 'none':
      # Initialize VQAE
      vqae = Autoencoder(mode=autoencoder_mode, plot_class=plot_class)

    # Initialize processor
    processor = MalmoProcessor(autoencoder=vqae,
                               autoencoder_mode=autoencoder_mode,
                               plot_class=plot_class,
                               action_space=action_space)

    if autoencoder_mode == 'training':
      # Use Random agent to train the VQAE
      agent = RandomAgent(num_actions=env.available_actions, processor=processor)
    else:
      if action_space == 'discrete' :
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
                               grayscale=opt.grayscale,
                               width=opt.width,
                               height=opt.height,
                               num_actions=env.available_actions)
          else:
            model = DQN_Model(window_length=opt.dqn_window_length,
                              grayscale=opt.grayscale,
                              width=opt.width,
                              height=opt.height,
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
                                                 grayscale=opt.grayscale,
                                                 width=opt.width,
                                                 height=opt.height,
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

    # Start training or testing agent
    if agent_mode == 'training':
        agent.fit(env=env, num_steps=steps, weights_path=weights_path, visualize=False)
        agent.save(weights_path)
    else:
        agent.load(weights_path)
        agent.test(env=env, num_episodes=10, visualize=False)
