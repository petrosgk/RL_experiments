import os
from argparse import ArgumentParser
import options as opt
from environments.malmo.missions.rooms import Rooms, RoomsEnvironment, RoomsStateBuilder
from environments.malmo.missions.classroom import Classroom, ClassroomEnvironment, ClassroomStateBuilder
from agents.dqn import DQN
from environments.malmo.processor import MalmoProcessor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from policies.seek_novelty import SeekNovelStatesPolicy
from models.state_autoencoder import model as state_model
import tensorflow as tf
import keras.backend as K
import utility.plot as plot

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
arg_parser.add_argument('--exploration-policy', type=str, default='e-greedy',
                        choices=['e-greedy', 'seek-novel'])
arg_parser.add_argument('--action-space', default='discrete',
                        help='Action space to use (discrete, continuous)')
arg_parser.add_argument('--mode', default='training',
                        help='Training or testing mode')
args = arg_parser.parse_args()

ms_per_tick = args.ms_per_tick
clients = args.clients
steps = args.steps
action_space = args.action_space
mode = args.mode

mission = Rooms(ms_per_tick)
mission_agent_names = mission.agent_names
mission_name = mission.mission_name
mission_xml = mission.mission_xml

clients = open(clients, 'r').read().splitlines()
print('Clients: {}'.format(clients))
assert len(clients) >= len(mission_agent_names), '1 Malmo client for each agent must be specified in clients.txt'
clients = parse_clients_args(clients)

# Setup tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(session=sess)

for e, name in enumerate(mission_agent_names):
    # Setup plotting
    if args.exploration_policy == 'seek-novel':
      plot_vqvae = True
    else:
      plot_vqvae = False
    if opt.plot_frequence > 0:
      plot_class = plot.Plot(plot_vqvae=plot_vqvae)
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
                           remotes=clients, state_builder=state_builder, role=e, recording_path=recording_path)

    # Initialize processor
    processor = MalmoProcessor(abs_max_reward=env.abs_max_reward, plot_class=plot_class)

    # Setup exploration policy
    if args.exploration_policy == 'seek-novel':
      autoencoder = state_model(channels=1 if opt.grayscale else 3,
                                sequence_length=opt.dqn_window_length,
                                state_vector_length=opt.state_vector_length,
                                state_matrix_features=opt.state_matrix_features,
                                temperature_per_step=opt.temperature_per_step)
      policy = LinearAnnealedPolicy(SeekNovelStatesPolicy(num_actions=env.available_actions,
                                                          autoencoder=autoencoder,
                                                          plot_class=plot_class),
                                    attr='eps', value_max=opt.eps_value_max, value_min=opt.eps_value_min,
                                    value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)
    else:
      policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                    attr='eps', value_max=opt.eps_value_max,
                                    value_min=opt.eps_value_min,
                                    value_test=opt.eps_value_test, nb_steps=opt.eps_decay_steps)

    # Setup agent
    agent = DQN(num_actions=env.available_actions,
                policy=policy, test_policy=policy, processor=processor)

    print(mission_name + ' initialized.')

    # Setup weights path
    path = os.path.join('weights', 'Malmo', '{}'.format(mission_name))
    if not os.path.exists(path):
      os.makedirs(path)
    weights_path = os.path.join(path, '{}.hdf5'.format(name))

    # Start training or testing agent
    if mode == 'training':
        agent.fit(env=env, num_steps=steps, weights_path=weights_path, visualize=False)
        agent.save(weights_path)
    else:
        agent.load(weights_path)
        agent.test(env=env, num_episodes=10, visualize=False)
