import numpy as np

# General options
plot = True
plot_replay = False
recurrent = False  # Use recurrent NN

# State properties
width, height, grayscale = 84, 84, False

# Tabular-Q agent properties
tabular_q_nb_steps_warmup = 1000
tabular_q_learning_rate = 0.1
tabular_q_gamma = 0.9

# DQN agent properties
dqn_window_length = 4
dqn_action_repetition = 4
dqn_test_action_repetition = 4
dqn_train_interval = 4
dqn_replay_memory_size = 100000
eps_value_max = 1.0
eps_value_min = 0.05
eps_value_test = 0.05
eps_decay_steps = 1000000
dqn_batch_size = 32
dqn_learning_rate = 0.0025
dqn_nb_steps_warmup = 10000
dqn_gamma = .99
dqn_delta_clip = np.inf
dqn_target_model_update = 10000
enable_double_dqn = True
enable_dueling_network = True

# DDPG agent properties
ddpg_window_length = 4
ddpg_action_repetition = 4
ddpg_train_interval = 4
ddpg_replay_memory_size = 100000
ddpg_learning_rate_actor = 1e-4
ddpg_learning_rate_critic = 1e-3
ddpg_batch_size = 16
ddpg_nb_steps_warmup_actor = 1000
ddpg_nb_steps_warmup_critic = 1000
ddpg_gamma = .99
ddpg_target_model_update = 1e-3

# Training options
visualize = False
log_interval = 10000
test_interval = 10000
test_nb_episodes = 5

# Exploration policy options
state_vector_length = 4096  # Must be >= 64
epochs = 25
batch_size = 256
frame_buffer_size = 100000
learning_rate = 1e-4
loss_threshold = 0.01
beta = 1.0
use_quantized_observations = False
use_intrinsic_rewards = True
