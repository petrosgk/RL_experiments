# General options
plot_frequence = 10

# State properties
width, height, grayscale = 84, 84, False

# Random agent properties
random_action_repetition = 1

# DQN agent properties
dqn_window_length = 4
dqn_action_repetition = 1
dqn_train_interval = 4
dqn_replay_memory_size = 1000000
eps_value_max = 1.0
eps_value_min = 0.05
eps_value_test = 0.005
eps_decay_steps = 1000000
dqn_batch_size = 32
dqn_learning_rate = .00025
dqn_nb_steps_warmup = 50000
dqn_gamma = .99
dqn_delta_clip = 1.0
dqn_target_model_update = 10000
enable_double_dqn = False
enable_dueling_network = False

# Training options
visualize = True
test_interval = 0

# Exploration policy options
state_vector_length = 32
state_matrix_features = 32
epochs = 10
batch_size = 32
steps_warmup = 5000
learning_rate = 1e-3
temperature_per_step = [[0, 1.0], [50000, 0.1]]
losses_buffer_size = 10