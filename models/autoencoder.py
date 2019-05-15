import os
from collections import deque
import numpy as np
from models.vqae_model import model as vqae
import options as opt


class Autoencoder:
  def __init__(self, mode='training', plot_class=None):
    if mode == 'training':
      frame_buffer_size = opt.frame_buffer_size
    else:
      frame_buffer_size = 1
    self.frame_buffer = deque(maxlen=frame_buffer_size)
    self.model = vqae(channels=1 if opt.grayscale else 3,
                      state_vector_length=opt.state_vector_length)
    if os.path.exists(os.path.join('weights', 'state_autoencoder.hdf5')):
      print('Loading pre-trained autoencoder weights...')
      self.model.load_weights(os.path.join('weights', 'state_autoencoder.hdf5'))
    self.last_quantized_inputs = np.zeros(shape=opt.state_vector_length, dtype=np.float32)
    self.last_decoded_inputs = np.zeros(shape=(opt.height, opt.width, 1 if opt.grayscale else 3), dtype=np.float32)

    self.plot_class = plot_class

  def construct_model_inputs(self):
    frames = np.asarray(self.frame_buffer, dtype=np.uint8)
    if opt.grayscale:
      frames = np.expand_dims(frames, axis=-1)
    return frames

  def train(self):
    inputs = self.construct_model_inputs()
    train_history = self.model.fit(inputs, inputs, batch_size=opt.batch_size, epochs=opt.epochs,
                                   shuffle=True, verbose=1)
    self.save_weights()
    if self.plot_class:
      losses = train_history.history['loss']
      for loss in losses:
        self.plot_class.plot_autoencoder_train_loss_buffer.append(loss)

  def save_weights(self):
    self.model.save_weights(os.path.join('weights', 'state_autoencoder.hdf5'))

  def get_outputs(self):
    inputs = self.construct_model_inputs()
    decoded_inputs, quantized_inputs = self.model.predict_on_batch(inputs)
    inputs = np.squeeze(inputs, axis=0)  # Remove batch dimension
    decoded_inputs = np.squeeze(decoded_inputs, axis=0)  # Remove batch dimension
    quantized_inputs = np.squeeze(quantized_inputs, axis=0)  # Remove batch dimension
    quantized_inputs = quantized_inputs.astype(np.uint8)
    loss = self.L2_loss(decoded_inputs, inputs)
    if self.plot_class:
      self.plot_class.plot_autoencoder_test_loss_buffer.append(loss)
    if loss > opt.loss_threshold:
      quantized_inputs = None
    else:
      self.plot_class.state_visit_counts[np.argmax(quantized_inputs)] += 1
      if opt.plot_replay:
        self.plot_class.state_buffer.append(inputs)
        self.plot_class.reconstructed_state_buffer.append((decoded_inputs * 255.0).astype('uint8'))
        self.plot_class.state_index_buffer.append(np.argmax(quantized_inputs))
    return quantized_inputs

  @staticmethod
  def L1_loss(decoded_state, state):
    state = (state / 255.0).astype(np.float32)
    loss = np.mean(np.abs(decoded_state - state))
    return loss

  @staticmethod
  def L2_loss(decoded_state, state):
    state = (state / 255.0).astype(np.float32)
    loss = np.mean(np.square(decoded_state - state))
    return loss
