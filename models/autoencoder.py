import os
from collections import deque
import numpy as np
from models.vqae_model import model as vqae
import options as opt


class Autoencoder:
  def __init__(self, plot_class=None):
    frame_buffer_size = opt.frame_buffer_size
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
    train_history = self.model.fit(inputs, inputs, batch_size=opt.batch_size, verbose=0)
    if self.plot_class:
      losses = train_history.history['loss']
      snrs = train_history.history['dec_head_SNR']
      for loss, snr in zip(losses, snrs):
        self.plot_class.plot_autoencoder_train_loss_buffer.append(loss)
        self.plot_class.plot_autoencoder_train_snr_buffer.append(snr)

  def save_weights(self):
    self.model.save_weights(os.path.join('weights', 'state_autoencoder.hdf5'))

  def get_outputs(self, observation):
    inputs = observation
    if opt.grayscale:
      inputs = np.expand_dims(inputs, axis=-1)
    inputs = np.expand_dims(inputs, axis=0)
    decoded_inputs, quantized_inputs = self.model.predict_on_batch(inputs)
    inputs = np.squeeze(inputs, axis=0)  # Remove batch dimension
    decoded_inputs = np.squeeze(decoded_inputs, axis=0)  # Remove batch dimension
    quantized_inputs = np.squeeze(quantized_inputs, axis=0)  # Remove batch dimension
    quantized_inputs = quantized_inputs.astype(np.uint8)
    loss = self.L2_loss(decoded_inputs, inputs)
    snr = self.SNR(decoded_inputs, inputs)
    if self.plot_class:
      self.plot_class.plot_autoencoder_test_loss_buffer.append(loss)
      self.plot_class.plot_autoencoder_test_snr_buffer.append(snr)
    if snr < opt.snr_threshold_db:
      quantized_inputs = None
      decoded_inputs = np.zeros_like(decoded_inputs)
    else:
      self.plot_class.state_visit_counts[np.argmax(quantized_inputs)] += 1
    if opt.plot_replay:
      self.plot_class.state_buffer.append(inputs)
      self.plot_class.reconstructed_state_buffer.append((decoded_inputs * 255.0).astype('uint8'))
      if quantized_inputs is not None:
        state_index = np.argmax(quantized_inputs)
      else:
        state_index = 0
      self.plot_class.state_index_buffer.append(state_index)
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

  @staticmethod
  def SNR(decoded_state, state):
    state = (state / 255.0).astype(np.float32)
    mean_signal = np.mean(state)
    noise = np.abs(decoded_state - state)
    noise_std = np.std(noise)
    snr = 20 * np.log10(mean_signal / noise_std)
    return snr