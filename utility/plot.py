import matplotlib.pyplot as plt
from collections import deque
import options as opt
import numpy as np
import os
import cv2
import random


class Plot:
  def __init__(self, autoencoder):
    self.autoencoder = autoencoder

    self.reward_buffer = deque(maxlen=1000)
    self.episode_reward_buffer = deque()

    self.reward_fig = plt.figure(figsize=(5, 5))
    self.reward_fig.show()
    self.reward_fig.canvas.draw()

    if self.autoencoder:
        self.plot_autoencoder_train_loss_buffer = deque(maxlen=10000)
        self.plot_autoencoder_train_snr_buffer = deque(maxlen=10000)
        self.plot_autoencoder_test_loss_buffer = deque(maxlen=1000)
        self.plot_autoencoder_test_snr_buffer = deque(maxlen=1000)
        self.state_visit_counts = np.zeros(shape=opt.state_vector_length)
        self.vqae_fig = plt.figure(figsize=(10, 15))
        self.vqae_fig.show()
        self.vqae_fig.canvas.draw()
        if opt.plot_replay:
          self.state_buffer = deque()
          self.reconstructed_state_buffer = deque()
          self.state_index_buffer = deque()
          self.state_coordinates_buffer = deque()
          self.img_fig = plt.figure(figsize=(5, 10))
          self.img_fig.show()
          self.img_fig.canvas.draw()

    self.step = 0

  @staticmethod
  def draw_fig(fig):
    fig.tight_layout()
    fig.canvas.draw()
    fig.clear()

  def plot(self, isTraining=True):
    if not isTraining:
      self.reward_buffer.append(np.sum(self.episode_reward_buffer))
      self.episode_reward_buffer.clear()
      ax_0 = self.reward_fig.add_subplot(1, 1, 1)
      ax_0.set_title('reward per episode')
      ax_0.plot(self.reward_buffer)
      self.draw_fig(self.reward_fig)

    if self.autoencoder:

      ax_0 = self.vqae_fig.add_subplot(5, 1, 1)
      ax_0.set_title('VQ-AE train loss')
      ax_0.plot(self.plot_autoencoder_train_loss_buffer)
      ax_1 = self.vqae_fig.add_subplot(5, 1, 2)
      ax_1.set_title('VQ-AE train SNR (db)')
      ax_1.plot(self.plot_autoencoder_train_snr_buffer)
      ax_2 = self.vqae_fig.add_subplot(5, 1, 3)
      ax_2.set_title('VQ-AE test loss')
      ax_2.plot(self.plot_autoencoder_test_loss_buffer)
      ax_3 = self.vqae_fig.add_subplot(5, 1, 4)
      ax_3.set_title('VQ-AE test SNR (dB)')
      ax_3.plot(self.plot_autoencoder_test_snr_buffer)
      ax_4 = self.vqae_fig.add_subplot(5, 1, 5)
      ax_4.set_title('state visit distribution')
      ax_4.plot(self.state_visit_counts, 'b+')
      self.draw_fig(self.vqae_fig)

      if opt.plot_replay:
        state_coordinates_file = open('state_coordinates.txt', 'w')

        for state, reconstructed_state, state_index, state_coordinates in zip(self.state_buffer,
                                                                              self.reconstructed_state_buffer,
                                                                              self.state_index_buffer,
                                                                              self.state_coordinates_buffer):

          ax_0 = self.img_fig.add_subplot(2, 1, 1)
          ax_0.set_title('frame')
          frame = state
          if opt.grayscale:
            frame = np.squeeze(frame, axis=-1)
            ax_0.imshow(frame, cmap='gray')
          else:
            ax_0.imshow(frame)

          state_coordinates_string = 'state={0:1d}, x={1:2.1f}, z={2:2.1f}, yaw={3:2.1f}'.format(
            state_index,
            state_coordinates[0],
            state_coordinates[1],
            state_coordinates[2])
          state_coordinates_file.write(state_coordinates_string + '\n')

          ax_1 = self.img_fig.add_subplot(2, 1, 2)
          ax_1.set_title('reconstructed_frame')
          ax_1.text(3.0, 5.0, state_coordinates_string, color='g', size=10, bbox=dict(facecolor='white'))
          reconstructed_frame = reconstructed_state
          if opt.grayscale:
            reconstructed_frame = np.squeeze(reconstructed_frame, axis=-1)
            ax_1.imshow(reconstructed_frame, cmap='gray')
          else:
            ax_1.imshow(reconstructed_frame)

          self.draw_fig(self.img_fig)

          frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
          filename = os.path.join('outputs', 'frames', str(state_index) + '.' + str(random.randint(0, 1000)) + '.png')
          cv2.imwrite(filename, frame)
          reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR)
          filename = os.path.join('outputs', 'decoded_frames', str(state_index) + '.' + str(random.randint(0, 1000)) + '.png')
          cv2.imwrite(filename, reconstructed_frame)

        self.state_buffer.clear()
        self.reconstructed_state_buffer.clear()
        self.state_index_buffer.clear()
        self.state_coordinates_buffer.clear()

      with open('train_snr.csv', 'w+') as f:
          for snr in self.plot_autoencoder_train_snr_buffer:
            f.write(str(snr) + '\n')
      with open('test_snr.csv', 'w+') as f:
          for snr in self.plot_autoencoder_test_snr_buffer:
            f.write(str(snr + 7) + '\n')

    self.step += 1
