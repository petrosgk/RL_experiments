import matplotlib.pyplot as plt
from collections import deque
import options as opt
import numpy as np


class Plot(object):
  def __init__(self, plot_vqvae):
    self.reward_buffer = deque()
    self.episode_reward_buffer = deque()

    self.reward_fig = plt.figure(figsize=(5, 5))
    self.reward_fig.show()
    self.reward_fig.canvas.draw()

    if plot_vqvae:
      self.plot_autoencoder_train_loss_buffer = deque(maxlen=1000)
      self.plot_autoencoder_test_loss_buffer = deque(maxlen=1000)
      self.plot_autoencoder_temperature_buffer = deque(maxlen=1000)
      self.state = None
      self.state_visit_counts = None

      self.vqvae_fig = plt.figure(figsize=(10, 10))
      self.vqvae_fig.show()
      self.vqvae_fig.canvas.draw()

      self.img_fig = plt.figure(figsize=(5, 5))
      self.img_fig.show()
      self.img_fig.canvas.draw()

    self.plot_vqvae = plot_vqvae

  def plot(self):
    if self.plot_vqvae:
      ax_0 = self.vqvae_fig.add_subplot(4, 1, 1)
      ax_0.set_title('VQVAE train loss')
      ax_0.plot(self.plot_autoencoder_train_loss_buffer)
      ax_1 = self.vqvae_fig.add_subplot(4, 1, 2)
      ax_1.set_title('VQVAE test loss')
      ax_1.plot(self.plot_autoencoder_test_loss_buffer)
      ax_2 = self.vqvae_fig.add_subplot(4, 1, 3)
      ax_2.set_title('VQVAE softmax temperature')
      ax_2.plot(self.plot_autoencoder_temperature_buffer)
      ax_3 = self.vqvae_fig.add_subplot(4, 1, 4)
      ax_3.set_title('state visit distribution')
      if self.state_visit_counts is not None:
        ax_3.plot(self.state_visit_counts, 'b+')
      self.vqvae_fig.tight_layout()
      self.vqvae_fig.canvas.draw()
      self.vqvae_fig.clear()

      ax_0 = self.img_fig.add_subplot(1, 1, 1)
      ax_0.set_title('reconstructed state')
      if self.state is not None:
        if opt.grayscale:
          state = self.state[:, :, -1:]
          state = np.squeeze(state, axis=-1)
          ax_0.imshow(state, cmap='gray')
        else:
          state = self.state[:, :, -3:]
          ax_0.imshow(state)
      self.img_fig.tight_layout()
      self.img_fig.canvas.draw()
      self.img_fig.clear()

    ax_0 = self.reward_fig.add_subplot(1, 1, 1)
    ax_0.set_title('reward per episode')
    ax_0.plot(self.reward_buffer)
    self.reward_fig.tight_layout()
    self.reward_fig.canvas.draw()
    self.reward_fig.clear()
