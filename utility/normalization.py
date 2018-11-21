import numpy as np


class FeatureNormalization(object):
  def __init__(self, width, height, sequence_length, channels):
    self.width = width
    self.height = height
    self.sum = np.zeros(shape=(width, height, sequence_length * channels))
    self.squared_sum = np.zeros(shape=(width, height, sequence_length * channels))
    self.mean = None
    self.std = None
    self.count = 0

  def update_statistics(self, data):
    # Update sum(X) and sum(X^2)
    self.sum = np.add(self.sum, data)
    self.squared_sum = np.add(self.squared_sum, np.square(data))
    self.count += 1
    # Update mean = E[X]
    self.mean = self.sum / self.count
    # Update std = sqrt(E[X^2] - E[X]^2)
    mean_of_squares = self.squared_sum / self.count
    self.std = np.sqrt(mean_of_squares - np.square(self.mean) + 1e-8)

  def normalize(self, data):
    # Update running statistics
    self.update_statistics(data)
    # Normalize to zero mean and unit variance
    normalized_data = (data - self.mean) / self.std
    return normalized_data.astype(np.float32)

  def denormalize(self, normalized_data):
    data = (normalized_data * self.std) + self.mean
    return data.astype(np.float32)