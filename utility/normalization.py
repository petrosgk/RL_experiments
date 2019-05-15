import numpy as np


class Normalization(object):
  def __init__(self):
    self.sum = 0
    self.squared_sum = 0
    self.mean = 1.0
    self.std = 1.0
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

  def normalize_mean(self, data):
    # Normalize by dividing with the running mean BEFORE updating mean with current data
    normalized_data = data / self.mean
    # Update running statistics
    self.update_statistics(data)
    return normalized_data.astype(np.float32)

  def normalize_zmuv(self, data):
    # Update running statistics
    self.update_statistics(data)
    # Normalize to zero mean and unit variance
    normalized_data = (data - self.mean) / self.std
    return normalized_data.astype(np.float32)

  def denormalize(self, normalized_data):
    data = (normalized_data * self.std) + self.mean
    return data.astype(np.float32)