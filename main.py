import numpy as np

class NeuralNetwork:
  def __init__(self):
    self.learning_rate = 0.1
    self.weights_ih = np.random.normal(0.0, 0.1, (2, 2))
    self.weights_ho = np.random.normal(0.0, 0.1, (2, 2))
    self.bias_h = np.random.normal(0.0, 0.1, (2, 1))
    self.bias_o = np.random.normal(0.0, 0.1, (2, 1))

