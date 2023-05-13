import numpy as np

class NeuralNetwork:
  def __init__(self):
    self.learning_rate = 0.1
    self.weights_ih = np.random.normal(0.0, 0.1, (2, 2))
    self.weights_ho = np.random.normal(0.0, 0.1, (2, 2))
    self.bias_h = np.random.normal(0.0, 0.1, (2, 1))
    self.bias_o = np.random.normal(0.0, 0.1, (2, 1))

  def tanh(self, x):
    return np.tanh(x)

  def derivative_tanh(self, x):
    return 1 - np.tanh(x) ** 2

  def forward(self, input):
    self.hidden = self.tanh(np.dot(self.weights_ih, input) + self.bias_h)
    self.output = self.tanh(np.dot(self.weights_ho, self.hidden) + self.bias_o)
    return self.output

  def backward(self, input, target):
    output_errors = target - self.output
    hidden_errors = np.dot(self.weights_ho.T, output_errors)

    # Delta rule for updating weights and biases
    self.weights_ho += self.learning_rate * np.dot((output_errors * self.derivative_tanh(self.output)), self.hidden.T)
    self.bias_o += self.learning_rate * (output_errors * self.derivative_tanh(self.output))
    self.weights_ih += self.learning_rate * np.dot((hidden_errors * self.derivative_tanh(self.hidden)), input.T)
    self.bias_h += self.learning_rate * (hidden_errors * self.derivative_tanh(self.hidden))

