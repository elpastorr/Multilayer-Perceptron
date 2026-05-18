import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # init weights w/ small random value
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

        self.input = None
        self.z = None
        self.output = None

    def forward(self, inputs):
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.z
    
    def update(self, dw, db, learning_rate):
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
