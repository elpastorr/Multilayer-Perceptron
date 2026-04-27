import numpy as np
from layer import DenseLayer

class MLP:
    def __init__(self, input_dim = 30, hidden_dim = 24, output_dim = 2, batch_size = 8):
        self.batch_size = batch_size

        self.layers = [
            DenseLayer(input_dim, hidden_dim),
            DenseLayer(hidden_dim, hidden_dim),
            DenseLayer(hidden_dim, output_dim)
        ]

    def fit(self, )