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

    def fit(self, X_train, Y_train, X_val, Y_val, epochs = 50, learning_rate = 0.5, break_count = 2, min_delta = 0.0001):
        self.batch_size= X_train.shape[0]
        self.best_val_loss = 100
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        count = 0

        for epoch in range(epochs):
            # 1 - Forward Pass
            predictions = self.feedforward(X_train)

            # 2 - Compute Loss
            loss = self.compute_loss(predictions, Y_train)
            acc = self.compute_accuracy(predictions, Y_train)

            # 3 - Back Propagation + Update weights
            self.back_prop(X_train, Y_train, learning_rate)

            # 4 - Validation with unseen data




    def feedforward(self, X):
        activation = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(activation)

            if i == len(self.layers) - 1:
                activation = self.softmax(z) # last layer
            else:
                activation = self.sigmoid(z) # hidden layers

            layer.output = activation
        return activation

    def compute_loss(self, predictions, Y_true):
        predictions = np.clip(predictions, 1e-15, 1) # cliping to prevent undefined log(0)
        # Binary Cross Entropy Loss formula : -1/N * (Y * log(P) + (1 - Y) * log(1 - P))
        return -np.mean(Y_true * np.log(predictions) + (1 - Y_true) * np.log(1 - predictions))

    def compute_accuracy(self, predictions, Y_true):
        pred_table = np.argmax(predictions, axis = 1)
        true_table = np.argmax(Y_true, axis = 1)

        return np.mean(pred_table == true_table)

    def back_prop(self, X, Y_true, learning_rate):
        output_layer = self.layers[-1]
        prev_layer_output = self.layers[-2].output

        dZ = output_layer.output - Y_true

        dW = (prev_layer_output.T @ dZ) / self.batch_size
        db = np.mean(dZ, axis = 0, keepdims = True)

        # dA_prev = dZ @ output_layer.weights.T

        output_layer.update(dW, db, learning_rate)










    def softmax(self, z):
        exp_Z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))