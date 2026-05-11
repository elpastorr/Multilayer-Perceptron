import numpy as np
import os
from layer import DenseLayer


class MLP:
    def __init__(self, input_dim=30, hidden_dim=24, output_dim=2, batch_size=8):
        self.batch_size = batch_size

        self.layers = [
            DenseLayer(input_dim, hidden_dim),
            DenseLayer(hidden_dim, hidden_dim),
            DenseLayer(hidden_dim, output_dim)
        ]

    def train(self, X_train, Y_train, X_val, Y_val, epochs, learning_rate, break_count=2, min_delta=0.0001):
        self.batch_size= X_train.shape[0]
        self.best_val_loss = 100
        history = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": []}
        count = 0

        for epoch in range(epochs):
            # Forward Pass
            predictions = self.feedforward(X_train)

            # Compute Loss
            loss = self.compute_loss(predictions, Y_train)
            acc = self.compute_accuracy(predictions, Y_train)

            # Back Propagation + Update weights
            self.back_prop(X_train, Y_train, learning_rate)

            # Validation with unseen data
            preds_val = self.feedforward(X_val)
            # print(X_val)
            # print(preds_val)
            loss_val = self.compute_loss(preds_val, Y_val)
            acc_val = self.compute_accuracy(preds_val, Y_val)

            history["loss_train"].append(loss)
            history["loss_val"].append(loss_val)
            history["acc_train"].append(acc)
            history["acc_val"].append(acc_val)

            if loss_val < (self.best_val_loss - min_delta):
                self.best_val_loss = loss_val
                count = 0
            else:
                count += 1
                if count >= break_count:
                    print(f"Epoch: {epoch}: Loss {loss:.3f} - Accuracy {acc:.2%}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}: Loss {loss:.3f} - Accuracy {acc:.2%}")
        
        return history


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
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15) # cliping to prevent undefined log(0)
        # Binary Cross Entropy Loss formula : -1/N * (Y * log(P) + (1 - Y) * log(1 - P))
        return -np.mean(Y_true * np.log(predictions) + (1 - Y_true) * np.log(1 - predictions))

    def compute_accuracy(self, predictions, Y_true):
        pred_table = np.argmax(predictions, axis=1)
        true_table = np.argmax(Y_true, axis=1)

        return np.mean(pred_table == true_table)

    def back_prop(self, X, Y_true, learning_rate):
        output_layer = self.layers[-1]
        prev_layer_output = self.layers[-2].output

        dZ = output_layer.output - Y_true

        dW = (prev_layer_output.T @ dZ) / self.batch_size
        db = np.mean(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ output_layer.weights.T

        output_layer.update(dW, db, learning_rate)

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]

            if i > 0:
                input_to_layer = self.layers[i - 1].output
            else:
                input_to_layer = X
        
            dZ_hidden = dA_prev * (layer.output * (1 - layer.output))

            dW_hidden = (input_to_layer.T @ dZ_hidden) / self.batch_size
            db_hidden = np.mean(dZ_hidden, axis=0, keepdims=True)

            dA_prev = dZ_hidden @ layer.weights.T

            layer.update(dW_hidden, db_hidden, learning_rate)

    def save_model(self, scaler, filename):
        model = {"layers": [{"w": layer.weights, "b": layer.biases} for layer in self.layers],
                 "scaler": {"min": scaler.min, "max": scaler.max}}

        if not os.path.isdir("model"):
            os.makedirs("model")
        np.save(filename, model)
        print("Model saved to", filename)

    def softmax(self, z):
        exp_Z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))