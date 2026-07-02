# MLP Project

## Overview
This project implements a simple Multi-Layer Perceptron (MLP) from scratch in Python, using NumPy for the neural network math and scikit-learn only for dataset splitting and class compatibility.

The main goal is to build, train, save, and evaluate a feedforward neural network for binary classification using a dataset stored in CSV format.

## What the project does
- Loads raw CSV data with `pandas`
- Splits data into training, validation, and test sets
- Normalizes input features with a custom min-max scaler
- Builds an MLP with dense layers and sigmoid/softmax activation
- Trains the network by computing loss and backpropagating gradients
- Saves the trained model weights and scaler values to `model/model.npy`
- Evaluates accuracy and prediction performance on a test dataset
- Plots training and validation loss and accuracy curves

## Core files
- `srcs/mlp.py` - `MLP` class containing the main network implementation
- `srcs/layer.py` - `DenseLayer` implementation with forward pass and weight updates
- `srcs/min_max_scaler.py` - custom scaler for normalizing features and one-hot encoding labels
- `srcs/load_data.py` - simple CSV loader using `pandas`
- `srcs/train.py` - training script that builds the dataset split, scales data, trains the MLP, saves the model, and plots history
- `srcs/predict.py` - prediction script that loads the saved model and test data, computes accuracy, and prints confusion metrics
- `srcs/plot_history.py` - plotting helper for training/validation loss and accuracy curves

## Dependencies
Installed via `requirements.txt`:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `PyQt6` (required by some matplotlib backends)

## Usage
1. Create and activate the virtual environment:
   ```bash
   ./setup_env.sh
   source .venv/bin/activate
   ```

2. Train the model with your dataset:
   ```bash
   python srcs/train.py --dataset ./data/data.csv
   ```
   This will:
   - split the dataset into train/validation/test sets
   - normalize features using min-max scaling
   - train an MLP with 2 hidden layers by default
   - save the trained model to `./model/model.npy`
   - display training history plots

3. Run predictions using the saved model:
   ```bash
   python srcs/predict.py --model ./model/model.npy --test_data ./data/test_data.csv --hidlayer 2
   ```

## Key concepts learned
- **Neural network architecture**: how to stack dense layers and connect them with activation functions.
- **Forward propagation**: computing layer inputs, applying activation functions, and obtaining output probabilities with softmax.
- **Activation functions**:
  - `sigmoid` for hidden layers
  - `softmax` for the output layer in a binary classification setting
- **Loss function**: binary cross-entropy loss for comparing predicted probabilities to true labels.
- **Backpropagation**: deriving gradients through the network and using them to update weights and biases.
- **Gradient descent**: applying a learning rate to move weights in the direction that reduces loss.
- **Data preprocessing**:
  - splitting into training, validation, and test sets
  - normalizing input features with min-max scaling
  - converting labels into one-hot encoded vectors for classification
- **Model persistence**: saving model weights and scaler parameters so predictions can be made later without retraining.
- **Performance tracking**: recording training and validation loss/accuracy and visualizing the learning curves.
- **Practical debugging**: handling numerical stability with clipping before log operations and using small constants to avoid division by zero.

## Notes
- The network currently expects 30 input features and a binary label in the dataset.
- The training loop includes early stopping logic based on validation loss not improving for a few epochs.
- The `MLP` class supports up to 5 hidden layers, but the code recommends using 2 hidden layers for the configured dimensions.
