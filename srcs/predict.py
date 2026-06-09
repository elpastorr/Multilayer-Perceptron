import argparse
import numpy as np
from mlp import MLP
from load_data import load_data


def main():
    parser = argparse.ArgumentParser(description="Predict Args Parser")
    parser.add_argument('--model', type=str, default="./model/model.npy")
    parser.add_argument('--test_data', type=str, default="./data/test_data.csv")
    parser.add_argument('--hidlayer', type=int, default=2)

    args = parser.parse_args()
    try:
        model_save = np.load(args.model, allow_pickle=True).item()
    except:
        print("Error: Failed to load", args.model)
        return

    try:
        raw = load_data(args.test_data)
    except:
        print("Error: Failed to load", args.test_data)
        return

    label = raw[0]

    y_one_hot = np.column_stack((label == 'B', label == 'M')).astype(int)
    try:
        data = raw.loc[:, 1:].to_numpy().astype(float)
    except:
        print("Error: Failed to read", args.test_data)
        return
    model = MLP(input_dim=30, hidden_dim=24, output_dim=2, hidlayer=args.hidlayer)

    for i, layer in enumerate(model.layers):
        layer.weights = model_save["layers"][i]["w"]
        layer.biases = model_save["layers"][i]["b"]

    min = np.array(model_save["scaler"]["min"]).flatten()
    max = np.array(model_save["scaler"]["max"]).flatten()

    X_scaled = (data - min) / (max - min + 1e-8)

    probs = model.feedforward(X_scaled)
    predictions = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_one_hot, axis=1)

    accuracy = np.mean(predictions == true_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    true_pos = np.sum((predictions == 1) & (true_labels == 1))
    true_neg = np.sum((predictions == 0) & (true_labels == 0))
    false_pos = np.sum((predictions == 1) & (true_labels == 0))
    false_neg = np.sum((predictions == 0) & (true_labels == 1))
    print(f"True positive: {true_pos}, True negative: {true_neg}, False positive: {false_pos}, False negative: {false_neg}")

    loss = model.compute_loss(predictions, true_labels)
    print("Loss =", loss)


if __name__ == "__main__":
    main()
