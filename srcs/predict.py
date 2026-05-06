import numpy as np
from mlp import MLP
from load_data import load_data


def main():
    model_save = np.load("model/model.npy", allow_pickle=True).item()

    try:
        raw = load_data("data/test_data.csv")
    except:
        print("Error: Failed to load data/test_data.csv")
        return

    label = raw[0]

    y_one_hot = np.column_stack((label == 'B', label == 'M')).astype(int)
    data = raw.loc[:, 1:].to_numpy().astype(float)

    model = MLP(input_dim=30, hidden_dim=24, output_dim=2)

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