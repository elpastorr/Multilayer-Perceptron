import argparse
import pandas as pd
import numpy as np
from load_data import load_data
from sklearn.model_selection import train_test_split
from plot_history import plot_history
from min_max_scaler import min_max_scaler
from mlp import MLP


def main():
    parser = argparse.ArgumentParser(description="Train Args Parser")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_data', type=str, default="./data/test_data.csv")
    parser.add_argument('--hidlayer', type=int, default=2)

    args = parser.parse_args()
    try:
        raw = load_data(args.dataset)
    except:
        print("Error: Failed to load", args.dataset)
        return

    labels = raw[1]
    data = raw.loc[:, 2:]

    X_train, X_temp, Y_train, Y_temp = train_test_split(data, labels, stratify=labels, random_state=42, test_size=0.4)
    
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, stratify=Y_temp, random_state=42, test_size=0.5)

    test_dataset = pd.concat([Y_test, X_test], axis=1)

    test_dataset.to_csv(args.test_data, index=False, header=False)
    print("Test dataset saved to", args.test_data)

    scaler = min_max_scaler()
    scaler.set(X_train, Y_train)

    X_train_scaled = scaler.normalise(X_train)

    X_val_scaled = scaler.normalise(X_validation)

    Y_train_oh = scaler.y_one_hot

    Y_val_oh = np.column_stack((Y_validation == 'B', Y_validation == 'M')).astype(int)

    model = MLP(input_dim=30, hidden_dim=24, output_dim=2, batch_size=len(X_train_scaled), hidlayer=args.hidlayer)

    history = model.train(X_train_scaled, Y_train_oh, X_val_scaled, Y_val_oh, epochs=1000, learning_rate=0.4)

    model.save_model(scaler, "./model/model.npy")

    plot_history(history)


if __name__ == "__main__":
    main()
