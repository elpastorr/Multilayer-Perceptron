import numpy as np
from mlp import MLP
from load_data import load_data


def main():
    try:
        raw = load_data("data/test_data.csv")
    except:
        print("Error: Failed to load data/test_data.csv")
        return


if __name__ == "__main__":
    main()