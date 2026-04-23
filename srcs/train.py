import argparse
from load_data import load_data


def main():
    parser = argparse.ArgumentParser(description="EEG Data Parser")
    parser.add_argument('--dataset', type = str, required=True)

    args = parser.parse_args()
    try:
        raw = load_data(args.dataset)
    except:
        print("Error: Failed to load", args.dataset)
        return

    labels = raw[1]
    data = raw.loc[:, 2:]




if __name__ == "__main__":
    main()