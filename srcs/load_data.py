import pandas as pd


def load_data(path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)

    return raw