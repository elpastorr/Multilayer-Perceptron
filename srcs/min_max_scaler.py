import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class min_max_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def set(self, X, Y):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.y_one_hot = np.column_stack((Y == 'B', Y == 'M')).astype(int)
        return self

    def normalise(self, X):
        # 1e-8 to avoid division by zero
        return (X - self.min) / (self.max - self.min + 1e-8)
