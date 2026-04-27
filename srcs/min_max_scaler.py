import numpy as np
from sklearn.base import BaseEstimator, TransgormerMixin

class min_max_scaler(BaseEstimator, TransgormerMixin):
    def __init__(self):
        return
    
    def fit(self, X, Y):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.y = np.column_stack((y == 'B', y == 'M')).astype(int)
        return self
    
    def normalise(self, X):
        # 1e-8 to avoid division by zero
        return (X - self.min) / (self.max - self.min + 1e-8)