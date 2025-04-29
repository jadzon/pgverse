import numpy as np

def generate_default_data():
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X[:, 0]) + X[:, 0] ** 2
    return X, y
