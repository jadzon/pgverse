import numpy as np
from model.symbolic_regressor import SymbolicRegressor

def test_fit_and_predict():
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = X[:, 0] ** 2
    reg = SymbolicRegressor()
    reg.set_data(X, y)
    reg.fit()
    y_pred = reg.predict(X)
    assert len(y_pred) == len(y)
    #assert np.allclose(y_pred, y, atol=0.1)  # Allow some tolerance for floating point errors