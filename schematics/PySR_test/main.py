import numpy as np
from model.symbolic_regressor import SymbolicRegressor

if __name__ == "__main__":
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X[:, 0]) + X[:, 0] ** 2

    reg = SymbolicRegressor()
    reg.set_data(X, y)
    reg.fit()

    formula, latex_formula = reg.get_formula()
    print("Najlepszy wzór:", formula)
    reg.plot()