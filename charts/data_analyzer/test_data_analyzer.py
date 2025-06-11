import numpy as np
from charts.data_analyzer.symbolic_regressor import SymbolicRegressor


if __name__ == "__main__":
    #0 – Przykład: sin(x) + x^2
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.sin(X[:, 0]) + X[:, 0] ** 2
    
    # #1 – Przeregulowanie: sin z dużym szumem
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.sin(3 * X[:, 0]) + 0.8 * np.random.randn(100)

    # #2 – Prosta funkcja kwadratowa
    # X = np.linspace(-2, 2, 100).reshape(-1, 1)
    # y = X[:, 0] ** 2

    # #3 – Funkcja wykładnicza
    # X = np.linspace(0, 3, 100).reshape(-1, 1)
    # y = np.exp(X[:, 0])

    # #4 – Sygnał typu sin(x) + x
    # X = np.linspace(-5, 5, 100).reshape(-1, 1)
    # y = np.sin(X[:, 0]) + X[:, 0]

    # #5 – Funkcja tangens (ograniczony zakres) !!!
    # X = np.linspace(-1.2, 1.2, 100).reshape(-1, 1)
    # y = np.tan(X[:, 0])

    # #6 – Funkcja logarytmiczna (dla X > 0)
    # X = np.linspace(0.1, 5, 100).reshape(-1, 1)
    # y = np.log(X[:, 0])

    # #7 – Skok jednostkowy + szum ?!
    # X = np.linspace(-5, 5, 100).reshape(-1, 1)
    # y = (X[:, 0] > 0).astype(float) + 0.1 * np.random.randn(100)

    # #8 – Funkcja absolutna
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.abs(X[:, 0])

    # #9 – Wielomian 3. stopnia z szumem !!
    # X = np.linspace(-2, 2, 100).reshape(-1, 1)
    # y = X[:, 0] ** 3 - 2 * X[:, 0] ** 2 + X[:, 0] + 0.2 * np.random.randn(100)

    # #10 – Złożona: sin(x) * exp(-x^2)
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.sin(X[:, 0]) * np.exp(-X[:, 0] ** 2)

    # #11 – Silna nieliniowość: x * sin(x^2) !
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = X[:, 0] * np.sin(X[:, 0] ** 2)

    # #12 – Stopniowanie funkcji logicznej: X > 1 ?!
    # X = np.linspace(-2, 2, 100).reshape(-1, 1)
    # y = (X[:, 0] > 1).astype(float)

    # #13 – Sygnał schodkowy ?!!
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.floor(X[:, 0])

    # #14 – Funkcja złożona z ifów
    # X = np.linspace(-3, 3, 100).reshape(-1, 1)
    # y = np.where(X[:, 0] < 0, X[:, 0] ** 2, np.sqrt(X[:, 0]))

    # #15 – Składana sinusoida
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X[:, 0]) + 0.5 * np.sin(5 * X[:, 0])
    reg = SymbolicRegressor()
    reg.set_data(X, y)
    reg.fit()

    formula, latex_formula = reg.get_formula()
    print("Najlepszy wzór:", formula)
    reg.plot()