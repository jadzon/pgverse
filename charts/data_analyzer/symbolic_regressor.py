import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import latex, simplify,Float
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
def round_constants(expr, n_digits=4):
        return expr.xreplace({
            x: Float(round(x, n_digits))
            for x in expr.atoms(Float)
        })
class SymbolicRegressor:
    def __init__(self, niterations=50, maxsize=20):
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "square", "abs", "tanh", "cube" , "sqrt","log2"],
            extra_sympy_mappings={"cube": lambda x: x**3},
            model_selection="best",
            constraints={"abs": 0},
            elementwise_loss="loss(x, y) = (x - y)^2",
            maxsize=maxsize,
            verbosity=0,
            #complexity_penalty=1e-4,
            should_simplify=True,
            random_state=42,
            deterministic=True,
            parallelism="serial",
            temp_equation_file=False,
            output_directory=None
        )
        self.fitted = False
        self.X = None
        self.y = None
        self.latex_formula = ""

    def set_data(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self):
        if self.X is None or self.y is None:
            raise ValueError("Dane wejściowe X i y nie zostały ustawione.")
        self.model.fit(self.X, self.y)
        self.fitted = True
        formula = simplify(self.model.sympy())
        formula = round_constants(formula, n_digits=4)
        self.latex_formula = latex(formula)

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model nie został dopasowany. Wywołaj najpierw metodę `fit()`.")
        return self.model.predict(X)

    def get_formula(self, simplify_result=True, n_digits=4):
        if not self.fitted:
            raise ValueError("Model nie został dopasowany.")
        formula = self.model.sympy()
        if simplify_result:
            formula = simplify(formula)
            formula = round_constants(formula, n_digits=n_digits)
        return formula, latex(formula)
    
    def score(self, X=None, y=None):
        if not self.fitted:
            raise ValueError("Model nie został dopasowany.")

        # Użycie danych treningowych domyślnie
        if X is None or y is None:
            if self.X is None or self.y is None:
                raise ValueError("Dane nie są dostępne.")
            X = self.X
            y = self.y

        y_pred = self.predict(X)

        return {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred)
        }
    def compare_with_csv(self, csv_path):
        if not self.fitted:
            raise ValueError("Model must be trained before comparison.")

        df = pd.read_csv(csv_path)
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError("CSV must contain 'x' and 'y' columns.")

        x_csv = df["x"].values.reshape(-1, 1)
        y_csv = df["y"].values

        y_pred = self.predict(x_csv)

        metrics = {
            "MSE": mean_squared_error(y_csv, y_pred),
            "MAE": mean_absolute_error(y_csv, y_pred),
            "R2": r2_score(y_csv, y_pred)
        }

        print("Porównanie z danymi z CSV:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        return metrics

        

    def plot(self, save_path=None):
            if not self.fitted:
                raise ValueError("Model nie został dopasowany.")
            
            y_pred = self.predict(self.X)

            plt.figure()
            plt.scatter(self.X, self.y, label="Punkty rzeczywiste", alpha=0.5)
            plt.plot(self.X, y_pred, color="red", label="Odwzorowana funkcja")
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Odwzorowanie wzoru z rozszerzonymi operatorami')

            plt.text(
                0.5, 0.9,
                f"${self.latex_formula}$",
                transform=plt.gca().transAxes,
                fontsize=12, color="black",
                ha="center", bbox=dict(facecolor='white', alpha=0.8)
            )

            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
