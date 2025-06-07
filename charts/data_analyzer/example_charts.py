import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Foldery na dane i obrazy
DATA_DIR = "dane"
IMG_DIR = "charts_examples"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Lista funkcji testowych
functions = [
    ("1", lambda X: np.sin(X) + X**2, np.linspace(-3, 3, 100)),
    ("2", lambda X: np.sin(3 * X) + 0.8 * np.random.randn(len(X)), np.linspace(-3, 3, 100)),
    ("3", lambda X: X**2, np.linspace(-2, 2, 100)),
    ("4", lambda X: np.exp(X), np.linspace(0, 3, 100)),
    ("5", lambda X: np.sin(X) + X, np.linspace(-5, 5, 100)),
    ("6", lambda X: np.tan(X), np.linspace(-1.2, 1.2, 100)),
    ("7", lambda X: np.log(X), np.linspace(0.1, 5, 100)),
    ("8", lambda X: (X > 0).astype(float) + 0.1 * np.random.randn(len(X)), np.linspace(-5, 5, 100)),
    ("9", lambda X: np.abs(X), np.linspace(-3, 3, 100)),
    ("10", lambda X: X**3 - 2*X**2 + X + 0.2 * np.random.randn(len(X)), np.linspace(-2, 2, 100)),
    ("11", lambda X: np.sin(X) * np.exp(-X**2), np.linspace(-3, 3, 100)),
    ("12", lambda X: X * np.sin(X**2), np.linspace(-3, 3, 100)),
    ("13", lambda X: (X > 1).astype(float), np.linspace(-2, 2, 100)),
    ("14", lambda X: np.floor(X), np.linspace(-3, 3, 100)),
    ("15", lambda X: np.where(X < 0, X**2, np.sqrt(np.clip(X, 0, None))), np.linspace(-3, 3, 100)),
    ("16", lambda X: np.sin(X) + 0.5 * np.sin(5 * X), np.linspace(0, 10, 100))
]

for idx, (name, func, x_range) in enumerate(functions):
    X = x_range.reshape(-1, 1)
    y = func(X[:, 0])

    # Nazwy plików
    base_name = f"{name}"
    csv_path = os.path.join(DATA_DIR, base_name + ".csv")
    img_path = os.path.join(IMG_DIR, base_name + ".JPG")

    # Zapis do CSV
    df = pd.DataFrame({"x": X[:, 0], "y": y})
    df.to_csv(csv_path, index=False)

    # Wykres
    plt.figure()
    plt.plot(X, y, label="f(x)")
    plt.title(f"{base_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

print("✅ Wygenerowano dane i obrazy.")
