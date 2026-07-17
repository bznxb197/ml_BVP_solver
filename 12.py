import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.generate_dataset_V4 import build_bspline_basis

df = pd.read_csv("/Users/aleksandrcernyh/numeric_ML/ml_BVP_solver/dataset_V4.csv")
Y_raw = df.iloc[:, 25:].values
mask = (np.abs(Y_raw) < 10).all(axis=1)
df = df[mask]

x = np.linspace(0, 1, 500)
Phi = build_bspline_basis(x)

# Рисуем 12 случайных решений из датасета
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
idx = np.random.choice(len(df), 12, replace=False)

for ax, i in zip(axes.flat, idx):
    coeffs = df.iloc[i, 25:].values
    y = Phi @ coeffs
    eps = 10**df.iloc[i, 0]
    ax.plot(x, y)
    ax.set_title(f"eps={eps:.2e}", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dataset_samples.png", dpi=100)
print("Сохранено: dataset_samples.png")