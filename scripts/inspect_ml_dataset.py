from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = str(ROOT / "data" / "processed" / "rd_dataset.npz")

data = np.load(DATA_FILE)
X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("F range in data:", y[:, 0].min(), y[:, 0].max())
print("k range in data:", y[:, 1].min(), y[:, 1].max())

fig, axes = plt.subplots(3, 4, figsize=(10, 8))
axes = axes.ravel()

for i in range(12):
    axes[i].imshow(X[i, 0], cmap="gray")
    axes[i].set_title(f"F={y[i,0]:.4f}\nk={y[i,1]:.4f}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()