import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "data/processed/rd_dataset.npz"
MODEL_FILE = "models/cnn_regressor_best.pt"
OUTPUT_DIR = "models/eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EXAMPLES = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# Simulation settings
# -----------------------------
N = 200
Du = 0.16
Dv = 0.08
steps = 2500
dt = 1.0
sim_seed = 123
check_interval = 100
stability_threshold = 1e-5


# -----------------------------
# Dataset
# -----------------------------
class ReactionDiffusionDataset(Dataset):
    def __init__(self, data_file: str):
        data = np.load(data_file)

        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)

        self.F_min = float(data["F_min"])
        self.F_max = float(data["F_max"])
        self.K_min = float(data["K_min"])
        self.K_max = float(data["K_max"])

        self.y_norm = np.zeros_like(self.y, dtype=np.float32)
        self.y_norm[:, 0] = (self.y[:, 0] - self.F_min) / (self.F_max - self.F_min)
        self.y_norm[:, 1] = (self.y[:, 1] - self.K_min) / (self.K_max - self.K_min)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y_norm[idx], dtype=torch.float32)
        return x, y

    def denormalize_labels(self, y_norm: np.ndarray) -> np.ndarray:
        y = np.zeros_like(y_norm, dtype=np.float32)
        y[:, 0] = y_norm[:, 0] * (self.F_max - self.F_min) + self.F_min
        y[:, 1] = y_norm[:, 1] * (self.K_max - self.K_min) + self.K_min
        return y


# -----------------------------
# Model
# -----------------------------
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


# -----------------------------
# RD simulator
# -----------------------------
def laplacian(Z: np.ndarray) -> np.ndarray:
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )


def initialize_grid(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    U = np.ones((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    r = 20
    c = n // 2
    U[c-r:c+r, c-r:c+r] = 0.50
    V[c-r:c+r, c-r:c+r] = 0.25

    noise = 0.02
    U += noise * rng.random((n, n))
    V += noise * rng.random((n, n))

    U = np.clip(U, 0, 1)
    V = np.clip(V, 0, 1)
    return U, V


def simulate(Du: float, Dv: float, F: float, k: float, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, V = initialize_grid(N, rng)

    prev_V = V.copy()

    for i in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        uvv = U * V * V

        U += (Du * Lu - uvv + F * (1 - U)) * dt
        V += (Dv * Lv + uvv - (F + k) * V) * dt

        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

        if i % check_interval == 0 and i > 0:
            diff = np.mean(np.abs(V - prev_V))
            if diff < stability_threshold:
                break
            prev_V = V.copy()

    return V.astype(np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = ReactionDiffusionDataset(DATA_FILE)

    total_size = len(dataset)
    test_size = int(total_size * TEST_RATIO)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    model = CNNRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    chosen_indices = list(range(min(NUM_EXAMPLES, len(test_set))))

    for out_idx, ds_idx in enumerate(chosen_indices):
        x_tensor, y_norm_true = test_set[ds_idx]
        x_batch = x_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            y_norm_pred = model(x_batch).cpu().numpy()

        y_true = dataset.denormalize_labels(y_norm_true.unsqueeze(0).numpy())[0]
        y_pred = dataset.denormalize_labels(y_norm_pred)[0]

        true_F, true_k = float(y_true[0]), float(y_true[1])
        pred_F, pred_k = float(y_pred[0]), float(y_pred[1])

        # low-res image from dataset
        img_low = x_tensor[0].numpy()

        # regenerate high-res patterns
        V_true = simulate(Du, Dv, true_F, true_k, steps, seed=sim_seed)
        V_pred = simulate(Du, Dv, pred_F, pred_k, steps, seed=sim_seed)

        V_true_n = normalize(V_true)
        V_pred_n = normalize(V_pred)
        diff = np.abs(V_true_n - V_pred_n)

        pattern_mse = mse(V_true_n, V_pred_n)

        print(
            f"[{out_idx}] True F={true_F:.4f}, k={true_k:.4f} | "
            f"Pred F={pred_F:.4f}, k={pred_k:.4f} | "
            f"pattern_mse={pattern_mse:.6f}"
        )

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_low, cmap="gray")
        axes[0].set_title("Dataset Input (64x64)")
        axes[0].axis("off")

        axes[1].imshow(V_true_n, cmap="gray")
        axes[1].set_title(f"True Sim\nF={true_F:.4f}, k={true_k:.4f}")
        axes[1].axis("off")

        axes[2].imshow(V_pred_n, cmap="gray")
        axes[2].set_title(f"Pred Sim\nF={pred_F:.4f}, k={pred_k:.4f}")
        axes[2].axis("off")

        axes[3].imshow(diff, cmap="hot")
        axes[3].set_title(f"Abs Diff\nMSE={pattern_mse:.5f}")
        axes[3].axis("off")

        plt.tight_layout()
        out_file = os.path.join(OUTPUT_DIR, f"reconstruction_{out_idx:02d}.png")
        plt.savefig(out_file, dpi=180)
        plt.close()

    print(f"\nSaved reconstructions to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()