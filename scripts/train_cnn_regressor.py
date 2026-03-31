import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = str(ROOT / "data" / "processed" / "rd_dataset.npz")
MODEL_DIR = str(ROOT / "scripts" / "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# Dataset
# -----------------------------
class ReactionDiffusionDataset(Dataset):
    def __init__(self, data_file: str):
        data = np.load(data_file)

        self.X = data["X"].astype(np.float32)   # shape: (N, 1, 64, 64)
        self.y = data["y"].astype(np.float32)   # shape: (N, 2)

        self.F_min = float(data["F_min"])
        self.F_max = float(data["F_max"])
        self.K_min = float(data["K_min"])
        self.K_max = float(data["K_max"])

        # Normalize labels into [0, 1]
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
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # output: normalized F, k
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        total_loss += loss.item() * X_batch.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, all_preds, all_targets


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
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

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = os.path.join(MODEL_DIR, "cnn_regressor_best.pt")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, _, _ = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print("\nBest model saved to:", best_model_path)

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, preds_norm, targets_norm = evaluate(model, test_loader, criterion)

    preds_real = dataset.denormalize_labels(preds_norm)
    targets_real = dataset.denormalize_labels(targets_norm)

    F_mae = mae(preds_real[:, 0], targets_real[:, 0])
    k_mae = mae(preds_real[:, 1], targets_real[:, 1])

    F_mse = mse_np(preds_real[:, 0], targets_real[:, 0])
    k_mse = mse_np(preds_real[:, 1], targets_real[:, 1])

    print("\n=== Test Results ===")
    print(f"Normalized test loss: {test_loss:.6f}")
    print(f"F MAE: {F_mae:.6f}")
    print(f"F MSE: {F_mse:.6f}")
    print(f"k MAE: {k_mae:.6f}")
    print(f"k MSE: {k_mse:.6f}")

    # Save a few prediction examples
    print("\nSample predictions:")
    for i in range(min(10, len(preds_real))):
        print(
            f"True:  F={targets_real[i,0]:.4f}, k={targets_real[i,1]:.4f} | "
            f"Pred: F={preds_real[i,0]:.4f}, k={preds_real[i,1]:.4f}"
        )

    # Plot losses
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CNN Regressor Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cnn_training_curve.png"), dpi=180)
    plt.show()


if __name__ == "__main__":
    main()