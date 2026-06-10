"""
Dataset-size experiment (3k / 10k / 20k)
========================================

A single, size-parameterized pipeline so the 3,000 / 10,000 / 20,000-sample
runs are produced under identical settings and written to size-tagged paths
(no overwrites). Used to benchmark how dataset size affects the CNN
parameter-regressor.

All artifacts are tagged by sample count N and use a consistent V2 architecture
(matching scripts/analyzer_core.py), so the three runs are apples-to-apples:

  dataset : data/processed/rd_dataset_{N}.npz
  model   : scripts/models/cnn_v2_{N}.pt
  curve   : scripts/models/cnn_v2_{N}_curve.png
  eval    : scripts/models/eval_v2_{N}/reconstruction_*.png
  metrics : scripts/models/metrics_v2_{N}.json

Usage (from repo root):
    # 20k: dataset already exists -> just train + evaluate
    python3 scripts/size_experiment.py --samples 20000 --train --eval

    # 3k: generate, then train + evaluate
    python3 scripts/size_experiment.py --samples 3000 --generate --train --eval

    # 10k: dataset exists -> train + evaluate for a fair comparison
    python3 scripts/size_experiment.py --samples 10000 --train --eval

    # build comparison table + plot from all metrics_v2_*.json
    python3 scripts/size_experiment.py --compare

    # quick smoke test (tiny, throwaway)
    python3 scripts/size_experiment.py --samples 30 --generate --train --eval \
        --steps 150 --epochs 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Reuse the shared core (model architecture, simulator, helpers).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyzer_core as core  # noqa: E402

ROOT = core.ROOT
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "scripts" / "models"

# Training / data settings (held constant across sizes)
F_MIN, F_MAX = 0.0200, 0.0350
K_MIN, K_MAX = 0.0550, 0.0625
OUTPUT_SIZE = 64
GEN_SEED = 42

BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

DEVICE = core.DEVICE


# -----------------------------
# Size-tagged paths
# -----------------------------
def data_path(n: int) -> Path:
    return DATA_DIR / f"rd_dataset_{n}.npz"


def model_path(n: int) -> Path:
    return MODEL_DIR / f"cnn_v2_{n}.pt"


def curve_path(n: int) -> Path:
    return MODEL_DIR / f"cnn_v2_{n}_curve.png"


def eval_dir(n: int) -> Path:
    return MODEL_DIR / f"eval_v2_{n}"


def metrics_path(n: int) -> Path:
    return MODEL_DIR / f"metrics_v2_{n}.json"


# -----------------------------
# 1) Dataset generation
# -----------------------------
def generate_dataset(n: int, steps: int) -> None:
    out = data_path(n)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GEN_SEED)

    X = np.zeros((n, 1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)

    print(f"[generate] N={n}, steps={steps} -> {out}")
    for i in range(n):
        F = rng.uniform(F_MIN, F_MAX)
        k = rng.uniform(K_MIN, K_MAX)
        V = core.simulate(F, k, steps=steps, seed=GEN_SEED + i)
        X[i, 0] = core.to_64(core.normalize(V))
        y[i, 0] = F
        y[i, 1] = k
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"[generate]   {i + 1}/{n}")

    np.savez_compressed(
        out, X=X, y=y,
        F_min=F_MIN, F_max=F_MAX, K_min=K_MIN, K_max=K_MAX,
    )
    print(f"[generate] saved {out}  X={X.shape}")


# -----------------------------
# 2) Dataset wrapper
# -----------------------------
class RDDataset(Dataset):
    def __init__(self, npz_path: Path, augment: bool = False):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.F_min, self.F_max = float(data["F_min"]), float(data["F_max"])
        self.K_min, self.K_max = float(data["K_min"]), float(data["K_max"])
        self.augment = augment

        self.y_norm = np.zeros_like(self.y, dtype=np.float32)
        self.y_norm[:, 0] = (self.y[:, 0] - self.F_min) / (self.F_max - self.F_min)
        self.y_norm[:, 1] = (self.y[:, 1] - self.K_min) / (self.K_max - self.K_min)

    def __len__(self):
        return len(self.X)

    def _aug(self, x):
        img = x[0]
        if np.random.random() < 0.5:
            img = np.fliplr(img)
        if np.random.random() < 0.5:
            img = np.flipud(img)
        img = np.rot90(img, k=int(np.random.randint(0, 4)))
        return np.ascontiguousarray(img, dtype=np.float32)[None, :, :]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = self._aug(x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y_norm[idx], dtype=torch.float32),
        )

    def denormalize(self, y_norm: np.ndarray) -> np.ndarray:
        y = np.zeros_like(y_norm, dtype=np.float32)
        y[:, 0] = y_norm[:, 0] * (self.F_max - self.F_min) + self.F_min
        y[:, 1] = y_norm[:, 1] * (self.K_max - self.K_min) + self.K_min
        return y


def _splits(total: int):
    test_size = int(total * TEST_RATIO)
    val_size = int(total * VAL_RATIO)
    train_size = total - val_size - test_size
    return random_split(
        range(total), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )


# -----------------------------
# 3) Train
# -----------------------------
def train(n: int, epochs: int) -> None:
    npz = data_path(n)
    if not npz.exists():
        raise SystemExit(f"[train] dataset missing: {npz} (run with --generate)")

    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    base = RDDataset(npz, augment=False)
    train_idx, val_idx, _ = _splits(len(base))

    train_ds = Subset(RDDataset(npz, augment=True), train_idx.indices)
    val_ds = Subset(RDDataset(npz, augment=False), val_idx.indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = core.CNNRegressorV2().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] N={n}, epochs={epochs}, device={DEVICE}, "
          f"train={len(train_ds)}, val={len(val_ds)}")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * xb.size(0)
        tr = tot / len(train_ds)

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vtot += criterion(model(xb), yb).item() * xb.size(0)
        vl = vtot / len(val_ds)

        train_losses.append(tr)
        val_losses.append(vl)
        print(f"[train]   epoch {ep+1:02d}/{epochs} | train={tr:.6f} | val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), model_path(n))

    print(f"[train] best val={best_val:.6f} -> {model_path(n)}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title(f"CNN Regressor V2 — {n} samples")
    plt.legend(); plt.tight_layout()
    plt.savefig(curve_path(n), dpi=180)
    plt.close()


# -----------------------------
# 4) Evaluate
# -----------------------------
def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def evaluate(n: int, num_examples: int = 8, recon_steps: int = 2500) -> None:
    npz = data_path(n)
    mp = model_path(n)
    if not mp.exists():
        raise SystemExit(f"[eval] model missing: {mp} (run with --train)")

    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    ds = RDDataset(npz, augment=False)
    _, _, test_idx = _splits(len(ds))
    test_ds = Subset(ds, test_idx.indices)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = core.CNNRegressorV2().to(DEVICE)
    model.load_state_dict(torch.load(mp, map_location=DEVICE))
    model.eval()

    criterion = nn.MSELoss()
    tot = 0.0
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            tot += criterion(out, yb).item() * xb.size(0)
            preds.append(out.cpu().numpy())
            targs.append(yb.cpu().numpy())
    norm_loss = tot / len(test_ds)
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)

    preds_r = ds.denormalize(preds)
    targs_r = ds.denormalize(targs)

    metrics = {
        "samples": n,
        "test_size": len(test_ds),
        "norm_test_loss": norm_loss,
        "F_mae": _mae(preds_r[:, 0], targs_r[:, 0]),
        "k_mae": _mae(preds_r[:, 1], targs_r[:, 1]),
        "F_mse": _mse(preds_r[:, 0], targs_r[:, 0]),
        "k_mse": _mse(preds_r[:, 1], targs_r[:, 1]),
        "model": str(model_path(n).relative_to(ROOT)),
        "dataset": str(npz.relative_to(ROOT)),
    }
    metrics_path(n).write_text(json.dumps(metrics, indent=2))
    print(f"[eval] N={n}: F_mae={metrics['F_mae']:.5f} k_mae={metrics['k_mae']:.5f} "
          f"norm_loss={norm_loss:.6f} -> {metrics_path(n)}")

    # Reconstruction figures (re-simulate predicted vs true at full resolution)
    out_dir = eval_dir(n)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_examples, len(test_ds))):
        x_t, y_norm_true = test_ds[i]
        with torch.no_grad():
            y_norm_pred = model(x_t.unsqueeze(0).to(DEVICE)).cpu().numpy()
        yt = ds.denormalize(y_norm_true.unsqueeze(0).numpy())[0]
        yp = ds.denormalize(y_norm_pred)[0]

        V_true = core.normalize(core.simulate(float(yt[0]), float(yt[1]), steps=recon_steps, seed=123))
        V_pred = core.normalize(core.simulate(float(yp[0]), float(yp[1]), steps=recon_steps, seed=123))
        diff = np.abs(V_true - V_pred)
        pmse = _mse(V_true, V_pred)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(x_t[0].numpy(), cmap="gray"); ax[0].set_title("Dataset input (64×64)")
        ax[1].imshow(V_true, cmap="gray"); ax[1].set_title(f"True\nF={yt[0]:.4f}, k={yt[1]:.4f}")
        ax[2].imshow(V_pred, cmap="gray"); ax[2].set_title(f"Pred\nF={yp[0]:.4f}, k={yp[1]:.4f}")
        ax[3].imshow(diff, cmap="hot"); ax[3].set_title(f"Abs diff\nMSE={pmse:.5f}")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"reconstruction_{i:02d}.png", dpi=160)
        plt.close()
    print(f"[eval] saved reconstructions -> {out_dir}")


# -----------------------------
# 5) Compare
# -----------------------------
def compare() -> None:
    rows = []
    for mf in sorted(MODEL_DIR.glob("metrics_v2_*.json")):
        rows.append(json.loads(mf.read_text()))
    if not rows:
        raise SystemExit("[compare] no metrics_v2_*.json found. Run --eval first.")
    rows.sort(key=lambda r: r["samples"])

    # CSV
    csv = MODEL_DIR / "size_comparison.csv"
    header = ["samples", "F_mae", "k_mae", "F_mse", "k_mse", "norm_test_loss"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))
    csv.write_text("\n".join(lines) + "\n")

    print("\n=== Dataset-size comparison ===")
    print(f"{'N':>7} | {'F_MAE':>9} | {'k_MAE':>9} | {'norm_loss':>10}")
    for r in rows:
        print(f"{r['samples']:>7} | {r['F_mae']:>9.5f} | {r['k_mae']:>9.5f} | {r['norm_test_loss']:>10.6f}")

    # Plot
    ns = [r["samples"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ns, [r["F_mae"] for r in rows], "o-", label="F MAE")
    ax[0].plot(ns, [r["k_mae"] for r in rows], "s-", label="k MAE")
    ax[0].set_xlabel("Training samples"); ax[0].set_ylabel("MAE (real units)")
    ax[0].set_title("Parameter error vs dataset size"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ns, [r["norm_test_loss"] for r in rows], "^-", color="purple")
    ax[1].set_xlabel("Training samples"); ax[1].set_ylabel("Normalized test loss")
    ax[1].set_title("Normalized test loss vs dataset size"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "size_comparison.png", dpi=180)
    plt.close()
    print(f"\nSaved -> {csv}")
    print(f"Saved -> {MODEL_DIR / 'size_comparison.png'}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Dataset-size experiment (3k/10k/20k)")
    ap.add_argument("--samples", type=int, help="dataset size N")
    ap.add_argument("--generate", action="store_true", help="generate the dataset")
    ap.add_argument("--train", action="store_true", help="train the CNN")
    ap.add_argument("--eval", action="store_true", help="evaluate + save metrics/figures")
    ap.add_argument("--compare", action="store_true", help="build comparison table + plot")
    ap.add_argument("--steps", type=int, default=3000, help="sim steps for generation")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="training epochs")
    ap.add_argument("--recon-steps", type=int, default=2500, help="sim steps for eval recon")
    args = ap.parse_args()

    if args.compare:
        compare()
        return

    if not args.samples:
        raise SystemExit("Provide --samples N (or --compare).")

    if args.generate:
        generate_dataset(args.samples, steps=args.steps)
    if args.train:
        train(args.samples, epochs=args.epochs)
    if args.eval:
        evaluate(args.samples, recon_steps=args.recon_steps)

    if not (args.generate or args.train or args.eval):
        raise SystemExit("Nothing to do. Pass --generate / --train / --eval / --compare.")


if __name__ == "__main__":
    main()
