import os
import numpy as np
from typing import Tuple

# -----------------------------
# Simulation settings
# -----------------------------
N = 200
Du = 0.16
Dv = 0.08
steps = 3000
dt = 1.0
seed = 42
check_interval = 100
stability_threshold = 1e-5

# Parameter ranges
F_MIN, F_MAX = 0.0200, 0.0350
K_MIN, K_MAX = 0.0550, 0.0625

# Dataset settings
NUM_SAMPLES = 10000
OUTPUT_SIZE = 64
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rd_dataset.npz")


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


def simulate(Du: float, Dv: float, F: float, k: float, steps: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
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


def resize_nearest(img: np.ndarray, out_size: int) -> np.ndarray:
    """
    Simple nearest-neighbor resize without external dependencies.
    """
    h, w = img.shape
    row_idx = np.linspace(0, h - 1, out_size).astype(int)
    col_idx = np.linspace(0, w - 1, out_size).astype(int)
    return img[row_idx][:, col_idx]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(seed)

    X = np.zeros((NUM_SAMPLES, 1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    y = np.zeros((NUM_SAMPLES, 2), dtype=np.float32)

    for i in range(NUM_SAMPLES):
        F = rng.uniform(F_MIN, F_MAX)
        k = rng.uniform(K_MIN, K_MAX)

        # sample-specific deterministic seed
        sim_seed = seed + i

        V = simulate(Du, Dv, F, k, steps, sim_seed)
        V = normalize(V)
        V_small = resize_nearest(V, OUTPUT_SIZE)

        X[i, 0] = V_small
        y[i, 0] = F
        y[i, 1] = k

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES}")

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        F_min=F_MIN,
        F_max=F_MAX,
        K_min=K_MIN,
        K_max=K_MAX,
    )

    print(f"\nSaved dataset to: {OUTPUT_FILE}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()