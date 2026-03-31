from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "outputs" / "target"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Grid size
N = 200

# Fixed parameters
Du = 0.16
Dv = 0.08
F = 0.0275
k = 0.0600

# Simulation settings
steps = 5000
dt = 1.0
seed = 42
check_interval = 100
stability_threshold = 1e-5


def laplacian(Z: np.ndarray) -> np.ndarray:
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )


def initialize_grid(n: int, rng: np.random.Generator):
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


def simulate(Du, Dv, F, k, steps):
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
                print(f"Stabilized at step {i}")
                break
            prev_V = V.copy()

    return U, V


# Run simulation
U, V = simulate(Du, Dv, F, k, steps)

# Save raw array too
np.save(TARGET_DIR / "target_pattern.npy", V)

# Save image
plt.figure(figsize=(6, 6))
plt.imshow(V, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.savefig(TARGET_DIR / "target_pattern.png", dpi=200, bbox_inches="tight", pad_inches=0)
plt.show()