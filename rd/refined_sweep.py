import os
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(ROOT / "outputs" / "sweeps" / "refined_sweep_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid size
N = 200

# Fixed diffusion parameters
Du = 0.16
Dv = 0.08

# Refined search region
F_values = np.arange(0.020, 0.0451, 0.0025)
k_values = np.arange(0.050, 0.0651, 0.0025)

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


def simulate(Du: float, Dv: float, F: float, k: float, steps: int):
    rng = np.random.default_rng(seed)
    U, V = initialize_grid(N, rng)

    prev_V = V.copy()
    steps_used = steps

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
                steps_used = i
                break
            prev_V = V.copy()

    return U, V, steps_used


def compute_features(V: np.ndarray):
    mean_val = float(np.mean(V))
    std_val = float(np.std(V))

    gx = np.abs(np.diff(V, axis=1)).mean()
    gy = np.abs(np.diff(V, axis=0)).mean()
    gradient_strength = float(gx + gy)

    threshold = float(np.mean(V) + np.std(V))
    active_ratio = float(np.mean(V > threshold))

    return mean_val, std_val, gradient_strength, active_ratio


rows = []

for F in F_values:
    for k in k_values:
        print(f"Running F={F:.4f}, k={k:.4f}")
        U, V, steps_used = simulate(Du, Dv, F, k, steps)

        mean_val, std_val, gradient_strength, active_ratio = compute_features(V)

        filename = f"pattern_F{F:.4f}_k{k:.4f}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        plt.figure(figsize=(4, 4))
        plt.imshow(V, cmap="plasma", interpolation="nearest")
        plt.title(f"F={F:.4f}, k={k:.4f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        rows.append({
            "Du": Du,
            "Dv": Dv,
            "F": round(float(F), 4),
            "k": round(float(k), 4),
            "steps_used": steps_used,
            "mean_V": mean_val,
            "std_V": std_val,
            "gradient_strength": gradient_strength,
            "active_ratio": active_ratio,
            "image": filename
        })

csv_path = os.path.join(OUTPUT_DIR, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Results saved to: {OUTPUT_DIR}")