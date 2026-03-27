import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Output folder
OUTPUT_DIR = "sweep_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid size
N = 200

# Fixed diffusion parameters
Du = 0.16
Dv = 0.08

# Try different feed/kill values
F_values = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]
k_values = [0.045, 0.050, 0.055, 0.060, 0.065, 0.070]

# Simulation settings
steps = 3000
dt = 1.0
seed = 42


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

    prev_V = V.copy()  # 👈 buraya ekledik

    for i in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)

        uvv = U * V * V

        U += (Du * Lu - uvv + F * (1 - U)) * dt
        V += (Dv * Lv + uvv - (F + k) * V) * dt

        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

        # 👇 HER 100 ADIMDA KONTROL
        if i % 100 == 0:
            diff = np.mean(np.abs(V - prev_V))

            if diff < 1e-5:
                print(f"Stabil oldu (step {i}) → erken durdu")
                break

            prev_V = V.copy()

    return U, V

def classify_pattern(V: np.ndarray) -> str:
    """
    Very rough manual-style heuristic classification.
    This is not scientifically perfect, just useful for notes.
    """
    std = np.std(V)
    mean = np.mean(V)

    # crude edge-like activity estimate
    gx = np.abs(np.diff(V, axis=1)).mean()
    gy = np.abs(np.diff(V, axis=0)).mean()
    grad = gx + gy

    if std < 0.03:
        return "almost uniform"
    if grad > 0.10 and 0.05 < mean < 0.35:
        return "complex / maze-like"
    if std > 0.08 and mean < 0.20:
        return "spots or mixed"
    return "stripe-like or mixed"


rows = []

for F in F_values:
    for k in k_values:
        print(f"Running F={F:.3f}, k={k:.3f}")
        U, V = simulate(Du, Dv, F, k, steps)

        filename = f"pattern_F{F:.3f}_k{k:.3f}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        plt.figure(figsize=(4, 4))
        plt.imshow(V, cmap="plasma", interpolation="nearest")
        plt.title(f"F={F:.3f}, k={k:.3f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        pattern_type = classify_pattern(V)

        rows.append({
            "Du": Du,
            "Dv": Dv,
            "F": F,
            "k": k,
            "steps": steps,
            "image": filename,
            "pattern_note": pattern_type
        })

csv_path = os.path.join(OUTPUT_DIR, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Images and CSV saved in: {OUTPUT_DIR}")