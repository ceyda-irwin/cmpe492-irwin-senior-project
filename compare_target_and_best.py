import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Fixed settings
# -----------------------------
N = 200
Du = 0.16
Dv = 0.08
steps = 5000
dt = 1.0
seed = 42
check_interval = 100
stability_threshold = 1e-5

# Target params
TARGET_F = 0.0275
TARGET_K = 0.0600

# GA best params
BEST_F = 0.0234
BEST_K = 0.0577


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
                break
            prev_V = V.copy()

    return V


target = simulate(Du, Dv, TARGET_F, TARGET_K, steps)
best = simulate(Du, Dv, BEST_F, BEST_K, steps)
diff = np.abs(target - best)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(target, cmap="gray")
axes[0].set_title(f"Target\nF={TARGET_F}, k={TARGET_K}")
axes[0].axis("off")

axes[1].imshow(best, cmap="gray")
axes[1].set_title(f"GA Best\nF={BEST_F}, k={BEST_K}")
axes[1].axis("off")

axes[2].imshow(diff, cmap="hot")
axes[2].set_title("Absolute Difference")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("target_vs_best_comparison.png", dpi=200)
plt.show()