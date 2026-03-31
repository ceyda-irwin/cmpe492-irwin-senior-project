from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIR = ROOT / "outputs" / "visual_checks"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

N = 200
Du = 0.16
Dv = 0.08
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
                break
            prev_V = V.copy()

    return V


def normalize(img: np.ndarray) -> np.ndarray:
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-12:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def labeled_components(img: np.ndarray, min_size: int = 8):
    img_n = normalize(img)
    thresh = np.mean(img_n) + 0.5 * np.std(img_n)
    binary = img_n > thresh

    structure = np.ones((3, 3), dtype=int)
    labeled, num = ndimage.label(binary, structure=structure)

    if num == 0:
        return binary, np.zeros_like(labeled), 0

    component_areas = ndimage.sum(binary, labeled, index=np.arange(1, num + 1))
    component_areas = np.array(component_areas)

    cleaned = np.zeros_like(labeled)
    new_label = 1

    for old_label, area in enumerate(component_areas, start=1):
        if area >= min_size:
            cleaned[labeled == old_label] = new_label
            new_label += 1

    return binary, cleaned, new_label - 1


patterns = [
    ("target", 0.0275, 0.0600),
    ("close", 0.0250, 0.0600),
    ("confuser", 0.0200, 0.0550),
    ("stripe_like", 0.0245, 0.0570),
]

fig, axes = plt.subplots(len(patterns), 3, figsize=(10, 14))

for row, (label, F, k) in enumerate(patterns):
    V = simulate(Du, Dv, F, k, steps)
    binary, labeled, count = labeled_components(V)

    axes[row, 0].imshow(V, cmap="gray")
    axes[row, 0].set_title(f"{label}\nF={F}, k={k}")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(binary, cmap="gray")
    axes[row, 1].set_title("Thresholded")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(labeled, cmap="nipy_spectral")
    axes[row, 2].set_title(f"Components: {count}")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig(VISUAL_DIR / "component_visual_check.png", dpi=200)
plt.show()