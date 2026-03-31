from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def radial_fft_profile(img: np.ndarray, max_radius: int = 40) -> np.ndarray:
    img_n = normalize(img)
    img_zm = img_n - np.mean(img_n)

    fft2 = np.fft.fft2(img_zm)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shifted))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = np.floor(r).astype(int)

    profile = []
    for radius in range(max_radius):
        mask = (r_int == radius)
        profile.append(np.mean(magnitude[mask]) if np.any(mask) else 0.0)

    return normalize(np.array(profile))


patterns = [
    ("target", 0.0275, 0.0600),
    ("close", 0.0300, 0.0600),
    ("confuser", 0.0200, 0.0550),
    ("far", 0.0450, 0.0500),
]

plt.figure(figsize=(8, 5))

for label, F, k in patterns:
    V = simulate(Du, Dv, F, k, steps)
    profile = radial_fft_profile(V, max_radius=40)
    plt.plot(profile, marker="o", label=f"{label} ({F}, {k})")

plt.xlabel("Radial frequency bin")
plt.ylabel("Normalized energy")
plt.title("Radial FFT Profiles")
plt.legend()
plt.tight_layout()
plt.savefig(VISUAL_DIR / "radial_profile_check.png", dpi=200)
plt.show()