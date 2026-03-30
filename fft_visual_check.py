import numpy as np
import matplotlib.pyplot as plt

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


def fft_signature(img: np.ndarray, crop_size: int = 64) -> np.ndarray:
    img_n = normalize(img)
    img_zm = img_n - np.mean(img_n)

    fft2 = np.fft.fft2(img_zm)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shifted)
    log_mag = np.log1p(magnitude)

    h, w = log_mag.shape
    ch, cw = h // 2, w // 2
    r = crop_size // 2

    cropped = log_mag[ch-r:ch+r, cw-r:cw+r]
    return normalize(cropped)


patterns = [
    ("target", 0.0275, 0.0600),
    ("close", 0.0300, 0.0600),
    ("confuser", 0.0200, 0.0550),
    ("far", 0.0450, 0.0500),
]

fig, axes = plt.subplots(len(patterns), 2, figsize=(8, 14))

for row, (label, F, k) in enumerate(patterns):
    V = simulate(Du, Dv, F, k, steps)
    fft_img = fft_signature(V)

    axes[row, 0].imshow(V, cmap="gray")
    axes[row, 0].set_title(f"{label}: F={F}, k={k}")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(fft_img, cmap="magma")
    axes[row, 1].set_title(f"{label} FFT")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig("fft_visual_check.png", dpi=200)
plt.show()