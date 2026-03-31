from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Fixed global settings
# -----------------------------
N = 200
Du = 0.16
Dv = 0.08
steps = 5000
dt = 1.0
seed = 42
check_interval = 100
stability_threshold = 1e-5

# Target parameters
TARGET_F = 0.0275
TARGET_K = 0.0600

# Load target pattern
target = np.load(ROOT / "outputs" / "target" / "target_pattern.npy")


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


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_features(V: np.ndarray):
    mean_val = float(np.mean(V))
    std_val = float(np.std(V))
    gx = float(np.abs(np.diff(V, axis=1)).mean())
    gy = float(np.abs(np.diff(V, axis=0)).mean())
    gradient_strength = gx + gy
    threshold = mean_val + std_val
    active_ratio = float(np.mean(V > threshold))

    return {
        "mean": mean_val,
        "std": std_val,
        "grad": gradient_strength,
        "active": active_ratio,
    }


def radial_fft_profile(img: np.ndarray, max_radius: int = 40) -> np.ndarray:
    img_n = normalize(img)
    img_zm = img_n - np.mean(img_n)

    fft2 = np.fft.fft2(img_zm)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.log1p(np.abs(fft_shifted))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.floor(r).astype(int)

    profile = []
    for radius in range(max_radius):
        mask = (r_int == radius)
        profile.append(np.mean(magnitude[mask]) if np.any(mask) else 0.0)

    return normalize(np.array(profile, dtype=np.float64))


def connected_component_features(img: np.ndarray, min_size: int = 8):
    """
    Threshold image and compute connected-component statistics.
    Small components below min_size are ignored as noise.
    """
    img_n = normalize(img)
    thresh = np.mean(img_n) + 0.5 * np.std(img_n)
    binary = img_n > thresh

    structure = np.ones((3, 3), dtype=int)
    labeled, num = ndimage.label(binary, structure=structure)

    if num == 0:
        return {
            "count": 0,
            "mean_area": 0.0,
            "max_area": 0.0,
        }

    component_areas = ndimage.sum(binary, labeled, index=np.arange(1, num + 1))
    component_areas = np.array(component_areas, dtype=np.float64)

    # remove tiny noise blobs
    component_areas = component_areas[component_areas >= min_size]

    if len(component_areas) == 0:
        return {
            "count": 0,
            "mean_area": 0.0,
            "max_area": 0.0,
        }

    return {
        "count": int(len(component_areas)),
        "mean_area": float(np.mean(component_areas)),
        "max_area": float(np.max(component_areas)),
    }


target_n = normalize(target)
target_feat = compute_features(target)
target_radial = radial_fft_profile(target, max_radius=40)
target_cc = connected_component_features(target)

print("Target connected components:", target_cc)


def fitness(candidate: np.ndarray) -> float:
    candidate_n = normalize(candidate)
    candidate_feat = compute_features(candidate)
    candidate_radial = radial_fft_profile(candidate, max_radius=40)
    candidate_cc = connected_component_features(candidate)

    mse_error = mse(candidate_n, target_n)
    std_penalty = abs(candidate_feat["std"] - target_feat["std"])
    grad_penalty = abs(candidate_feat["grad"] - target_feat["grad"])
    active_penalty = abs(candidate_feat["active"] - target_feat["active"])
    radial_penalty = mse(candidate_radial, target_radial)

    cand_count = candidate_cc["count"]
    cand_mean_area = candidate_cc["mean_area"]
    cand_max_area = candidate_cc["max_area"]

    target_count = target_cc["count"]
    target_mean_area = target_cc["mean_area"]
    target_max_area = target_cc["max_area"]
    
    count_penalty = abs(np.log1p(cand_count) - np.log1p(target_count))
    mean_area_penalty = abs(np.log1p(cand_mean_area) - np.log1p(target_mean_area))
    max_area_penalty = abs(np.log1p(cand_max_area) - np.log1p(target_max_area))

    total_error = (
        0.7 * mse_error
        + 1.0 * std_penalty
        + 1.0 * grad_penalty
        + 0.7 * active_penalty
        + 1.8 * radial_penalty
        + 1.0 * count_penalty
        + 0.8 * mean_area_penalty
        + 0.6 * max_area_penalty
    )

    return 1.0 / (total_error + 1e-8)


test_params = [
    ("exact_target", 0.0275, 0.0600),
    ("very_close_1", 0.0275, 0.0575),
    ("very_close_2", 0.0300, 0.0600),
    ("close_3",      0.0250, 0.0600),
    ("medium_far_1", 0.0350, 0.0600),
    ("medium_far_2", 0.0200, 0.0550),
    ("far_1",        0.0450, 0.0500),
    ("far_2",        0.0200, 0.0650),
]

print("\nTesting component-enhanced fitness against target pattern:\n")
print(f"Target parameters: F={TARGET_F}, k={TARGET_K}\n")

results = []

for label, F, k in test_params:
    print(f"Running {label}: F={F}, k={k}")
    candidate = simulate(Du, Dv, F, k, steps)
    cc = connected_component_features(candidate)
    score = fitness(candidate)
    results.append((label, F, k, score, cc))

print("\nResults:")
for label, F, k, score, cc in sorted(results, key=lambda x: x[3], reverse=True):
    print(
        f"{label:12s} | F={F:.4f}, k={k:.4f} | fitness={score:.6f} | "
        f"count={cc['count']}, mean_area={cc['mean_area']:.2f}, max_area={cc['max_area']:.2f}"
    )