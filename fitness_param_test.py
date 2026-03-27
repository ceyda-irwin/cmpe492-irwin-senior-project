import numpy as np

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
target = np.load("target_pattern.npy")


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


def fitness(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate_n = normalize(candidate)
    target_n = normalize(target)

    error = mse(candidate_n, target_n)
    return 1.0 / (error + 1e-8)


# Test parameter sets
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

print("Testing fitness against target pattern:\n")
print(f"Target parameters: F={TARGET_F}, k={TARGET_K}\n")

results = []

for label, F, k in test_params:
    print(f"Running {label}: F={F}, k={k}")
    candidate = simulate(Du, Dv, F, k, steps)
    score = fitness(candidate, target)
    results.append((label, F, k, score))

print("\nResults:")
for label, F, k, score in sorted(results, key=lambda x: x[3], reverse=True):
    print(f"{label:12s} | F={F:.4f}, k={k:.4f} | fitness={score:.6f}")