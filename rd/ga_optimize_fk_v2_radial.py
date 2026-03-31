import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TARGET_NPY = ROOT / "outputs" / "target" / "target_pattern.npy"
GA_OUT = ROOT / "outputs" / "ga"
GA_BEST_RADIAL_DIR = GA_OUT / "best_radial_gen"
GA_BEST_RADIAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Fixed simulation settings
# -----------------------------
N = 200
Du = 0.16
Dv = 0.08
steps = 3000
dt = 1.0
seed = 42
check_interval = 100
stability_threshold = 1e-5

# Search bounds
F_MIN, F_MAX = 0.0200, 0.0350
K_MIN, K_MAX = 0.0550, 0.0625

# GA settings
POP_SIZE = 8
GENERATIONS = 10
ELITE_COUNT = 2
MUTATION_RATE = 0.35
MUTATION_SCALE_F = 0.0012
MUTATION_SCALE_K = 0.0012

# Load target
target = np.load(TARGET_NPY)


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


target_n = normalize(target)
target_feat = compute_features(target)
target_radial = radial_fft_profile(target, max_radius=40)


def fitness(candidate: np.ndarray) -> float:
    candidate_n = normalize(candidate)
    candidate_feat = compute_features(candidate)
    candidate_radial = radial_fft_profile(candidate, max_radius=40)

    mse_error = mse(candidate_n, target_n)
    std_penalty = abs(candidate_feat["std"] - target_feat["std"])
    grad_penalty = abs(candidate_feat["grad"] - target_feat["grad"])
    active_penalty = abs(candidate_feat["active"] - target_feat["active"])
    radial_penalty = mse(candidate_radial, target_radial)

    total_error = (
        0.8 * mse_error
        + 1.2 * std_penalty
        + 1.2 * grad_penalty
        + 0.8 * active_penalty
        + 2.5 * radial_penalty
    )

    return 1.0 / (total_error + 1e-8)


def random_individual():
    return {
        "F": random.uniform(F_MIN, F_MAX),
        "k": random.uniform(K_MIN, K_MAX),
    }


def clamp(value, low, high):
    return max(low, min(high, value))


def crossover(parent1, parent2):
    alpha = random.random()
    child_F = alpha * parent1["F"] + (1 - alpha) * parent2["F"]
    child_k = alpha * parent1["k"] + (1 - alpha) * parent2["k"]
    return {"F": child_F, "k": child_k}


def mutate(individual):
    if random.random() < MUTATION_RATE:
        individual["F"] += random.gauss(0, MUTATION_SCALE_F)
    if random.random() < MUTATION_RATE:
        individual["k"] += random.gauss(0, MUTATION_SCALE_K)

    individual["F"] = clamp(individual["F"], F_MIN, F_MAX)
    individual["k"] = clamp(individual["k"], K_MIN, K_MAX)
    return individual


def evaluate_population(population):
    evaluated = []
    for ind in population:
        V = simulate(Du, Dv, ind["F"], ind["k"], steps)
        score = fitness(V)
        evaluated.append({
            "F": ind["F"],
            "k": ind["k"],
            "fitness": score,
            "pattern": V
        })
    evaluated.sort(key=lambda x: x["fitness"], reverse=True)
    return evaluated


def select_parents(evaluated):
    top_half = evaluated[: max(2, len(evaluated) // 2)]
    return random.choice(top_half), random.choice(top_half)


def make_next_generation(evaluated):
    next_population = []

    for elite in evaluated[:ELITE_COUNT]:
        next_population.append({"F": elite["F"], "k": elite["k"]})

    while len(next_population) < POP_SIZE:
        p1, p2 = select_parents(evaluated)
        child = crossover(p1, p2)
        child = mutate(child)
        next_population.append(child)

    return next_population


def save_best_image(best_result, generation):
    plt.figure(figsize=(6, 6))
    plt.imshow(best_result["pattern"], cmap="gray")
    plt.title(
        f"Gen {generation} | F={best_result['F']:.4f}, k={best_result['k']:.4f}\n"
        f"fitness={best_result['fitness']:.4f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GA_BEST_RADIAL_DIR / f"best_radial_gen_{generation:02d}.png", dpi=180, bbox_inches="tight")
    plt.close()


def main():
    random.seed(42)
    np.random.seed(42)

    population = [random_individual() for _ in range(POP_SIZE)]
    best_history = []
    global_best = None

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen} ===")
        evaluated = evaluate_population(population)

        best = evaluated[0]
        best_history.append(best["fitness"])

        if global_best is None or best["fitness"] > global_best["fitness"]:
            global_best = best

        print(
            f"Best -> F={best['F']:.4f}, k={best['k']:.4f}, "
            f"fitness={best['fitness']:.6f}"
        )

        save_best_image(best, gen)
        population = make_next_generation(evaluated)

    print("\n=== Final Best ===")
    print(
        f"F={global_best['F']:.4f}, k={global_best['k']:.4f}, "
        f"fitness={global_best['fitness']:.6f}"
    )

    plt.figure(figsize=(8, 4))
    plt.plot(best_history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Optimization Progress (Radial FFT Fitness)")
    plt.tight_layout()
    plt.savefig(GA_OUT / "ga_fitness_progress_radial.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()