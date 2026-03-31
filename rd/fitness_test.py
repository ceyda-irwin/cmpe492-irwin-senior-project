from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Load target pattern
target = np.load(ROOT / "outputs" / "target" / "target_pattern.npy")


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

    # Higher fitness should mean better result
    return 1.0 / (error + 1e-8)


# Test 1: exact same image
score_same = fitness(target, target)
print("Fitness(target, target) =", score_same)

# Test 2: random image
random_img = np.random.random(target.shape)
score_random = fitness(random_img, target)
print("Fitness(random, target) =", score_random)