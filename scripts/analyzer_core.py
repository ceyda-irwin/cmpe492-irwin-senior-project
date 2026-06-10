"""
Analyzer core
=============

Shared building blocks for the Pattern-to-Parameter Analyzer. Both the CLI
(`predict_from_image.py`) and the interactive app (`app/app.py`) import from
here so there is a single source of truth for:

  * the trained CNN architecture (must match `train_cnn_regressor_v2.py`),
  * the forward Gray-Scott simulator (must match `generate_ml_dataset.py`),
  * image pre-processing, prediction, reconstruction, retrieval and a simple
    morphology-family heuristic.

This module adds NO new method. It only packages the existing forward simulator
and trained CNN into a reusable analysis API.

Honest framing for the report: a predicted (F, k) is "the Gray-Scott regime
whose morphology most closely matches the input", NOT a claim about true
biological parameters. The training range is narrow
(F in [0.020, 0.035], k in [0.0550, 0.0625]); inputs outside this regime
(e.g. clean parallel zebra stripes) saturate at the range boundary -- which is
itself a useful demonstration of the model's generalization limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

# -----------------------------
# Paths / config
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "scripts" / "models"
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "analyzer"


def _resolve_model_file():
    """
    Pick the best available CNNRegressorV2 checkpoint. Prefers the largest
    size-tagged `cnn_v2_{N}.pt` (trained on the most data), then falls back to
    the canonical `cnn_regressor_v2_best.pt`.
    """
    def _n(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    tagged = sorted(MODEL_DIR.glob("cnn_v2_*.pt"), key=_n)
    if tagged:
        return tagged[-1]
    return MODEL_DIR / "cnn_regressor_v2_best.pt"


MODEL_FILE = _resolve_model_file()


def _resolve_data_file():
    """
    Locate a dataset npz for retrieval. Prefers the canonical `rd_dataset.npz`,
    then a 10k size-tagged set (good speed/quality balance), then the largest
    available `rd_dataset_{N}.npz`. Returns None if nothing is present.
    """
    canonical = DATA_DIR / "rd_dataset.npz"
    if canonical.exists():
        return canonical

    def _n(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    tagged = sorted(DATA_DIR.glob("rd_dataset_*.npz"), key=_n)
    if not tagged:
        return None
    for p in tagged:
        if _n(p) == 10000:
            return p
    return tagged[-1]  # largest N


DATA_FILE = _resolve_data_file()

# Default (F, k) ranges -- must match generate_ml_dataset.py.
# Overridden by the dataset npz if it is present (see load_ranges()).
F_MIN, F_MAX = 0.0200, 0.0350
K_MIN, K_MAX = 0.0550, 0.0625

# Simulation settings (match the dataset generator)
N = 200
Du = 0.16
Dv = 0.08
STEPS = 3000
DT = 1.0
SIM_SEED = 42
CHECK_INTERVAL = 100
STABILITY_THRESHOLD = 1e-5
OUTPUT_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model (must match train_cnn_regressor_v2.py)
# -----------------------------
class CNNRegressorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 16 -> 8
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),                   # normalized F, k in [0, 1]
        )

    def forward(self, x):
        return self.regressor(self.features(x))

    def embed(self, x):
        """Return the flattened feature embedding (for retrieval)."""
        return torch.flatten(self.features(x), 1)


# -----------------------------
# Range / model loading
# -----------------------------
def load_ranges() -> Tuple[float, float, float, float]:
    """Return (F_min, F_max, k_min, k_max), pulled from the dataset if present."""
    if DATA_FILE is not None and DATA_FILE.exists():
        d = np.load(DATA_FILE)
        return (
            float(d["F_min"]), float(d["F_max"]),
            float(d["K_min"]), float(d["K_max"]),
        )
    return F_MIN, F_MAX, K_MIN, K_MAX


def model_available() -> bool:
    return MODEL_FILE.exists()


def dataset_available() -> bool:
    return DATA_FILE is not None and DATA_FILE.exists()


def load_model() -> CNNRegressorV2:
    """Load the trained CNN. Raises FileNotFoundError if weights are missing."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_FILE}. "
            f"Train it first with scripts/train_cnn_regressor_v2.py."
        )
    model = CNNRegressorV2().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    return model


# -----------------------------
# Simulator
# -----------------------------
def laplacian(Z: np.ndarray) -> np.ndarray:
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )


def initialize_grid(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    U = np.ones((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)
    r = 20
    c = n // 2
    U[c - r:c + r, c - r:c + r] = 0.50
    V[c - r:c + r, c - r:c + r] = 0.25
    noise = 0.02
    U += noise * rng.random((n, n))
    V += noise * rng.random((n, n))
    return np.clip(U, 0, 1), np.clip(V, 0, 1)


def simulate(F: float, k: float, steps: int = STEPS, seed: int = SIM_SEED) -> np.ndarray:
    """Run the forward Gray-Scott model and return the final V field (float32)."""
    rng = np.random.default_rng(seed)
    U, V = initialize_grid(N, rng)
    prev_V = V.copy()
    for i in range(steps):
        uvv = U * V * V
        U += (Du * laplacian(U) - uvv + F * (1 - U)) * DT
        V += (Dv * laplacian(V) + uvv - (F + k) * V) * DT
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)
        if i % CHECK_INTERVAL == 0 and i > 0:
            if np.mean(np.abs(V - prev_V)) < STABILITY_THRESHOLD:
                break
            prev_V = V.copy()
    return V.astype(np.float32)


# -----------------------------
# Image helpers
# -----------------------------
def normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def otsu_threshold(img: np.ndarray) -> float:
    """Otsu's method on a [0,1] image; returns a threshold in [0,1]."""
    hist, edges = np.histogram(img, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    w_b = np.cumsum(hist)
    w_f = total - w_b
    sum_total = np.sum(hist * centers)
    sum_b = np.cumsum(hist * centers)
    valid = (w_b > 0) & (w_f > 0)
    if not np.any(valid):
        return 0.5
    mean_b = np.where(w_b > 0, sum_b / np.maximum(w_b, 1e-12), 0.0)
    mean_f = np.where(w_f > 0, (sum_total - sum_b) / np.maximum(w_f, 1e-12), 0.0)
    between = w_b * w_f * (mean_b - mean_f) ** 2
    between[~valid] = -1
    return float(centers[int(np.argmax(between))])


def preprocess_array(arr: np.ndarray, binarize: bool = False) -> np.ndarray:
    """Grayscale 2D array -> 64x64 -> [0,1] (optionally Otsu-binarized)."""
    img = Image.fromarray(np.asarray(arr, dtype=np.float32))
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
    out = normalize(np.asarray(img, dtype=np.float32))
    if binarize:
        out = (out > otsu_threshold(out)).astype(np.float32)
    return out


def preprocess_image(source, binarize: bool = False) -> np.ndarray:
    """
    Load any image -> grayscale -> 64x64 -> [0,1] (optionally binarized).
    `source` may be a path or anything PIL.Image.open accepts (e.g. a file-like
    object from a web upload).
    """
    img = Image.open(source).convert("L")
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
    arr = normalize(np.asarray(img, dtype=np.float32))
    if binarize:
        arr = (arr > otsu_threshold(arr)).astype(np.float32)
    return arr


def to_64(img: np.ndarray) -> np.ndarray:
    """Downsample a (normalized) field to a 64x64 [0,1] image."""
    out = Image.fromarray((normalize(img) * 255).astype(np.uint8))
    out = out.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
    return normalize(np.asarray(out, dtype=np.float32))


def dominant_wavelength(img: np.ndarray) -> float:
    """
    Estimate the dominant spatial wavelength (in pixels) via the radial power
    spectrum. Context for the report's 'surface area / scale calibration'
    discussion: how many feature repeats span the domain.
    """
    z = img - img.mean()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial[0] = 0.0  # drop DC
    half = max(2, len(radial) // 2)
    peak = int(np.argmax(radial[1:half])) + 1
    return float(h / peak) if peak > 0 else float("nan")


def classify_morphology(img: np.ndarray) -> str:
    """
    Cheap heuristic family label (spot / stripe / mixed) from connected-component
    shape statistics. Used only for a human-readable hint in the UI -- not a
    rigorous classifier.
    """
    try:
        from scipy import ndimage
    except Exception:
        return "unknown"

    n = normalize(img)
    binary = n > (n.mean() + 0.25 * n.std())
    labeled, num = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return "uniform / empty"

    # Elongation: compare each component's bounding-box aspect ratio.
    elong = []
    for sl in ndimage.find_objects(labeled):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h > 0 and w > 0:
            elong.append(max(h, w) / min(h, w))
    if not elong:
        return "uniform / empty"

    mean_elong = float(np.mean(elong))
    active = float(binary.mean())

    if mean_elong > 2.2 or active > 0.45:
        return "stripe / labyrinth"
    if mean_elong < 1.6 and num >= 8:
        return "spots"
    return "mixed"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# -----------------------------
# Prediction / reconstruction
# -----------------------------
@dataclass
class AnalysisResult:
    query_64: np.ndarray                 # preprocessed input (64x64, [0,1])
    F_pred: float
    k_pred: float
    norm_pred: Tuple[float, float]       # normalized predictions in [0,1]
    saturated: List[str]                 # subset of ["F", "k"] hitting the boundary
    wavelength_px: float
    family: str
    recon_full: Optional[np.ndarray] = None   # full NxN reconstruction
    recon_64: Optional[np.ndarray] = None      # 64x64 reconstruction
    diff_64: Optional[np.ndarray] = None
    recon_mse: Optional[float] = None
    neighbors: List[dict] = field(default_factory=list)  # retrieval results


def predict(model: CNNRegressorV2, query_64: np.ndarray,
            ranges: Optional[Tuple[float, float, float, float]] = None,
            edge: float = 0.02) -> AnalysisResult:
    """Run the CNN on a preprocessed 64x64 image and package the prediction."""
    f_min, f_max, k_min, k_max = ranges if ranges is not None else load_ranges()

    with torch.no_grad():
        x = torch.tensor(query_64[None, None], dtype=torch.float32, device=DEVICE)
        norm_pred = model(x).cpu().numpy()[0]

    F_pred = float(norm_pred[0] * (f_max - f_min) + f_min)
    k_pred = float(norm_pred[1] * (k_max - k_min) + k_min)

    saturated = [
        name for name, val in zip(["F", "k"], norm_pred)
        if val <= edge or val >= 1 - edge
    ]

    return AnalysisResult(
        query_64=query_64,
        F_pred=F_pred,
        k_pred=k_pred,
        norm_pred=(float(norm_pred[0]), float(norm_pred[1])),
        saturated=saturated,
        wavelength_px=dominant_wavelength(query_64),
        family=classify_morphology(query_64),
    )


def reconstruct(result: AnalysisResult, steps: int = STEPS,
                seed: int = SIM_SEED) -> AnalysisResult:
    """Re-simulate with the predicted (F, k) and fill in reconstruction fields."""
    V = normalize(simulate(result.F_pred, result.k_pred, steps=steps, seed=seed))
    recon_64 = to_64(V)
    result.recon_full = V
    result.recon_64 = recon_64
    result.diff_64 = np.abs(result.query_64 - recon_64)
    result.recon_mse = mse(result.query_64, recon_64)
    return result


# -----------------------------
# Stage 2: optimization-based refinement (Nelder-Mead)
# -----------------------------
@dataclass
class RefineResult:
    F0: float                       # CNN warm-start F
    k0: float                       # CNN warm-start k
    F_refined: float
    k_refined: float
    error0: float                   # objective at the warm start
    error_refined: float            # objective after refinement
    n_evals: int                    # number of forward simulations used
    metric: str
    recon_full: Optional[np.ndarray] = None
    recon_64: Optional[np.ndarray] = None
    diff_64: Optional[np.ndarray] = None
    recon_mse: Optional[float] = None

    @property
    def improvement(self) -> float:
        """Fractional reduction in the objective (0 = none, 1 = perfect)."""
        if self.error0 <= 0:
            return 0.0
        return (self.error0 - self.error_refined) / self.error0


def _radial_fft_profile(img: np.ndarray, max_radius: int = 40) -> np.ndarray:
    """Normalized radial profile of the log-magnitude FFT spectrum."""
    z = normalize(img)
    z = z - z.mean()
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(z))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.floor(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(int)
    prof = np.array([
        mag[r == rad].mean() if np.any(r == rad) else 0.0
        for rad in range(max_radius)
    ], dtype=np.float64)
    return normalize(prof)


def pattern_error(sim_64: np.ndarray, target_64: np.ndarray, metric: str = "mse") -> float:
    """
    Objective comparing a simulated 64x64 field to the target.
      * "mse"     : plain pixel MSE (fast, default).
      * "spectral": pixel MSE + FFT radial-profile MSE (scale/texture aware).
    """
    if metric == "spectral":
        px = mse(sim_64, target_64)
        spec = mse(_radial_fft_profile(sim_64), _radial_fft_profile(target_64))
        return float(px + spec)
    return mse(sim_64, target_64)


def refine(target_64: np.ndarray, F0: float, k0: float, *,
           ranges: Optional[Tuple[float, float, float, float]] = None,
           metric: str = "mse", steps: int = STEPS, seed: int = SIM_SEED,
           maxiter: int = 40) -> RefineResult:
    """
    Stage 2: starting from a (CNN) warm start (F0, k0), run bounded Nelder-Mead
    over (F, k) so the *simulated* pattern matches the target. Gradient-free; each
    function evaluation is one forward Gray-Scott simulation.
    """
    from scipy.optimize import minimize

    f_min, f_max, k_min, k_max = ranges if ranges is not None else load_ranges()
    span_f = max(f_max - f_min, 1e-12)
    span_k = max(k_max - k_min, 1e-12)

    counter = {"n": 0}

    def objective(norm_params: np.ndarray) -> float:
        counter["n"] += 1
        nf = float(np.clip(norm_params[0], 0.0, 1.0))
        nk = float(np.clip(norm_params[1], 0.0, 1.0))
        F = nf * span_f + f_min
        k = nk * span_k + k_min
        sim_64 = to_64(normalize(simulate(F, k, steps=steps, seed=seed)))
        return pattern_error(sim_64, target_64, metric=metric)

    x0 = np.array([(F0 - f_min) / span_f, (k0 - k_min) / span_k], dtype=np.float64)
    x0 = np.clip(x0, 0.0, 1.0)
    error0 = objective(x0)

    res = minimize(
        objective, x0, method="Nelder-Mead",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5},
    )

    nf, nk = np.clip(res.x, 0.0, 1.0)
    F_ref = float(nf * span_f + f_min)
    k_ref = float(nk * span_k + k_min)

    out = RefineResult(
        F0=F0, k0=k0, F_refined=F_ref, k_refined=k_ref,
        error0=float(error0), error_refined=float(res.fun),
        n_evals=counter["n"], metric=metric,
    )

    # Fill in a reconstruction at the refined parameters for display.
    V = normalize(simulate(F_ref, k_ref, steps=steps, seed=seed))
    out.recon_full = V
    out.recon_64 = to_64(V)
    out.diff_64 = np.abs(target_64 - out.recon_64)
    out.recon_mse = mse(target_64, out.recon_64)
    return out


def hybrid_invert(model: CNNRegressorV2, query_64: np.ndarray, *,
                  metric: str = "mse", steps: int = STEPS, seed: int = SIM_SEED,
                  maxiter: int = 40,
                  ranges: Optional[Tuple[float, float, float, float]] = None
                  ) -> Tuple[AnalysisResult, RefineResult]:
    """
    Full two-stage inverse: CNN coarse prediction (Stage 1) -> Nelder-Mead
    refinement against the simulated pattern (Stage 2). Returns both results so
    the caller can show a CNN-only vs refined comparison.
    """
    coarse = predict(model, query_64, ranges=ranges)
    coarse = reconstruct(coarse, steps=steps, seed=seed)
    refined = refine(
        query_64, coarse.F_pred, coarse.k_pred,
        ranges=ranges, metric=metric, steps=steps, seed=seed, maxiter=maxiter,
    )
    return coarse, refined


# -----------------------------
# Retrieval from dataset
# -----------------------------
def retrieve_similar(model: CNNRegressorV2, query_64: np.ndarray,
                     top_k: int = 5):
    """
    Return the top_k most similar dataset patterns in CNN-embedding space.
    Returns (neighbors, X) where neighbors is a list of dicts and X is the
    dataset image stack, or None if the dataset is unavailable.
    """
    if DATA_FILE is None or not DATA_FILE.exists():
        return None

    data = np.load(DATA_FILE)
    X = data["X"].astype(np.float32)   # (Nsamp, 1, 64, 64)
    y = data["y"].astype(np.float32)   # (Nsamp, 2)

    model.eval()
    with torch.no_grad():
        q = torch.tensor(query_64[None, None], dtype=torch.float32, device=DEVICE)
        q_emb = model.embed(q).cpu().numpy()[0]
        q_emb /= (np.linalg.norm(q_emb) + 1e-8)

        embs = []
        for s in range(0, len(X), 256):
            batch = torch.tensor(X[s:s + 256], dtype=torch.float32, device=DEVICE)
            embs.append(model.embed(batch).cpu().numpy())
        embs = np.concatenate(embs, axis=0)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    sims = embs @ q_emb
    idx = np.argsort(-sims)[:top_k]
    neighbors = [
        {"idx": int(i), "sim": float(sims[i]), "F": float(y[i, 0]), "k": float(y[i, 1])}
        for i in idx
    ]
    return neighbors, X


def analyze(model: CNNRegressorV2, query_64: np.ndarray, *,
            steps: int = STEPS, seed: int = SIM_SEED,
            do_reconstruct: bool = True, do_retrieve: bool = False,
            top_k: int = 5,
            ranges: Optional[Tuple[float, float, float, float]] = None) -> AnalysisResult:
    """One-call convenience: predict (+ reconstruct + retrieve)."""
    result = predict(model, query_64, ranges=ranges)
    if do_reconstruct:
        result = reconstruct(result, steps=steps, seed=seed)
    if do_retrieve:
        retrieved = retrieve_similar(model, query_64, top_k=top_k)
        if retrieved is not None:
            neighbors, X = retrieved
            for nb in neighbors:
                nb["image"] = X[nb["idx"], 0]
            result.neighbors = neighbors
    return result
