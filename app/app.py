"""
Pattern-to-Parameter Analyzer -- interactive demo
=================================================

A small Streamlit front-end over the existing Gray-Scott inverse pipeline.
Upload (or pick) a pattern image; the app:

  * pre-processes it to the 64x64 representation the CNN was trained on,
  * predicts the most likely generating parameters (F, k),
  * re-simulates the Gray-Scott model and shows input vs reconstruction vs diff,
  * (optionally) retrieves the most similar dataset patterns.

No new method is introduced -- this is a usability layer over
`scripts/analyzer_core.py`.

Run from the repo root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make scripts/ importable so we reuse the shared analyzer core.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyzer_core as core  # noqa: E402


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return core.load_model()


@st.cache_data(show_spinner=False)
def get_ranges():
    return core.load_ranges()


@st.cache_data(show_spinner="Re-simulating Gray-Scott ...")
def cached_simulate(F: float, k: float, steps: int, seed: int) -> np.ndarray:
    return core.normalize(core.simulate(F, k, steps=steps, seed=seed))


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Pattern-to-Parameter Analyzer", layout="wide")
st.title("🌀 Pattern-to-Parameter Analyzer")
st.caption(
    "Map a Turing-like pattern back to the Gray-Scott reaction-diffusion regime "
    "(feed F, kill k) most likely to generate it, then reconstruct it."
)

# --- Environment status ---------------------------------------------------
model_ok = core.model_available()
data_ok = core.dataset_available()

cols = st.columns(2)
cols[0].metric("Trained model", "ready ✅" if model_ok else "missing ❌")
cols[1].metric("Dataset (for retrieval)", "ready ✅" if data_ok else "missing ❌")

if not model_ok:
    st.error(
        f"Trained model not found at `{core.MODEL_FILE}`.\n\n"
        "Train it first:\n```\npython3 scripts/train_cnn_regressor_v2.py\n```"
    )
    st.stop()

f_min, f_max, k_min, k_max = get_ranges()
st.info(
    f"Trained parameter range: **F ∈ [{f_min:.4f}, {f_max:.4f}]**, "
    f"**k ∈ [{k_min:.4f}, {k_max:.4f}]**.  Inputs whose morphology falls outside "
    "this regime (e.g. clean parallel zebra stripes) will saturate at the "
    "boundary — a deliberate signal of the model's generalization limits."
)

# --- Sidebar controls -----------------------------------------------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload a pattern image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    binarize = st.checkbox(
        "Otsu-binarize (recommended for real photos / drawings)", value=False
    )

    st.header("Reconstruction")
    steps = st.slider("Simulation steps", 500, 5000, core.STEPS, step=500)
    seed = st.number_input("Seed", value=core.SIM_SEED, step=1)

    st.header("Retrieval")
    do_retrieve = st.checkbox(
        "Find similar dataset patterns", value=False, disabled=not data_ok,
        help=None if data_ok else "Dataset npz not present.",
    )
    top_k = st.slider("Top-k neighbors", 3, 8, 5, disabled=not do_retrieve)

if uploaded is None:
    st.warning("⬅️ Upload a pattern image from the sidebar to begin.")
    st.stop()

# --- Run the pipeline -----------------------------------------------------
query = core.preprocess_image(uploaded, binarize=binarize)
model = get_model()

result = core.predict(model, query, ranges=(f_min, f_max, k_min, k_max))

# Reconstruction uses the cached simulator so re-runs are instant.
V = cached_simulate(result.F_pred, result.k_pred, int(steps), int(seed))
recon_64 = core.to_64(V)
diff_64 = np.abs(result.query_64 - recon_64)
recon_mse = core.mse(result.query_64, recon_64)

# --- Prediction summary ---------------------------------------------------
st.subheader("Prediction")
m = st.columns(4)
m[0].metric("Predicted F", f"{result.F_pred:.4f}")
m[1].metric("Predicted k", f"{result.k_pred:.4f}")
m[2].metric("Morphology family", result.family)
m[3].metric("Reconstruction MSE", f"{recon_mse:.4f}")

if result.saturated:
    st.warning(
        f"⚠️ {', '.join(result.saturated)} saturated at the trained-range "
        "boundary — the input pattern is likely outside the Gray-Scott regime "
        "the model has seen. Treat the prediction as a nearest in-range guess."
    )

st.caption(
    f"Dominant wavelength ≈ {result.wavelength_px:.1f}px "
    f"(~{core.OUTPUT_SIZE / result.wavelength_px:.1f} feature repeats across 64px). "
    "Use this for the surface-area / scale calibration discussion."
)

# --- Visual comparison ----------------------------------------------------
st.subheader("Input → Reconstruction → Difference")
v = st.columns(4)
v[0].image(result.query_64, caption="Input (preprocessed 64×64)",
           clamp=True, use_container_width=True)
v[1].image(V, caption=f"Reconstruction (full {core.N}×{core.N})",
           clamp=True, use_container_width=True)
v[2].image(recon_64, caption="Reconstruction (64×64)",
           clamp=True, use_container_width=True)
# Diff as a heat-style map (red = larger error).
diff_rgb = np.zeros((*diff_64.shape, 3), dtype=np.float32)
diff_rgb[..., 0] = diff_64 / (diff_64.max() + 1e-8)
v[3].image(diff_rgb, caption=f"Abs diff (MSE={recon_mse:.4f})",
           clamp=True, use_container_width=True)

# --- Retrieval gallery ----------------------------------------------------
if do_retrieve:
    st.subheader("Similar dataset patterns")
    with st.spinner("Searching dataset in CNN-embedding space ..."):
        retrieved = core.retrieve_similar(model, result.query_64, top_k=int(top_k))
    if retrieved is None:
        st.info("Dataset not available for retrieval.")
    else:
        neighbors, X = retrieved
        gcols = st.columns(len(neighbors))
        for col, nb in zip(gcols, neighbors):
            col.image(
                X[nb["idx"], 0],
                caption=f"sim={nb['sim']:.2f}\nF={nb['F']:.3f}, k={nb['k']:.3f}",
                clamp=True, use_container_width=True,
            )

st.divider()
st.caption(
    "Honest framing: the predicted (F, k) is the Gray-Scott regime whose "
    "morphology most closely matches the input — not a measurement of true "
    "biological parameters."
)
