"""
Pattern-to-Parameter Analyzer -- interactive demo
=================================================

A polished Streamlit front-end over the existing Gray-Scott inverse pipeline.
Upload a pattern image and the app:

  * pre-processes it to the 64x64 representation the CNN was trained on,
  * predicts the most likely generating parameters (F, k),
  * re-simulates the Gray-Scott model (input vs reconstruction vs difference),
  * optionally refines the guess (CNN -> Nelder-Mead) and retrieves similar
    dataset patterns.

No new method is introduced -- this is a usability layer over
`scripts/analyzer_core.py`.

Run from the repo root:
    python3 -m streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Make scripts/ importable so we reuse the shared analyzer core.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyzer_core as core  # noqa: E402

ACCENT = "#7C5CFF"
PATTERN_CMAP = "magma"


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
# Visual helpers
# -----------------------------
def colorize(arr: np.ndarray, cmap: str = PATTERN_CMAP) -> np.ndarray:
    """Map a [0,1] grayscale array to an RGB uint8 image via a matplotlib cmap."""
    a = np.atleast_2d(np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0))
    rgb = plt.get_cmap(cmap)(a)[..., :3]
    return (rgb * 255).astype(np.uint8)


def diff_image(diff: np.ndarray) -> np.ndarray:
    d = diff / (diff.max() + 1e-8)
    return colorize(d, "inferno")


def param_map(ranges, cnn_pt, refined_pt=None):
    """Scatter the predicted (F, k) inside the trained parameter rectangle."""
    f_min, f_max, k_min, k_max = ranges
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("#0e1117")
    ax.add_patch(plt.Rectangle((f_min, k_min), f_max - f_min, k_max - k_min,
                               fill=True, color=ACCENT, alpha=0.08))
    ax.add_patch(plt.Rectangle((f_min, k_min), f_max - f_min, k_max - k_min,
                               fill=False, color=ACCENT, lw=1.5, ls="--"))
    ax.scatter(*cnn_pt, s=180, color="#4DA3FF", edgecolor="white",
               zorder=5, label="CNN")
    if refined_pt is not None:
        ax.scatter(*refined_pt, s=200, marker="*", color="#5CFFA6",
                   edgecolor="white", zorder=6, label="Refined")
        ax.annotate("", xy=refined_pt, xytext=cnn_pt,
                    arrowprops=dict(arrowstyle="->", color="white", alpha=0.6))
    ax.set_xlim(f_min - 0.001, f_max + 0.001)
    ax.set_ylim(k_min - 0.001, k_max + 0.001)
    ax.set_xlabel("feed  F", color="#cfd2dc")
    ax.set_ylabel("kill  k", color="#cfd2dc")
    ax.set_title("Trained parameter space", color="#cfd2dc")
    ax.tick_params(colors="#8b8fa3")
    for s in ax.spines.values():
        s.set_color("#2a2f3a")
    ax.legend(facecolor="#0e1117", edgecolor="#2a2f3a", labelcolor="#cfd2dc")
    fig.tight_layout()
    return fig


# -----------------------------
# Page setup + styling
# -----------------------------
st.set_page_config(
    page_title="Pattern-to-Parameter Analyzer",
    page_icon="🌀",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
      .hero {{
        background: linear-gradient(135deg, {ACCENT} 0%, #3A2A8C 100%);
        border-radius: 16px; padding: 26px 30px; margin-bottom: 18px;
        color: white; box-shadow: 0 8px 30px rgba(124,92,255,0.25);
      }}
      .hero h1 {{ margin: 0; font-size: 2rem; }}
      .hero p {{ margin: 6px 0 0; opacity: 0.92; font-size: 1.02rem; }}
      .pill {{
        display:inline-block; padding:3px 12px; border-radius:999px;
        font-size:0.8rem; margin-right:8px; background:rgba(255,255,255,0.18);
      }}
      div[data-testid="stMetric"] {{
        background: #161a23; border: 1px solid #262b36;
        border-radius: 12px; padding: 14px 16px;
      }}
      div[data-testid="stImage"] img {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌀 Pattern-to-Parameter Analyzer</h1>
      <p>Map a Turing-like pattern back to the Gray–Scott reaction–diffusion
      regime (feed <b>F</b>, kill <b>k</b>) that most likely generated it —
      then reconstruct, refine, and retrieve.</p>
      <div style="margin-top:12px;">
        <span class="pill">CNN inverse</span>
        <span class="pill">Nelder–Mead refine</span>
        <span class="pill">embedding retrieval</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Environment status ---------------------------------------------------
model_ok = core.model_available()
data_ok = core.dataset_available()

if not model_ok:
    st.error(
        f"Trained model not found at `{core.MODEL_FILE}`.\n\n"
        "Train it first:\n```\npython3 scripts/train_cnn_regressor_v2.py\n```"
    )
    st.stop()

f_min, f_max, k_min, k_max = get_ranges()

# --- Sidebar controls -----------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    sc = st.columns(2)
    sc[0].caption("Model")
    sc[0].markdown("**ready ✅**" if model_ok else "**missing ❌**")
    sc[1].caption("Dataset")
    sc[1].markdown("**ready ✅**" if data_ok else "**missing ❌**")
    st.divider()

    st.markdown("#### Input")
    uploaded = st.file_uploader(
        "Pattern image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    binarize = st.checkbox(
        "Otsu-binarize (real photos / drawings)", value=False
    )

    st.markdown("#### Reconstruction")
    steps = st.slider(
        "Simulation steps", 500, 5000, core.STEPS, step=500,
        help="Keep this at the value the input was generated with (the dataset "
             "uses 3000). Using more steps does NOT reduce error and often "
             "increases it, because the pattern keeps evolving past the target.",
    )
    seed = st.number_input("Seed", value=core.SIM_SEED, step=1)

    st.markdown("#### Refinement (Stage 2)")
    do_refine = st.checkbox(
        "Refine CNN guess (Nelder–Mead)", value=False,
        help="Warm-start a gradient-free optimizer from the CNN prediction and "
             "tune (F, k) so the simulated pattern matches the input.",
    )
    metric = st.selectbox("Objective", ["mse", "spectral"], disabled=not do_refine)
    maxiter = st.slider("Max iterations", 10, 80, 40, step=5, disabled=not do_refine)

    st.markdown("#### Retrieval")
    do_retrieve = st.checkbox(
        "Find similar dataset patterns", value=False, disabled=not data_ok,
        help=None if data_ok else "Dataset npz not present.",
    )
    top_k = st.slider("Top-k neighbors", 3, 8, 5, disabled=not do_retrieve)

# --- Empty state ----------------------------------------------------------
if uploaded is None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("How it works")
        st.markdown(
            "1. **Upload** a pattern image in the sidebar.\n"
            "2. The **CNN** predicts the generating parameters *(F, k)*.\n"
            "3. The model is **re-simulated** and compared to your input.\n"
            "4. Optionally **refine** the guess and **retrieve** similar patterns."
        )
        st.info(
            f"Trained range: **F ∈ [{f_min:.4f}, {f_max:.4f}]**, "
            f"**k ∈ [{k_min:.4f}, {k_max:.4f}]**. Out-of-regime inputs "
            "(e.g. clean parallel zebra stripes) saturate at the boundary — a "
            "deliberate signal of the model's generalization limits."
        )
    with c2:
        demo = cached_simulate(0.026, 0.058, core.STEPS, core.SIM_SEED)
        st.image(colorize(demo), caption="Example Gray–Scott pattern",
                 use_container_width=True)
    st.stop()

# --- Run the pipeline -----------------------------------------------------
query = core.preprocess_image(uploaded, binarize=binarize)
model = get_model()
result = core.predict(model, query, ranges=(f_min, f_max, k_min, k_max))

V = cached_simulate(result.F_pred, result.k_pred, int(steps), int(seed))
recon_64 = core.to_64(V)
diff_64 = np.abs(result.query_64 - recon_64)
recon_mse = core.mse(result.query_64, recon_64)

# Compute refinement up-front (so the parameter map can show it).
ref = None
if do_refine:
    with st.spinner("Refining (simulate-and-match) ..."):
        ref = core.refine(
            result.query_64, result.F_pred, result.k_pred,
            ranges=(f_min, f_max, k_min, k_max),
            metric=metric, steps=int(steps), seed=int(seed), maxiter=int(maxiter),
        )

# --- Top metric cards -----------------------------------------------------
m = st.columns(4)
m[0].metric("Predicted F", f"{result.F_pred:.4f}")
m[1].metric("Predicted k", f"{result.k_pred:.4f}")
m[2].metric("Morphology", result.family)
m[3].metric("Reconstruction MSE", f"{recon_mse:.4f}")

if result.saturated:
    st.warning(
        f"⚠️ {', '.join(result.saturated)} saturated at the trained-range "
        "boundary — the input is likely outside the regime the model has seen. "
        "Treat it as a nearest in-range guess."
    )

# --- Tabs -----------------------------------------------------------------
tab_recon, tab_map, tab_refine, tab_similar, tab_about = st.tabs(
    ["🔍 Reconstruction", "🗺️ Parameter map", "🎯 Refine", "🖼️ Similar", "ℹ️ About"]
)

with tab_recon:
    v = st.columns(4)
    v[0].image(colorize(result.query_64), caption="Input (64×64)",
               use_container_width=True)
    v[1].image(colorize(V), caption=f"Reconstruction ({core.N}×{core.N})",
               use_container_width=True)
    v[2].image(colorize(recon_64), caption="Reconstruction (64×64)",
               use_container_width=True)
    v[3].image(diff_image(diff_64), caption=f"Abs diff (MSE={recon_mse:.4f})",
               use_container_width=True)
    st.caption(
        f"Dominant wavelength ≈ {result.wavelength_px:.1f}px "
        f"(~{core.OUTPUT_SIZE / result.wavelength_px:.1f} feature repeats across "
        "64px) — useful for the surface-area / scale-calibration discussion."
    )
    if not result.saturated and recon_mse > 0.02:
        st.info(
            "ℹ️ A large pixel difference here does **not** mean the prediction is "
            "wrong. Gray–Scott is highly sensitive: a change of ~0.001 in *k* "
            "reorganizes the exact spot/stripe layout, so two patterns from the "
            "**same morphology family** can differ a lot pixel-by-pixel. This is "
            "the project's central *sensitivity / non-uniqueness* finding — not a "
            "bug. (Tip: keep simulation steps at the value the input was made "
            "with; raising it increases the difference.)"
        )

with tab_map:
    mc1, mc2 = st.columns([1, 1])
    with mc1:
        refined_pt = (ref.F_refined, ref.k_refined) if ref else None
        st.pyplot(param_map((f_min, f_max, k_min, k_max),
                            (result.F_pred, result.k_pred), refined_pt))
    with mc2:
        st.markdown("**Where the prediction lands**")
        st.markdown(
            "- 🔵 **CNN** — the network's instant estimate.\n"
            "- ⭐ **Refined** — after Nelder–Mead (enable refinement).\n\n"
            "Points near the dashed boundary mean the input is at the edge of "
            "the trained regime."
        )
        if ref:
            st.markdown(
                f"The optimizer moved the estimate by "
                f"**ΔF = {ref.F_refined - ref.F0:+.4f}**, "
                f"**Δk = {ref.k_refined - ref.k0:+.4f}**."
            )

with tab_refine:
    if not do_refine:
        st.info("Enable **Refine CNN guess** in the sidebar to run Stage 2.")
    else:
        rc = st.columns(4)
        rc[0].metric("Refined F", f"{ref.F_refined:.4f}",
                     delta=f"{ref.F_refined - ref.F0:+.4f}")
        rc[1].metric("Refined k", f"{ref.k_refined:.4f}",
                     delta=f"{ref.k_refined - ref.k0:+.4f}")
        rc[2].metric("Recon MSE", f"{ref.recon_mse:.4f}",
                     delta=f"{ref.recon_mse - recon_mse:+.4f}", delta_color="inverse")
        rc[3].metric("Objective ↓", f"{ref.improvement * 100:.0f}%",
                     help=f"{ref.error0:.4f} → {ref.error_refined:.4f} in "
                          f"{ref.n_evals} simulations")
        rv = st.columns(3)
        rv[0].image(colorize(recon_64),
                    caption=f"CNN only (F={ref.F0:.4f}, k={ref.k0:.4f})",
                    use_container_width=True)
        rv[1].image(colorize(ref.recon_64),
                    caption=f"Refined (F={ref.F_refined:.4f}, k={ref.k_refined:.4f})",
                    use_container_width=True)
        rv[2].image(diff_image(ref.diff_64), caption="Abs diff (refined)",
                    use_container_width=True)
        st.caption(
            f"Refinement used **{ref.n_evals}** forward simulations "
            f"(objective: {metric}). The CNN supplies the warm start; the "
            "optimizer tunes (F, k) against the actual simulated pattern."
        )

with tab_similar:
    if not do_retrieve:
        st.info("Enable **Find similar dataset patterns** in the sidebar "
                "(requires the dataset npz).")
    else:
        with st.spinner("Searching dataset in CNN-embedding space ..."):
            retrieved = core.retrieve_similar(model, result.query_64, top_k=int(top_k))
        if retrieved is None:
            st.info("Dataset not available for retrieval.")
        else:
            neighbors, X = retrieved
            gcols = st.columns(len(neighbors))
            for col, nb in zip(gcols, neighbors):
                col.image(
                    colorize(X[nb["idx"], 0]),
                    caption=f"sim={nb['sim']:.2f}\nF={nb['F']:.3f}, k={nb['k']:.3f}",
                    use_container_width=True,
                )

with tab_about:
    st.markdown(
        "#### About this tool\n"
        "This packages the project's existing **forward Gray–Scott simulator** "
        "and **trained CNN** into a usable inverse-analysis tool — no new method "
        "is introduced.\n\n"
        "**Pipeline:** preprocess → CNN prediction *(Stage 1)* → optional "
        "Nelder–Mead refinement *(Stage 2)* → reconstruction & retrieval.\n\n"
        "> **Honest framing:** the predicted *(F, k)* is the Gray–Scott regime "
        "whose morphology most closely matches the input — *not* a measurement "
        "of true biological parameters."
    )
