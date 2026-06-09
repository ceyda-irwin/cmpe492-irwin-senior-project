"""
Pattern-to-Parameter Analyzer (CLI)
===================================

Take an arbitrary input image (a synthetic Gray-Scott pattern, a hand drawing,
or a real animal-skin photo) and:

  1. Pre-process it into the same 64x64 normalized representation the CNN was
     trained on (grayscale -> resize -> [0,1] normalize, optional binarize).
  2. Predict the most likely generating parameters (F, k) with the trained CNN.
  3. Re-simulate the Gray-Scott model with the predicted (F, k).
  4. Show input | reconstruction | abs-diff side by side and report an MSE.
  5. (Optional) Retrieve the nearest dataset patterns ("similar morphology").

This does NOT add a new method. It packages the existing forward simulator +
trained CNN into a usable analysis tool. All heavy lifting lives in
`scripts/analyzer_core.py`, shared with the interactive app.

Usage (from repo root):
    python3 scripts/predict_from_image.py path/to/image.png
    python3 scripts/predict_from_image.py path/to/leopard.jpg --binarize
    python3 scripts/predict_from_image.py img.png --retrieve   # needs dataset npz

Honest framing: the predicted (F, k) is "the Gray-Scott regime whose morphology
most closely matches the input", NOT a claim about true biological parameters.
The training range is narrow; inputs outside it (e.g. clean parallel zebra
stripes) saturate at the range boundary -- a useful demonstration of the
model's generalization limits.
"""

import argparse
import sys
from pathlib import Path

# Allow running as "python3 scripts/predict_from_image.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

import analyzer_core as core


def main():
    ap = argparse.ArgumentParser(description="Pattern-to-Parameter Analyzer (CLI)")
    ap.add_argument("image", type=str, help="path to an input image")
    ap.add_argument("--binarize", action="store_true",
                    help="Otsu-binarize the input (good for real photos / drawings)")
    ap.add_argument("--retrieve", action="store_true",
                    help="also retrieve nearest dataset patterns (needs rd_dataset.npz)")
    ap.add_argument("--steps", type=int, default=core.STEPS,
                    help="simulation steps for reconstruction")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = core.load_model()
    query = core.preprocess_image(img_path, binarize=args.binarize)

    result = core.analyze(
        model, query,
        steps=args.steps,
        do_reconstruct=True,
        do_retrieve=args.retrieve,
    )

    sat_note = ""
    if result.saturated:
        which = ", ".join(result.saturated)
        sat_note = f"  [WARNING: {which} saturated at range boundary -> input likely outside trained regime]"

    print(f"Input             : {img_path.name}")
    print(f"Morphology family : {result.family}")
    print(f"Dominant wavelength (px): {result.wavelength_px:.1f}  "
          f"(~{core.OUTPUT_SIZE / result.wavelength_px:.1f} feature repeats across 64px)")
    print(f"Predicted F       : {result.F_pred:.4f}   (norm {result.norm_pred[0]:.3f})")
    print(f"Predicted k       : {result.k_pred:.4f}   (norm {result.norm_pred[1]:.3f}){sat_note}")
    print(f"Reconstruction MSE (input vs regen): {result.recon_mse:.5f}")

    # Main analysis figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(result.query_64, cmap="gray"); axes[0].set_title("Input (preprocessed 64x64)")
    axes[1].imshow(result.recon_full, cmap="gray"); axes[1].set_title(f"Reconstruction (full {core.N}x{core.N})")
    axes[2].imshow(result.recon_64, cmap="gray"); axes[2].set_title("Reconstruction (64x64)")
    axes[3].imshow(result.diff_64, cmap="hot"); axes[3].set_title(f"Abs diff\nMSE={result.recon_mse:.4f}")
    for a in axes:
        a.axis("off")
    fig.suptitle(
        f"Pattern-to-Parameter Analyzer  |  Pred F={result.F_pred:.4f}, k={result.k_pred:.4f}{sat_note}",
        fontsize=12,
    )
    plt.tight_layout()
    out = core.OUTPUT_DIR / f"analysis_{img_path.stem}.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"Saved analysis figure -> {out}")

    # Optional retrieval gallery
    if args.retrieve:
        if not result.neighbors:
            print(f"[retrieve] dataset not found at {core.DATA_FILE}; skipped retrieval.")
        else:
            print("\nNearest dataset patterns (cosine sim in CNN-embedding space):")
            nbs = result.neighbors
            fig, axes = plt.subplots(1, len(nbs) + 1, figsize=(3 * (len(nbs) + 1), 3))
            axes[0].imshow(result.query_64, cmap="gray"); axes[0].set_title("query"); axes[0].axis("off")
            for j, nb in enumerate(nbs):
                print(f"  #{j+1}  sim={nb['sim']:.3f}  F={nb['F']:.4f} k={nb['k']:.4f}  (idx {nb['idx']})")
                axes[j + 1].imshow(nb["image"], cmap="gray")
                axes[j + 1].set_title(f"sim={nb['sim']:.2f}\nF={nb['F']:.3f},k={nb['k']:.3f}", fontsize=8)
                axes[j + 1].axis("off")
            plt.tight_layout()
            out2 = core.OUTPUT_DIR / f"retrieval_{img_path.stem}.png"
            plt.savefig(out2, dpi=170, bbox_inches="tight")
            plt.close()
            print(f"Saved retrieval gallery -> {out2}")


if __name__ == "__main__":
    main()
