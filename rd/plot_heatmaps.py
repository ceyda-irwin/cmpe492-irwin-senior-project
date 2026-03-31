from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFINED_DIR = ROOT / "outputs" / "sweeps" / "refined_sweep_outputs"

df = pd.read_csv(REFINED_DIR / "results.csv")

for metric in ["std_V", "gradient_strength", "active_ratio", "steps_used"]:
    pivot = df.pivot(index="k", columns="F", values=metric)

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pivot.columns)), [f"{x:.4f}" for x in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f"{y:.4f}" for y in pivot.index])
    plt.xlabel("F")
    plt.ylabel("k")
    plt.title(f"Heatmap of {metric}")
    plt.tight_layout()
    plt.savefig(REFINED_DIR / f"heatmap_{metric}.png", dpi=150)
    plt.close()

print("Heatmaps saved.")
