"""Standard ML evaluation: can each signal discriminate flat vs. geometric
DTD categories, and does combining them beat either alone?

Reads outputs/dtd_analysis.csv (produced by analysis.py) and frames the
qualitative pipeline as a binary classification problem:
  label = 1 if the image's DTD category is in GEOMETRIC_CATEGORIES, else 0
  (categories in "ambiguous" are excluded here, same as in analysis.py)

For three candidate "classifiers" -- each is just a single per-image scalar
score, no training involved -- we report ROC-AUC (0.5 = no better than
random, 1.0 = perfect separation):

  1. wavelet energy alone   (the frequency-only baseline, e.g. FLAIR/WEGE)
  2. normal variation alone (the geometry-only baseline, e.g. DSINE alone)
  3. combined score = wavelet energy * normal variation
     (a per-image scalar analogue of the pixel-wise fusion in src/fusion.py;
     not identical to the real fused attention map, since that operates
     per-pixel and per-image-normalized, but same spirit: only high when
     BOTH signals are high)

Chapter 1's claim, made quantitative: wavelet energy alone should be close
to AUC=0.5 (can't tell texture from geometry), normal variation alone should
be well above 0.5 (DSINE alone already discriminates reasonably), and the
combined score should be at or above the best individual signal.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

CSV_PATH = Path("outputs/dtd_analysis.csv")
OUTPUT_PLOT = Path("outputs/roc_curves.png")


def load_rows():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["group"] not in ("flat", "geometric"):
                continue
            rows.append(
                {
                    "label": 1 if r["group"] == "geometric" else 0,
                    "energy": float(r["mean_wavelet_energy"]),
                    "normvar": float(r["mean_normal_variation"]),
                }
            )
    return rows


def main():
    rows = load_rows()
    y = np.array([r["label"] for r in rows])
    energy = np.array([r["energy"] for r in rows])
    normvar = np.array([r["normvar"] for r in rows])
    combined = energy * normvar

    signals = {
        "wavelet energy alone (frequency-only baseline)": energy,
        "normal variation alone (DSINE-only baseline)": normvar,
        "combined (energy x normvar, our method's spirit)": combined,
    }

    print(f"n = {len(y)} images ({y.sum()} geometric, {len(y) - y.sum()} flat)\n")
    print(f"{'signal':<52}{'AUC':>8}")
    print("-" * 60)

    plt.figure(figsize=(6, 6))
    for name, score in signals.items():
        auc = roc_auc_score(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        print(f"{name:<52}{auc:>8.3f}")
        plt.plot(fpr, tpr, label=f"{name.split(' (')[0]} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Discriminating real geometry from flat texture (DTD categories)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nsaved ROC plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
