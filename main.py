"""Entry point for the geometry-vs-texture proof of concept (Chapter 3 pipeline).

For each image in IMAGE_FILES, runs the 3-stage pipeline described in Chapter 3
(no training anywhere -- DWT is a deterministic transform, DSINE is a frozen
pretrained checkpoint, fusion is a fixed formula) and saves a side-by-side
figure with 4 panels, all directly comparable (same heatmap style):
  original photo -> wavelet-only attention (frequency-only baseline) ->
  DSINE-only attention (normal variation, the geometry-only baseline) ->
  fused attention (wavelet x DSINE, our method).

Showing all three attention signals side by side (instead of DSINE's raw
normal-direction output, which is not itself an attention map) makes the
ablation explicit: you can see directly whether the fused result adds
anything over either baseline alone, not just read it off the aggregate
AUC numbers in roc_analysis.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dsine_infer import load_dsine, predict_normals
from wavelet_energy import wavelet_energy_map, normalize_for_display
from fusion import normal_variation_map, fuse

CHECKPOINT = "external/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
OUTPUT_DIR = Path("outputs")
MAX_SIDE = 768  # keep inference fast on CPU/MPS; DSINE handles arbitrary resolution

# Paths are relative to the project root. These are examples pulled from
# dtd_data/ (the same DTD images used for the quantitative validation in
# analysis.py), chosen for how clearly they illustrate the pipeline.
IMAGE_FILES = [
    "dtd_data/dtd/images/bumpy/bumpy_0106.jpg",
    "dtd_data/dtd/images/dotted/dotted_0041.jpg",
    "dtd_data/dtd/images/banded/banded_0023.jpg",
    "dtd_data/dtd/images/fibrous/fibrous_0095.jpg",
]


def process_image(model, device, image_path):
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    image_gray = np.asarray(img.convert("L")).astype(np.float64) / 255.0
    energy_map, _ = wavelet_energy_map(image_gray)

    normals = predict_normals(model, img, device)
    norm_var = normal_variation_map(normals)
    attention = fuse(energy_map, norm_var)

    return {
        "original": np.asarray(img),
        "energy": normalize_for_display(energy_map, percentile=99),
        "norm_var": normalize_for_display(norm_var, percentile=99),
        "attention": normalize_for_display(attention, percentile=99),
    }


def make_figure(results, title, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        ("original", "Original"),
        ("energy", "Wavelet-only Attention\n(frequency-only baseline)"),
        ("norm_var", "DSINE-only Attention\n(geometry-only baseline)"),
        ("attention", "Fused Attention\n(wavelet x DSINE, our method)"),
    ]
    for ax, (key, label) in zip(axes, panels):
        cmap = None if key == "original" else "inferno"
        ax.imshow(results[key], cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    model = load_dsine(CHECKPOINT, device)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename in IMAGE_FILES:
        image_path = Path(filename)
        print(f"processing {image_path}")
        results = process_image(model, device, image_path)
        out_path = OUTPUT_DIR / f"{image_path.stem}_pipeline.png"
        make_figure(results, image_path.stem, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
