"""Entry point for the geometry-vs-texture proof of concept (Chapter 3 pipeline).

For each image in images/, runs the 3-stage pipeline described in Chapter 3
(no training anywhere -- DWT is a deterministic transform, DSINE is a frozen
pretrained checkpoint, fusion is a fixed formula) and saves a side-by-side
figure: original -> wavelet energy (DWT) -> surface normals (DSINE) -> fused
geometrically grounded attention map.
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
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
MAX_SIDE = 768  # keep inference fast on CPU/MPS; DSINE handles arbitrary resolution

IMAGE_FILES = [
    "flat_pattern.jpg",
    "wrinkled_fabric.jpg",
    "corrugated_cardboard.jpg",
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
        "normals": (normals + 1) / 2,
        "attention": normalize_for_display(attention, percentile=99),
    }


def make_figure(results, title, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        ("original", "Original"),
        ("energy", "Wavelet Energy (DWT)"),
        ("normals", "Surface Normals (DSINE)"),
        ("attention", "Fused Attention Map"),
    ]
    for ax, (key, label) in zip(axes, panels):
        cmap = None if key in ("original", "normals") else "inferno"
        ax.imshow(results[key], cmap=cmap)
        ax.set_title(label, fontsize=11)
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
        image_path = IMAGES_DIR / filename
        print(f"processing {image_path}")
        results = process_image(model, device, image_path)
        out_path = OUTPUT_DIR / f"{image_path.stem}_pipeline.png"
        make_figure(results, image_path.stem, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
