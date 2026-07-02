"""Quick standalone sanity check that chains all 3 stages in memory:
DWT wavelet energy -> DSINE normals -> explicit fusion.

Usage:
    uv run python test_fusion.py [path/to/image.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from PIL import Image

from dsine_infer import load_dsine, predict_normals
from wavelet_energy import wavelet_energy_map, normalize_for_display
from fusion import normal_variation_map, fuse

CHECKPOINT = "external/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
DEFAULT_IMAGE = "DSINE-main/notes/example/000000_img.png"
OUTPUT_DIR = Path("outputs")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    img = Image.open(image_path).convert("RGB")

    # Stage 1: wavelet energy map (grayscale, in-memory array)
    image_gray = np.asarray(img.convert("L")).astype(np.float64) / 255.0
    energy_map, _ = wavelet_energy_map(image_gray)
    print(f"energy_map shape: {energy_map.shape}")

    # Stage 2: DSINE normals (same image, in-memory array)
    model = load_dsine(CHECKPOINT, device)
    normals = predict_normals(model, img, device)
    print(f"normals shape: {normals.shape}")

    # Stage 3: fuse the two in-memory arrays directly (no files in between)
    norm_var = normal_variation_map(normals)
    attention = fuse(energy_map, norm_var)
    print(f"attention shape: {attention.shape}, range: [{attention.min():.3f}, {attention.max():.3f}]")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    Image.fromarray((normalize_for_display(norm_var) * 255).astype(np.uint8)).save(
        OUTPUT_DIR / f"{stem}_normal_variation.png"
    )
    Image.fromarray((normalize_for_display(attention, percentile=99) * 255).astype(np.uint8)).save(
        OUTPUT_DIR / f"{stem}_attention.png"
    )
    print(f"saved: {OUTPUT_DIR / f'{stem}_normal_variation.png'}")
    print(f"saved: {OUTPUT_DIR / f'{stem}_attention.png'}")


if __name__ == "__main__":
    main()
