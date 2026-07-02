"""Quick standalone sanity check for the wavelet energy map (src/wavelet_energy.py).

Usage:
    uv run python test_wavelet.py [path/to/image.png]

If no image is given, runs on the example image bundled with the DSINE repo.
Saves the energy map (grayscale) and the low-frequency reconstruction to outputs/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from PIL import Image

from wavelet_energy import wavelet_energy_map, normalize_for_display

DEFAULT_IMAGE = "DSINE-main/notes/example/000000_img.png"
OUTPUT_DIR = Path("outputs")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    print(f"running wavelet energy map on: {image_path}")

    img = Image.open(image_path).convert("L")
    image_gray = np.asarray(img).astype(np.float64) / 255.0

    energy_map, low_freq = wavelet_energy_map(image_gray)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    energy_vis = (normalize_for_display(energy_map) * 255).astype(np.uint8)
    Image.fromarray(energy_vis).save(OUTPUT_DIR / f"{stem}_wavelet_energy.png")

    low_freq_vis = (np.clip(low_freq, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(low_freq_vis).save(OUTPUT_DIR / f"{stem}_low_freq.png")

    print(f"saved: {OUTPUT_DIR / f'{stem}_wavelet_energy.png'}")
    print(f"saved: {OUTPUT_DIR / f'{stem}_low_freq.png'}")


if __name__ == "__main__":
    main()
