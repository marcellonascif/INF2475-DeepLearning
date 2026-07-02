"""Quick standalone sanity check for the DSINE loader (src/dsine_infer.py).

Equivalent in spirit to DSINE-main/projects/dsine/test_minimal.py, but runs on
CPU/MPS via our own vendored, device-agnostic loader instead of the official
CUDA-only script.

Usage:
    uv run python test_dsine.py [path/to/image.png]

If no image is given, runs on the example image bundled with the DSINE repo.
Saves the predicted normal map (visualized as RGB) to outputs/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from PIL import Image

from dsine_infer import load_dsine, predict_normals

CHECKPOINT = "external/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
DEFAULT_IMAGE = "DSINE-main/notes/example/000000_img.png"
OUTPUT_DIR = Path("outputs")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading checkpoint: {CHECKPOINT}")
    model = load_dsine(CHECKPOINT, device)

    print(f"running inference on: {image_path}")
    img = Image.open(image_path)
    normals = predict_normals(model, img, device)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{Path(image_path).stem}_normal_pred.png"
    vis = (((normals + 1) * 0.5) * 255).astype(np.uint8)
    Image.fromarray(vis).save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
