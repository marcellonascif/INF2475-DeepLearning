"""Quantitative support for the qualitative pipeline results, using DTD
(Describable Textures Dataset, Cimpoi et al. 2014) as a proxy dataset.

This is NOT a benchmark against pixel-level ground-truth geometry/texture
labels (no such dataset exists off-the-shelf for this exact task). Instead,
it uses DTD's human-annotated texture *categories* as a coarse, independent
proxy: some categories are inherently flat/printed patterns (chequered,
dotted, striped, polka-dotted -- no real 3D surface relief), others are
inherently about real 3D surface relief (bumpy, cracked, wrinkled, pitted).
This category label was assigned by DTD's human annotators for a completely
different purpose (texture recognition), which is exactly what makes it a
fair, independent check here -- we are not the ones who decided which images
are "flat" vs "geometric".

For each image, two proxy statistics are computed directly from the raw
wavelet energy and normal variation signals (see analysis in main.py /
src/fusion.py for the full pipeline):

  1. Correlation: Pearson correlation between wavelet energy and normal
     variation, over all pixels. Low -> the two signals disagree (expected
     for pure texture). High -> the two signals agree (expected for real
     geometry).

  2. Enrichment ratio: mean normal variation among the top 10% highest
     wavelet-energy pixels (what a frequency-only method would flag as
     candidate geometric detail), divided by the image-wide mean normal
     variation. ~1 -> those candidate pixels are no more geometrically real
     than a random pixel (texture false positive). >> 1 -> they really do
     coincide with real geometry.

The core prediction: FLAT_CATEGORIES should show lower correlation and lower
enrichment ratio than GEOMETRIC_CATEGORIES, on average, across many images
per category (not just the 1 hand-picked example per case used in main.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from PIL import Image

from dsine_infer import load_dsine, predict_normals
from wavelet_energy import wavelet_energy_map
from fusion import normal_variation_map

CHECKPOINT = "external/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
DTD_IMAGES_DIR = Path("dtd_data/dtd/images")
MAX_SIDE = 384  # smaller than main.py's demo figures -- speed matters here, we need many images
TOP_PERCENT = 10

# All 47 DTD categories, manually bucketed by semantic judgment (not by
# running the pipeline first and picking favorable ones) into: clearly flat
# printed/painted patterns with no real 3D relief, clearly real 3D surface
# relief, or genuinely ambiguous at the category-name level (e.g. "veined"
# could mean marble veining, which is flat, or leaf veining, which is real
# relief). Ambiguous categories are still processed (nothing is silently
# dropped from the dataset), but excluded from the flat-vs-geometric summary
# comparison since we can't independently say which side they belong on.
FLAT_CATEGORIES = [
    "banded", "blotchy", "chequered", "dotted", "flecked", "freckled",
    "grid", "lined", "marbled", "paisley", "polka-dotted", "smeared",
    "stained", "striped", "swirly", "zigzagged",
]
GEOMETRIC_CATEGORIES = [
    "braided", "bubbly", "bumpy", "cobwebbed", "cracked", "crystalline",
    "fibrous", "frilly", "grooved", "honeycombed", "interlaced", "knitted",
    "lacelike", "matted", "meshed", "perforated", "pitted", "pleated",
    "porous", "potholed", "scaly", "stratified", "studded", "waffled",
    "wrinkled",
]
AMBIGUOUS_CATEGORIES = ["crosshatched", "gauzy", "spiralled", "sprinkled", "veined", "woven"]

OUTPUT_CSV = Path("outputs/dtd_analysis.csv")


def analyze_image(model, device, image_path):
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    image_gray = np.asarray(img.convert("L")).astype(np.float64) / 255.0
    energy_map, _ = wavelet_energy_map(image_gray, level=2)

    normals = predict_normals(model, img, device)
    norm_var = normal_variation_map(normals)

    energy_flat = energy_map.flatten()
    normvar_flat = norm_var.flatten()
    correlation = np.corrcoef(energy_flat, normvar_flat)[0, 1]

    threshold = np.percentile(energy_map, 100 - TOP_PERCENT)
    top_mask = energy_map >= threshold
    mean_normvar_top = norm_var[top_mask].mean()
    mean_normvar_all = norm_var.mean()
    enrichment = mean_normvar_top / mean_normvar_all if mean_normvar_all > 0 else np.nan

    # Raw (not per-image-normalized) mean normal variation. This is the
    # metric that actually turned out to carry the signal: DSINE's absolute
    # normal-variation magnitude differs by orders of magnitude between flat
    # and geometric surfaces, but that difference gets erased by the
    # per-image relative "enrichment ratio" above (which divides by each
    # image's own mean, discarding exactly the cross-image scale difference
    # that matters). Kept both so this is documented, not hidden.
    return correlation, enrichment, mean_normvar_all, energy_map.mean()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    model = load_dsine(CHECKPOINT, device)

    rows = []
    all_categories = (
        [("flat", c) for c in FLAT_CATEGORIES]
        + [("geometric", c) for c in GEOMETRIC_CATEGORIES]
        + [("ambiguous", c) for c in AMBIGUOUS_CATEGORIES]
    )

    t_start = time.time()
    n_done = 0
    for group, category in all_categories:
        cat_dir = DTD_IMAGES_DIR / category
        image_paths = sorted(cat_dir.glob("*.jpg"))
        print(f"[{group}] {category}: {len(image_paths)} images")

        for image_path in image_paths:
            try:
                correlation, enrichment, mean_normvar, mean_energy = analyze_image(model, device, image_path)
            except Exception as e:
                print(f"  skipping {image_path.name}: {e}")
                continue
            rows.append(
                {
                    "group": group,
                    "category": category,
                    "image": image_path.name,
                    "correlation": correlation,
                    "enrichment_ratio": enrichment,
                    "mean_normal_variation": mean_normvar,
                    "mean_wavelet_energy": mean_energy,
                }
            )
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t_start
                print(f"  ... {n_done} images done, {elapsed:.0f}s elapsed, {elapsed / n_done:.1f}s/image avg")

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        f.write("group,category,image,correlation,enrichment_ratio,mean_normal_variation,mean_wavelet_energy\n")
        for r in rows:
            f.write(
                f"{r['group']},{r['category']},{r['image']},{r['correlation']:.4f},"
                f"{r['enrichment_ratio']:.4f},{r['mean_normal_variation']:.6f},{r['mean_wavelet_energy']:.6f}\n"
            )
    print(f"\nsaved {len(rows)} rows to {OUTPUT_CSV}")

    print()
    header = f"{'group':<12}{'n':>6}{'mean corr':>12}{'mean enrich':>14}{'median normvar':>16}{'mean normvar':>14}"
    print(header)
    print("-" * len(header))
    for group in ["flat", "geometric", "ambiguous"]:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        corrs = np.array([r["correlation"] for r in group_rows])
        enrichs = np.array([r["enrichment_ratio"] for r in group_rows])
        normvars = np.array([r["mean_normal_variation"] for r in group_rows])
        tag = "  (excluded from flat-vs-geometric comparison)" if group == "ambiguous" else ""
        print(
            f"{group:<12}{len(group_rows):>6}{corrs.mean():>12.3f}{enrichs.mean():>14.2f}"
            f"{np.median(normvars):>16.6f}{normvars.mean():>14.6f}{tag}"
        )

    flat_normvars = np.array([r["mean_normal_variation"] for r in rows if r["group"] == "flat"])
    geo_normvars = np.array([r["mean_normal_variation"] for r in rows if r["group"] == "geometric"])
    ratio = np.median(geo_normvars) / np.median(flat_normvars)
    print(f"\nmedian normal-variation ratio (geometric / flat): {ratio:.1f}x")


if __name__ == "__main__":
    main()
