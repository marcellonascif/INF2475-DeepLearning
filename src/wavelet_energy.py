"""WEGE-style per-pixel wavelet energy map (classical DWT, no learning involved).

Follows the formulation described in FLAIR's Wavelet-Energy-Guided Encoding
(WEGE): the image is decomposed via a multi-level 2D discrete wavelet
transform, then reconstructed using only the coarsest approximation
coefficients (all detail coefficients zeroed out). The residual between the
original image and this purely low-frequency reconstruction is the
per-pixel high-frequency energy: it is large wherever the image has strong
high-frequency content, regardless of whether that content comes from real
geometry or from flat texture (that disambiguation happens later, in the
fusion stage with DSINE's normals).

Using a multi-level decomposition (rather than a single level) aggregates
detail across scales into one map, functioning as a classical, deterministic
analogue of the multi-scale feature pyramids used in CNN detectors
(FPN/SPPF) -- no network, no training, just recursive wavelet decomposition.
"""
import numpy as np
import pywt


def _match_shape(arr, shape):
    """Crop or pad `arr` to `shape` (wavelet reconstruction can be a few px off)."""
    out = np.zeros(shape, dtype=arr.dtype)
    h, w = min(arr.shape[0], shape[0]), min(arr.shape[1], shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def wavelet_energy_map(image_gray, wavelet="db4", level=3):
    """Compute a per-pixel high-frequency wavelet energy map.

    Args:
        image_gray: 2D float array (H, W), grayscale image in [0, 1].
        wavelet: wavelet family (Daubechies-4 by default, a common choice
            for image analysis; smooth enough to avoid ringing, compact
            enough to localize edges).
        level: number of decomposition levels. Higher level = the "low
            frequency" reference is smoother, so more scales of detail are
            captured as high frequency energy.

    Returns:
        energy_map: (H, W) float array, per-pixel squared high-frequency
            residual (not normalized).
        low_freq: (H, W) float array, the purely low-frequency (approximation
            only) reconstruction, useful for sanity-checking the decomposition.
    """
    h, w = image_gray.shape
    coeffs = pywt.wavedec2(image_gray, wavelet=wavelet, level=level)
    cA = coeffs[0]

    zeroed_details = [
        tuple(np.zeros_like(d) for d in detail_level) for detail_level in coeffs[1:]
    ]
    low_freq = pywt.waverec2([cA] + zeroed_details, wavelet=wavelet)
    low_freq = _match_shape(low_freq, (h, w))

    residual = image_gray - low_freq
    energy_map = residual ** 2
    return energy_map, low_freq


def normalize_for_display(arr, percentile=100):
    """Normalize to [0, 1] for visualization.

    `percentile` caps the upper reference at a given percentile instead of
    the true max, then clips. Sparse maps (e.g. gradient magnitudes) tend to
    have a few extremely bright outlier pixels that crush everything else
    toward black under plain min-max normalization; using e.g. percentile=99
    trades those outliers (clipped to 1.0) for much better contrast in the
    rest of the map.
    """
    lo = arr.min()
    hi = np.percentile(arr, percentile) if percentile < 100 else arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)
