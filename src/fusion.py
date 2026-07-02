"""Explicit, non-learned fusion of wavelet energy and surface-normal variation.

Implements the rule described in Chapter 3: a pixel with high wavelet energy
AND significant local normal variation is a strong candidate for genuine
geometric detail; a pixel with high wavelet energy but a locally constant
normal field is more likely texture. This is a fixed formula (normalize +
multiply), not a trained model -- there are no learned parameters anywhere
in this file.
"""
import cv2
import numpy as np

from wavelet_energy import normalize_for_display


def normal_variation_map(normals):
    """Per-pixel local variation of the normal field (analogue of wavelet energy).

    Args:
        normals: (H, W, 3) float array, unit surface normal vectors in [-1, 1].

    Returns:
        (H, W) float array: sum, over the 3 normal channels, of the squared
        Sobel gradient magnitude. Flat surfaces (even if strongly textured)
        have near-zero normal variation; real geometric edges/creases produce
        a spike here because the surface orientation itself changes.
    """
    variation = np.zeros(normals.shape[:2], dtype=np.float64)
    for c in range(3):
        channel = normals[:, :, c].astype(np.float64)
        gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        variation += gx ** 2 + gy ** 2
    return variation


def fuse(energy_map, normal_var_map):
    """Combine wavelet energy and normal variation into a geometrically grounded
    attention map.

    Both inputs are independently min-max normalized to [0, 1], then
    multiplied pixel-wise. The product behaves like a soft logical AND: a
    pixel only ends up with high attention if it scores high on *both*
    signals. High wavelet energy with near-zero normal variation (the
    texture case) collapses toward zero, exactly as described in Chapter 3.

    Returns:
        (H, W) float array in [0, 1].
    """
    e = normalize_for_display(energy_map)
    n = normalize_for_display(normal_var_map)
    return e * n
