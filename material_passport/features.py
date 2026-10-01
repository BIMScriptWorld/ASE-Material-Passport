"""Appearance feature computation from collected pixel samples and crops."""
from __future__ import annotations

import numpy as np


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB(0-255) -> HSV with H in [0,360), S,V in [0,1]."""
    r, g, b = (rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0)
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-9
    # max == r
    idx = mask & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    idx = mask & (mx == g)
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    idx = mask & (mx == b)
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
    s = np.where(mx > 1e-9, diff / np.maximum(mx, 1e-9), 0.0)
    v = mx
    return np.stack([h, s, v], axis=1)


def compute_color_features(rgb_samples: np.ndarray) -> dict:
    """Aggregate color statistics from an (N, 3) uint8 sample array."""
    rgb = rgb_samples.astype(np.float64)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    hsv = _rgb_to_hsv(rgb_samples)
    # Circular mean hue.
    hue_rad = np.deg2rad(hsv[:, 0])
    hue_mean = float((np.rad2deg(np.arctan2(
        np.sin(hue_rad).mean(), np.cos(hue_rad).mean())) + 360) % 360)
    return {
        "n_samples": int(rgb.shape[0]),
        "mean_rgb": rgb.mean(axis=0).tolist(),
        "std_rgb": rgb.std(axis=0).tolist(),
        "median_rgb": np.median(rgb, axis=0).tolist(),
        "luminance_mean": float(lum.mean()),
        "luminance_std": float(lum.std()),
        "saturation_mean": float(hsv[:, 1].mean()),
        "value_mean": float(hsv[:, 2].mean()),
        "hue_mean": hue_mean,
        # Texture proxy from scattered surface samples.
        "color_variation": float(rgb.std(axis=0).mean()),
    }


def laplacian_variance(gray: np.ndarray) -> float:
    """Edge/texture energy via the variance of a 3x3 Laplacian response."""
    g = gray.astype(np.float64)
    lap = (
        -4.0 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    inner = lap[1:-1, 1:-1]
    return float(inner.var()) if inner.size else 0.0
