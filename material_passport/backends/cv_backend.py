"""Heuristic computer-vision material backend.

Maps aggregated appearance features (color, saturation, texture) to the closed
material vocabulary using deterministic per-class rules. Requires no external
models or network access, so it always runs and serves as the default /
fallback backend.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from ..association import EntityObservation
from .base import MaterialBackend, Prediction


class CVMaterialBackend(MaterialBackend):
    name = "cv"

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        if not ob.observed:
            return None
        f = ob.features
        cls = ob.entity.cls
        if cls == "wall":
            label, conf = self._wall(f)
        elif cls == "door":
            label, conf = self._door(f)
        else:
            label, conf = self._window(f)
        return Prediction(label=label, confidence=conf, method=self.name,
                          raw={"features": _slim(f)})

    # -- per-class rules -------------------------------------------------
    def _wall(self, f: dict) -> tuple[str, float]:
        sat = f["saturation_mean"]
        val = f["value_mean"]
        texture = f.get("texture_laplacian", 0.0)
        color_var = f["color_variation"]
        hue = f["hue_mean"]
        r, g, b = f["mean_rgb"]

        # Glass-like: bright, low saturation, low texture, slightly blue.
        if val > 0.78 and sat < 0.12 and texture < 40 and b >= r:
            return "glass", 0.55
        # Brick: reddish hue, mid saturation, high texture.
        if (hue < 25 or hue > 345) and sat > 0.30 and texture > 120:
            return "brick", 0.7
        # Wood / wood paneling: warm orange-brown hue, decent saturation.
        if 18 <= hue <= 45 and sat > 0.28:
            return ("wood_paneling", 0.62) if texture > 90 else ("wood", 0.55)
        # Tile: high texture + lowish saturation + bright.
        if texture > 150 and sat < 0.25 and val > 0.55:
            return "tile", 0.55
        # Stone / concrete: greyish (low saturation), medium-dark.
        if sat < 0.15:
            if texture > 80 or color_var > 28:
                return "stone", 0.5
            return "concrete", 0.55
        # Wallpaper: moderate saturation with patterning (texture or color var).
        if sat >= 0.20 and (texture > 70 or color_var > 30):
            return "wallpaper", 0.5
        # Default smooth painted surface.
        return "painted_plaster", 0.6

    def _door(self, f: dict) -> tuple[str, float]:
        sat = f["saturation_mean"]
        val = f["value_mean"]
        texture = f.get("texture_laplacian", 0.0)
        hue = f["hue_mean"]
        r, g, b = f["mean_rgb"]
        # Glass door: bright, desaturated.
        if val > 0.75 and sat < 0.12:
            return "glass", 0.55
        # Metal: low saturation, neutral/cool, smooth.
        if sat < 0.16 and texture < 80:
            return "metal", 0.55
        # Wood: warm hue with saturation.
        if 15 <= hue <= 50 and sat > 0.22:
            return "wood", 0.7
        # Otherwise composite.
        return "composite", 0.5

    def _window(self, f: dict) -> tuple[str, float]:
        # Window material = frame material (glazing is always glass).
        sat = f["saturation_mean"]
        val = f["value_mean"]
        hue = f["hue_mean"]
        # Warm wood frame.
        if 15 <= hue <= 50 and sat > 0.22:
            return "wood", 0.6
        # Bright neutral -> aluminium; darker neutral -> pvc.
        if val > 0.6:
            return "aluminum", 0.55
        return "pvc", 0.5


def _slim(f: dict) -> dict:
    keys = ("mean_rgb", "saturation_mean", "value_mean", "hue_mean",
            "color_variation", "texture_laplacian")
    return {k: f[k] for k in keys if k in f}
