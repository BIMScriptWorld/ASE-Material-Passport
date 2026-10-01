"""Condition synthesis backends.

ASE renders are pristine synthetic surfaces, so there is no genuine wear signal
in the pixels. We therefore synthesize a *plausible* pseudo-label per entity:
  * ``rule``  - deterministic function of appearance stats + a seeded jitter so
                results are stable and reproducible per entity.
  * ``vlm``   - reuses the multimodal model response (shared with the material
                VLM backend).
The same multi-method consensus machinery used for materials applies here.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from . import config
from .association import EntityObservation
from .backends.base import BackendUnavailable, Prediction
from .backends import vlm_backend


def _seeded_unit(key: str) -> float:
    """Deterministic value in [0,1) from an entity key."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class ConditionBackend:
    name = "base"

    def available(self) -> bool:
        return True

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        raise NotImplementedError


class RuleConditionBackend(ConditionBackend):
    name = "rule"

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        if not ob.observed:
            return None
        f = ob.features
        # Wear score in [0,1]: darker, higher color spread and more texture
        # read as more worn. Values are gentle because ASE is pristine.
        lum = f["luminance_mean"] / 255.0
        color_var = min(f["color_variation"] / 60.0, 1.0)
        texture = min(f.get("texture_laplacian", 0.0) / 400.0, 1.0)
        wear = 0.45 * (1.0 - lum) + 0.30 * color_var + 0.25 * texture
        # Seeded jitter keeps a plausible spread without being random per run.
        wear = 0.7 * wear + 0.3 * _seeded_unit(ob.entity.key)
        wear = max(0.0, min(1.0, wear))

        if wear < 0.30:
            label, conf = "new", 0.55
        elif wear < 0.55:
            label, conf = "good", 0.6
        elif wear < 0.78:
            label, conf = "worn", 0.55
        else:
            label, conf = "damaged", 0.5
        return Prediction(label=label, confidence=conf, method=self.name,
                          raw={"wear_score": round(wear, 3)})


class VLMConditionBackend(ConditionBackend):
    name = "vlm"

    def available(self) -> bool:
        return VLMConditionBackend._vlm_ok()

    @staticmethod
    def _vlm_ok() -> bool:
        try:
            vlm_backend._check_available()
            return True
        except BackendUnavailable:
            return False

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        if not ob.observed:
            return None
        result = vlm_backend.query_vlm(ob)
        if not result or not result.get("condition"):
            return None
        return Prediction(label=result["condition"],
                          confidence=result.get("confidence", 0.6),
                          method=self.name, raw=result)


_REGISTRY = {
    "rule": RuleConditionBackend,
    "vlm": VLMConditionBackend,
}

ALL_CONDITION_METHODS = list(_REGISTRY)


def get_condition_backend(name: str) -> ConditionBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown condition method '{name}'. "
                       f"Choices: {ALL_CONDITION_METHODS}")
    return _REGISTRY[name]()
