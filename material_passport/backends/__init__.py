"""Backend registry for material classification methods."""
from __future__ import annotations

from .base import BackendUnavailable, MaterialBackend, Prediction
from .cv_backend import CVMaterialBackend
from .dms_backend import DMSMaterialBackend
from .vlm_backend import VLMMaterialBackend

_REGISTRY = {
    "cv": CVMaterialBackend,
    "vlm": VLMMaterialBackend,
    "dms": DMSMaterialBackend,
}

ALL_METHODS = list(_REGISTRY)


def get_material_backend(name: str) -> MaterialBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown material method '{name}'. "
                       f"Choices: {ALL_METHODS}")
    return _REGISTRY[name]()


def needs_crops(methods: list[str]) -> bool:
    """vlm and dms operate on saved image crops; cv does not."""
    return any(m in ("vlm", "dms") for m in methods)
