"""Dense material-classification backend (MINC, via a Hugging Face model).

Runs the pretrained SigLIP material classifier
``prithivMLmods/Minc-Materials-23`` (fine-tuned from
``google/siglip2-base-patch16-224``) on the entity's RGB crops and maps the
predicted MINC material category into the closed vocabulary.

Requires ``torch`` and ``transformers``. The model is downloaded from the
Hugging Face Hub on first use (cached afterwards). Override the checkpoint via
``MP_DMS_MODEL`` (a hub id or a local path). If the dependencies or weights are
unavailable the backend reports itself unavailable and the pipeline falls back
to other methods.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import numpy as np
from PIL import Image

from .. import config
from ..association import EntityObservation
from .base import BackendUnavailable, MaterialBackend, Prediction

DEFAULT_MODEL = "prithivMLmods/Minc-Materials-23"
_CACHED_MODEL_NAME = None
_CACHED_MODEL = None
_CACHED_PROCESSOR = None
_CACHED_TORCH = None
_CACHED_ID2LABEL = None
_CACHED_DEVICE = None

# MINC-2500 material categories (index order used by the model's id2label).
# Kept as a fallback if the loaded config lacks id2label.
MINC_CLASSES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile", "wallpaper",
    "water", "wood",
]

# MINC category -> our vocab, resolved per ASE class at lookup time.
_MINC_TO_VOCAB = {
    "brick": {"wall": "brick"},
    "ceramic": {"wall": "tile"},
    "tile": {"wall": "tile"},
    "glass": {"wall": "glass", "door": "glass", "window": "aluminum"},
    "metal": {"wall": "metal", "door": "metal", "window": "aluminum"},
    "painted": {"wall": "painted_plaster", "door": "wood"},
    "wallpaper": {"wall": "wallpaper"},
    "polishedstone": {"wall": "stone"},
    "stone": {"wall": "stone", "door": "wood"},
    "wood": {"wall": "wood_paneling", "door": "wood", "window": "wood"},
    "plastic": {"door": "wood", "window": "pvc"},
    "fabric": {"wall": "wallpaper"},
    "carpet": {"wall": "wallpaper"},
}

# MINC category -> coarse constituent material name, used to describe a
# composite surface (one made of two or more distinct materials).
_MINC_TO_BASE = {
    "brick": "brick",
    "ceramic": "ceramic",
    "tile": "tile",
    "glass": "glass",
    "mirror": "glass",
    "metal": "metal",
    "stone": "stone",
    "polishedstone": "stone",
    "wood": "wood",
    "plastic": "plastic",
    "fabric": "fabric",
    "carpet": "fabric",
    "leather": "leather",
    "paper": "paper",
}


def _map_minc(minc_label: str, cls: str) -> Optional[str]:
    mapping = _MINC_TO_VOCAB.get(minc_label)
    if not mapping:
        return None
    if cls in mapping:
        return mapping[cls]
    # Fall back to any defined target if it is valid for this class.
    for cand in mapping.values():
        if cand in config.MATERIAL_VOCAB[cls]:
            return cand
    return None


def _from_pretrained(model_cls, proc_cls, model_name):
    """Load model + processor, preferring the local cache.

    Once the weights have been downloaded, loading with
    ``local_files_only=True`` avoids any network round-trip to the Hugging
    Face Hub (and the associated 'unauthenticated request' warning). If the
    model is not cached yet, fall back to a normal (downloading) load.
    """
    try:
        model = model_cls.from_pretrained(model_name, local_files_only=True)
        proc = proc_cls.from_pretrained(model_name, local_files_only=True)
        return model, proc
    except Exception:
        model = model_cls.from_pretrained(model_name)
        proc = proc_cls.from_pretrained(model_name)
        return model, proc


class DMSMaterialBackend(MaterialBackend):
    name = "dms"

    def __init__(self):
        self._model = None
        self._processor = None
        self._torch = None
        self._id2label = None
        self._device = None

    def _load(self):
        global _CACHED_MODEL_NAME, _CACHED_MODEL, _CACHED_PROCESSOR
        global _CACHED_TORCH, _CACHED_ID2LABEL, _CACHED_DEVICE

        if self._model is not None:
            return
        model_name = os.environ.get("MP_DMS_MODEL", DEFAULT_MODEL)
        if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name:
            self._torch = _CACHED_TORCH
            self._device = _CACHED_DEVICE
            self._model = _CACHED_MODEL
            self._processor = _CACHED_PROCESSOR
            self._id2label = _CACHED_ID2LABEL
            return
        try:
            import torch
            from transformers import (AutoImageProcessor,
                                      SiglipForImageClassification)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(
                f"torch/transformers not installed: {exc}")
        try:
            model, processor = _from_pretrained(
                SiglipForImageClassification, AutoImageProcessor, model_name)
        except Exception as exc:
            raise BackendUnavailable(
                f"could not load DMS model '{model_name}': {exc}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = device
        self._model = model.eval().to(device)
        self._processor = processor
        id2label = getattr(model.config, "id2label", None) or {}
        # Config keys may be strings; normalise to int -> label.
        self._id2label = {int(k): v for k, v in id2label.items()} \
            if id2label else {i: c for i, c in enumerate(MINC_CLASSES)}
        _CACHED_MODEL_NAME = model_name
        _CACHED_MODEL = self._model
        _CACHED_PROCESSOR = self._processor
        _CACHED_TORCH = self._torch
        _CACHED_ID2LABEL = self._id2label
        _CACHED_DEVICE = self._device

    def available(self) -> bool:
        try:
            self._load()
            return True
        except BackendUnavailable:
            return False

    def _infer_crop(self, path: str) -> Optional[str]:
        torch = self._torch
        img = Image.open(path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        idx = int(logits.reshape(-1).argmax().item())
        return self._id2label.get(idx)

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        if not ob.observed or not ob.crop_paths:
            return None
        self._load()
        cls = ob.entity.cls
        votes = Counter()
        base_votes = Counter()
        for p in ob.crop_paths:
            minc = self._infer_crop(p)
            if not minc:
                continue
            mapped = _map_minc(minc, cls)
            if mapped:
                votes[mapped] += 1
            base = _MINC_TO_BASE.get(minc)
            if base:
                base_votes[base] += 1
        if not votes:
            return None

        label, n = votes.most_common(1)[0]
        conf = n / sum(votes.values())
        raw = {"votes": dict(votes), "base_votes": dict(base_votes)}

        # A surface showing two or more distinct constituent materials (each a
        # valid material for this class) is reported as composite, listing the
        # detected constituents.
        constituents = [b for b in base_votes
                        if b in config.MATERIAL_VOCAB[cls] and b != "composite"]
        if "composite" in config.MATERIAL_VOCAB[cls] and len(constituents) >= 2:
            ordered = [b for b, _ in base_votes.most_common()
                       if b in constituents]
            comp_votes = sum(base_votes[b] for b in constituents)
            label = "composite"
            conf = comp_votes / max(sum(base_votes.values()), 1)
            raw["components"] = ordered[:3]
        elif label == "composite":
            ordered = [b for b, _ in base_votes.most_common()]
            if ordered:
                raw["components"] = ordered[:3]

        return Prediction(label=label, confidence=float(conf),
                          method=self.name, raw=raw)
