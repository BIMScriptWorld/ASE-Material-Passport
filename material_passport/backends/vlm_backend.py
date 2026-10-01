"""Vision-language-model material backend (optional).

Supports two providers:
  * ``ollama`` - local Ollama vision model (default model: gemma4)
  * ``hf``     - Hugging Face Transformers vision-language model
                 (default model: google/gemma-4-E4B-it)

Environment:
  MP_VLM_PROVIDER - auto | ollama | hf (default: auto)
  MP_VLM_MODEL    - provider model name / path
  OLLAMA_HOST     - optional, Ollama server URL
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .. import config
from ..association import EntityObservation
from .base import BackendUnavailable, MaterialBackend, Prediction

# Cache full {material_type, condition, confidence} responses per entity key so
# the material and condition stages share a single model call.
_RESULT_CACHE: dict[str, dict] = {}
_HF_MODEL = None
_HF_PROCESSOR = None
_HF_TORCH = None
_HF_DEVICE = None


def clear_result_cache() -> None:
    """Drop per-entity VLM responses while keeping the loaded model resident."""
    _RESULT_CACHE.clear()


def _provider() -> str:
    return os.environ.get("MP_VLM_PROVIDER", "auto").lower()


def _model_name(provider: str | None = None) -> str:
    provider = provider or _provider()
    default = "google/gemma-4-E4B-it" if provider == "hf" else "gemma4"
    return os.environ.get("MP_VLM_MODEL", default)


def _check_ollama_available() -> None:
    """Raise BackendUnavailable if ollama or the model cannot be reached."""
    try:
        import ollama  # noqa
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailable(f"ollama package not installed: {exc}")
    try:
        names = {m.get("model") or m.get("name")
                 for m in ollama.list().get("models", [])}
    except Exception as exc:
        raise BackendUnavailable(f"cannot reach Ollama server: {exc}")
    model = _model_name("ollama")
    # Accept either an exact match or the bare name (e.g. "gemma4" vs
    # "gemma4:latest").
    if names and model not in names and \
            not any(str(n).split(":", 1)[0] == model for n in names if n):
        raise BackendUnavailable(
            f"Ollama model '{model}' not found. Available: {sorted(names)}")


def _load_hf():
    global _HF_MODEL, _HF_PROCESSOR, _HF_TORCH, _HF_DEVICE
    if _HF_MODEL is not None:
        return _HF_MODEL, _HF_PROCESSOR, _HF_TORCH, _HF_DEVICE
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailable(f"torch/transformers not installed: {exc}")
    model_name = _model_name("hf")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype)
    except Exception as exc:
        raise BackendUnavailable(
            f"could not load HF VLM model '{model_name}': {exc}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _HF_MODEL = model.eval().to(device)
    _HF_PROCESSOR = processor
    _HF_TORCH = torch
    _HF_DEVICE = device
    return _HF_MODEL, _HF_PROCESSOR, _HF_TORCH, _HF_DEVICE


def _check_hf_available() -> None:
    _load_hf()


def _check_available() -> None:
    provider = _provider()
    if provider == "ollama":
        _check_ollama_available()
        return
    if provider == "hf":
        _check_hf_available()
        return
    errors = []
    for check in (_check_ollama_available, _check_hf_available):
        try:
            check()
            return
        except BackendUnavailable as exc:
            errors.append(str(exc))
    raise BackendUnavailable("; ".join(errors))


def _prompt(cls: str, vocab: list[str]) -> str:
    return (
        f"These image crops show a building {cls} surface from a synthetic "
        f"indoor scene. Choose the single best material from this fixed list: "
        f"{vocab}. For a window, the material refers to the frame (glazing is "
        f"glass). Also rate its condition as one of {config.CONDITION_VOCAB}. "
        f"If (and only if) the material is 'composite', add a \"components\" "
        f"array naming the two main constituent materials, and do not add 'composite' as a component, e.g. "
        f'["wood", "metal"], ["glass", "metal"], and not ["composite", "metal"].'
        f"Respond ONLY as compact JSON: "
        f'{{"material_type": <one of the list>, '
        f'"condition": <one of the conditions>, '
        f'"components": [<material>, ...], '
        f'"confidence": <0..1 float>}}.'
    )


def _query_ollama(prompt: str, image_paths: list[str]) -> str:
    from ollama import chat
    images = list(image_paths[: config.N_CROP_FRAMES])
    response = chat(
        model=_model_name("ollama"),
        messages=[{"role": "user", "content": prompt, "images": images}],
        format="json",
        options={"temperature": 0.0},
    )
    return (response.message.content or "").strip()


def _query_hf(prompt: str, image_paths: list[str]) -> str:
    from PIL import Image

    model, processor, torch, device = _load_hf()
    images = [Image.open(p).convert("RGB")
              for p in image_paths[: config.N_CROP_FRAMES]]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    if hasattr(processor, "apply_chat_template"):
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    else:
        inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated = output[0][prompt_len:] if prompt_len else output[0]
    return processor.decode(generated, skip_special_tokens=True).strip()


def query_vlm(ob: EntityObservation) -> Optional[dict]:
    """Return a cached/fresh {material_type, condition, confidence} dict."""
    if not ob.crop_paths:
        return None
    scene_key = os.path.abspath(os.path.dirname(os.path.dirname(ob.crop_paths[0])))
    key = f"{scene_key}:{ob.entity.key}"
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    cls = ob.entity.cls
    vocab = config.MATERIAL_VOCAB[cls]
    prompt = _prompt(cls, vocab)
    provider = _provider()
    if provider == "hf":
        text = _query_hf(prompt, ob.crop_paths)
    elif provider == "ollama":
        text = _query_ollama(prompt, ob.crop_paths)
    else:
        try:
            text = _query_ollama(prompt, ob.crop_paths)
        except Exception:
            text = _query_hf(prompt, ob.crop_paths)
    result = _parse_json(text, cls, vocab)
    _RESULT_CACHE[key] = result
    return result


def _parse_json(text: str, cls: str, vocab: list[str]) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        data = {}
    mat = data.get("material_type")
    if mat not in vocab:
        mat = None
    cond = data.get("condition")
    if cond not in config.CONDITION_VOCAB:
        cond = None
    try:
        conf = float(data.get("confidence", 0.6))
    except (TypeError, ValueError):
        conf = 0.6
    components = data.get("components") or []
    if not isinstance(components, list):
        components = []
    components = [str(c) for c in components if c][:3] if mat == "composite" \
        else []
    return {"material_type": mat, "condition": cond,
            "components": components,
            "confidence": max(0.0, min(1.0, conf))}


class VLMMaterialBackend(MaterialBackend):
    name = "vlm"

    def available(self) -> bool:
        try:
            _check_available()
            return True
        except BackendUnavailable:
            return False

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        if not ob.observed:
            return None
        result = query_vlm(ob)
        if not result or not result.get("material_type"):
            return None
        return Prediction(label=result["material_type"],
                          confidence=result["confidence"],
                          method=self.name,
                          raw={"components": result.get("components", []),
                               **result})
