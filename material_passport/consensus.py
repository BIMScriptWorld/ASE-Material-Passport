"""Multi-method decision logic for combining backend predictions.

Strategies (apply identically to material and condition):
  * ``priority``        - take the first method (in priority order) whose
                          prediction meets the confidence threshold; otherwise
                          fall back to the highest-confidence prediction.
  * ``max_confidence``  - take the single highest-confidence prediction.
  * ``vote``            - majority label across methods; ties broken by summed
                          confidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .backends.base import Prediction


@dataclass
class Decision:
    label: Optional[str]
    confidence: float
    method: str                       # winning method, or "consensus"/"none"
    candidates: list = field(default_factory=list)  # list[Prediction]
    components: list = field(default_factory=list)  # composite constituents


def _components_of(pred: Optional[Prediction]) -> list:
    if pred is None:
        return []
    return list(pred.raw.get("components") or [])


def decide(predictions: list[Prediction], strategy: str, threshold: float,
           priority: list[str]) -> Decision:
    preds = [p for p in predictions if p is not None and p.label]
    if not preds:
        return Decision(label=None, confidence=0.0, method="none",
                        candidates=[])
    if len(preds) == 1:
        p = preds[0]
        return Decision(p.label, p.confidence, p.method, list(preds),
                        _components_of(p))

    if strategy == "max_confidence":
        best = max(preds, key=lambda p: p.confidence)
        return Decision(best.label, best.confidence, best.method, list(preds),
                        _components_of(best))

    if strategy == "vote":
        score = defaultdict(float)
        count = defaultdict(int)
        for p in preds:
            score[p.label] += p.confidence
            count[p.label] += 1
        best_label = max(score, key=lambda k: (count[k], score[k]))
        conf = score[best_label] / max(count[best_label], 1)
        winner = next((p for p in preds if p.label == best_label
                       and p.raw.get("components")), None) \
            or next((p for p in preds if p.label == best_label), None)
        return Decision(best_label, conf, "consensus", list(preds),
                        _components_of(winner))

    # Default: priority.
    by_method = {p.method: p for p in preds}
    for method in priority:
        p = by_method.get(method)
        if p is not None and p.confidence >= threshold:
            return Decision(p.label, p.confidence, p.method, list(preds),
                            _components_of(p))
    best = max(preds, key=lambda p: p.confidence)
    return Decision(best.label, best.confidence, best.method, list(preds),
                    _components_of(best))
