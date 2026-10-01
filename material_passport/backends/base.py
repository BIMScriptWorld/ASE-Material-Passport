"""Backend interfaces and shared types for material classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..association import EntityObservation


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependencies / weights / credentials are absent."""


@dataclass
class Prediction:
    label: str
    confidence: float
    method: str
    raw: dict = field(default_factory=dict)


class MaterialBackend:
    """Maps an :class:`EntityObservation` to a material in the closed vocab."""

    name = "base"

    def available(self) -> bool:  # pragma: no cover - trivial
        return True

    def predict(self, ob: EntityObservation) -> Optional[Prediction]:
        raise NotImplementedError
