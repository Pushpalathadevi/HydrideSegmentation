"""Model registry for predictor backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.microseg.core import Predictor
from src.microseg.domain import ModelSpec


@dataclass(frozen=True)
class _Entry:
    spec: ModelSpec
    factory: Callable[[], Predictor]


class ModelRegistry:
    """Registry mapping model identifiers to predictor factories."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, spec: ModelSpec, factory: Callable[[], Predictor]) -> None:
        if spec.model_id in self._entries:
            raise ValueError(f"model already registered: {spec.model_id}")
        self._entries[spec.model_id] = _Entry(spec=spec, factory=factory)

    def build(self, model_id: str) -> Predictor:
        try:
            entry = self._entries[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model id: {model_id}") from exc
        return entry.factory()

    def specs(self) -> list[ModelSpec]:
        return [entry.spec for entry in self._entries.values()]

    def model_ids(self) -> list[str]:
        return [spec.model_id for spec in self.specs()]
