"""Predictor adapters backed by current hydride implementation."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path

from hydride_segmentation.segmentation_mask_creation import run_model as run_conv_model
from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS

from src.microseg.domain import ModelSpec, SegmentationOutput
from src.microseg.plugins import find_repo_root, frozen_checkpoint_map
from src.microseg.plugins import ModelRegistry


_logger = logging.getLogger(__name__)


def _resolve_registry_weights_path(model_id: str) -> str | None:
    """Resolve model checkpoint path from frozen-checkpoint metadata if available."""

    try:
        records = frozen_checkpoint_map()
    except Exception as exc:
        _logger.debug("Could not load frozen checkpoint registry: %s", exc)
        return None

    rec = records.get(model_id)
    if rec is None:
        return None
    hint = str(rec.checkpoint_path_hint).strip()
    if not hint or hint.lower().startswith("n/a"):
        return None

    hinted_path = Path(hint)
    if hinted_path.is_absolute():
        return str(hinted_path) if hinted_path.exists() else None

    try:
        root = find_repo_root(Path(__file__))
    except Exception:
        root = Path.cwd()
    resolved = (root / hinted_path).resolve()
    return str(resolved) if resolved.exists() else None


class HydrideConventionalPredictor:
    """Adapter for the current conventional hydride segmentation path."""

    model_id = "hydride_conventional"

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        cfg = deepcopy(DEFAULT_CONVENTIONAL_PARAMS)
        if params:
            cfg.update(params)
        image, mask = run_conv_model(image_path, cfg)
        return SegmentationOutput(image=image, mask=mask)


class HydrideMLPredictor:
    """Adapter for the current ML hydride inference path."""

    model_id = "hydride_ml"

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        # Lazy import keeps non-ML paths usable without ML-only dependencies.
        from hydride_segmentation.inference import run_model as run_ml_model

        params = dict(params or {})
        weights_path = str(params.get("weights_path", "")).strip()
        if not weights_path:
            reg_path = _resolve_registry_weights_path(self.model_id)
            if reg_path:
                params["weights_path"] = reg_path
                weights_path = reg_path
                _logger.info("Hydride ML resolved checkpoint from frozen registry: %s", reg_path)
        image, mask = run_ml_model(image_path, params=params, weights_path=weights_path) if weights_path else run_ml_model(image_path, params=params)
        return SegmentationOutput(image=image, mask=mask)


def build_hydride_registry(registry: ModelRegistry | None = None) -> ModelRegistry:
    """Register hydride predictors into a model registry."""

    reg = registry or ModelRegistry()
    reg.register(
        ModelSpec(
            model_id="hydride_conventional",
            display_name="Hydride Conventional",
            feature_family="hydride",
            description="CLAHE + adaptive threshold + morphology",
            details=(
                "Classical CPU-first pipeline for hydride-like contrast patterns. "
                "Includes CLAHE normalization, adaptive thresholding, and morphology. "
                "Best for quick baseline runs and environments without ML weights."
            ),
        ),
        factory=HydrideConventionalPredictor,
    )
    reg.register(
        ModelSpec(
            model_id="hydride_ml",
            display_name="Hydride ML",
            feature_family="hydride",
            description="UNet-based hydride segmentation model",
            details=(
                "UNet-like learned predictor for hydride segmentation. "
                "Requires model weights and is generally more flexible for "
                "heterogeneous image contrast; validate with correction workflow."
            ),
        ),
        factory=HydrideMLPredictor,
    )
    return reg
