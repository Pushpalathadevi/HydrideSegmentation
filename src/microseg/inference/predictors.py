"""Predictor adapters backed by current hydride implementation."""

from __future__ import annotations

from copy import deepcopy

from hydride_segmentation.segmentation_mask_creation import run_model as run_conv_model
from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS

from src.microseg.domain import ModelSpec, SegmentationOutput
from src.microseg.plugins import ModelRegistry


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

        params = params or {}
        weights_path = params.get("weights_path")
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
        ),
        factory=HydrideConventionalPredictor,
    )
    reg.register(
        ModelSpec(
            model_id="hydride_ml",
            display_name="Hydride ML",
            feature_family="hydride",
            description="UNet-based hydride segmentation model",
        ),
        factory=HydrideMLPredictor,
    )
    return reg
