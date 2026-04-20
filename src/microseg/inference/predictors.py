"""Predictor adapters backed by conventional and unified trained-model inference loaders."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS
from hydride_segmentation.segmentation_mask_creation import run_model as run_conv_model

from src.microseg.domain import ModelSpec, SegmentationOutput
from src.microseg.inference.trained_model_loader import (
    InferenceModelReference,
    discover_inference_references,
    load_reference_from_registry,
    load_reference_from_run_dir,
    run_reference_inference,
)
from src.microseg.plugins import ModelRegistry


@dataclass(frozen=True)
class _DynamicModelBinding:
    model_id: str
    display_name: str
    description: str
    details: str
    reference: InferenceModelReference


class HydrideConventionalPredictor:
    """Adapter for conventional hydride segmentation path."""

    model_id = "hydride_conventional"

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        cfg = deepcopy(DEFAULT_CONVENTIONAL_PARAMS)
        if params:
            cfg.update(params)
        image, mask = run_conv_model(image_path, cfg)
        return SegmentationOutput(image=image, mask=mask, manifest={"inference_backend": "conventional"})


class HydrideMLPredictor:
    """Legacy ML adapter now routed through unified architecture-aware loader."""

    model_id = "hydride_ml"

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        cfg = dict(params or {})
        resolve_started = time.perf_counter()
        run_dir = str(cfg.get("run_dir", "")).strip()
        registry_model_id = str(cfg.get("registry_model_id", "")).strip()
        checkpoint_path = str(cfg.get("checkpoint_path", cfg.get("weights_path", ""))).strip()

        if run_dir:
            ref = load_reference_from_run_dir(run_dir)
        elif registry_model_id:
            ref = load_reference_from_registry(registry_model_id)
        elif checkpoint_path:
            ref = InferenceModelReference(
                reference_id=f"checkpoint::{checkpoint_path}",
                display_name="Direct checkpoint",
                source="checkpoint_path",
                checkpoint_path=checkpoint_path,
                architecture=str(cfg.get("model_architecture", "")).strip().lower() or "unknown",
                backend_label=str(cfg.get("backend", "")).strip().lower() or "custom",
            )
        else:
            raise ValueError(
                "hydride_ml requires one of: params.run_dir, params.registry_model_id, or params.checkpoint_path"
            )

        resolve_seconds = max(0.0, time.perf_counter() - resolve_started)
        image, mask, manifest = run_reference_inference(
            image_path,
            ref,
            enable_gpu=bool(cfg.get("enable_gpu", False)),
            device_policy=str(cfg.get("device_policy", "cpu")),
            preprocess_config=cfg.get("gui_preprocess"),
        )
        timings = dict(manifest.get("timing", {}))
        timings["model_resolution_seconds"] = float(resolve_seconds)
        manifest["timing"] = timings
        return SegmentationOutput(image=image, mask=mask, manifest=manifest)


class ReferencePredictor:
    """Predictor bound to a resolved inference reference."""

    def __init__(self, reference: InferenceModelReference) -> None:
        self.reference = reference

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        cfg = dict(params or {})
        image, mask, manifest = run_reference_inference(
            image_path,
            self.reference,
            enable_gpu=bool(cfg.get("enable_gpu", False)),
            device_policy=str(cfg.get("device_policy", "cpu")),
            preprocess_config=cfg.get("gui_preprocess"),
        )
        timings = dict(manifest.get("timing", {}))
        timings.setdefault("model_resolution_seconds", 0.0)
        manifest["timing"] = timings
        return SegmentationOutput(image=image, mask=mask, manifest=manifest)


def discover_dynamic_ml_model_bindings() -> tuple[list[_DynamicModelBinding], list[str]]:
    """Discover inference-ready run/registry models for GUI and pipeline registry."""

    refs, warnings = discover_inference_references(include_registry=True)
    bindings: list[_DynamicModelBinding] = []
    reserved_ids = {"hydride_conventional", "hydride_ml"}
    for ref in refs:
        model_id = ref.reference_id
        if ref.source == "registry" and ref.reference_id.startswith("registry::"):
            model_id = ref.reference_id.removeprefix("registry::") or ref.reference_id
        if model_id in reserved_ids:
            continue
        bindings.append(
            _DynamicModelBinding(
                model_id=model_id,
                display_name=ref.display_name,
                description=f"Trained {ref.architecture} model ({ref.source})",
                details=(
                    f"Architecture={ref.architecture}, backend={ref.backend_label}, "
                    f"checkpoint={ref.checkpoint_path}"
                ),
                reference=ref,
            )
        )
    return bindings, warnings


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
                "Includes CLAHE normalization, adaptive thresholding, and morphology."
            ),
        ),
        factory=HydrideConventionalPredictor,
    )
    reg.register(
        ModelSpec(
            model_id="hydride_ml",
            display_name="Hydride ML (UNet)",
            feature_family="hydride",
            description="Default trained UNet checkpoint",
            details=(
                "Repo-native trained checkpoint routed through the unified architecture-aware loader. "
                "Uses the frozen-checkpoint registry entry hydride_ml by default."
            ),
        ),
        factory=HydrideMLPredictor,
    )

    bindings, _warnings = discover_dynamic_ml_model_bindings()
    for binding in bindings:
        reg.register(
            ModelSpec(
                model_id=binding.model_id,
                display_name=binding.display_name,
                feature_family="hydride",
                description=binding.description,
                details=binding.details,
            ),
            factory=lambda ref=binding.reference: ReferencePredictor(ref),
        )
    return reg
