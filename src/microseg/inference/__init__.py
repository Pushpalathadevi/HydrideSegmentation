"""Inference orchestration and predictor adapters."""

from .predictors import (
    HydrideConventionalPredictor,
    HydrideMLPredictor,
    build_hydride_registry,
    discover_dynamic_ml_model_bindings,
)
from .trained_model_loader import (
    InferenceModelReference,
    ModelWarmLoadStatus,
    discover_inference_references,
    get_or_load_reference_bundle,
    load_reference_from_registry,
    load_reference_from_run_dir,
    run_reference_inference,
    supported_trainable_architectures,
    warm_load_reference_bundle,
)

__all__ = [
    "HydrideConventionalPredictor",
    "HydrideMLPredictor",
    "InferenceModelReference",
    "ModelWarmLoadStatus",
    "build_hydride_registry",
    "discover_dynamic_ml_model_bindings",
    "discover_inference_references",
    "get_or_load_reference_bundle",
    "load_reference_from_registry",
    "load_reference_from_run_dir",
    "run_reference_inference",
    "supported_trainable_architectures",
    "warm_load_reference_bundle",
]
