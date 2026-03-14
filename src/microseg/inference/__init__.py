"""Inference orchestration and predictor adapters."""

from .predictors import (
    HydrideConventionalPredictor,
    HydrideMLPredictor,
    build_hydride_registry,
    discover_dynamic_ml_model_bindings,
)
from .trained_model_loader import (
    InferenceModelReference,
    discover_inference_references,
    load_reference_from_registry,
    load_reference_from_run_dir,
    run_reference_inference,
    supported_trainable_architectures,
)

__all__ = [
    "HydrideConventionalPredictor",
    "HydrideMLPredictor",
    "InferenceModelReference",
    "build_hydride_registry",
    "discover_dynamic_ml_model_bindings",
    "discover_inference_references",
    "load_reference_from_registry",
    "load_reference_from_run_dir",
    "run_reference_inference",
    "supported_trainable_architectures",
]
