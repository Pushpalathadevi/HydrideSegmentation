import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from src.microseg.core import resolve_torch_device
from src.microseg.inference.trained_model_loader import (
    InferenceModelReference,
    load_reference_from_registry,
    load_reference_from_run_dir,
    run_reference_inference,
)

DEFAULT_MODEL_DIR = os.getenv("HYDRIDE_MODEL_PATH", "/opt/models/hydride_segmentation/")
DEFAULT_WEIGHTS = os.path.join(DEFAULT_MODEL_DIR, "model.pt")
DEFAULT_ENABLE_GPU = os.getenv("MICROSEG_ENABLE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_DEVICE_POLICY = os.getenv("MICROSEG_DEVICE_POLICY", "cpu")

_logger = logging.getLogger(__name__)


def _resolve_device(params: dict | None = None) -> tuple[bool, str]:
    cfg = params or {}
    enable_gpu = bool(cfg.get("enable_gpu", DEFAULT_ENABLE_GPU))
    policy = str(cfg.get("device_policy", DEFAULT_DEVICE_POLICY))
    resolved = resolve_torch_device(enable_gpu=enable_gpu, policy=policy)
    if resolved.fallback_used:
        _logger.info(resolved.reason)
    return enable_gpu, policy


def _reference_from_params(params: dict | None = None, weights_path: str = "") -> InferenceModelReference:
    cfg = dict(params or {})
    if str(cfg.get("run_dir", "")).strip():
        return load_reference_from_run_dir(str(cfg["run_dir"]).strip())
    if str(cfg.get("registry_model_id", "")).strip():
        return load_reference_from_registry(str(cfg["registry_model_id"]).strip())
    if str(cfg.get("checkpoint_path", "")).strip() or str(weights_path).strip():
        ckpt = str(cfg.get("checkpoint_path") or weights_path).strip()
        if not ckpt:
            raise ValueError("checkpoint path resolved empty")
        return InferenceModelReference(
            reference_id=f"checkpoint::{Path(ckpt).name}",
            display_name=f"Checkpoint: {Path(ckpt).name}",
            source="checkpoint_path",
            checkpoint_path=ckpt,
            architecture=str(cfg.get("model_architecture", "")).strip().lower() or "unknown",
            backend_label=str(cfg.get("backend", "")).strip().lower() or "custom",
        )
    return InferenceModelReference(
        reference_id="checkpoint::default",
        display_name="Checkpoint: default",
        source="checkpoint_path",
        checkpoint_path=DEFAULT_WEIGHTS,
        architecture="unknown",
        backend_label="legacy_default",
    )


def run_model(
    image_path: str,
    params: dict | None = None,
    weights_path: str = DEFAULT_WEIGHTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run architecture-aware segmentation on an image."""

    runtime_params = params or {}
    enable_gpu, policy = _resolve_device(runtime_params)
    reference = _reference_from_params(runtime_params, weights_path)
    image, mask, manifest = run_reference_inference(
        image_path,
        reference,
        enable_gpu=enable_gpu,
        device_policy=policy,
    )
    _logger.info(
        "ML inference complete: source=%s reference=%s architecture=%s checkpoint=%s",
        manifest.get("source"),
        manifest.get("reference_id"),
        manifest.get("architecture"),
        manifest.get("checkpoint_path"),
    )
    return image, mask
