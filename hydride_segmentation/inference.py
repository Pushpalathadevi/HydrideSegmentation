import os
import logging
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

from src.microseg.core import resolve_torch_device

# Default directory for model weights
DEFAULT_MODEL_DIR = os.getenv("HYDRIDE_MODEL_PATH", "/opt/models/hydride_segmentation/")
DEFAULT_WEIGHTS = os.path.join(DEFAULT_MODEL_DIR, "model.pt")
DEFAULT_ENABLE_GPU = os.getenv("MICROSEG_ENABLE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_DEVICE_POLICY = os.getenv("MICROSEG_DEVICE_POLICY", "cpu")

_logger = logging.getLogger(__name__)
_model_cache: dict[tuple[str, str], torch.nn.Module] = {}


def _normalize_device(device: str | None) -> str:
    if not device:
        return "cpu"
    return str(device).strip().lower()


def _resolve_device(params: dict | None = None) -> str:
    cfg = params or {}
    explicit = cfg.get("device")
    if explicit:
        return _normalize_device(str(explicit))
    enable_gpu = bool(cfg.get("enable_gpu", DEFAULT_ENABLE_GPU))
    policy = str(cfg.get("device_policy", DEFAULT_DEVICE_POLICY))
    resolved = resolve_torch_device(enable_gpu=enable_gpu, policy=policy)
    if resolved.fallback_used:
        _logger.info(resolved.reason)
    return resolved.selected_device


def _load_model(weights_path: str = DEFAULT_WEIGHTS, *, device: str = "cpu") -> torch.nn.Module:
    """Load segmentation model on the resolved device."""
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=3, classes=1)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def get_model(weights_path: str = DEFAULT_WEIGHTS, *, device: str = "cpu") -> torch.nn.Module:
    """Return cached model instance for ``weights_path`` + ``device`` pair."""

    dev = _normalize_device(device)
    key = (os.path.abspath(weights_path), dev)
    if key not in _model_cache:
        try:
            _model_cache[key] = _load_model(weights_path, device=dev)
        except Exception:
            if dev != "cpu":
                _logger.exception("Failed to initialize model on '%s', retrying on CPU.", dev)
                cpu_key = (os.path.abspath(weights_path), "cpu")
                if cpu_key not in _model_cache:
                    _model_cache[cpu_key] = _load_model(weights_path, device="cpu")
                return _model_cache[cpu_key]
            raise
    return _model_cache[key]

def run_model(image_path: str, params: dict | None = None,
              weights_path: str = DEFAULT_WEIGHTS) -> Tuple[np.ndarray, np.ndarray]:
    """Run hydride segmentation on an image.

    Parameters
    ----------
    image_path:
        Path to the input image.
    params:
        Unused placeholder for compatibility with the GUI.
    weights_path:
        Optional path to model weights. Defaults to ``HYDRIDE_MODEL_PATH``.

    Returns
    -------
    tuple
        Tuple of ``(image, mask)`` numpy arrays where ``mask`` is ``uint8``.
    """
    runtime_params = params or {}
    device = _resolve_device(runtime_params)
    model = get_model(weights_path, device=device)
    model_device = next(model.parameters()).device.type
    rgb = Image.open(image_path).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(model_device)

    with torch.no_grad():
        out = model(tensor)[0, 0]
        mask = torch.sigmoid(out).cpu().numpy()

    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    return np.array(rgb), mask_bin
