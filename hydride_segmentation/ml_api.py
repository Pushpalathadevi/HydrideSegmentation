"""Convenience API for `ml_server` integration."""
from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image

from .inference import get_model, DEFAULT_WEIGHTS
from src.microseg.core import resolve_torch_device


def run_inference_from_image(
    image: np.ndarray,
    weights_path: Optional[str] = None,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> np.ndarray:
    """Return segmentation mask for a given ``image`` array."""
    import torch

    resolved = resolve_torch_device(enable_gpu=enable_gpu, policy=device_policy)
    model = get_model(weights_path or DEFAULT_WEIGHTS, device=resolved.selected_device)
    model_device = next(model.parameters()).device.type
    rgb = Image.fromarray(image).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(model_device)
    with torch.no_grad():
        out = model(tensor)[0, 0]
        mask = torch.sigmoid(out).cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255
