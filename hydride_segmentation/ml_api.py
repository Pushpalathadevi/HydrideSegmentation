"""Convenience API for `ml_server` integration."""
from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image

from .inference import get_model, DEFAULT_WEIGHTS


def run_inference_from_image(image: np.ndarray, weights_path: Optional[str] = None) -> np.ndarray:
    """Return segmentation mask for a given ``image`` array."""
    model = get_model(weights_path or DEFAULT_WEIGHTS)
    rgb = Image.fromarray(image).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    import torch
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)[0, 0]
        mask = torch.sigmoid(out).cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255

