"""Convenience API for `ml_server` integration."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .inference import run_model


def run_inference_from_image(
    image: np.ndarray,
    weights_path: Optional[str] = None,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
    run_dir: str = "",
    registry_model_id: str = "",
) -> np.ndarray:
    """Return segmentation mask for a given ``image`` array."""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(image).convert("RGB").save(tmp.name)
        _, mask = run_model(
            tmp.name,
            params={
                "enable_gpu": enable_gpu,
                "device_policy": device_policy,
                "run_dir": run_dir,
                "registry_model_id": registry_model_id,
                "checkpoint_path": weights_path or "",
            },
            weights_path=weights_path or "",
        )
    return mask
