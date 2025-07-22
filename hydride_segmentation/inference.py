import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

# Default directory for model weights
DEFAULT_MODEL_DIR = os.getenv("HYDRIDE_MODEL_PATH", "/opt/models/hydride_segmentation/")
DEFAULT_WEIGHTS = os.path.join(DEFAULT_MODEL_DIR, "model.pt")

_model = None

def _load_model(weights_path: str = DEFAULT_WEIGHTS) -> torch.nn.Module:
    """Load the segmentation model with CPU-only torch."""
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=3, classes=1)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def get_model(weights_path: str = DEFAULT_WEIGHTS) -> torch.nn.Module:
    """Return a cached model instance."""
    global _model
    if _model is None:
        _model = _load_model(weights_path)
    return _model

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
    model = get_model(weights_path)
    rgb = Image.open(image_path).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)[0, 0]
        mask = torch.sigmoid(out).cpu().numpy()

    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    return np.array(rgb), mask_bin
