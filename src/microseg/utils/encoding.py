"""Binary/text encoding helpers for image artifacts."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def image_to_png_base64(image: Image.Image | np.ndarray) -> str:
    """Encode a PIL image or ndarray as PNG base64 text."""

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
