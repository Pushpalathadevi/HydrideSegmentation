"""Utility helpers for image conversion."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load grayscale image from raw bytes."""
    with Image.open(BytesIO(data)) as img:
        gray = img.convert("L")
        return np.array(gray)


def image_to_png_base64(image: Union[np.ndarray, Image.Image]) -> str:
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
