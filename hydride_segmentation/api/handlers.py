"""Request handlers for the segmentation API."""
from __future__ import annotations

import imghdr
from typing import Dict

from .schema import SegmentParams
from hydride_segmentation.core.utils import (
    load_image_from_bytes,
    image_to_png_base64,
)
from hydride_segmentation.core.conventional import segment, ConventionalParams
from hydride_segmentation.core.analysis import compute_metrics, analyze_mask

ALLOWED_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
MAX_SIZE_MB = 20


def _validate_and_read_file(file_storage) -> bytes:
    if file_storage is None or file_storage.filename == "":
        raise ValueError("file is required")
    filename = file_storage.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("unsupported file type")
    data = file_storage.read()
    if len(data) > MAX_SIZE_MB * 1024 * 1024:
        raise ValueError("file too large")
    kind = imghdr.what(None, h=data)
    if kind not in ALLOWED_EXTS:
        raise ValueError("unsupported file type")
    return data


def process_request(file_storage, params: SegmentParams) -> Dict:
    """Process request and return response dictionary."""
    if params.model == "ml":
        return {
            "ok": True,
            "model": "ml",
            "message": "ML model will be available soon",
            "metrics": {},
            "images": {},
        }

    data = _validate_and_read_file(file_storage)
    image = load_image_from_bytes(data)

    conv_params = ConventionalParams(
        clahe_clip_limit=params.clahe_clip_limit,
        clahe_tile_grid=params.clahe_tile_grid,
        adaptive_window=params.adaptive_window,
        adaptive_offset=params.adaptive_offset,
        morph_kernel=params.morph_kernel,
        morph_iters=params.morph_iters,
    )
    mask, overlay = segment(image, conv_params)
    metrics = compute_metrics(mask)
    analysis_imgs = analyze_mask(mask)
    response = {
        "ok": True,
        "model": "conventional",
        "metrics": metrics,
        "images": {
            "input_png_b64": image_to_png_base64(image),
            "mask_png_b64": image_to_png_base64(mask),
            "overlay_png_b64": image_to_png_base64(overlay),
            "orientation_map_png_b64": analysis_imgs["orientation_map_png_b64"],
            "size_histogram_png_b64": analysis_imgs["size_histogram_png_b64"],
            "angle_histogram_png_b64": analysis_imgs["angle_histogram_png_b64"],
        },
        "message": "success",
    }
    return response
