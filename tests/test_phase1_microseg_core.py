"""Phase 1 tests for microseg core contracts and adapters."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from hydride_segmentation.core.image_processing import run_segmentation
from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS
from hydride_segmentation.microseg_adapter import run_pipeline
from hydride_segmentation.segmentation_mask_creation import run_model as run_conv_model
from src.microseg.domain import SegmentationRequest
from src.microseg.inference import build_hydride_registry
from src.microseg.pipelines import SegmentationPipeline


def _synthetic_image() -> np.ndarray:
    img = np.zeros((120, 120), dtype=np.uint8)
    img[15:110, 20:40] = 255
    img[30:95, 70:95] = 255
    return img


def _write_temp_image(image: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(tmp.name)
    return tmp.name


def test_hydride_registry_contains_phase1_models() -> None:
    reg = build_hydride_registry()
    assert "hydride_conventional" in reg.model_ids()
    assert "hydride_ml" in reg.model_ids()


def test_microseg_pipeline_conventional_matches_legacy_mask() -> None:
    path = _write_temp_image(_synthetic_image())
    try:
        legacy_image, legacy_mask = run_conv_model(path, DEFAULT_CONVENTIONAL_PARAMS)

        pipeline = SegmentationPipeline.with_hydride_defaults()
        request = SegmentationRequest(
            image_path=path,
            model_id="hydride_conventional",
            params=DEFAULT_CONVENTIONAL_PARAMS,
            include_analysis=True,
        )
        result = pipeline.run(request)
    finally:
        Path(path).unlink(missing_ok=True)

    assert result.ok is True
    assert result.model_id == "hydride_conventional"
    assert np.array_equal(result.image, legacy_image)
    assert np.array_equal(result.mask, legacy_mask)
    assert "mask_area_fraction" in result.metrics
    assert "hydride_count" in result.metrics
    assert result.manifest["pipeline"] == "microseg.segmentation"


def test_gui_image_processing_conventional_uses_microseg_path() -> None:
    path = _write_temp_image(_synthetic_image())
    try:
        legacy_image, legacy_mask = run_conv_model(path, DEFAULT_CONVENTIONAL_PARAMS)

        gui_params = dict(DEFAULT_CONVENTIONAL_PARAMS)
        gui_params["image_path"] = path

        input_img, mask_img, overlay_img = run_segmentation(gui_params, "Conventional Model")
    finally:
        Path(path).unlink(missing_ok=True)

    assert np.array_equal(np.array(input_img), legacy_image)
    assert np.array_equal(np.array(mask_img), legacy_mask)
    assert np.array(overlay_img).shape[2] == 3


def test_hydride_microseg_adapter_returns_pipeline_result() -> None:
    path = _write_temp_image(_synthetic_image())
    try:
        result = run_pipeline(path, model_id="hydride_conventional", params=DEFAULT_CONVENTIONAL_PARAMS)
    finally:
        Path(path).unlink(missing_ok=True)

    assert result.ok is True
    assert result.model_id == "hydride_conventional"
    assert "input_png_b64" in result.images_b64
    assert "overlay_png_b64" in result.images_b64
