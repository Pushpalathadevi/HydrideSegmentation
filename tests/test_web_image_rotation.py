"""Tests for the browser orientation editor and its server-side rotation.

Fn is measured against the image axes, so a micrograph that was not captured
with the circumferential direction horizontal has to be turned before its angles
mean anything. These tests cover the rotation the browser submits, the preview
endpoint that lets the user see what they are turning, and the record of the
applied angle that keeps a result reproducible.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydride_segmentation.web import create_app
from hydride_segmentation.web.library import render_thumbnail_bytes
from hydride_segmentation.web.segmentation import (
    SegmentationRequestError,
    normalize_rotation,
    prepare_image,
    rotate_image_array,
)

WEB_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hydride_segmentation" / "web"


@pytest.fixture(scope="module")
def client():
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _landmark_array(width: int = 120, height: int = 40) -> np.ndarray:
    """Return an image whose corners identify its orientation."""

    array = np.full((height, width, 3), 90, dtype=np.uint8)
    array[0:6, 0:6] = 255  # top-left marker
    array[0:6, width - 6 :] = 0  # top-right marker
    return array


def _encode(array: np.ndarray, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


# -- angle normalization -------------------------------------------------


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        ("", 0.0),
        (None, 0.0),
        (0, 0.0),
        (90, 90.0),
        (-90, -90.0),
        (180, 180.0),
        (-180, 180.0),
        (270, -90.0),
        (370, 10.0),
        (-370, -10.0),
        ("12.5", 12.5),
    ],
)
def test_normalize_rotation_reduces_into_a_half_turn(submitted, expected) -> None:
    assert normalize_rotation(submitted) == pytest.approx(expected)


@pytest.mark.parametrize("submitted", ["sideways", "nan", "inf", "90deg"])
def test_normalize_rotation_rejects_values_that_are_not_finite_numbers(submitted) -> None:
    with pytest.raises(SegmentationRequestError):
        normalize_rotation(submitted)


# -- rotating the array --------------------------------------------------


def test_quarter_turn_is_lossless_and_swaps_the_axes() -> None:
    array = _landmark_array()
    rotated = rotate_image_array(array, 90)

    assert rotated.shape == (array.shape[1], array.shape[0], 3)
    # A counter-clockwise quarter turn sends the top-left marker to the bottom
    # left, and the values themselves are untouched.
    assert rotated[-1, 0].tolist() == [255, 255, 255]
    assert sorted(np.unique(rotated).tolist()) == sorted(np.unique(array).tolist())


def test_opposite_quarter_turns_restore_the_original_exactly() -> None:
    array = _landmark_array()
    assert np.array_equal(rotate_image_array(rotate_image_array(array, 90), -90), array)


def test_half_turn_flips_both_axes() -> None:
    array = _landmark_array()
    assert np.array_equal(rotate_image_array(array, 180), array[::-1, ::-1])


def test_zero_rotation_returns_the_input_untouched() -> None:
    array = _landmark_array()
    assert rotate_image_array(array, 0) is array


def test_free_angle_is_cropped_inside_the_turned_image() -> None:
    # A uniform image leaves no doubt: any padded corner would show up as a
    # value the source never contained.
    array = np.full((80, 160, 3), 120, dtype=np.uint8)
    rotated = rotate_image_array(array, 15)

    assert rotated.shape[0] >= 1 and rotated.shape[1] >= 1
    assert rotated.shape[0] < array.shape[0]
    assert rotated.shape[1] < array.shape[1]
    assert int(rotated.min()) == 120 and int(rotated.max()) == 120


# -- prepare_image -------------------------------------------------------


def test_prepare_image_reports_the_rotation_and_both_sets_of_dimensions() -> None:
    data = _encode(_landmark_array(120, 40))
    prepared = prepare_image(data, expected_extension="png", rotation_deg=90)

    assert prepared.width == 40 and prepared.height == 120
    assert prepared.uploaded_width == 120 and prepared.uploaded_height == 40

    metadata = prepared.to_metadata()
    assert metadata["rotation_deg"] == pytest.approx(90.0)
    assert metadata["rotation_applied"] is True
    assert metadata["uploaded_width"] == 120
    assert metadata["uploaded_height"] == 40
    # Downstream restoration uses these, so they must describe the rotated image.
    assert metadata["output_width"] == 40 and metadata["output_height"] == 120


def test_prepare_image_without_a_rotation_is_unchanged() -> None:
    data = _encode(_landmark_array(120, 40))
    prepared = prepare_image(data, expected_extension="png")

    assert prepared.width == 120 and prepared.height == 40
    metadata = prepared.to_metadata()
    assert metadata["rotation_deg"] == 0.0
    assert metadata["rotation_applied"] is False


def test_prepare_image_rotates_before_downscaling() -> None:
    data = _encode(_landmark_array(200, 100))
    prepared = prepare_image(
        data, expected_extension="png", rotation_deg=90, max_long_side_px=50
    )

    assert prepared.downscaled is True
    # The long side after the quarter turn is the original height.
    assert prepared.original_width == 100 and prepared.original_height == 200
    assert max(prepared.width, prepared.height) == 50


def test_prepare_image_rejects_a_rotation_that_is_not_a_number() -> None:
    data = _encode(_landmark_array())
    with pytest.raises(SegmentationRequestError):
        prepare_image(data, expected_extension="png", rotation_deg="sideways")


# -- preview endpoint ----------------------------------------------------


def test_preview_endpoint_renders_an_uploaded_tiff(client) -> None:
    data = _encode(_landmark_array(), fmt="TIFF")
    response = client.post(
        "/api/preview",
        data={"image": (io.BytesIO(data), "micrograph.tif")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["Cache-Control"] == "no-store"
    with Image.open(io.BytesIO(response.data)) as preview:
        assert preview.width > 0 and preview.height > 0


def test_preview_endpoint_requires_an_image(client) -> None:
    response = client.post("/api/preview", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "NO_IMAGE"


def test_preview_endpoint_rejects_an_unsupported_extension(client) -> None:
    response = client.post(
        "/api/preview",
        data={"image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_render_thumbnail_bytes_shrinks_a_large_micrograph() -> None:
    payload = render_thumbnail_bytes(_encode(_landmark_array(2000, 1200)))
    with Image.open(io.BytesIO(payload)) as preview:
        assert max(preview.width, preview.height) <= 2000
        assert preview.format == "JPEG"


# -- request plumbing ----------------------------------------------------


def test_jobs_endpoint_accepts_a_rotation(client) -> None:
    data = _encode(_landmark_array(96, 64))
    response = client.post(
        "/api/jobs",
        data={
            "image": (io.BytesIO(data), "micrograph.png"),
            "model_id": "hydride_conventional",
            "rotation_deg": "90",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202


def test_jobs_endpoint_rejects_a_rotation_that_is_not_a_number(client) -> None:
    data = _encode(_landmark_array(96, 64))
    response = client.post(
        "/api/jobs",
        data={
            "image": (io.BytesIO(data), "micrograph.png"),
            "model_id": "hydride_conventional",
            "rotation_deg": "sideways",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "VALIDATION"


# -- browser assets ------------------------------------------------------


def test_index_page_offers_the_orientation_editor(client) -> None:
    body = client.get("/").get_data(as_text=True)
    assert 'id="rotate-panel"' in body
    assert 'id="rotate-slider"' in body
    assert 'id="rotate-canvas"' in body
    assert "circumferential" in body


def test_app_script_submits_the_rotation_and_falls_back_to_a_server_preview() -> None:
    script = (WEB_PACKAGE_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'form.append("rotation_deg"' in script
    assert "api/preview" in script
    assert "largestInscribedRect" in script
    # A library TIFF cannot be posted back, so its cached thumbnail is the
    # fallback the preview falls through to.
    assert "previewFallbackUrl" in script
    assert "image.thumb_url" in script


def test_help_page_documents_the_orientation_assumption(client) -> None:
    body = client.get("/help").get_data(as_text=True)
    assert 'id="orientation"' in body
    assert "circumferential" in body
