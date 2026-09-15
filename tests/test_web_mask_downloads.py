"""Mask downloads keep class IDs apart from display values.

``*_mask_labels.png`` holds class numbers {0, 1}; ``*_mask_preview.png`` holds display values
{0, 255}; both come from one label array with identical geometry, and the ZIP's metadata says so.
Non-binary masks are refused rather than silently binarized.
"""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import time
import zipfile

import numpy as np
from PIL import Image
import pytest

from hydride_segmentation.web import create_app
from src.microseg.io.mask_download import (
    MASK_DOWNLOAD_SCHEMA,
    MaskEncodingError,
    binary_labels,
    build_mask_bundle,
    build_mask_downloads,
)

H, W = 40, 60
FOREGROUND = 10 * 40 + 3 * 6


def _mask(value=255, dtype=np.uint8) -> np.ndarray:
    mask = np.zeros((H, W), dtype=dtype)
    mask[5:15, 10:50] = value
    mask[30:33, 2:8] = value
    return mask


def _decode(data: bytes) -> tuple[str, tuple[int, int], np.ndarray, dict]:
    with Image.open(BytesIO(data)) as image:
        return image.mode, image.size, np.asarray(image), dict(image.info)


def _values(arr: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(arr)}


@pytest.mark.parametrize("value, dtype", [(255, np.uint8), (1, np.uint8), (128, np.uint8), (True, np.bool_),
                                          (65535, np.uint16)], ids=["0-255", "0-1", "0-128", "bool", "16-bit"])
def test_labels_hold_only_zero_one_and_preview_only_zero_255_with_identical_geometry(value, dtype):
    mask = _mask(value, dtype)
    downloads = build_mask_downloads(mask, source_name="s.png", base_name="s", app_version="9.9.9")

    mode, size, labels, _ = _decode(downloads.labels_png)
    assert mode == "L" and size == (W, H) and _values(labels) == {0, 1}
    pmode, psize, preview, _ = _decode(downloads.preview_png)
    assert pmode == "L" and psize == (W, H) and _values(preview) == {0, 255}

    assert np.array_equal(labels == 1, preview == 255), "labels and preview must describe the same pixels"
    assert np.array_equal(labels == 1, mask.astype(bool)), "no pixel may move, grow or vanish"
    assert int(labels.sum()) == FOREGROUND


def test_normalization_reuses_the_repository_binary_rules():
    assert binary_labels(_mask(255))[1]["rule"] == "canonical {0,255}->{0,1} (to_index_mask)"
    assert binary_labels(_mask(1))[1]["rule"] == "none: values already {0,1}"
    assert binary_labels(_mask(128))[1]["rule"] == "two_value_zero_background: 128 -> 1"
    labels, info = binary_labels(np.repeat(_mask(255)[:, :, None], 3, axis=2))
    assert int(labels.sum()) == FOREGROUND and info["notes"] == ["identical RGB channels read as greyscale"]


def test_empty_and_full_masks_stay_within_their_encodings():
    empty = build_mask_downloads(np.zeros((H, W), np.uint8), source_name="e", base_name="e", app_version="x")
    assert _values(_decode(empty.labels_png)[2]) == {0} and _values(_decode(empty.preview_png)[2]) == {0}
    full = build_mask_downloads(np.full((H, W), 255, np.uint8), source_name="f", base_name="f", app_version="x")
    assert _values(_decode(full.labels_png)[2]) == {1} and _values(_decode(full.preview_png)[2]) == {255}


RAMP = np.tile(np.arange(W, dtype=np.uint8), (H, 1))
COLOUR = np.zeros((H, W, 3), np.uint8)
COLOUR[5:15, 10:50] = (220, 30, 30)


@pytest.mark.parametrize("mask", [RAMP, COLOUR, np.full((H, W), 7, np.uint8), _mask(2) + 1,
                                  np.linspace(0, 1, W * H, dtype=np.float32).reshape(H, W)],
                         ids=["many-levels", "colour", "single-7", "no-background", "probabilities"])
def test_non_binary_masks_are_refused_not_binarized(mask):
    with pytest.raises(MaskEncodingError):
        binary_labels(mask)


def test_metadata_describes_class_ids_display_mapping_and_both_files():
    downloads = build_mask_downloads(_mask(255), source_name="sample micrograph.png", base_name="sample_micrograph",
                                     app_version="1.2.3", model_id="hydride_conventional", job_meta={"job_id": "j1"})
    meta = downloads.metadata
    assert meta["schema_version"] == MASK_DOWNLOAD_SCHEMA
    assert meta["class_map"] == [{"index": 0, "name": "background"}, {"index": 1, "name": "hydride"}]
    assert meta["display_mapping"] == {"0": 0, "1": 255}
    assert (meta["width"], meta["height"]) == (W, H) and meta["resized"] is False
    assert meta["files"]["labels"]["name"] == "sample_micrograph_mask_labels.png"
    assert meta["files"]["labels"]["values"] == [0, 1]
    assert meta["files"]["preview"]["name"] == "sample_micrograph_mask_preview.png"
    assert meta["files"]["preview"]["values"] == [0, 255]
    assert meta["files"]["labels"]["sha256"] == hashlib.sha256(downloads.labels_png).hexdigest()
    assert meta["files"]["preview"]["sha256"] == hashlib.sha256(downloads.preview_png).hexdigest()
    raw = _decode(downloads.labels_png)[2]
    assert meta["label_sha256"] == hashlib.sha256(f"{W}x{H}:".encode() + raw.tobytes()).hexdigest()
    assert meta["normalization"]["source_values"] == [0, 255]
    assert meta["foreground_pixels"] == FOREGROUND
    assert meta["foreground_fraction"] == pytest.approx(FOREGROUND / (W * H))
    assert (meta["model_id"], meta["job"], meta["app"]["version"]) == ("hydride_conventional", {"job_id": "j1"}, "1.2.3")
    info = _decode(downloads.labels_png)[3]
    assert info["Software"] == "HydrideSegmentation 1.2.3" and info["microseg.mask_encoding"] == "labels"


def test_bundle_holds_both_masks_and_matching_metadata():
    downloads = build_mask_downloads(_mask(255), source_name="s.png", base_name="s", app_version="x")
    with zipfile.ZipFile(BytesIO(build_mask_bundle(downloads))) as archive:
        assert set(archive.namelist()) == {"s_mask_labels.png", "s_mask_preview.png", "s_mask_metadata.json",
                                           "README.txt"}
        labels_png, preview_png = archive.read("s_mask_labels.png"), archive.read("s_mask_preview.png")
        meta = json.loads(archive.read("s_mask_metadata.json"))
        readme = archive.read("README.txt").decode("utf-8")
    assert labels_png == downloads.labels_png and preview_png == downloads.preview_png
    assert meta["files"]["labels"]["sha256"] == hashlib.sha256(labels_png).hexdigest()
    assert meta["files"]["preview"]["sha256"] == hashlib.sha256(preview_png).hexdigest()
    assert "0 = background, 1 = hydride" in readme and "255" in readme


# ----------------------------------------------------------------------------- web routes
def _png_b64(arr: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _completed_job(mask: np.ndarray):
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    result = {"source_name": "sample micrograph.png", "model_id": "hydride_conventional",
              "images": {"input_png_b64": _png_b64(np.full((H, W), 128, np.uint8)), "mask_png_b64": _png_b64(mask)},
              "metrics": {}, "manifest": {}}
    job = app.extensions["microseg_web"]["jobs"].submit(lambda _progress: result)
    assert job is not None
    deadline = time.monotonic() + 5
    while job.state != "completed" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.state == "completed"
    return app, job


def test_job_mask_endpoints_return_labels_preview_and_bundle():
    app, job = _completed_job(_mask(255))
    with app.test_client() as client:
        labels = client.get(f"/api/jobs/{job.job_id}/mask_labels.png")
        preview = client.get(f"/api/jobs/{job.job_id}/mask_preview.png")
        bundle = client.get(f"/api/jobs/{job.job_id}/masks.zip")

    assert labels.status_code == 200 and labels.mimetype == "image/png"
    assert "sample_micrograph_mask_labels.png" in labels.headers["Content-Disposition"]
    assert _values(_decode(labels.data)[2]) == {0, 1}
    assert preview.status_code == 200 and preview.mimetype == "image/png"
    assert "sample_micrograph_mask_preview.png" in preview.headers["Content-Disposition"]
    assert _values(_decode(preview.data)[2]) == {0, 255}
    assert _decode(labels.data)[1] == _decode(preview.data)[1] == (W, H)
    assert bundle.status_code == 200 and bundle.mimetype == "application/zip"
    assert "sample_micrograph_masks.zip" in bundle.headers["Content-Disposition"]
    with zipfile.ZipFile(BytesIO(bundle.data)) as archive:
        meta = json.loads(archive.read("sample_micrograph_mask_metadata.json"))
    assert meta["job"]["job_id"] == job.job_id and meta["source_name"] == "sample micrograph.png"


def test_job_mask_endpoint_explains_a_non_binary_mask():
    app, job = _completed_job(RAMP)
    with app.test_client() as client:
        response = client.get(f"/api/jobs/{job.job_id}/mask_labels.png")
    assert response.status_code == 422
    assert "MASK_NOT_BINARY" in response.get_data(as_text=True)
    assert "single foreground value" in response.get_data(as_text=True)


def test_unknown_job_mask_download_is_not_found():
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        assert client.get("/api/jobs/no-such-job/masks.zip").status_code == 404


def test_workspace_offers_the_mask_choices_and_keeps_existing_downloads():
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        body = client.get("/").get_data(as_text=True)
        script = client.get("/static/js/app.js").get_data(as_text=True)
    for element in ("download-mask-labels", "download-mask-preview", "download-mask-bundle", "download-mask",
                    "download-overlay", "download-report", "download-bundle"):
        assert f'id="{element}"' in body, element
    assert '"mask_labels.png"' in script and '"mask_preview.png"' in script and '"masks.zip"' in script
    assert 'download("mask_png_b64", "mask")' in script, "the original mask download must keep working"
