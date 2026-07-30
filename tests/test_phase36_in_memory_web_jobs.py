"""Regression tests for validated, memory-only web jobs."""

from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from hydride_segmentation.web import create_app
from hydride_segmentation.web.config import load_web_config
from hydride_segmentation.web.segmentation import SegmentationRequestError, prepare_image


def _image_bytes(*, image_format: str = "PNG") -> bytes:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[15:32, 8:56] = 235
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=image_format)
    return buffer.getvalue()


def test_packaged_upload_limit_is_five_mb() -> None:
    assert load_web_config().max_upload_mb == 5


def test_workspace_exposes_upload_preview_and_sample_thumbnails() -> None:
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        body = client.get("/").get_data(as_text=True)
        script = client.get("/static/js/app.js").get_data(as_text=True)

    assert 'id="selection-preview"' in body
    assert 'class="sample-btn"' in body
    assert 'data-sample-url="/api/samples/' in body
    assert '<img src="/api/samples/' in body
    assert "URL.createObjectURL(file)" in script
    assert "showPreview(sampleUrl" in script


def test_prepare_image_rejects_extension_content_mismatch() -> None:
    try:
        prepare_image(_image_bytes(), expected_extension="jpg")
    except SegmentationRequestError as exc:
        assert "filename says" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("mismatched image content was accepted")


def test_prepare_image_records_memory_safe_provenance() -> None:
    prepared = prepare_image(_image_bytes(), expected_extension="png")
    metadata = prepared.to_metadata()

    assert metadata["input_transport"] == "memory"
    assert metadata["byte_size"] > 0
    assert len(metadata["sha256"]) == 64
    assert metadata["frame_count"] == 1


def test_background_job_reports_progress_and_terminal_result() -> None:
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post(
            "/api/jobs",
            data={
                "model_id": "hydride_conventional",
                "include_analysis": "false",
                "image": (io.BytesIO(_image_bytes()), "micrograph.png"),
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 202
        created = response.get_json()
        assert created["privacy"]["source_persisted"] is False

        status_url = created["status_url"]
        snapshots = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            payload = client.get(status_url).get_json()
            snapshots.append(payload)
            if payload["terminal"]:
                break
            time.sleep(0.05)

    final = snapshots[-1]
    assert final["state"] == "completed"
    assert final["percent"] == 100
    assert final["events"]
    assert final["result"]["manifest"]["privacy"]["input_transport"] == "memory"
    assert final["result"]["manifest"]["privacy"]["source_persisted"] is False
    assert final["result"]["images"]["mask_png_b64"]


def test_job_submission_rejects_invalid_bytes_before_queueing() -> None:
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post(
            "/api/jobs",
            data={
                "model_id": "hydride_conventional",
                "image": (io.BytesIO(b"not an image"), "micrograph.png"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "VALIDATION"
