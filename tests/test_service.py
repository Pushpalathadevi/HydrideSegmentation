"""Tests for legacy Flask service request validation and response behavior."""

from __future__ import annotations

import sys
import types
from io import BytesIO

import numpy as np
from PIL import Image

# Stub optional ML inference dependency so service module can import in CPU-only test env.
fake_inference = types.ModuleType("hydride_segmentation.inference")


def _fake_ml_run_model(image_path: str, params=None):  # noqa: ANN001
    return np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)


fake_inference.run_model = _fake_ml_run_model  # type: ignore[attr-defined]
sys.modules.setdefault("hydride_segmentation.inference", fake_inference)

import hydride_segmentation.service as service  # noqa: E402


def _png_bytes() -> bytes:
    image = np.zeros((24, 24), dtype=np.uint8)
    image[8:16, 8:16] = 255
    with BytesIO() as buf:
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()


def test_service_rejects_unknown_model_name() -> None:
    client = service.app.test_client()
    payload = {"image": (BytesIO(_png_bytes()), "test.png"), "model": "mystery"}
    response = client.post("/infer", data=payload, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Unsupported model" in response.get_json()["error"]


def test_service_rejects_invalid_conventional_literal(monkeypatch) -> None:
    observed = {"called": False}

    def _fake_segment(image_path: str, model: str, params: dict):  # noqa: ANN001
        observed["called"] = True
        return np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr(service, "_segment", _fake_segment)

    client = service.app.test_client()
    payload = {
        "image": (BytesIO(_png_bytes()), "test.png"),
        "model": "conv",
        "clahe": "{bad_json}",
    }
    response = client.post("/infer", data=payload, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Invalid value for 'clahe'" in response.get_json()["error"]
    assert observed["called"] is False


def test_service_conventional_model_alias_runs_inference(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _fake_segment(image_path: str, model: str, params: dict):  # noqa: ANN001
        observed["model"] = model
        observed["params"] = params
        return np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr(service, "_segment", _fake_segment)

    client = service.app.test_client()
    payload = {"image": (BytesIO(_png_bytes()), "test.png"), "model": "conventional"}
    response = client.post("/infer", data=payload, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.content_type.startswith("image/png")
    assert observed["model"] == "conventional"
    assert isinstance(observed["params"], dict)
