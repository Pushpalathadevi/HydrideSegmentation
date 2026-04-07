"""Phase 0 baseline regression tests.

These tests lock current behavior before larger refactors.
"""

from __future__ import annotations

import importlib
import json
from io import BytesIO
from pathlib import Path
import sys

from flask import Flask
import numpy as np
from PIL import Image

from hydride_segmentation.api import create_blueprint
from hydride_segmentation.core.analysis import compute_metrics
from hydride_segmentation.core.conventional import ConventionalParams, segment
from hydride_segmentation.legacy_api import segment_hydride_image


SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "base_zero_snapshot.json"


def _load_snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _synthetic_image() -> np.ndarray:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:90, 10:30] = 255
    img[20:80, 60:80] = 255
    return img


def test_package_import_does_not_import_gui_module() -> None:
    """Importing the package should not require GUI extras at import time."""
    sys.modules.pop("hydride_segmentation", None)
    sys.modules.pop("hydride_segmentation.core.gui_app", None)

    mod = importlib.import_module("hydride_segmentation")

    assert mod is not None
    assert "hydride_segmentation.core.gui_app" not in sys.modules


def test_core_conventional_matches_base_zero_snapshot() -> None:
    """Conventional core metrics should remain stable for baseline synthetic input."""
    snap = _load_snapshot()["core_conventional"]

    mask, _ = segment(_synthetic_image(), ConventionalParams())
    metrics = compute_metrics(mask)

    assert metrics["hydride_count"] == snap["hydride_count"]
    assert metrics["mask_area_fraction"] == snap["mask_area_fraction"]
    assert int((mask > 0).sum()) == snap["foreground_pixels"]
    assert int(mask.sum()) == snap["mask_sum"]


def test_api_segment_conventional_matches_base_zero_snapshot() -> None:
    """Blueprint `/segment` response contract and metrics should remain stable."""
    snap = _load_snapshot()["api_segment_conventional"]

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(), url_prefix="/api/v1/hydride_segmentation")
    client = app.test_client()

    with BytesIO() as buf:
        Image.fromarray(_synthetic_image()).save(buf, format="PNG")
        payload = {"file": (BytesIO(buf.getvalue()), "baseline.png")}

    res = client.post(
        "/api/v1/hydride_segmentation/segment",
        data=payload,
        content_type="multipart/form-data",
    )

    assert res.status_code == 200
    out = res.get_json()

    assert out["ok"] == snap["ok"]
    assert out["model"] == snap["model"]
    assert out["metrics"] == snap["metrics"]
    assert sorted(out["images"].keys()) == snap["image_keys"]


def test_legacy_public_api_conventional_mode_contract() -> None:
    """Legacy callable should keep returning the expected object contract."""
    result = segment_hydride_image(_synthetic_image(), mode="conv")

    assert sorted(result.keys()) == sorted(
        [
            "original",
            "mask",
            "overlay",
            "orientation_map",
            "distribution_plot",
            "angle_distribution",
            "hydride_area_fraction",
        ]
    )
    assert 0.0 <= result["hydride_area_fraction"] <= 1.0
