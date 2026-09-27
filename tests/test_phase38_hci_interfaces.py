"""Phase 38: HCI through the CLI, the web API, reports and Help (interface parity)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydride_segmentation.web import create_app
from hydride_segmentation.web.reporting import build_excel_workbook, build_pdf_report, build_report_bundle
from hydride_segmentation.web.segmentation import SegmentationRequestError, build_hci_request
from src.microseg.evaluation.continuity import Calibration, ContinuityAnalysisConfig, analyze_continuity

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "test_data" / "hci_synthetic_v1" / "masks"


@pytest.fixture(scope="module")
def client():
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _mask_png(path: Path) -> bytes:
    """A benchmark mask rendered as a dark-hydride grey micrograph."""

    mask = np.array(Image.open(path)) > 0
    img = np.where(mask, 40, 210).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


# -- request parsing ------------------------------------------------------


def test_hci_request_defaults_to_enabled_auto_min() -> None:
    req = build_hci_request({})
    assert req.enabled and req.bridge_mode == "auto_min"
    assert req.config.max_bridge_distance == "auto" and req.config.auto_size_statistic == "min"
    assert not req.calibration.calibrated


def test_hci_request_modes_and_validation() -> None:
    fixed = build_hci_request({"hci_bridge_mode": "fixed", "hci_max_bridge_distance": "10", "hci_pixel_size_um": "0.5"})
    assert fixed.config.max_bridge_distance == 10.0 and fixed.calibration.pixel_size_um == 0.5
    median = build_hci_request({"hci_bridge_mode": "auto_median"})
    assert median.config.auto_size_statistic == "percentile" and median.config.auto_size_percentile == 50.0
    assert not build_hci_request({"hci_enabled": "false"}).enabled
    for bad in (
        {"hci_bridge_mode": "fixed"},
        {"hci_bridge_mode": "fixed", "hci_max_bridge_distance": "-1"},
        {"hci_bridge_mode": "nope"},
        {"hci_pixel_size_um": "abc"},
        {"hci_reference_length_radial_um": "600"},
    ):
        with pytest.raises(SegmentationRequestError):
            build_hci_request(bad)


# -- web API --------------------------------------------------------------


def test_segment_returns_hci_by_default(client) -> None:
    data = {"model_id": "hydride_conventional", "image": (io.BytesIO(_mask_png(BENCH / "gap_sweep" / "gap_sweep__gap_03px__r00.png")), "chains.png")}
    response = client.post("/api/segment", data=data)
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    hci = payload["hci"]
    assert hci["enabled"] and hci["status"] == "ok"
    assert hci["schema_version"] == "microseg.hci_result.v1"
    for d in ("radial", "circumferential", "isotropic"):
        assert 0 < hci["specimen"]["HCI"][d] <= 1
    for key in ("hci_clusters_png_b64", "hci_curve_png_b64", "hci_topology_png_b64", "hci_path_png_b64"):
        assert payload["images"].get(key)
    assert payload["manifest"]["hci"]["enabled"] is True
    assert payload["timing"]["hci_seconds"] >= 0


def test_segment_without_hci_and_fixed_mode(client) -> None:
    png = _mask_png(BENCH / "arrangement" / "arrangement__radial_stringers__r00.png")
    off = client.post("/api/segment", data={"model_id": "hydride_conventional", "hci_enabled": "false", "image": (io.BytesIO(png), "a.png")}).get_json()
    assert off["hci"] == {"enabled": False}
    assert "hci_curve_png_b64" not in off["images"]
    fixed = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "hci_bridge_mode": "fixed", "hci_max_bridge_distance": "10", "hci_pixel_size_um": "0.5", "image": (io.BytesIO(png), "a.png")},
    ).get_json()
    assert fixed["hci"]["specimen"]["max_bridge_distance"] == pytest.approx(10.0)
    assert fixed["hci"]["unit"] == "um"
    bad = client.post("/api/segment", data={"model_id": "hydride_conventional", "hci_bridge_mode": "fixed", "image": (io.BytesIO(png), "a.png")})
    assert bad.status_code == 400


def test_workspace_and_help_expose_hci(client) -> None:
    page = client.get("/").get_data(as_text=True)
    assert 'id="chk-hci" checked' in page and 'id="hci-panel"' in page
    assert 'data-view="hci_curve_png_b64"' in page
    helptext = client.get("/help").get_data(as_text=True)
    assert 'id="hci"' in helptext and "hci_concept.svg" in helptext and "hci_algorithm_flow.svg" in helptext
    assert r"\mathrm{HCI}_u" in helptext and "N_E = \\beta + 2C - 2\\mu" in helptext
    for asset in ("img/hci_concept.svg", "img/hci_algorithm_flow.svg"):
        assert client.get(f"/static/{asset}").status_code == 200


# -- reports --------------------------------------------------------------


def _web_result_with_hci() -> dict:
    from hydride_segmentation.web.segmentation import hci_web_summary
    from src.microseg.evaluation.continuity import visualization as viz
    import base64

    mask = np.array(Image.open(BENCH / "topology" / "topology__honeycomb_mesh__r00.png"))
    req = build_hci_request({"hci_pixel_size_um": "0.5"})
    res = analyze_continuity(mask, req.config, req.calibration)
    png = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return {
        "source_name": "mesh.png",
        "model_id": "hydride_conventional",
        "model_display_name": "Conventional",
        "fn": {},
        "metrics": {"area_fraction": float((mask > 0).mean())},
        "hci": hci_web_summary(res, req, 0.1),
        "images": {"mask_png_b64": png(viz.to_png_bytes((mask > 0).astype(np.uint8) * 255)), "hci_curve_png_b64": png(viz.curve_figure(res)), "hci_clusters_png_b64": png(viz.to_png_bytes(viz.cluster_overlay(res)))},
        "manifest": {"image": {}, "quantification": {}},
        "timing": {},
    }


def test_pdf_xlsx_and_bundle_include_hci() -> None:
    from pypdf import PdfReader

    result = _web_result_with_hci()
    meta = {"job_id": "job-hci"}
    reader = PdfReader(io.BytesIO(build_pdf_report(result, app_version="2.0.0", job_meta=meta)))
    assert len(reader.pages) == 3
    assert "Hydride Connectivity Index" in (reader.pages[2].extract_text() or "")
    with zipfile.ZipFile(io.BytesIO(build_excel_workbook(result, app_version="2.0.0", job_meta=meta))) as xlsx:
        assert "Connectivity" in xlsx.read("xl/workbook.xml").decode("utf-8")
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(result, app_version="2.0.0", job_meta=meta))) as bundle:
        manifest = json.loads(bundle.read("mesh_results.json"))
        assert manifest["hci"]["status"] == "ok"
        assert "images/hci_curve.png" in bundle.namelist()


# -- CLI parity -----------------------------------------------------------


def test_cli_matches_library(tmp_path: Path) -> None:
    mask_path = BENCH / "arrangement" / "arrangement__radial_stepped__r00.png"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "microseg_cli.py"), "hci", "--mask", str(mask_path), "--output-dir", str(tmp_path), "--pixel-size-um", "0.5", "--max-bridge-distance", "10", "--no-overlays"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    cli = json.loads((tmp_path / mask_path.stem / "hci_result.json").read_text(encoding="utf-8"))
    lib = analyze_continuity(np.array(Image.open(mask_path)), ContinuityAnalysisConfig(max_bridge_distance=10.0), Calibration(0.5, "cli")).to_dict()
    assert cli["specimen"]["HCI"] == pytest.approx(lib["specimen"]["HCI"], abs=1e-12)
    assert cli["specimen"]["critical_linking_distance"] == lib["specimen"]["critical_linking_distance"]
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["results"][0]["status"] == "ok"
    assert (tmp_path / "summary.csv").exists()
