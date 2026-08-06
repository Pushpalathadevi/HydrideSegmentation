"""Regression tests for the web download catalog, reports, and inspection UI."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from pypdf import PdfReader

from hydride_segmentation.web import create_app
from hydride_segmentation.web.config import WebServerConfig
from hydride_segmentation.web.downloads import DownloadCatalog
from hydride_segmentation.web.reporting import build_pdf_report, build_report_bundle


def _png_b64(color: tuple[int, int, int]) -> str:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = color
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result() -> dict:
    images = {
        "input_png_b64": _png_b64((190, 190, 190)),
        "mask_png_b64": _png_b64((255, 255, 255)),
        "overlay_png_b64": _png_b64((220, 80, 80)),
        "orientation_map_png_b64": _png_b64((60, 120, 180)),
        "size_histogram_png_b64": _png_b64((80, 140, 200)),
        "angle_histogram_png_b64": _png_b64((230, 150, 50)),
    }
    return {
        "source_name": "sample micrograph.png",
        "model_id": "hydride_conventional",
        "model_display_name": "Conventional segmentation",
        "images": images,
        "metrics": {
            "hydride_count": 3,
            "hydride_area_fraction": 0.125,
            "orientation_mean_deg": 31.2,
            "size_mean_pixels": 28.0,
        },
        "fn": {"available": True, "fn_length_weighted": 0.18, "fn_count": 0.3333},
        "manifest": {
            "image": {"width": 120, "height": 80, "downscaled": False},
            "quantification": {"fn_angle_threshold_deg": 45.0, "min_feature_pixels": 1},
            "privacy": {"source_persisted": False, "result_persisted": False},
        },
        "timing": {"total_seconds": 1.25},
        "analysis_data": {
            "sizes_px": [12, 27, 45],
            "lengths_px": [6.0, 12.5, 19.0],
            "orientations_deg": [10.0, 33.0, 58.0],
            "fn_exceeding_angle": [False, False, True],
            "orientation_histogram": {"counts": [1, 1, 1], "bin_edges_deg": [0, 30, 60, 90]},
            "size_histogram": {"counts": [1, 1, 1], "bin_edges_px": [0, 15, 30, 45]},
        },
    }


def test_download_catalog_is_metadata_driven_and_rejects_escaping_paths(tmp_path: Path) -> None:
    metadata = tmp_path / "downloads" / "metadata"
    metadata.mkdir(parents=True)
    asset = tmp_path / "paper.pdf"
    asset.write_bytes(b"%PDF-test")
    valid = {
        "schema_version": "microseg.download.v1",
        "asset_id": "paper",
        "display_name": "Research paper",
        "help_text": "Scientific background",
        "repo_path": "paper.pdf",
    }
    (metadata / "paper.json").write_text(json.dumps(valid), encoding="utf-8")
    invalid = dict(valid, asset_id="escape", repo_path="../outside.pdf")
    (metadata / "escape.json").write_text(json.dumps(invalid), encoding="utf-8")

    assets = DownloadCatalog(tmp_path).assets()

    assert [item.asset_id for item in assets] == ["paper"]
    assert assets[0].available is True
    assert assets[0].sha256 and len(assets[0].sha256) == 64


def test_download_page_and_route_use_catalog_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "downloads" / "metadata"
    metadata.mkdir(parents=True)
    payload_path = tmp_path / "notes.txt"
    payload_path.write_text("release evidence", encoding="utf-8")
    (metadata / "notes.json").write_text(json.dumps({
        "schema_version": "microseg.download.v1", "asset_id": "notes",
        "display_name": "Release evidence", "help_text": "Traceability notes",
        "repo_path": "notes.txt", "download_name": "MicroSeg_notes.txt",
    }), encoding="utf-8")
    app = create_app(config=WebServerConfig(repo_root=str(tmp_path)), preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        page = client.get("/downloads")
        download = client.get("/downloads/notes")

    assert page.status_code == 200
    assert "Release evidence" in page.get_data(as_text=True)
    assert download.data == b"release evidence"
    assert "MicroSeg_notes.txt" in download.headers["Content-Disposition"]


def test_workspace_places_run_button_near_input_and_exposes_split_zoom_and_version() -> None:
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        body = client.get("/").get_data(as_text=True)
        script = client.get("/static/js/app.js").get_data(as_text=True)

    assert body.index('id="run-btn"') < body.index('id="model-select"')
    assert 'id="compare-input-image"' in body
    assert 'id="compare-mask-image"' in body
    assert 'id="zoom-toggle"' in body
    assert 'id="download-report"' in body and 'id="download-bundle"' in body
    assert "v1.0.0" in body and ">Downloads<" in body
    assert 'event.key === "Escape"' in script
    assert "applySynchronizedZoom" in script


def test_pdf_and_zip_exports_are_scientific_and_reproducible() -> None:
    result = _result()
    job_meta = {"job_id": "job-123", "created_utc": "2026-08-05T10:00:00Z", "finished_utc": "2026-08-05T10:00:02Z"}

    pdf = build_pdf_report(result, app_version="1.0.0", job_meta=job_meta)
    reader = PdfReader(BytesIO(pdf))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert len(reader.pages) == 2
    assert "Detailed Segmentation Report" in text
    assert "hydride_conventional" in text
    assert "Fn (length-weighted)" in text

    bundle = build_report_bundle(result, app_version="1.0.0", job_meta=job_meta)
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert "sample_micrograph_detailed_report.pdf" in names
        assert "sample_micrograph_scientific_data.xlsx" in names
        assert "sample_micrograph_results.json" in names
        assert "images/input.png" in names and "images/mask.png" in names
        workbook = archive.read("sample_micrograph_scientific_data.xlsx")
        with zipfile.ZipFile(BytesIO(workbook)) as xlsx:
            workbook_xml = xlsx.read("xl/workbook.xml").decode("utf-8")
            assert all(name in workbook_xml for name in ("Summary", "Metrics", "Features", "Histograms", "Metadata"))


def test_completed_job_report_endpoints_return_attachments() -> None:
    app = create_app(preload=False)
    app.config.update(TESTING=True)
    manager = app.extensions["microseg_web"]["jobs"]
    job = manager.submit(lambda _progress: _result())
    assert job is not None
    deadline = time.monotonic() + 5
    while job.state != "completed" and time.monotonic() < deadline:
        time.sleep(0.01)

    with app.test_client() as client:
        report = client.get(f"/api/jobs/{job.job_id}/report.pdf")
        bundle = client.get(f"/api/jobs/{job.job_id}/bundle.zip")

    assert report.status_code == 200 and report.data.startswith(b"%PDF")
    assert report.mimetype == "application/pdf"
    assert bundle.status_code == 200 and bundle.data.startswith(b"PK")
    assert bundle.mimetype == "application/zip"
