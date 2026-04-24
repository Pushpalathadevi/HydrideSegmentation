"""Phase 23 tests for hydride statistics and desktop results reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.app import DesktopResultExportConfig, DesktopResultExporter, DesktopWorkflowManager
from src.microseg.corrections import CorrectionSession
from src.microseg.evaluation import (
    HydrideVisualizationConfig,
    compute_hydride_statistics,
    render_hydride_visualizations,
)
from src.microseg.utils import calibration_from_manual_line, metadata_calibration_from_image


def _synthetic_mask() -> np.ndarray:
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[20:45, 10:90] = 1
    mask[70:95, 40:110] = 1
    return mask


def _synthetic_image() -> np.ndarray:
    image = np.zeros((120, 120), dtype=np.uint8)
    image[20:45, 10:90] = 220
    image[70:95, 40:110] = 180
    return image


def _write_image(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _conventional_model_name(workflow: DesktopWorkflowManager) -> str:
    return next(name for name in workflow.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase23_compute_hydride_statistics_and_visuals() -> None:
    mask = _synthetic_mask()
    stats = compute_hydride_statistics(mask, orientation_bins=12, size_bins=10, min_feature_pixels=4)
    assert int(stats.scalar_metrics["hydride_count"]) == 2
    assert float(stats.scalar_metrics["hydride_area_fraction"]) > 0.0
    assert len(stats.orientation_hist_counts) == 12
    assert len(stats.size_hist_counts) == 10

    visuals = render_hydride_visualizations(
        stats,
        HydrideVisualizationConfig(
            orientation_bins=12,
            size_bins=10,
            min_feature_pixels=4,
            orientation_cmap="viridis",
            size_scale="linear",
        ),
    )
    assert sorted(visuals.keys()) == [
        "orientation_distribution_rgb",
        "orientation_map_rgb",
        "size_distribution_rgb",
    ]
    for arr in visuals.values():
        assert arr.ndim == 3
        assert arr.shape[2] == 3


def test_phase23_statistics_with_micron_calibration() -> None:
    mask = _synthetic_mask()
    stats = compute_hydride_statistics(
        mask,
        orientation_bins=10,
        size_bins=8,
        min_feature_pixels=3,
        microns_per_pixel=0.5,
    )
    assert stats.microns_per_pixel == 0.5
    assert "size_mean_um2" in stats.scalar_metrics
    assert "equivalent_diameter_mean_um" in stats.scalar_metrics
    assert len(stats.sizes_um2) == int(stats.scalar_metrics["hydride_count"])
    assert len(stats.size_hist_counts_um2) == 8


def test_phase23_manual_and_metadata_calibration(tmp_path: Path) -> None:
    manual = calibration_from_manual_line(pixel_distance=200.0, known_length_value=100.0, known_length_unit="um")
    assert abs(manual.microns_per_pixel - 0.5) < 1e-9
    assert manual.source == "manual_line"

    tif_path = tmp_path / "calib.tiff"
    Image.fromarray(_synthetic_image()).save(tif_path, format="TIFF", dpi=(25400, 25400))
    auto = metadata_calibration_from_image(tif_path)
    assert auto is not None
    assert auto.source in {"tiff_metadata", "dpi_metadata"}
    assert abs(float(auto.microns_per_pixel) - 1.0) < 0.05


def test_phase23_desktop_result_export_package(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)

    image_path = tmp_path / "input.png"
    _write_image(image_path, _synthetic_image())

    record = workflow.run_single(
        str(image_path),
        model_name=model_name,
        params={"image_path": str(image_path)},
        include_analysis=True,
    )
    sess = CorrectionSession(np.array(record.mask_image))
    sess.apply_brush(x=12, y=12, radius=6, mode="add", class_index=1)

    exporter = DesktopResultExporter()
    out_dir = exporter.export(
        record,
        output_dir=tmp_path / "results",
        corrected_mask=sess.current_mask,
        annotator="qa",
        notes="phase23",
        config=DesktopResultExportConfig(
            orientation_bins=16,
            size_bins=14,
            min_feature_pixels=3,
            orientation_cmap="plasma",
            size_scale="linear",
            microns_per_pixel=0.5,
            calibration_source="manual_line",
            calibration_notes="test calibration",
            write_html_report=True,
            write_pdf_report=True,
            compute_extended_metrics=True,
            write_distribution_charts=True,
            write_physical_calibration_metrics=True,
        ),
    )

    assert out_dir.exists()
    assert (out_dir / "results_summary.json").exists()
    assert (out_dir / "results_report.html").exists()
    assert (out_dir / "results_report.pdf").exists()
    assert (out_dir / "results_metrics.csv").exists()
    assert (out_dir / "artifacts_manifest.json").exists()
    assert (out_dir / "predicted_mask_indexed.png").exists()
    assert (out_dir / "corrected_mask_indexed.png").exists()
    assert (out_dir / "predicted_orientation_distribution.png").exists()
    assert (out_dir / "corrected_orientation_distribution.png").exists()

    payload = json.loads((out_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.desktop_results.v2"
    assert payload["annotator"] == "qa"
    assert "predicted_stats" in payload
    assert "corrected_stats" in payload
    assert float(payload["spatial_calibration"]["microns_per_pixel"]) == 0.5
    assert "applied_export_criteria" in payload
    assert payload["report_outputs"]["metrics_csv"] == "results_metrics.csv"


def test_phase23_desktop_result_export_skips_optional_distribution_charts(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)

    image_path = tmp_path / "input_optional.png"
    _write_image(image_path, _synthetic_image())
    record = workflow.run_single(
        str(image_path),
        model_name=model_name,
        params={"image_path": str(image_path)},
        include_analysis=False,
    )

    out_dir = DesktopResultExporter().export(
        record,
        output_dir=tmp_path / "results_optional",
        config=DesktopResultExportConfig(write_pdf_report=False),
    )

    assert (out_dir / "predicted_orientation_map.png").exists()
    assert not (out_dir / "predicted_size_distribution.png").exists()
    assert not (out_dir / "predicted_orientation_distribution.png").exists()
    payload = json.loads((out_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert payload["analysis_config"]["postprocessing_options"]["write_distribution_charts"] is False
    assert payload["applied_export_criteria"]["compute_extended_metrics"] is False
