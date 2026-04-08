"""Phase 27 smoke tests for Qt settings integration (offscreen)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from PIL import Image


def test_phase27_qt_window_applies_ui_config(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    cfg_path = tmp_path / "desktop_ui.custom.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "microseg.desktop_ui_config.v1",
                "appearance": {
                    "base_font_size": 18,
                    "heading_font_size": 20,
                    "monospace_font_size": 15,
                    "menu_font_size": 19,
                    "tab_font_size": 20,
                    "toolbar_font_size": 21,
                    "status_font_size": 17,
                    "control_padding_px": 7,
                    "panel_spacing_px": 10,
                    "table_row_padding_px": 8,
                    "table_min_row_height_px": 30,
                    "high_contrast": True,
                },
                "window": {
                    "initial_width": 3000,
                    "initial_height": 2400,
                    "minimum_width": 1280,
                    "minimum_height": 800,
                    "remember_geometry": False,
                    "clamp_to_screen": True,
                },
                "export_defaults": {
                    "report_profile": "audit",
                    "write_html_report": True,
                    "write_pdf_report": False,
                    "write_csv_report": True,
                    "top_k_key_metrics": 9,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow(ui_config_path=str(cfg_path))
    style = win.styleSheet()
    assert "font-size: 18px" in style
    assert "font-size: 19px" in style
    assert "font-size: 20px" in style
    assert "font-size: 21px" in style
    assert win.chk_report_csv.isChecked() is True
    assert win.chk_report_pdf.isChecked() is False
    assert win.report_profile_combo.currentText().lower() == "audit"
    assert win.btn_thumb_up.text() == "👍"
    assert win.btn_thumb_down.text() == "👎"
    assert "Feedback:" in win.feedback_rating_label.text()
    assert win.model_combo.minimumWidth() >= 320
    assert win.inference_options_group.isCheckable() is True
    assert win.inference_options_group.isChecked() is False
    assert win.correction_tools_group.isCheckable() is True
    assert win.export_group.isCheckable() is True
    assert win.workflow_aux_group.isCheckable() is True
    screen = app.primaryScreen()
    assert screen is not None
    assert win.width() <= screen.availableGeometry().width()
    assert win.height() <= screen.availableGeometry().height()
    export_cfg = win._results_export_config_from_ui()  # noqa: SLF001
    assert export_cfg.report_profile == "audit"
    assert export_cfg.write_csv_report is True
    win.close()


def test_phase27_qt_inference_launches_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()

    image_path = tmp_path / "test_image.png"
    Image.new("RGB", (32, 32), (120, 120, 120)).save(image_path)
    win.path_edit.setText(str(image_path))
    win.orch_infer_image_edit.setText(str(image_path))
    model_name = win.model_combo.currentText()
    monkeypatch.setattr(win, "_selected_model_id", lambda _name: "hydride_ml_Unet")

    captured = {}

    def _fake_start(*, label, model_name, image_path, cfg):
        captured["label"] = label
        captured["model_name"] = model_name
        captured["image_path"] = image_path
        captured["cfg"] = dict(cfg)

    monkeypatch.setattr(win, "_start_inference_subprocess", _fake_start)
    win.on_run_segmentation()
    assert captured["label"] == "single"
    assert captured["model_name"] == model_name
    assert captured["image_path"] == str(image_path)
    assert captured["cfg"]["operator_id"] == str(win.feedback_writer.config.operator_id)
    win.close()


def test_phase27_load_exported_cli_run_round_trips(tmp_path: Path) -> None:
    from src.microseg.app.desktop_workflow import load_exported_run

    run_dir = tmp_path / "sample_run"
    run_dir.mkdir()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(run_dir / "input.png")
    Image.new("L", (16, 16), 0).save(run_dir / "prediction.png")
    Image.new("RGB", (16, 16), (40, 50, 60)).save(run_dir / "overlay.png")
    Image.new("RGB", (16, 16), (70, 80, 90)).save(run_dir / "orientation_map.png")
    (run_dir / "metrics.json").write_text('{"dice": 0.91}', encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        """
{
  "run_id": "20260408T162308",
  "image_path": "test_data/example.png",
  "image_name": "example.png",
  "model_name": "Registry: hydride_unet_optical_v1 (unet_binary)",
  "model_id": "hydride_ml_Unet",
  "started_utc": "2026-04-08T16:23:06.661119+00:00",
  "finished_utc": "2026-04-08T16:23:08.185992+00:00",
  "metrics": {"dice": 0.91},
  "manifest": {"params": {"enable_gpu": false}}
}
""".strip(),
        encoding="utf-8",
    )

    record = load_exported_run(run_dir)
    assert record.run_id == "20260408T162308"
    assert record.model_id == "hydride_ml_Unet"
    assert record.metrics["dice"] == 0.91
    assert record.input_image.size == (16, 16)
    assert record.mask_image.size == (16, 16)
    assert record.overlay_image.size == (16, 16)
    assert "orientation_map" in record.analysis_images_b64
