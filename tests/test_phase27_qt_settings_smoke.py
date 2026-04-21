"""Phase 27 smoke tests for Qt settings integration (offscreen)."""

from __future__ import annotations

import os
import base64
from io import BytesIO
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
    assert win.inference_options_group.isChecked() is True
    assert win.setup_status_box.isCheckable() is True
    assert win.correction_tools_group.isCheckable() is True
    assert win.export_group.isCheckable() is True
    assert win.workflow_aux_group.isCheckable() is True
    assert win.active_run_box.isHidden() is True
    assert win.history_box.isHidden() is True
    assert win.log_panel.isHidden() is False
    assert win.model_combo.count() == 2
    assert win.model_combo.itemText(0) == "Hydride ML (UNet)"
    assert win.model_combo.itemText(1) == "Hydride Conventional"
    assert win.model_combo.currentText() == "Hydride ML (UNet)"
    screen = app.primaryScreen()
    assert screen is not None
    assert win.width() <= screen.availableGeometry().width()
    assert win.height() <= screen.availableGeometry().height()
    export_cfg = win._results_export_config_from_ui()  # noqa: SLF001
    assert export_cfg.report_profile == "audit"
    assert export_cfg.write_csv_report is True
    assert "Model:" in win.current_model_summary_label.text()
    win.close()


def test_phase27_qt_inference_uses_background_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _fake_start(*, label, job, on_finished):
        captured["label"] = label
        captured["job"] = job
        captured["on_finished"] = on_finished

    monkeypatch.setattr(win, "_start_background_job", _fake_start)
    win.on_run_segmentation()
    assert captured["label"] == "single"
    assert callable(captured["job"])
    assert callable(captured["on_finished"])
    win.close()


def test_phase27_qt_history_selection_defers_results_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow
    from src.microseg.app.desktop_workflow import DesktopRunRecord

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    mask = Image.new("L", (32, 32), 0)
    overlay = Image.new("RGB", (32, 32), (100, 80, 60))
    record = DesktopRunRecord(
        run_id="run_lazy_dashboard",
        image_path="test_data/example.png",
        image_name="example.png",
        model_name=win.model_combo.currentText(),
        model_id="hydride_ml",
        started_utc="2026-04-20T00:00:00+00:00",
        finished_utc="2026-04-20T00:00:01+00:00",
        input_image=image,
        mask_image=mask,
        overlay_image=overlay,
        metrics={},
        manifest={},
    )
    win.workflow.append_history(record)
    win.history_list.addItem(record.history_label)

    calls = {"count": 0}

    def _fake_update() -> None:
        calls["count"] += 1

    monkeypatch.setattr(win, "_update_results_dashboard", _fake_update)
    win.tabs.setCurrentWidget(win.input_view)
    win._show_record(record)  # noqa: SLF001
    assert calls["count"] == 0
    assert win._results_dirty is True  # noqa: SLF001
    win.close()


def test_phase27_qt_results_dashboard_uses_background_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow
    from src.microseg.app.desktop_workflow import DesktopRunRecord

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    mask = Image.new("L", (32, 32), 0)
    overlay = Image.new("RGB", (32, 32), (100, 80, 60))
    record = DesktopRunRecord(
        run_id="run_async_dashboard",
        image_path="test_data/example.png",
        image_name="example.png",
        model_name=win.model_combo.currentText(),
        model_id="hydride_ml",
        started_utc="2026-04-20T00:00:00+00:00",
        finished_utc="2026-04-20T00:00:01+00:00",
        input_image=image,
        mask_image=mask,
        overlay_image=overlay,
        metrics={},
        manifest={},
    )
    win.state.current_run = record
    win.state.correction_session = None

    calls: dict[str, object] = {}

    def _fake_start_results_dashboard_worker(**kwargs) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(win, "_start_results_dashboard_worker", _fake_start_results_dashboard_worker)
    win._update_results_dashboard()  # noqa: SLF001
    assert calls["run_id"] == "run_async_dashboard"
    assert "cache_key" in calls
    assert win.results_summary_label.text() in {
        "Results: computing analysis in background...",
        "Results: run segmentation to populate dashboard",
    } or "computing analysis" in win.results_summary_label.text()
    win.close()


def test_phase27_qt_results_dashboard_uses_run_record_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow
    from src.microseg.app.desktop_workflow import DesktopRunRecord

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    mask = Image.new("L", (32, 32), 0)
    overlay = Image.new("RGB", (32, 32), (100, 80, 60))

    def _b64_png(rgb: tuple[int, int, int]) -> str:
        buf = BytesIO()
        Image.new("RGB", (16, 16), rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    record = DesktopRunRecord(
        run_id="run_fast_path_dashboard",
        image_path="test_data/example.png",
        image_name="example.png",
        model_name=win.model_combo.currentText(),
        model_id="hydride_ml",
        started_utc="2026-04-20T00:00:00+00:00",
        finished_utc="2026-04-20T00:00:01+00:00",
        input_image=image,
        mask_image=mask,
        overlay_image=overlay,
        metrics={"hydride_area_fraction_percent": 12.3456, "hydride_count": 4},
        manifest={},
        analysis_images_b64={
            "orientation_map_png_b64": _b64_png((255, 0, 0)),
            "size_histogram_png_b64": _b64_png((0, 255, 0)),
            "angle_histogram_png_b64": _b64_png((0, 0, 255)),
        },
    )
    win.state.current_run = record
    win.state.correction_session = None

    called = {"worker": False}

    def _fake_start_results_dashboard_worker(**kwargs) -> None:
        called["worker"] = True

    monkeypatch.setattr(win, "_start_results_dashboard_worker", _fake_start_results_dashboard_worker)
    win._update_results_dashboard()  # noqa: SLF001
    assert called["worker"] is False
    assert "predicted area=12.35" in win.results_summary_label.text()
    win.close()


def test_phase27_qt_startup_requests_model_warm_load(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    calls: list[str] = []
    original = QtSegmentationMainWindow._start_model_warm_load

    def _tracked(self, model_name: str) -> None:
        calls.append(str(model_name))
        return original(self, model_name)

    monkeypatch.setattr(QtSegmentationMainWindow, "_start_model_warm_load", _tracked)
    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    app.processEvents()
    assert any(name == win.model_combo.currentText() for name in calls)
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


def test_phase27_qt_batch_progress_updates_status_banner() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow
    from src.microseg.app import DesktopBatchProgress

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    win._set_segmentation_busy(True, label="batch")  # noqa: SLF001
    win._on_background_job_status(  # noqa: SLF001
        DesktopBatchProgress(
            stage="infer",
            message="[1/4] Inference complete for sample_a.png.",
            completed_steps=2,
            total_steps=10,
            completed_images=1,
            total_images=4,
            percent_complete=20,
            elapsed_seconds=5.0,
            eta_seconds=15.0,
            current_image="sample_a.png",
        )
    )
    assert win.segmentation_progress_label.text() == "[1/4] Inference complete for sample_a.png."
    assert "Processed 1/4 images" in win.segmentation_detail_label.text()
    assert win.segmentation_progress_bar.value() == 20
    assert "ETA:" in win.segmentation_eta_label.text()
    win._set_segmentation_busy(False)  # noqa: SLF001
    win.close()
