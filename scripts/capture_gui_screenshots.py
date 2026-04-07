"""Capture representative screenshots of the Qt app UI.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/capture_gui_screenshots.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from hydride_segmentation.qt.main_window import QtSegmentationMainWindow


def _tab_index(tab_widget, name: str) -> int:
    for idx in range(tab_widget.count()):
        if tab_widget.tabText(idx) == name:
            return idx
    raise ValueError(f"tab not found: {name}")


def _save(win: QtSegmentationMainWindow, app: QApplication, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    app.processEvents()
    win.repaint()
    app.processEvents()
    win.grab().save(str(path))


def _pick_default_model(win: QtSegmentationMainWindow) -> str:
    options = win.workflow.model_options()
    for name in options:
        if "conventional" in name.lower():
            return name
    if not options:
        raise RuntimeError("No model options available")
    return options[0]


def _prepare_demo_reports(repo_root: Path) -> tuple[Path, Path]:
    out = repo_root / "outputs" / "ui_review_demo"
    out.mkdir(parents=True, exist_ok=True)

    a = out / "baseline_eval.json"
    b = out / "candidate_eval.json"

    payload_a = {
        "schema_version": "microseg.pixel_eval.v2",
        "backend": "unet_binary",
        "runtime_device": "cpu",
        "runtime_seconds": 42.0,
        "runtime_human": "42s",
        "config_sha256": "demo_cfg_hash_001",
        "samples_evaluated": 25,
        "metrics": {"pixel_accuracy": 0.901, "macro_f1": 0.842, "mean_iou": 0.781},
        "tracked_samples": [{"sample_name": "val_001.png"}, {"sample_name": "val_009.png"}],
    }
    payload_b = {
        "schema_version": "microseg.pixel_eval.v2",
        "backend": "unet_binary",
        "runtime_device": "cpu",
        "runtime_seconds": 39.0,
        "runtime_human": "39s",
        "config_sha256": "demo_cfg_hash_001",
        "samples_evaluated": 25,
        "metrics": {"pixel_accuracy": 0.919, "macro_f1": 0.861, "mean_iou": 0.803},
        "tracked_samples": [{"sample_name": "val_001.png"}, {"sample_name": "val_009.png"}, {"sample_name": "val_013.png"}],
    }
    a.write_text(json.dumps(payload_a, indent=2), encoding="utf-8")
    b.write_text(json.dumps(payload_b, indent=2), encoding="utf-8")
    return a, b


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "artifacts" / "screenshots"
    test_image = repo_root / "data" / "sample_images" / "hydride_optical_sample.png"
    if not test_image.exists():
        test_image = repo_root / "test_data" / "3PB_SRT_data_generation_1817_OD_side1_8.png"

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    win.resize(1780, 1080)
    win.show()
    app.processEvents()

    # Run one segmentation so correction/overlay views are populated.
    try:
        model_name = _pick_default_model(win)
        record = win.workflow.run_single(
            str(test_image),
            model_name=model_name,
            params={"image_path": str(test_image)},
            include_analysis=True,
        )
        win._add_record(record)  # noqa: SLF001 - screenshot utility intentionally uses internal UI path
        win._show_record(record)  # noqa: SLF001
    except Exception:
        # Keep screenshot capture going even if runtime environment cannot execute segmentation.
        pass

    # 1) Main segmentation view.
    win.tabs.setCurrentIndex(_tab_index(win.tabs, "Input"))
    _save(win, app, out_dir / "qt_gui_phase13_input_v0160.png")

    # 2) Correction split view.
    win.tabs.setCurrentIndex(_tab_index(win.tabs, "Correction Split View"))
    _save(win, app, out_dir / "qt_gui_phase13_correction_split_v0160.png")

    # 3) Workflow hub - training tab.
    win.tabs.setCurrentIndex(_tab_index(win.tabs, "Workflow Hub"))
    win.workflow_tabs.setCurrentIndex(_tab_index(win.workflow_tabs, "Training"))
    _save(win, app, out_dir / "qt_gui_phase13_workflow_training_v0160.png")

    # 4) Workflow hub - dataset prep + QA tab.
    win.workflow_tabs.setCurrentIndex(_tab_index(win.workflow_tabs, "Dataset Prep + QA"))
    win.orch_prepare_colormap.setPlainText('{"0":[0,0,0],"1":[255,0,0],"2":[0,255,0]}')
    win.dataset_preview_filter.setText("train")
    _save(win, app, out_dir / "qt_gui_phase13_workflow_dataset_prep_qa_v0160.png")

    # 5) Workflow hub - run review tab (with loaded demo reports).
    report_a, report_b = _prepare_demo_reports(repo_root)
    win.workflow_tabs.setCurrentIndex(_tab_index(win.workflow_tabs, "Run Review"))
    win.review_report_a_edit.setText(str(report_a))
    win.review_report_b_edit.setText(str(report_b))
    win.on_load_review_report_a()
    win.on_load_review_report_b()
    win.on_compare_review_reports()
    _save(win, app, out_dir / "qt_gui_phase13_workflow_run_review_v0160.png")

    # 6) Results dashboard tab.
    win.tabs.setCurrentIndex(_tab_index(win.tabs, "Results Dashboard"))
    win.results_orientation_bins.setValue(24)
    win.results_size_bins.setValue(24)
    win.results_min_feature.setValue(8)
    win.results_size_scale.setCurrentText("linear")
    win.results_cmap.setCurrentText("viridis")
    win._update_results_dashboard()  # noqa: SLF001 - screenshot utility drives UI state
    _save(win, app, out_dir / "qt_gui_phase23_results_dashboard_v0230.png")

    print("Screenshots written to:")
    for name in [
        "qt_gui_phase13_input_v0160.png",
        "qt_gui_phase13_correction_split_v0160.png",
        "qt_gui_phase13_workflow_training_v0160.png",
        "qt_gui_phase13_workflow_dataset_prep_qa_v0160.png",
        "qt_gui_phase13_workflow_run_review_v0160.png",
        "qt_gui_phase23_results_dashboard_v0230.png",
    ]:
        print(str(out_dir / name))


if __name__ == "__main__":
    main()
