"""Phase 27 smoke tests for Qt settings integration (offscreen)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


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
                    "control_padding_px": 7,
                    "panel_spacing_px": 10,
                    "table_row_padding_px": 8,
                    "table_min_row_height_px": 30,
                    "high_contrast": True,
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
    assert win.chk_report_csv.isChecked() is True
    assert win.chk_report_pdf.isChecked() is False
    assert win.report_profile_combo.currentText().lower() == "audit"
    assert win.btn_thumb_up.text() == "👍"
    assert win.btn_thumb_down.text() == "👎"
    assert "Feedback:" in win.feedback_rating_label.text()
    export_cfg = win._results_export_config_from_ui()  # noqa: SLF001
    assert export_cfg.report_profile == "audit"
    assert export_cfg.write_csv_report is True
    win.close()
