"""Phase 27 tests for desktop UI config validation and stylesheet generation."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.microseg.app.desktop_ui_config import (
    SCHEMA_VERSION,
    build_qt_stylesheet,
    default_desktop_ui_config,
    load_desktop_ui_config,
)


def test_phase27_ui_config_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    cfg, warnings, source = load_desktop_ui_config(tmp_path / "missing.yml")
    assert cfg.schema_version == SCHEMA_VERSION
    assert source is not None
    assert warnings
    assert cfg.appearance.base_font_size == default_desktop_ui_config().appearance.base_font_size


def test_phase27_ui_config_invalid_values_are_clamped(tmp_path: Path) -> None:
    path = tmp_path / "ui.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "appearance": {
                    "base_font_size": 1000,
                    "heading_font_size": -9,
                    "monospace_font_size": "bad",
                    "menu_font_size": 1,
                    "tab_font_size": 900,
                    "toolbar_font_size": "bad",
                    "status_font_size": 2,
                    "control_padding_px": 1,
                    "panel_spacing_px": 999,
                    "table_row_padding_px": -3,
                    "table_min_row_height_px": 500,
                    "high_contrast": "true",
                },
                "window": {
                    "initial_width": 99999,
                    "initial_height": 4,
                    "minimum_width": 1,
                    "minimum_height": 1,
                    "left_dock_width": 1,
                    "right_dock_width": 99999,
                    "workflow_dock_width": 4,
                    "remember_geometry": "true",
                    "clamp_to_screen": "false",
                    "start_maximized": "false",
                    "start_fullscreen": "false",
                },
                "export_defaults": {
                    "report_profile": "unknown_profile",
                    "write_csv_report": True,
                    "include_sections": ["metadata", "bad_section"],
                    "sort_metrics": "bad",
                    "top_k_key_metrics": -7,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg, warnings, _ = load_desktop_ui_config(path)
    assert warnings
    assert cfg.appearance.base_font_size == 30
    assert cfg.appearance.heading_font_size == 11
    assert cfg.appearance.menu_font_size == 10
    assert cfg.appearance.tab_font_size == 30
    assert cfg.appearance.toolbar_font_size == 15
    assert cfg.appearance.status_font_size == 10
    assert cfg.appearance.control_padding_px == 2
    assert cfg.appearance.panel_spacing_px == 24
    assert cfg.appearance.table_row_padding_px == 2
    assert cfg.appearance.table_min_row_height_px == 64
    assert cfg.appearance.high_contrast is True
    assert cfg.window.initial_width == 5120
    assert cfg.window.initial_height == 768
    assert cfg.window.minimum_width == 800
    assert cfg.window.minimum_height == 600
    assert cfg.window.left_dock_width == 220
    assert cfg.window.right_dock_width == 800
    assert cfg.window.workflow_dock_width == 600
    assert cfg.window.remember_geometry is True
    assert cfg.window.clamp_to_screen is False
    assert cfg.export_defaults.report_profile == "balanced"
    assert cfg.export_defaults.sort_metrics == "name"
    assert cfg.export_defaults.top_k_key_metrics == 1
    assert "metadata" in cfg.export_defaults.include_sections


def test_phase27_stylesheet_generation_uses_config_values(tmp_path: Path) -> None:
    path = tmp_path / "ui2.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "appearance": {
                    "base_font_size": 18,
                    "heading_font_size": 22,
                    "monospace_font_size": 16,
                    "menu_font_size": 19,
                    "tab_font_size": 20,
                    "toolbar_font_size": 21,
                    "status_font_size": 17,
                    "control_padding_px": 7,
                    "panel_spacing_px": 10,
                    "table_row_padding_px": 9,
                    "table_min_row_height_px": 30,
                    "high_contrast": False,
                },
                "window": {
                    "initial_width": 1440,
                    "initial_height": 900,
                    "minimum_width": 1280,
                    "minimum_height": 800,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg, warnings, _ = load_desktop_ui_config(path)
    assert not warnings
    style = build_qt_stylesheet(cfg)
    assert "font-size: 18px" in style
    assert "font-size: 22px" in style
    assert "font-size: 16px" in style
    assert "font-size: 19px" in style
    assert "font-size: 20px" in style
    assert "font-size: 21px" in style
    assert "min-height: 30px" in style
    assert "QTabBar::tab {" in style
    assert "QTabBar::tab:selected {" in style
    assert "QTabWidget::pane {" in style
    assert "QPushButton:hover" in style
    assert "QToolButton:hover" in style
    assert "selection-background-color" in style
    assert "QFrame#segmentationStatusPanel {" in style
    assert "QProgressBar {" in style
    assert "QProgressBar::chunk {" in style
    assert "#0D1117" in style
    assert "#161B22" in style
