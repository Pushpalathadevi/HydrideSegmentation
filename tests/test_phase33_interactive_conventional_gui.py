"""Regression tests for the interactive conventional segmentation workspace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image


def _window():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    QApplication.instance() or QApplication([])
    return QtSegmentationMainWindow()


def test_conventional_workspace_combines_input_output_and_parameter_help() -> None:
    win = _window()
    win.model_combo.setCurrentText("Hydride Conventional")

    assert win.tabs.indexOf(win.comparison_widget) >= 0
    assert win.tabs.indexOf(win.input_view) == -1
    assert win.tabs.indexOf(win.mask_view) == -1
    assert win.comparison_splitter.indexOf(win.input_view) == 0
    assert win.comparison_splitter.indexOf(win.mask_view) == 1
    assert "Result not ready yet" in win.mask_view.image_label.text()
    assert win.conventional_row_widget.isHidden() is False
    assert win.conv_clip_spin.toolTip()
    assert win.conv_block_spin.toolTip()
    assert win.conv_area_spin.toolTip()
    assert win.conv_crop_percent.isEnabled() is False

    win.conv_crop_check.setChecked(True)
    assert win.conv_crop_percent.isEnabled() is True
    win.close()


def test_conventional_parameter_change_queues_debounced_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    win = _window()
    image_path = tmp_path / "input.png"
    Image.new("L", (32, 32), 128).save(image_path)
    win.path_edit.setText(str(image_path))
    win.state.image_path = str(image_path)
    win.model_combo.setCurrentText("Hydride Conventional")

    calls = {"count": 0}
    monkeypatch.setattr(win, "on_run_segmentation", lambda: calls.__setitem__("count", calls["count"] + 1))
    win.conv_c_spin.setValue(win.conv_c_spin.value() + 1)

    assert win._live_conventional_timer.isActive() is True  # noqa: SLF001
    assert "Updating result" in win.mask_view.image_label.text()
    win._live_conventional_timer.stop()  # noqa: SLF001
    win._run_live_conventional_segmentation()  # noqa: SLF001
    assert calls["count"] == 1
    win.close()


def test_loading_new_image_resets_stale_result(tmp_path: Path) -> None:
    win = _window()
    image_path = tmp_path / "new.png"
    Image.new("RGB", (24, 20), (80, 90, 100)).save(image_path)

    win.path_edit.setText(str(image_path))
    win.state.current_run = object()  # type: ignore[assignment]
    win._reset_segmentation_result_view()  # noqa: SLF001

    assert win.state.current_run is None
    assert win.state.correction_session is None
    assert "Result not ready yet" in win.mask_view.image_label.text()
    win.close()


def test_conventional_run_renders_mask_beside_input(tmp_path: Path) -> None:
    win = _window()
    image_path = tmp_path / "hydrides.png"
    image = Image.new("L", (64, 64), 210)
    for x in range(10, 54):
        image.putpixel((x, 24), 25)
        image.putpixel((x, 25), 25)
    image.save(image_path)
    win.path_edit.setText(str(image_path))
    win.state.image_path = str(image_path)
    win.model_combo.setCurrentText("Hydride Conventional")

    record = win._run_desktop_segmentation_job(  # noqa: SLF001
        lambda _message: None,
        path=str(image_path),
        model_name="Hydride Conventional",
        params=win._collect_conventional_params(),  # noqa: SLF001
        include_analysis=False,
        resolved_config={},
    )
    win._show_record(record)  # noqa: SLF001

    assert win.state.current_run is record
    assert win.input_view.image_label.pixmap() is not None
    assert win.mask_view.image_label.pixmap() is not None
    assert not win.mask_view.image_label.pixmap().isNull()
    win.close()
