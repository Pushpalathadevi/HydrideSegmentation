"""Capture reproducible screenshots of the Phase 33 conventional GUI workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from hydride_segmentation.qt.main_window import QtSegmentationMainWindow


def main() -> int:
    """Render the not-ready and completed conventional-segmentation states."""
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "artifacts" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = repo_root / "test_data" / "syntheticHydrides.png"

    app = QApplication.instance() or QApplication([])
    window = QtSegmentationMainWindow()
    window.resize(1500, 900)
    window.model_combo.setCurrentText("Hydride Conventional")
    window._load_sample_path(image_path)  # noqa: SLF001
    window.show()
    app.processEvents()
    window.grab().save(str(output_dir / "qt_gui_conventional_live_not_ready_v033.png"))

    params = window._collect_conventional_params()  # noqa: SLF001
    record = window._run_desktop_segmentation_job(  # noqa: SLF001
        lambda _message: None,
        path=str(image_path),
        model_name="Hydride Conventional",
        params=params,
        include_analysis=False,
        resolved_config={},
    )
    window._show_record(record)  # noqa: SLF001
    app.processEvents()
    window.grab().save(str(output_dir / "qt_gui_conventional_live_result_v033.png"))
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
