"""Qt GUI entry point for microstructural segmentation desktop app."""

from __future__ import annotations


def launch_qt_gui() -> None:
    """Launch Qt desktop application."""

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PySide6 is required for Qt GUI. Install with `pip install PySide6`."
        ) from exc

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    win.show()
    app.exec()


def main() -> None:
    """Console entry point wrapper."""

    launch_qt_gui()


if __name__ == "__main__":  # pragma: no cover
    main()
