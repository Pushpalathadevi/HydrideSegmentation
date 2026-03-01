"""Qt GUI entry point for microstructural segmentation desktop app."""

from __future__ import annotations


def launch_qt_gui(*, ui_config_path: str | None = None) -> None:
    """Launch Qt desktop application."""

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PySide6 is required for Qt GUI. Install with `pip install PySide6`."
        ) from exc

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    app = QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow(ui_config_path=ui_config_path)
    win.show()
    app.exec()


def main() -> None:
    """Console entry point wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch MicroSeg Qt desktop app")
    parser.add_argument("--ui-config", type=str, default="", help="Optional desktop UI YAML config path")
    args = parser.parse_args()
    launch_qt_gui(ui_config_path=str(args.ui_config or "").strip() or None)


if __name__ == "__main__":  # pragma: no cover
    main()
