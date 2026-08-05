"""Qt GUI entry point for microstructural segmentation desktop app."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def launch_qt_gui(
    *,
    ui_config_path: str | None = None,
    smoke_test: bool = False,
    smoke_report_path: str | None = None,
) -> None:
    """Launch Qt desktop application.

    Parameters
    ----------
    ui_config_path:
        Optional desktop UI configuration override.
    smoke_test:
        Initialize the full main window, record a report, and exit without
        entering an interactive session. Used to verify packaged builds.
    smoke_report_path:
        Optional JSON report destination for packaged smoke validation.
    """

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PySide6 is required for Qt GUI. Install with `pip install PySide6`."
        ) from exc

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("MicroSeg")
    app.setApplicationName("MicroSegDesktop")
    win = QtSegmentationMainWindow(ui_config_path=ui_config_path)
    win.show()

    if smoke_test:
        from hydride_segmentation.version import __version__

        report = {
            "schema_version": "microseg.desktop_packaged_smoke.v1",
            "status": "passed",
            "app_version": __version__,
            "window_title": win.windowTitle(),
            "workspace_root": str(win.orchestrator.repo_root),
            "sample_image_count": len(win._sample_images),
            "cli_executable": str(win.orchestrator.cli_executable or ""),
        }
        if smoke_report_path:
            destination = Path(smoke_report_path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        QTimer.singleShot(250, win.close)
        QTimer.singleShot(500, app.quit)
    app.exec()


def main() -> None:
    """Console entry point wrapper."""
    import argparse

    # The packaged desktop folder contains two launchers built from the same
    # dependency graph. The console launcher dispatches to the unified CLI;
    # the windowed launcher continues into Qt below.
    if Path(sys.executable).stem.casefold() == "microsegcli":
        from scripts.microseg_cli import main as cli_main

        cli_main()
        return

    parser = argparse.ArgumentParser(description="Launch MicroSeg Qt desktop app")
    parser.add_argument("--ui-config", type=str, default="", help="Optional desktop UI YAML config path")
    parser.add_argument("--smoke-test", action="store_true", help="Initialize the packaged app and exit")
    parser.add_argument("--smoke-report", type=str, default="", help="JSON output path for --smoke-test")
    args = parser.parse_args()
    launch_qt_gui(
        ui_config_path=str(args.ui_config or "").strip() or None,
        smoke_test=bool(args.smoke_test),
        smoke_report_path=str(args.smoke_report or "").strip() or None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
