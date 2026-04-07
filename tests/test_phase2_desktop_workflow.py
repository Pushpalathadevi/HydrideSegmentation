"""Phase 2 tests for desktop workflow manager and export packaging."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from src.microseg.app.desktop_workflow import DesktopWorkflowManager
from hydride_segmentation.microseg_adapter import resolve_gui_model_id


def _synthetic_image_a() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[10:80, 12:28] = 255
    arr[20:75, 55:75] = 255
    return arr


def _synthetic_image_b() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[8:70, 8:24] = 255
    arr[30:86, 50:85] = 255
    return arr


def _tmp_image(image: np.ndarray) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(f.name)
    return f.name


def test_phase2_model_registry_options_available() -> None:
    mgr = DesktopWorkflowManager()
    options = mgr.model_options()
    assert options
    assert any(resolve_gui_model_id(name) == "hydride_conventional" for name in options)
    assert any(resolve_gui_model_id(name) == "hydride_ml" for name in options)


def test_phase2_single_run_and_export_package() -> None:
    mgr = DesktopWorkflowManager()
    conv_name = next(name for name in mgr.model_options() if resolve_gui_model_id(name) == "hydride_conventional")

    p = _tmp_image(_synthetic_image_a())
    out_dir = tempfile.mkdtemp(prefix="phase2_export_")
    try:
        record = mgr.run_single(p, model_name=conv_name, params={"image_path": p}, include_analysis=True)
        run_dir = mgr.export_run(record, out_dir)
    finally:
        Path(p).unlink(missing_ok=True)

    assert run_dir.exists()
    assert (run_dir / "input.png").exists()
    assert (run_dir / "prediction.png").exists()
    assert (run_dir / "overlay.png").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()


def test_phase2_batch_runs_recorded_in_history() -> None:
    mgr = DesktopWorkflowManager(max_history=10)
    conv_name = next(name for name in mgr.model_options() if resolve_gui_model_id(name) == "hydride_conventional")

    p1 = _tmp_image(_synthetic_image_a())
    p2 = _tmp_image(_synthetic_image_b())
    try:
        records = mgr.run_batch(
            [p1, p2],
            model_name=conv_name,
            params={"image_path": p1},
            include_analysis=False,
        )
    finally:
        Path(p1).unlink(missing_ok=True)
        Path(p2).unlink(missing_ok=True)

    assert len(records) == 2
    assert len(mgr.history()) == 2
    assert mgr.latest() is not None
