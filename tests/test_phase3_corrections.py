"""Phase 3 tests for correction sessions and correction dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.app.desktop_workflow import DesktopWorkflowManager
from src.microseg.corrections import CorrectionDatasetPackager, CorrectionExporter, CorrectionSession
from hydride_segmentation.microseg_adapter import resolve_gui_model_id


def _synthetic_image() -> np.ndarray:
    arr = np.zeros((90, 90), dtype=np.uint8)
    arr[8:70, 10:24] = 255
    arr[30:86, 52:80] = 255
    return arr


def _write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def _conventional_model_name(workflow: DesktopWorkflowManager) -> str:
    return next(name for name in workflow.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase3_correction_session_brush_and_undo_redo() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    session = CorrectionSession(mask)

    session.apply_brush(x=25, y=25, radius=5, mode="add")
    after_add = int(np.count_nonzero(session.current_mask))
    assert after_add > 0

    session.apply_brush(x=25, y=25, radius=3, mode="erase")
    after_erase = int(np.count_nonzero(session.current_mask))
    assert after_erase < after_add

    assert session.undo() is True
    assert int(np.count_nonzero(session.current_mask)) == after_add

    assert session.redo() is True
    assert int(np.count_nonzero(session.current_mask)) == after_erase

    report = session.report()
    assert report.actions_applied == 2


def test_phase3_correction_session_polygon_and_reset() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:30, 20:30] = 255
    session = CorrectionSession(mask)

    session.apply_polygon(points=[(5, 5), (25, 5), (15, 20)], mode="add")
    after_poly = int(np.count_nonzero(session.current_mask))
    assert after_poly > int(np.count_nonzero(mask))

    session.reset_to_initial()
    assert np.array_equal(session.current_mask, session.initial_mask)


def test_phase3_export_sample_writes_schema_record(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)

    img_path = tmp_path / "input.png"
    _write_image(img_path, _synthetic_image())

    record = workflow.run_single(str(img_path), model_name=model_name, params={"image_path": str(img_path)}, include_analysis=True)
    session = CorrectionSession(np.array(record.mask_image))
    session.apply_brush(x=10, y=10, radius=4, mode="add")

    exporter = CorrectionExporter()
    sample_dir = exporter.export_sample(record, session.current_mask, tmp_path / "exports", annotator="tester", notes="phase3")

    assert (sample_dir / "input.png").exists()
    assert (sample_dir / "predicted_mask.png").exists()
    assert (sample_dir / "corrected_mask.png").exists()
    assert (sample_dir / "corrected_overlay.png").exists()
    record_json = sample_dir / "correction_record.json"
    assert record_json.exists()

    payload = json.loads(record_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.correction.v1"
    assert payload["annotator"] == "tester"


def test_phase3_dataset_packager_builds_split_layout(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)
    exporter = CorrectionExporter()

    sample_roots = []
    for idx in range(4):
        arr = _synthetic_image().copy()
        arr[idx:idx + 5, idx:idx + 5] = 255
        p = tmp_path / f"input_{idx}.png"
        _write_image(p, arr)
        run = workflow.run_single(str(p), model_name=model_name, params={"image_path": str(p)}, include_analysis=False)
        sess = CorrectionSession(np.array(run.mask_image))
        sess.apply_brush(x=5 + idx, y=5 + idx, radius=3, mode="add")
        sample_roots.append(exporter.export_sample(run, sess.current_mask, tmp_path / "raw_exports"))

    out_dir = tmp_path / "packaged"
    packager = CorrectionDatasetPackager(seed=7)
    packaged = packager.package(sample_roots, out_dir, train_ratio=0.5, val_ratio=0.25)

    assert (packaged / "dataset_manifest.json").exists()
    for split in ["train", "val", "test"]:
        assert (packaged / split / "images").exists()
        assert (packaged / split / "masks").exists()
        assert (packaged / split / "metadata").exists()
