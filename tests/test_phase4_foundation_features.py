"""Phase 4 foundation tests: class maps, config overrides, and session persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.app import ProjectSaveRequest, ProjectStateStore
from src.microseg.app.desktop_workflow import DesktopWorkflowManager
from src.microseg.corrections import (
    DEFAULT_CLASS_MAP,
    SegmentationClass,
    SegmentationClassMap,
    colorize_index_mask,
    normalize_binary_index_mask,
    to_index_mask,
)
from src.microseg.corrections.session import CorrectionSession
from src.microseg.io import ConfigError, merge_dicts, parse_set_overrides, resolve_config


def _synthetic_image() -> np.ndarray:
    arr = np.zeros((70, 70), dtype=np.uint8)
    arr[6:30, 10:24] = 255
    arr[20:62, 35:58] = 255
    return arr


def _write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def _conv_model_name(mgr: DesktopWorkflowManager) -> str:
    return next(name for name in mgr.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase4_class_map_and_colorization() -> None:
    cmap = SegmentationClassMap(
        classes=(
            SegmentationClass(0, "background", (0, 0, 0)),
            SegmentationClass(1, "hydride", (255, 0, 0)),
            SegmentationClass(2, "gb", (0, 255, 0)),
        )
    )
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:8, 5:8] = 2

    rgb = colorize_index_mask(mask, cmap)
    assert tuple(rgb[1, 1].tolist()) == (255, 0, 0)
    assert tuple(rgb[6, 6].tolist()) == (0, 255, 0)


def test_phase4_session_feature_delete_and_relabel() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    mask[20:30, 20:30] = 1

    sess = CorrectionSession(mask)
    assert sess.delete_feature(6, 6) is True
    assert int(np.count_nonzero(sess.current_mask == 1)) < int(np.count_nonzero(mask == 1))

    assert sess.relabel_feature(24, 24, class_index=2) is True
    assert int(np.count_nonzero(sess.current_mask == 2)) > 0


def test_phase4_config_overrides_and_resolve(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
model_name: Hydride Conventional
params:
  area_threshold: 10
include_analysis: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_set_overrides(["params.area_threshold=22", "include_analysis=true"])
    merged = merge_dicts({"params": {"area_threshold": 10}}, parsed)
    assert merged["params"]["area_threshold"] == 22
    parsed_map = parse_set_overrides(['mask_colormap={"0":[0,0,0],"1":[255,0,0]}'])
    assert parsed_map["mask_colormap"]["1"] == [255, 0, 0]

    resolved = resolve_config(cfg_path, ["params.crop=true"])
    assert resolved["model_name"] == "Hydride Conventional"
    assert resolved["params"]["crop"] is True


def test_phase4_config_overrides_reject_invalid_json() -> None:
    with pytest.raises(ConfigError, match="invalid JSON override value"):
        parse_set_overrides(["mask_colormap={bad_json}"])


def test_phase4_project_state_roundtrip(tmp_path: Path) -> None:
    mgr = DesktopWorkflowManager()
    model_name = _conv_model_name(mgr)
    img_path = tmp_path / "input.png"
    _write_image(img_path, _synthetic_image())

    record = mgr.run_single(str(img_path), model_name=model_name, params={"image_path": str(img_path)}, include_analysis=False)
    record.feedback_record_dir = str(tmp_path / "feedback_record")
    record.feedback_record_id = "feedback_001"
    sess = CorrectionSession(to_index_mask(np.array(record.mask_image)))
    sess.apply_brush(x=8, y=8, radius=3, mode="add", class_index=1)

    store = ProjectStateStore()
    out = store.save(
        ProjectSaveRequest(
            record=record,
            corrected_mask=sess.current_mask,
            class_map=DEFAULT_CLASS_MAP,
            annotator="qa",
            notes="roundtrip",
            ui_state={"tool": "brush", "class_index": 1},
        ),
        tmp_path / "project",
    )

    loaded = store.load(out)
    assert loaded.annotator == "qa"
    assert loaded.notes == "roundtrip"
    assert np.array_equal(loaded.corrected_mask, sess.current_mask)
    assert loaded.record.feedback_record_dir == str(tmp_path / "feedback_record")
    assert loaded.record.feedback_record_id == "feedback_001"


def test_phase4_binary_mask_normalization_option() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[:, 3:] = 7

    preserved = normalize_binary_index_mask(src, mode="off")
    normalized = normalize_binary_index_mask(src, mode="two_value_zero_background")

    assert set(np.unique(preserved).tolist()) == {0, 7}
    assert set(np.unique(normalized).tolist()) == {0, 1}
