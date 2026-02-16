"""Phase 9 tests for dataset split planning and QA checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.dataops import (
    CorrectionSplitConfig,
    DatasetQualityConfig,
    plan_and_materialize_correction_splits,
    run_dataset_quality_checks,
)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _make_correction_sample(root: Path, sample_id: str, source_name: str, value: int) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((16, 16, 3), value, dtype=np.uint8)
    msk = np.zeros((16, 16), dtype=np.uint8)
    msk[:, 8:] = 1
    _write_png(sample_dir / "input.png", img)
    _write_png(sample_dir / "corrected_mask_indexed.png", msk)

    rec = {
        "schema_version": "microseg.correction.v1",
        "sample_id": sample_id,
        "source_image_path": f"/tmp/{source_name}.png",
        "model_id": "hydride_ml",
        "model_name": "Hydride ML",
        "run_id": "r1",
        "created_utc": "2026-02-16T00:00:00Z",
        "annotator": "tester",
        "notes": "",
        "files": {
            "input": "input.png",
            "corrected_mask_indexed": "corrected_mask_indexed.png",
        },
        "metrics": {},
    }
    (sample_dir / "correction_record.json").write_text(json.dumps(rec), encoding="utf-8")


def test_phase9_leakage_aware_split_keeps_source_group_together(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_correction_sample(raw, "a1", "source_a", 10)
    _make_correction_sample(raw, "a2", "source_a", 20)
    _make_correction_sample(raw, "b1", "source_b", 30)
    _make_correction_sample(raw, "b2", "source_b", 40)

    out = tmp_path / "dataset"
    result = plan_and_materialize_correction_splits(
        CorrectionSplitConfig(
            input_dir=str(raw),
            output_dir=str(out),
            train_ratio=0.5,
            val_ratio=0.25,
            seed=7,
            leakage_group="source_stem",
        )
    )
    assert Path(result.manifest_path).exists()

    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    sample_to_split = manifest["sample_to_split"]
    assert sample_to_split["a1"] == sample_to_split["a2"]
    assert sample_to_split["b1"] == sample_to_split["b2"]


def test_phase9_dataset_qa_detects_mismatch_and_duplicates(tmp_path: Path) -> None:
    ds = tmp_path / "dataset"

    # train pair
    img_train = np.zeros((16, 16, 3), dtype=np.uint8)
    msk_train = np.zeros((16, 16), dtype=np.uint8)
    _write_png(ds / "train" / "images" / "x.png", img_train)
    _write_png(ds / "train" / "masks" / "x.png", msk_train)

    # val pair with shape mismatch and duplicate image content
    img_val = img_train.copy()
    msk_val = np.zeros((15, 16), dtype=np.uint8)
    _write_png(ds / "val" / "images" / "x.png", img_val)
    _write_png(ds / "val" / "masks" / "x.png", msk_val)

    report = run_dataset_quality_checks(
        DatasetQualityConfig(
            dataset_dir=str(ds),
            output_path=str(tmp_path / "qa.json"),
            strict=False,
        )
    )
    assert report.ok is False
    assert report.dimension_mismatches
    assert report.duplicate_files
    assert Path(tmp_path / "qa.json").exists()
