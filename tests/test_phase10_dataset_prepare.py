"""Phase 10 tests for training dataset auto-prepare behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.dataops.training_dataset import (
    DatasetPrepareConfig,
    prepare_training_dataset_layout,
    preview_training_dataset_layout,
)


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_phase10_prepare_from_source_masks_adds_id_and_split(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    for i in range(10):
        img = np.full((16, 16, 3), i * 10, dtype=np.uint8)
        m = np.zeros((16, 16), dtype=np.uint8)
        m[:, 8:] = 1 if i % 2 else 2
        _write(root / "source" / f"img_{i}.png", img)
        _write(root / "masks" / f"img_{i}.png", m)

    out = tmp_path / "prepared"
    res = prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=7,
            id_width=6,
        )
    )

    assert res.prepared is True
    assert res.split_counts["train"] == 8
    assert res.split_counts["val"] == 1
    assert res.split_counts["test"] == 1
    manifest = out / "dataset_prepare_manifest.json"
    assert manifest.exists()

    names = sorted(p.name for p in out.glob("*/*/*.png") if p.parent.name in {"images", "masks"} and p.parent.parent.name in {"train", "val", "test"})
    image_names = sorted(set(p.name for p in out.glob("*/*/*.png") if p.parent.name == "images"))
    assert image_names
    ids = sorted(int(name.rsplit("_", 1)[1].replace(".png", "")) for name in image_names)
    assert ids == list(range(1, 11))
    assert "img_0_000001.png" in image_names
    assert len(names) == 20


def test_phase10_prepare_leakage_aware_keeps_augmented_groups_together(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stems = ["a_aug1", "a_aug2", "b_aug1", "b_aug2", "c", "d"]
    for i, stem in enumerate(stems):
        img = np.full((16, 16, 3), i * 20, dtype=np.uint8)
        m = np.zeros((16, 16), dtype=np.uint8)
        m[:, 8:] = 1
        _write(root / "source" / f"{stem}.png", img)
        _write(root / "masks" / f"{stem}.png", m)

    out = tmp_path / "prepared"
    prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=3,
            split_strategy="leakage_aware",
            leakage_group_mode="suffix_aware",
        )
    )

    import json

    manifest = json.loads((out / "dataset_prepare_manifest.json").read_text(encoding="utf-8"))
    group_to_splits: dict[str, set[str]] = {}
    for rec in manifest["mapping"]:
        group_to_splits.setdefault(str(rec["source_group"]), set()).add(str(rec["split"]))

    assert group_to_splits["a"] in ({"train"}, {"val"}, {"test"})
    assert group_to_splits["b"] in ({"train"}, {"val"}, {"test"})
    assert len(group_to_splits["a"]) == 1
    assert len(group_to_splits["b"]) == 1
    assert manifest["split_strategy"] == "leakage_aware"
    assert manifest["leakage_group_mode"] == "suffix_aware"


def test_phase10_prepare_supports_rgb_mask_colormap_conversion(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    mask_rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    mask_rgb[:, :4] = np.array([0, 0, 0], dtype=np.uint8)
    mask_rgb[:, 4:8] = np.array([255, 0, 0], dtype=np.uint8)
    mask_rgb[:, 8:] = np.array([0, 255, 0], dtype=np.uint8)
    _write(root / "source" / "sample.png", img)
    _write(root / "masks" / "sample.png", mask_rgb)

    out = tmp_path / "prepared"
    prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
            mask_input_type="rgb_colormap",
            mask_colormap={"0": [0, 0, 0], "1": [255, 0, 0], "2": [0, 255, 0]},
        )
    )

    out_mask = next(out.glob("*/masks/sample_*.png"))
    arr = np.asarray(Image.open(out_mask).convert("L"), dtype=np.uint8)
    uniq = set(np.unique(arr).tolist())
    assert uniq == {0, 1, 2}
    assert int(arr[0, 2]) == 0
    assert int(arr[0, 6]) == 1
    assert int(arr[0, 10]) == 2


def test_phase10_prepare_uses_existing_split_layout_when_present(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        m = np.zeros((8, 8), dtype=np.uint8)
        _write(root / split / "images" / f"{split}_a.png", img)
        _write(root / split / "masks" / f"{split}_a.png", m)

    out = tmp_path / "prepared"
    res = prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
        )
    )
    assert res.used_existing_splits is True
    assert res.prepared is False
    assert res.dataset_dir == str(root)


def test_phase10_preview_reports_unsplit_mapping_and_histogram(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[:, 5:] = 1
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m2[:, 2:8] = 2
    _write(root / "source" / "s1_aug1.png", img)
    _write(root / "masks" / "s1_aug1.png", m1)
    _write(root / "source" / "s1_aug2.png", img)
    _write(root / "masks" / "s1_aug2.png", m2)

    preview = preview_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(tmp_path / "prepared"),
            split_strategy="leakage_aware",
            leakage_group_mode="suffix_aware",
        )
    )
    assert preview.used_existing_splits is False
    assert preview.total_pairs == 2
    assert preview.leakage_groups == 1
    assert sum(preview.split_counts.values()) == 2
    assert preview.class_histogram["0"] > 0
    assert preview.class_histogram["1"] > 0
    assert preview.class_histogram["2"] > 0
    assert all(rec["source_group"] == "s1" for rec in preview.mapping)
