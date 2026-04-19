"""Phase 10 tests for training dataset auto-prepare behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.dataops.training_dataset import (
    DatasetPrepareConfig,
    generate_dataset_split_manifest_from_splits,
    prepare_training_dataset_layout,
    preview_training_dataset_layout,
)
from src.microseg.data_preparation.augmentation import (
    AugmentationConfig,
    AugmentationOperationConfig,
    parse_augmentation_config,
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


def test_phase10_prepare_binary_mask_normalization_two_value_zero_background(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 5:] = 255
    _write(root / "source" / "sample.png", img)
    _write(root / "masks" / "sample.png", mask)

    out = tmp_path / "prepared"
    prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
            binary_mask_normalization="two_value_zero_background",
        )
    )

    out_mask = next(out.glob("*/masks/sample_*.png"))
    arr = np.asarray(Image.open(out_mask).convert("L"), dtype=np.uint8)
    assert set(np.unique(arr).tolist()) == {0, 1}


def test_phase10_generate_dataset_manifest_from_existing_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    _write(dataset / "train" / "images" / "a.png", img)
    _write(dataset / "train" / "masks" / "a.png", mask)
    _write(dataset / "val" / "images" / "a.png", img)
    _write(dataset / "val" / "masks" / "a.png", mask)
    _write(dataset / "test" / "images" / "b.png", img)
    _write(dataset / "test" / "masks" / "b.png", mask)

    manifest_path = generate_dataset_split_manifest_from_splits(dataset)
    assert manifest_path.exists()

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "microseg.dataset_split_manifest.v1"
    assert manifest["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert manifest["sample_to_split"]["train/a"] == "train"
    assert manifest["sample_to_split"]["val/a"] == "val"
    assert manifest["sample_to_split"]["b"] == "test"


def test_phase10_augmentation_config_parsing() -> None:
    cfg = parse_augmentation_config(
        {
            "enabled": True,
            "seed": 7,
            "stage": "post_resize",
            "apply_splits": ["train"],
            "variants_per_sample": 2,
            "operations": [
                {"name": "shadow", "probability": 0.75, "parameters": {"count_range": [1, 2]}},
                {"name": "blur", "probability": 0.5, "parameters": {"kernel_size_range": [3, 5]}},
            ],
            "debug": {"enabled": True, "max_samples": 3},
        },
        default_seed=42,
    )

    assert cfg.enabled is True
    assert cfg.seed == 7
    assert cfg.stage == "post_resize"
    assert cfg.apply_splits == ("train",)
    assert cfg.variants_per_sample == 2
    assert [op.name for op in cfg.operations] == ["shadow", "blur"]
    assert cfg.debug.enabled is True
    assert cfg.debug.max_samples == 3


def test_phase10_prepare_augmentation_is_seeded_and_leakage_safe(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    for stem in ["a_aug1", "a_aug2", "b", "c"]:
        img = np.full((16, 16, 3), len(stem) * 10, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:, 4:12] = 1
        _write(root / "source" / f"{stem}.png", img)
        _write(root / "masks" / f"{stem}.png", mask)

    augmentation = AugmentationConfig(
        enabled=True,
        seed=11,
        apply_splits=("train",),
        variants_per_sample=2,
        operations=(
            AugmentationOperationConfig(
                name="shadow",
                probability=1.0,
                parameters={"count_range": [1, 1], "intensity_range": [30, 30]},
            ),
            AugmentationOperationConfig(
                name="blur",
                probability=1.0,
                parameters={"count_range": [1, 1], "kernel_size_range": [3, 3]},
            ),
        ),
    )

    out_a = tmp_path / "prepared_a"
    out_b = tmp_path / "prepared_b"
    cfg = DatasetPrepareConfig(
        dataset_dir=str(root),
        output_dir=str(out_a),
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=5,
        split_strategy="leakage_aware",
        leakage_group_mode="suffix_aware",
        augmentation=augmentation,
    )
    prepare_training_dataset_layout(cfg)
    prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out_b),
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=5,
            split_strategy="leakage_aware",
            leakage_group_mode="suffix_aware",
            augmentation=augmentation,
        )
    )

    train_aug_a = sorted((out_a / "train" / "images").glob("*_aug*.png"))
    train_aug_b = sorted((out_b / "train" / "images").glob("*_aug*.png"))
    assert train_aug_a
    assert [p.name for p in train_aug_a] == [p.name for p in train_aug_b]

    first_a = np.asarray(Image.open(train_aug_a[0]).convert("RGB"), dtype=np.uint8)
    first_b = np.asarray(Image.open(train_aug_b[0]).convert("RGB"), dtype=np.uint8)
    assert np.array_equal(first_a, first_b)

    import json

    manifest = json.loads((out_a / "dataset_prepare_manifest.json").read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = {}
    for row in manifest["mapping"]:
        groups.setdefault(str(row["source_group"]), set()).add(str(row["split"]))
    assert len(groups["a"]) == 1
    assert manifest["augmentation"]["generated_samples"] > 0


def test_phase10_prepare_existing_split_layout_can_materialize_augmented_copy(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        img = np.full((8, 8, 3), 120, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, 3:5] = 1
        _write(root / split / "images" / f"{split}_a.png", img)
        _write(root / split / "masks" / f"{split}_a.png", mask)

    out = tmp_path / "prepared"
    res = prepare_training_dataset_layout(
        DatasetPrepareConfig(
            dataset_dir=str(root),
            output_dir=str(out),
            augmentation=AugmentationConfig(
                enabled=True,
                seed=19,
                apply_splits=("train",),
                variants_per_sample=1,
                operations=(
                    AugmentationOperationConfig(
                        name="shadow",
                        probability=1.0,
                        parameters={"count_range": [1, 1], "intensity_range": [20, 20]},
                    ),
                ),
            ),
        )
    )

    assert res.prepared is True
    assert any((out / "train" / "images").glob("*_aug001.png"))
    assert not any((out / "val" / "images").glob("*_aug001.png"))
    assert (out / "dataset_prepare_manifest.json").exists()
