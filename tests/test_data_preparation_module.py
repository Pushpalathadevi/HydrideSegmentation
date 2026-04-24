"""Tests for the segmentation data preparation subsystem."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from src.microseg.data_preparation.binarization import MaskBinarizer
from src.microseg.data_preparation.cli import main as prep_main
from src.microseg.data_preparation.augmentation import AugmentationRunner, parse_augmentation_config
from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pairing import PairCollector
from src.microseg.data_preparation.pipeline import DatasetPreparer
from src.microseg.data_preparation.resizing import Resizer


def _build_paired_dataset(tmp_path: Path, n: int = 6) -> Path:
    input_dir = tmp_path / "pairs"
    input_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(Image.open(Path("test_data") / "syntheticHydrides.png").convert("L"), dtype=np.uint8)
    for i in range(n):
        img = np.stack([base, np.roll(base, i + 1, axis=1), np.roll(base, i + 2, axis=0)], axis=-1)
        mask = np.zeros_like(base, dtype=np.uint8)
        mask[base >= (100 + i)] = 255
        Image.fromarray(img).save(input_dir / f"sample_{i}.png")
        Image.fromarray(mask).save(input_dir / f"sample_{i}_mask.png")
    return input_dir


def _build_same_stem_cross_extension_dataset(tmp_path: Path, n: int = 4) -> Path:
    input_dir = tmp_path / "pairs_same_stem"
    input_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(Image.open(Path("test_data") / "syntheticHydrides.png").convert("L"), dtype=np.uint8)
    for i in range(n):
        img = np.stack([base, np.roll(base, i + 1, axis=1), np.roll(base, i + 2, axis=0)], axis=-1)
        mask = np.zeros_like(base, dtype=np.uint8)
        mask[base >= (100 + i)] = 255
        Image.fromarray(img).save(input_dir / f"sample_{i}.jpg")
        Image.fromarray(mask).save(input_dir / f"sample_{i}.png")
    return input_dir


def _build_nonbinary_edge_dataset(tmp_path: Path) -> Path:
    input_dir = tmp_path / "pairs_nonbinary"
    input_dir.mkdir(parents=True, exist_ok=True)

    base = np.asarray(Image.open(Path("data") / "sample_images" / "hydride_synthetic_sample.png").convert("L"), dtype=np.uint8)
    image = np.stack([base, np.roll(base, 1, axis=1), np.roll(base, 2, axis=0)], axis=-1)
    mask = np.zeros_like(base, dtype=np.uint8)
    mask[base >= 120] = 255

    # Simulate compression/anti-alias artifacts by introducing gray edge values.
    edge_map = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    noisy_mask = mask.copy()
    noisy_mask[edge_map] = 96

    Image.fromarray(image).save(input_dir / "sample_nonbinary.png")
    Image.fromarray(noisy_mask).save(input_dir / "sample_nonbinary_mask.png")
    return input_dir


def test_pair_collector_default_patterns(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=4)
    pairs = PairCollector(
        image_extensions=[".png", ".jpg"],
        mask_extensions=[".png"],
        mask_name_patterns=["{stem}.png", "{stem}_mask.png"],
        strict=True,
    ).collect(input_dir)
    assert len(pairs) == 4
    assert pairs[0].stem == "sample_0"
    assert pairs[0].mask_path.name == "sample_0_mask.png"


def test_pair_collector_same_stem_cross_extension_requires_opt_in(tmp_path: Path) -> None:
    input_dir = _build_same_stem_cross_extension_dataset(tmp_path, n=3)
    collector = PairCollector(
        image_extensions=[".png", ".jpg"],
        mask_extensions=[".png"],
        mask_name_patterns=["{stem}.png", "{stem}_mask.png"],
        strict=True,
    )
    try:
        collector.collect(input_dir)
    except ValueError as exc:
        assert "pairing mismatch detected" in str(exc)
    else:
        raise AssertionError("expected strict pairing mismatch for same-stem cross-extension layout without opt-in")


def test_pair_collector_same_stem_cross_extension_opt_in_pairs_successfully(tmp_path: Path) -> None:
    input_dir = _build_same_stem_cross_extension_dataset(tmp_path, n=3)
    pairs = PairCollector(
        image_extensions=[".png", ".jpg"],
        mask_extensions=[".png"],
        mask_name_patterns=["{stem}.png", "{stem}_mask.png"],
        same_stem_pairing_enabled=True,
        same_stem_image_extensions=[".jpg", ".jpeg"],
        same_stem_mask_extensions=[".png"],
        strict=True,
    ).collect(input_dir)
    assert len(pairs) == 3
    assert pairs[0].image_path.suffix.lower() == ".jpg"
    assert pairs[0].mask_path.suffix.lower() == ".png"
    assert pairs[0].image_path.stem == pairs[0].mask_path.stem


def test_mask_binarizer_modes() -> None:
    raw = np.array([[0, 10, 128, 255], [1, 120, 129, 255]], dtype=np.uint8)
    modes = {
        "nonzero": {0, 1},
        "threshold": {0, 1},
        "value_equals": {0, 1},
        "otsu": {0, 1},
        "percentile": {0, 1},
    }
    for mode, expected in modes.items():
        cfg = DatasetPrepConfig(
            input_dir="in",
            output_dir="out",
            binarization_mode=mode,  # type: ignore[arg-type]
            threshold=128,
            foreground_values=[255, 128],
            percentile=70,
        )
        out, stats = MaskBinarizer(cfg).apply(raw)
        assert set(np.unique(out).tolist()) == expected
        assert set(stats["unique_binary_values"]) == expected


def test_resizer_policies_shape_and_discrete_mask() -> None:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[:, 25:40] = 1
    for policy in ["letterbox_pad", "center_crop", "stretch", "keep_aspect_no_pad"]:
        cfg = DatasetPrepConfig(input_dir="in", output_dir="out", resize_policy=policy, target_size=(32, 32))  # type: ignore[arg-type]
        out_img, out_mask, _ = Resizer(cfg).apply(img, mask)
        if policy == "keep_aspect_no_pad":
            assert out_img.shape[0] <= 32 and out_img.shape[1] <= 32
        else:
            assert out_img.shape[:2] == (32, 32)
        assert set(np.unique(out_mask).tolist()).issubset({0, 1})


def test_pipeline_exports_manifest_and_debug_outputs(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=8)
    output_dir = tmp_path / "out"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["oxford", "mado"],
        "mask_name_patterns": ["{stem}_mask.png", "{stem}.png"],
        "debug": {"enabled": True, "limit_pairs": 5, "inspection_limit": 3},
        "target_size": (64, 64),
        "mask_ext": ".png",
        "image_ext": ".png",
    })
    result = DatasetPreparer(cfg).run()

    assert result.total_pairs == 5
    assert (output_dir / "oxford" / "images").exists()
    assert (output_dir / "oxford" / "annotations" / "trimaps").exists()
    assert (output_dir / "mado" / "train" / "images").exists()
    assert (output_dir / "debug_inspection").exists()
    assert len(list((output_dir / "debug_inspection").rglob("*_panel.png"))) == 3
    assert len(list((output_dir / "debug_inspection").rglob("*_mask_difference.png"))) == 3
    assert len(list((output_dir / "debug_inspection").rglob("*_criteria.json"))) == 3

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_counts"]["train"] + manifest["split_counts"]["val"] + manifest["split_counts"]["test"] == 5
    assert len(manifest["records"]) == 5
    assert "resolved_config" in manifest


def test_pipeline_warns_and_surfaces_nonbinary_mask_values(tmp_path: Path, caplog) -> None:
    input_dir = _build_nonbinary_edge_dataset(tmp_path)
    output_dir = tmp_path / "out_nonbinary"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["oxford", "mado"],
        "mask_name_patterns": ["{stem}_mask.png", "{stem}.png"],
        "debug": {"enabled": True, "limit_pairs": 2, "inspection_limit": 2},
        "target_size": (64, 64),
        "mask_ext": ".png",
        "image_ext": ".png",
    })

    caplog.set_level(logging.WARNING, logger="microseg.data_preparation")
    DatasetPreparer(cfg).run()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["records"][0]
    warnings = record["warnings"]
    assert warnings
    assert any("non-binary values" in warning for warning in warnings)
    assert any("sample_nonbinary" in warning for warning in manifest["warnings_summary"])
    assert any("non-binary values" in rec.getMessage() for rec in caplog.records)

    difference = list((output_dir / "debug_inspection").rglob("*_mask_difference.png"))
    assert difference
    assert record["mask_stats"]["non_binary_pixel_count"] > 0
    assert 96 in set(record["mask_stats"]["non_binary_values"])


def test_cli_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=3)
    output_dir = tmp_path / "dry"
    code = prep_main([
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--style",
        "oxford,mado",
        "--dry-run",
        "--seed",
        "11",
    ])
    assert code == 0
    assert (output_dir / "manifest.json").exists()
    assert not (output_dir / "oxford" / "images").exists()


def test_mask_binarizer_rgb_threshold_mode() -> None:
    raw = np.zeros((4, 4, 3), dtype=np.uint8)
    raw[:, :, 2] = np.array([
        [255, 220, 199, 100],
        [210, 205, 180, 0],
        [255, 255, 255, 255],
        [200, 201, 202, 203],
    ], dtype=np.uint8)
    raw[:, :, 1] = np.array([
        [0, 40, 30, 0],
        [61, 55, 10, 0],
        [0, 30, 60, 10],
        [0, 10, 20, 100],
    ], dtype=np.uint8)
    cfg = DatasetPrepConfig(
        input_dir="in",
        output_dir="out",
        rgb_mask_mode=True,
        mask_r_min=200,
        mask_g_max=60,
        mask_b_max=60,
        allow_red_dominance_fallback=False,
    )
    out, stats = MaskBinarizer(cfg).apply(raw)
    assert set(np.unique(out).tolist()) == {0, 1}
    assert stats["mode"] == "rgb_threshold"
    assert int(out[0, 0]) == 1
    assert int(out[0, 2]) == 0
    assert int(out[1, 0]) == 0


def test_mask_binarizer_rgb_mode_grayscale_binary_like_passthrough() -> None:
    raw = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    cfg = DatasetPrepConfig(input_dir="in", output_dir="out", rgb_mask_mode=True)
    out, stats = MaskBinarizer(cfg).apply(raw)
    assert set(np.unique(out).tolist()) == {0, 1}
    assert stats["mode"] == "grayscale_binary_passthrough"
    assert int(out[0, 1]) == 1


def test_mask_binarizer_rgb_red_dominance_fallback() -> None:
    raw = np.zeros((2, 4, 3), dtype=np.uint8)
    raw[0, 0] = np.array([0, 0, 25], dtype=np.uint8)
    raw[0, 1] = np.array([2, 3, 24], dtype=np.uint8)
    raw[0, 2] = np.array([15, 18, 25], dtype=np.uint8)
    raw[0, 3] = np.array([0, 0, 10], dtype=np.uint8)
    cfg = DatasetPrepConfig(
        input_dir="in",
        output_dir="out",
        rgb_mask_mode=True,
        mask_r_min=200,
        mask_g_max=60,
        mask_b_max=60,
        allow_red_dominance_fallback=True,
        mask_red_min_fallback=16,
        mask_red_dominance_margin=8,
        mask_red_dominance_ratio=1.5,
    )
    out, stats = MaskBinarizer(cfg).apply(raw)
    assert set(np.unique(out).tolist()).issubset({0, 1})
    assert int(out[0, 0]) == 1
    assert int(out[0, 1]) == 1
    assert int(out[0, 2]) == 0
    assert int(out[0, 3]) == 0
    assert int(stats["red_dominance_fg_pixel_count"]) >= 2


def test_mask_binarizer_auto_otsu_for_noisy_grayscale() -> None:
    raw = np.array([
        [0, 0, 2, 4, 220, 255],
        [0, 1, 3, 5, 230, 255],
        [0, 0, 2, 4, 210, 255],
        [0, 0, 0, 5, 200, 255],
    ], dtype=np.uint8)
    cfg = DatasetPrepConfig(
        input_dir="in",
        output_dir="out",
        auto_otsu_for_noisy_grayscale=True,
        noisy_grayscale_low_max=5,
        noisy_grayscale_high_min=200,
        noisy_grayscale_min_extreme_ratio=0.95,
    )
    out, stats = MaskBinarizer(cfg).apply(raw)
    assert stats["mode"] == "grayscale_auto_otsu"
    assert stats["auto_otsu_applied"] is True
    assert "otsu_threshold" in stats
    assert set(np.unique(out).tolist()).issubset({0, 1})


def test_resize_policy_short_side_to_target_crop_alignment() -> None:
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=np.uint8)
    img[:, 20:24, :] = 255
    mask[:, 20:24] = 1

    cfg = DatasetPrepConfig(
        input_dir="in",
        output_dir="out",
        resize_policy="short_side_to_target_crop",
        target_size=(16, 16),
        crop_mode_train="random",
        crop_mode_eval="center",
        seed=13,
    )  # type: ignore[arg-type]

    out_img, out_mask, _ = Resizer(cfg).apply(img, mask, split="train", sample_seed=777)
    assert out_img.shape[:2] == (16, 16)
    assert out_mask.shape == (16, 16)
    assert set(np.unique(out_mask).tolist()).issubset({0, 1})
    bright = out_img[:, :, 0] > 0
    assert np.array_equal(bright, out_mask > 0)


def test_pipeline_paired_jpg_rgb_png_outputs_mado(tmp_path: Path) -> None:
    input_dir = tmp_path / "pairs_rgb"
    input_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        image = np.full((48, 64, 3), 110 + i, dtype=np.uint8)
        mask_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        mask_rgb[:, 20:30, 0] = 0
        mask_rgb[:, 20:30, 1] = 20
        mask_rgb[:, 20:30, 2] = 230
        Image.fromarray(image).save(input_dir / f"pair_{i}.jpg")
        Image.fromarray(mask_rgb).save(input_dir / f"pair_{i}.png")

    output_dir = tmp_path / "out_rgb"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "image_extensions": [".jpg", ".jpeg"],
        "mask_extensions": [".png"],
        "mask_name_patterns": ["{stem}.png"],
        "rgb_mask_mode": True,
        "resize_policy": "short_side_to_target_crop",
        "target_size": 32,
        "crop_mode_train": "random",
        "crop_mode_eval": "center",
    })
    result = DatasetPreparer(cfg).run()
    assert result.total_pairs == 6

    for split in ["train", "val", "test"]:
        for path in (output_dir / "mado" / split / "masks").glob("*.png"):
            arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            assert arr.shape == (32, 32)
            assert set(np.unique(arr).tolist()).issubset({0, 255})

    qa = json.loads((output_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert qa["pairing"]["pair_count"] == 6
    assert qa["split_counts"]["train"] + qa["split_counts"]["val"] + qa["split_counts"]["test"] == 6


def test_pipeline_same_stem_cross_extension_opt_in_outputs_mado(tmp_path: Path) -> None:
    input_dir = _build_same_stem_cross_extension_dataset(tmp_path, n=6)
    output_dir = tmp_path / "out_same_stem"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "image_extensions": [".jpg", ".jpeg", ".png"],
        "mask_extensions": [".png"],
        "mask_name_patterns": ["{stem}_mask.png", "{stem}.png"],
        "same_stem_pairing": {
            "enabled": True,
            "image_extensions": [".jpg", ".jpeg"],
            "mask_extensions": [".png"],
        },
        "resize_policy": "short_side_to_target_crop",
        "target_size": 32,
        "crop_mode_train": "random",
        "crop_mode_eval": "center",
    })
    result = DatasetPreparer(cfg).run()
    assert result.total_pairs == 6

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_config"]["same_stem_pairing"]["enabled"] is True
    assert manifest["resolved_config"]["same_stem_pairing"]["image_extensions"] == [".jpg", ".jpeg"]
    assert manifest["resolved_config"]["same_stem_pairing"]["mask_extensions"] == [".png"]
    assert len(manifest["records"]) == 6

    qa = json.loads((output_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert qa["pairing"]["pair_count"] == 6
    assert qa["pairing"]["missing_masks"] == []
    assert qa["pairing"]["missing_images"] == []


def test_pipeline_split_caps_route_remainder_to_train(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=12)
    output_dir = tmp_path / "out_split_caps"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "train_pct": 0.8,
        "val_pct": 0.25,
        "max_val_examples": 1,
        "max_test_examples": 1,
        "target_size": 32,
    })
    result = DatasetPreparer(cfg).run()
    assert result.total_pairs == 12
    assert result.split_counts == {"train": 10, "val": 1, "test": 1}

    qa = json.loads((output_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert qa["split_counts"] == {"train": 10, "val": 1, "test": 1}
    assert qa["split_policy"]["max_val_examples"] == 1
    assert qa["split_policy"]["max_test_examples"] == 1


def test_pipeline_maps_grayscale_binary01_mask_to_255(tmp_path: Path) -> None:
    input_dir = tmp_path / "pairs_gray01"
    input_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, 10:30] = 1
    Image.fromarray(image).save(input_dir / "sample.jpg")
    Image.fromarray(mask).save(input_dir / "sample_mask.png")

    output_dir = tmp_path / "out_gray01"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "image_extensions": [".jpg", ".jpeg"],
        "mask_extensions": [".png"],
        "mask_name_patterns": ["{stem}_mask.png"],
        "rgb_mask_mode": True,
        "resize_policy": "short_side_to_target_crop",
        "target_size": 32,
        "crop_mode_train": "center",
        "crop_mode_eval": "center",
    })
    DatasetPreparer(cfg).run()
    out_mask_path = next((output_dir / "mado" / "train" / "masks").glob("*.png"))
    out_mask = np.asarray(Image.open(out_mask_path).convert("L"), dtype=np.uint8)
    assert out_mask.shape == (32, 32)
    assert set(np.unique(out_mask).tolist()) == {0, 255}


def test_pipeline_warns_when_output_mask_all_zeros(tmp_path: Path, caplog) -> None:
    input_dir = tmp_path / "pairs_empty_warn"
    input_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "sample.jpg")
    Image.fromarray(mask).save(input_dir / "sample_mask.png")

    output_dir = tmp_path / "out_empty_warn"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "image_extensions": [".jpg", ".jpeg"],
        "mask_extensions": [".png"],
        "mask_name_patterns": ["{stem}_mask.png"],
        "rgb_mask_mode": True,
        "empty_mask_action": "warn",
        "target_size": 32,
        "resize_policy": "short_side_to_target_crop",
        "crop_mode_train": "center",
        "crop_mode_eval": "center",
    })
    caplog.set_level(logging.WARNING, logger="microseg.data_preparation")
    DatasetPreparer(cfg).run()

    assert any("all zeros after preprocessing" in rec.getMessage() for rec in caplog.records)
    qa = json.loads((output_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert qa["empty_output_masks"]["count"] == 1
    assert qa["empty_output_masks"]["stems"] == ["sample"]


def test_pipeline_errors_when_output_mask_all_zeros(tmp_path: Path) -> None:
    input_dir = tmp_path / "pairs_empty_error"
    input_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "sample.jpg")
    Image.fromarray(mask).save(input_dir / "sample_mask.png")

    output_dir = tmp_path / "out_empty_error"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "image_extensions": [".jpg", ".jpeg"],
        "mask_extensions": [".png"],
        "mask_name_patterns": ["{stem}_mask.png"],
        "rgb_mask_mode": True,
        "empty_mask_action": "error",
        "target_size": 32,
        "resize_policy": "short_side_to_target_crop",
        "crop_mode_train": "center",
        "crop_mode_eval": "center",
    })
    try:
        DatasetPreparer(cfg).run()
    except ValueError as exc:
        assert "output mask is all zeros after preprocessing" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty output mask")


def test_pipeline_augmentation_generates_debug_and_metadata(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=4)
    output_dir = tmp_path / "out_aug"
    cfg = DatasetPrepConfig.from_dict({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "styles": ["mado"],
        "target_size": 32,
        "augmentation": {
            "enabled": True,
            "seed": 17,
            "stage": "post_resize",
            "apply_splits": ["train"],
            "variants_per_sample": 2,
            "operations": [
                {
                    "name": "shadow",
                    "probability": 1.0,
                    "parameters": {"count_range": [1, 1], "intensity_range": [25, 25]},
                },
                {
                    "name": "blur",
                    "probability": 1.0,
                    "parameters": {"count_range": [1, 1], "kernel_size_range": [3, 3]},
                },
            ],
            "debug": {"enabled": True, "max_samples": 2},
        },
    })

    result = DatasetPreparer(cfg).run()
    assert result.total_pairs > 4
    assert len(list((output_dir / "debug_augmentation").rglob("*_panel.png"))) == 2
    assert len(list((output_dir / "debug_augmentation").rglob("*_metadata.json"))) == 2

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    augmented = [row for row in manifest["records"] if "augmentation" in row]
    assert augmented
    assert manifest["resolved_config"]["augmentation"]["enabled"] is True
    qa = json.loads((output_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert qa["augmentation"]["generated_samples"] >= 1


def test_pipeline_augmentation_is_seeded(tmp_path: Path) -> None:
    input_dir = _build_paired_dataset(tmp_path, n=3)
    cfg_base = {
        "input_dir": str(input_dir),
        "styles": ["mado"],
        "target_size": 32,
        "augmentation": {
            "enabled": True,
            "seed": 29,
            "stage": "pre_resize",
            "apply_splits": ["train"],
            "variants_per_sample": 1,
            "operations": [
                {
                    "name": "shadow",
                    "probability": 1.0,
                    "parameters": {"count_range": [1, 1], "intensity_range": [30, 30]},
                },
            ],
        },
    }

    out_a = tmp_path / "seeded_a"
    out_b = tmp_path / "seeded_b"
    DatasetPreparer(DatasetPrepConfig.from_dict({**cfg_base, "output_dir": str(out_a)})).run()
    DatasetPreparer(DatasetPrepConfig.from_dict({**cfg_base, "output_dir": str(out_b)})).run()

    aug_a = sorted((out_a / "mado" / "train" / "images").glob("*_aug*.png"))
    aug_b = sorted((out_b / "mado" / "train" / "images").glob("*_aug*.png"))
    assert aug_a and aug_b
    assert [p.name for p in aug_a] == [p.name for p in aug_b]
    assert np.array_equal(
        np.asarray(Image.open(aug_a[0]).convert("RGB"), dtype=np.uint8),
        np.asarray(Image.open(aug_b[0]).convert("RGB"), dtype=np.uint8),
    )


def test_augmentation_scalar_and_range_parameters_are_sampled_and_recorded() -> None:
    image = np.full((32, 32, 3), 160, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    cfg = parse_augmentation_config(
        {
            "enabled": True,
            "seed": 123,
            "stage": "post_resize",
            "apply_splits": ["train"],
            "variants_per_sample": 5,
            "operations": [
                {
                    "name": "shadow",
                    "probability": 1.0,
                    "parameters": {
                        "count_range": [1, 1],
                        "intensity_range": [10, 10],
                        "radius": [100, 300],
                        "sigma": 120,
                    },
                },
                {
                    "name": "blur",
                    "probability": 1.0,
                    "parameters": {
                        "count_range": [1, 1],
                        "kernel_size": [3, 9],
                        "sigma": [10, 20],
                        "min_center_distance_ratio": [0.2, 0.6],
                    },
                },
            ],
        },
        default_seed=1,
    )

    variants = AugmentationRunner(cfg).generate_variants(
        image=image,
        mask=mask,
        split="train",
        source_name="sample.png",
        resolved_stage="post_resize",
    )

    assert len(variants) == 5
    for variant in variants:
        ops = {op.name: op.parameters for op in variant.metadata.applied_operations}
        shadow = ops["shadow"]
        blur = ops["blur"]
        assert 100 <= float(shadow["radius"]) <= 300
        assert float(shadow["sigma"]) == 120.0
        kernel = int(blur["blurs"][0]["kernel_size"])
        assert kernel in {3, 5, 7, 9}
        assert 10 <= float(blur["sigma"]) <= 20
        assert 0.2 <= float(blur["min_center_distance_ratio"]) <= 0.6


def test_augmentation_invalid_kernel_range_fails_clearly() -> None:
    image = np.full((16, 16, 3), 160, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    cfg = parse_augmentation_config(
        {
            "enabled": True,
            "seed": 7,
            "operations": [
                {
                    "name": "blur",
                    "probability": 1.0,
                    "parameters": {"count_range": [1, 1], "kernel_size": [2, 2]},
                }
            ],
        },
        default_seed=1,
    )

    try:
        AugmentationRunner(cfg).generate_variants(
            image=image,
            mask=mask,
            split="train",
            source_name="bad.png",
            resolved_stage="post_resize",
        )
    except ValueError as exc:
        assert "positive odd integer" in str(exc)
    else:
        raise AssertionError("expected invalid even-only kernel range to fail")
