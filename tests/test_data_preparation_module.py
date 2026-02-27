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
    )
    out, stats = MaskBinarizer(cfg).apply(raw)
    assert set(np.unique(out).tolist()) == {0, 1}
    assert stats["mode"] == "rgb_threshold"
    assert int(out[0, 0]) == 1
    assert int(out[0, 2]) == 0
    assert int(out[1, 0]) == 0


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
