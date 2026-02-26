"""Tests for the segmentation data preparation subsystem."""

from __future__ import annotations

import json
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

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_counts"]["train"] + manifest["split_counts"]["val"] + manifest["split_counts"]["test"] == 5
    assert len(manifest["records"]) == 5
    assert "resolved_config" in manifest


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
