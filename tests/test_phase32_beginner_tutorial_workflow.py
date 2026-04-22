"""Phase 32 tests for the beginner paired-dataset tutorial workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )


def test_phase32_tutorial_dataset_generator_writes_12_pairs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_pairs"
    _run("scripts/generate_tutorial_dataset.py", "--output-dir", str(raw_dir))

    images = sorted(raw_dir.glob("*.png"))
    pair_images = [path for path in images if not path.name.endswith("_mask.png")]
    pair_masks = [path for path in images if path.name.endswith("_mask.png")]
    assert len(pair_images) == 12
    assert len(pair_masks) == 12

    manifest = json.loads((raw_dir / "tutorial_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 12
    assert len(manifest["families"]) == 6


def test_phase32_paired_tutorial_prepare_dataset_is_leakage_aware(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_pairs"
    out_dir = tmp_path / "prepared_dataset"
    _run("scripts/generate_tutorial_dataset.py", "--output-dir", str(raw_dir))
    _run(
        "scripts/microseg_cli.py",
        "prepare_dataset",
        "--config",
        "configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml",
        "--set",
        f"input_dir={raw_dir}",
        "--set",
        f"output_dir={out_dir}",
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((out_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))

    assert manifest["source_split_counts"] == {"train": 8, "val": 2, "test": 2}
    assert manifest["split_counts"] == {"train": 16, "val": 2, "test": 2}
    assert manifest["resolved_config"]["split_strategy"] == "leakage_aware"
    assert manifest["leakage_groups"] == 6
    assert qa["base_split_counts"] == {"train": 8, "val": 2, "test": 2}
    assert qa["split_counts"] == {"train": 16, "val": 2, "test": 2}
    assert qa["augmentation"]["generated_samples"] == 8

    grouped: dict[str, set[str]] = {}
    for row in manifest["records"]:
        grouped.setdefault(str(row["source_group"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in grouped.values())

    augmented_rows = [row for row in manifest["records"] if row.get("augmentation")]
    assert len(augmented_rows) == 8
    train_aug = sorted((out_dir / "mado" / "train" / "images").glob("*_aug001.png"))
    val_aug = sorted((out_dir / "mado" / "val" / "images").glob("*_aug001.png"))
    test_aug = sorted((out_dir / "mado" / "test" / "images").glob("*_aug001.png"))
    assert len(train_aug) == 8
    assert not val_aug
    assert not test_aug
    assert (out_dir / "dataset_qa_report.html").exists()
    assert (out_dir / "debug_augmentation").exists()


def test_phase32_tutorial_training_and_docs_build(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_pairs"
    prepared_dir = tmp_path / "prepared_dataset"
    training_dir = tmp_path / "training_unet"

    _run("scripts/generate_tutorial_dataset.py", "--output-dir", str(raw_dir))
    _run(
        "scripts/microseg_cli.py",
        "prepare_dataset",
        "--config",
        "configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml",
        "--set",
        f"input_dir={raw_dir}",
        "--set",
        f"output_dir={prepared_dir}",
    )
    _run(
        "scripts/microseg_cli.py",
        "train",
        "--config",
        "configs/tutorials/train.tiny_unet_from_prepared.yml",
        "--set",
        f"dataset_dir={prepared_dir / 'mado'}",
        "--set",
        f"output_dir={training_dir}",
    )

    report = json.loads((training_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((training_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert report["config"]["dataset_dir"] == str(prepared_dir / "mado")
    assert manifest["best_checkpoint"]
    assert (training_dir / manifest["best_checkpoint"]).exists()
    assert (training_dir / "training_report.html").exists()
    assert (training_dir / "last_checkpoint.pt").exists()

    docs_out = tmp_path / "docs_html"
    _run("-m", "sphinx", "-b", "html", "docs", str(docs_out))
    assert (docs_out / "cli_windows_linux.html").exists()
    assert (docs_out / "tutorials" / "05_paired_dataset_preparation_and_training_cli.html").exists()
