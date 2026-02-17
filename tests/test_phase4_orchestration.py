"""Phase 4 orchestration tests for training/evaluation command stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.app import OrchestrationCommandBuilder
from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator
from src.microseg.training import PixelClassifierTrainer, PixelTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _build_dataset(root: Path) -> Path:
    ds = root / "dataset"

    # Train split: bright -> class 1, dark -> class 0
    for i in range(3):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        msk = np.zeros((32, 32), dtype=np.uint8)
        img[:, :16] = 20
        img[:, 16:] = 230
        msk[:, 16:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"train_{i}.png", img, msk)

    # Val split with same rule.
    for i in range(2):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        msk = np.zeros((32, 32), dtype=np.uint8)
        img[:16, :] = 220
        msk[:16, :] = 1
        _write_pair(ds / "val" / "images", ds / "val" / "masks", f"val_{i}.png", img, msk)

    return ds


def test_phase4_command_builder_constructs_expected_commands() -> None:
    builder = OrchestrationCommandBuilder.discover(start=Path(__file__))

    infer_cmd = builder.infer(config="configs/inference.default.yml", overrides=["a=1"], image="x.png")
    train_cmd = builder.train(config="configs/train.default.yml", dataset_dir="d", output_dir="o")
    eval_cmd = builder.evaluate(config="configs/evaluate.default.yml", model_path="m.joblib", dataset_dir="d")
    prep_cmd = builder.dataset_prepare(config="configs/dataset_prepare.default.yml", dataset_dir="d", output_dir="o")
    qa_cmd = builder.dataset_qa(config="configs/dataset_qa.default.yml", dataset_dir="d", strict=True)
    hpc_cmd = builder.hpc_ga_generate(config="configs/hpc_ga.default.yml", dataset_dir="d", output_dir="o")

    assert infer_cmd[0].endswith("python") or "python" in infer_cmd[0]
    assert infer_cmd[2] == "infer"
    assert "--set" in infer_cmd
    assert train_cmd[2] == "train"
    assert eval_cmd[2] == "evaluate"
    assert prep_cmd[2] == "dataset-prepare"
    assert qa_cmd[2] == "dataset-qa"
    assert hpc_cmd[2] == "hpc-ga-generate"
    assert qa_cmd[-1] == "--strict"


def test_phase4_train_and_evaluate_pixel_model(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    training_out = tmp_path / "training"

    trainer = PixelClassifierTrainer()
    trained = trainer.train(
        PixelTrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(training_out),
            train_split="train",
            max_samples=2000,
            max_iter=300,
            seed=7,
        )
    )

    model_path = Path(trained["model_path"])
    assert model_path.exists()
    assert (training_out / "training_manifest.json").exists()

    report_path = tmp_path / "eval" / "report.json"
    evaluator = PixelModelEvaluator()
    payload = evaluator.evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(dataset_dir),
            model_path=str(model_path),
            split="val",
            output_path=str(report_path),
        )
    )

    assert report_path.exists()
    assert payload["metrics"]["pixel_accuracy"] > 0.7
    assert "macro_f1" in payload["metrics"]

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert str(raw["schema_version"]).startswith("microseg.pixel_eval.v")
