"""Unified CLI for inference, training, evaluation, and dataset packaging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.app.desktop_workflow import DesktopWorkflowManager
from src.microseg.corrections import CorrectionDatasetPackager
from src.microseg.io import resolve_config
from src.microseg.training import PixelClassifierTrainer, PixelTrainingConfig
from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator


def _infer(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    image_path = args.image or cfg.get("image_path")
    if not image_path:
        raise ValueError("image path is required (--image or config:image_path)")

    model_name = args.model_name or cfg.get("model_name")
    if not model_name:
        raise ValueError("model name is required (--model-name or config:model_name)")

    out_dir = Path(args.output_dir or cfg.get("output_dir") or "outputs/inference")
    include_analysis = bool(cfg.get("include_analysis", True))
    params = dict(cfg.get("params", {}))

    mgr = DesktopWorkflowManager()
    record = mgr.run_single(
        str(image_path),
        model_name=model_name,
        params=params,
        include_analysis=include_analysis,
    )
    run_dir = mgr.export_run(record, out_dir)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"inference export: {run_dir}")
    return 0


def _train(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset directory is required (--dataset-dir or config:dataset_dir)")
    output_dir = args.output_dir or cfg.get("output_dir") or "outputs/training"

    trainer = PixelClassifierTrainer()
    result = trainer.train(
        PixelTrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            train_split=str(cfg.get("train_split", "train")),
            max_samples=int(cfg.get("max_samples", args.max_samples)),
            max_iter=int(cfg.get("max_iter", args.max_iter)),
            seed=int(cfg.get("seed", args.seed)),
        )
    )

    out_root = Path(output_dir)
    (out_root / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"training output: {out_root}")
    print(f"model: {result['model_path']}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    model_path = args.model_path or cfg.get("model_path")
    if not dataset_dir:
        raise ValueError("dataset directory is required (--dataset-dir or config:dataset_dir)")
    if not model_path:
        raise ValueError("model path is required (--model-path or config:model_path)")

    output_path = args.output_path or cfg.get("output_path") or "outputs/evaluation/pixel_eval_report.json"
    split = args.split or cfg.get("split") or "val"

    evaluator = PixelModelEvaluator()
    payload = evaluator.evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(dataset_dir),
            model_path=str(model_path),
            split=str(split),
            output_path=str(output_path),
        )
    )
    Path(output_path).with_name("resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"evaluation report: {output_path}")
    print(f"metrics: {payload['metrics']}")
    return 0


def _package(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    input_dir = Path(args.input_dir or cfg.get("input_dir"))
    output_dir = Path(args.output_dir or cfg.get("output_dir") or "outputs/packaged_dataset")
    train_ratio = float(cfg.get("train_ratio", args.train_ratio))
    val_ratio = float(cfg.get("val_ratio", args.val_ratio))
    seed = int(cfg.get("seed", args.seed))

    sample_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not sample_dirs:
        raise ValueError(f"no sample directories found under {input_dir}")

    out = CorrectionDatasetPackager(seed=seed).package(
        sample_dirs,
        output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"dataset package: {out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MicroSeg unified CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    infer = sub.add_parser("infer", help="Run inference and export artifacts")
    infer.add_argument("--config", type=str, help="YAML config path")
    infer.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    infer.add_argument("--image", type=str, help="Input image path")
    infer.add_argument("--model-name", type=str, help="Model display name from registry")
    infer.add_argument("--output-dir", type=str, help="Export output directory")
    infer.set_defaults(handler=_infer)

    train = sub.add_parser("train", help="Train baseline pixel classifier")
    train.add_argument("--config", type=str, help="YAML config path")
    train.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    train.add_argument("--dataset-dir", type=str, help="Dataset directory containing train/images + train/masks")
    train.add_argument("--output-dir", type=str, help="Training output directory")
    train.add_argument("--max-samples", type=int, default=250000)
    train.add_argument("--max-iter", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=_train)

    ev = sub.add_parser("evaluate", help="Evaluate trained baseline model on dataset split")
    ev.add_argument("--config", type=str, help="YAML config path")
    ev.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    ev.add_argument("--dataset-dir", type=str, help="Dataset directory containing split/images + split/masks")
    ev.add_argument("--model-path", type=str, help="Path to trained model artifact")
    ev.add_argument("--split", type=str, default="val", help="Split name to evaluate")
    ev.add_argument("--output-path", type=str, help="Report output path")
    ev.set_defaults(handler=_evaluate)

    pack = sub.add_parser("package", help="Package correction exports into dataset splits")
    pack.add_argument("--config", type=str, help="YAML config path")
    pack.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    pack.add_argument("--input-dir", type=str, help="Directory containing corrected sample folders")
    pack.add_argument("--output-dir", type=str, help="Dataset output directory")
    pack.add_argument("--train-ratio", type=float, default=0.8)
    pack.add_argument("--val-ratio", type=float, default=0.1)
    pack.add_argument("--seed", type=int, default=42)
    pack.set_defaults(handler=_package)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rc = args.handler(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
