"""Unified CLI for inference, training, evaluation, registry, dataops, and phase-gates."""

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
from src.microseg.dataops import (
    CorrectionSplitConfig,
    DatasetPrepareConfig,
    DatasetQualityConfig,
    plan_and_materialize_correction_splits,
    prepare_training_dataset_layout,
    run_dataset_quality_checks,
)
from src.microseg.io import resolve_config
from src.microseg.plugins import (
    load_frozen_checkpoint_records,
    validate_frozen_registry,
    write_registry_validation_report,
)
from src.microseg.quality import PhaseGateConfig, run_phase_gate
from src.microseg.training import (
    PixelClassifierTrainer,
    PixelTrainingConfig,
    TorchPixelClassifierTrainer,
    TorchPixelTrainingConfig,
    UNetBinaryTrainer,
    UNetBinaryTrainingConfig,
)
from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator


def _parse_name_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    sep = "|" if "|" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


def _parse_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{field_name} must be a JSON object string or YAML mapping; got: {text[:80]}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must resolve to a mapping/object")
    return {str(k): v for k, v in payload.items()}


def _build_dataset_prepare_config(
    *,
    cfg: dict[str, object],
    args: argparse.Namespace,
    dataset_dir: str,
    output_dir: str,
    fallback_seed: int,
) -> DatasetPrepareConfig:
    return DatasetPrepareConfig(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        train_ratio=float(cfg.get("split_train_ratio", args.split_train_ratio)),
        val_ratio=float(cfg.get("split_val_ratio", args.split_val_ratio)),
        test_ratio=float(cfg.get("split_test_ratio", args.split_test_ratio)),
        seed=int(cfg.get("split_seed", fallback_seed)),
        id_width=int(cfg.get("split_id_width", args.split_id_width)),
        split_strategy=str(cfg.get("split_strategy", args.split_strategy)),
        leakage_group_mode=str(cfg.get("leakage_group_mode", args.leakage_group_mode)),
        leakage_group_regex=str(cfg.get("leakage_group_regex", args.leakage_group_regex or "")),
        mask_input_type=str(cfg.get("mask_input_type", args.mask_input_type)),
        mask_colormap=_parse_mapping(
            cfg.get("mask_colormap", args.mask_colormap_json),
            field_name="mask_colormap",
        ),
        mask_colormap_strict=bool(cfg.get("mask_colormap_strict", args.mask_colormap_strict)),
    )


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
    params["enable_gpu"] = bool(cfg.get("enable_gpu", args.enable_gpu))
    params["device_policy"] = str(cfg.get("device_policy", args.device_policy))

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
    backend = str(cfg.get("backend", args.backend))
    enable_gpu = bool(cfg.get("enable_gpu", args.enable_gpu))
    device_policy = str(cfg.get("device_policy", args.device_policy))
    seed = int(cfg.get("seed", args.seed))
    val_tracking_fixed = tuple(_parse_name_list(cfg.get("val_tracking_fixed_samples", args.val_tracking_fixed_samples)))
    auto_prepare = bool(cfg.get("auto_prepare_dataset", args.auto_prepare_dataset))
    if auto_prepare:
        prep_out = str(
            cfg.get(
                "prepare_output_dir",
                args.prepare_output_dir or str(Path(output_dir) / "prepared_dataset"),
            )
        )
        prepared = prepare_training_dataset_layout(
            _build_dataset_prepare_config(
                cfg=cfg,
                args=args,
                dataset_dir=str(dataset_dir),
                output_dir=prep_out,
                fallback_seed=seed,
            )
        )
        dataset_dir = prepared.dataset_dir
    else:
        prepared = None

    if backend == "torch_pixel":
        trainer = TorchPixelClassifierTrainer()
        result = trainer.train(
            TorchPixelTrainingConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                train_split=str(cfg.get("train_split", "train")),
                max_samples=int(cfg.get("max_samples", args.max_samples)),
                epochs=int(cfg.get("epochs", args.epochs)),
                batch_size=int(cfg.get("batch_size", args.batch_size)),
                learning_rate=float(cfg.get("learning_rate", args.learning_rate)),
                seed=seed,
                enable_gpu=enable_gpu,
                device_policy=device_policy,
            )
        )
    elif backend == "unet_binary":
        trainer = UNetBinaryTrainer()
        result = trainer.train(
            UNetBinaryTrainingConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                train_split=str(cfg.get("train_split", "train")),
                val_split=str(cfg.get("val_split", "val")),
                epochs=int(cfg.get("epochs", args.epochs)),
                batch_size=int(cfg.get("batch_size", args.batch_size)),
                learning_rate=float(cfg.get("learning_rate", args.learning_rate)),
                weight_decay=float(cfg.get("weight_decay", args.weight_decay)),
                seed=seed,
                enable_gpu=enable_gpu,
                device_policy=device_policy,
                early_stopping_patience=int(cfg.get("early_stopping_patience", args.early_stopping_patience)),
                early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", args.early_stopping_min_delta)),
                checkpoint_every=int(cfg.get("checkpoint_every", args.checkpoint_every)),
                resume_checkpoint=str(cfg.get("resume_checkpoint", args.resume_checkpoint or "")),
                val_tracking_samples=int(cfg.get("val_tracking_samples", args.val_tracking_samples)),
                val_tracking_fixed_samples=val_tracking_fixed,
                val_tracking_seed=int(cfg.get("val_tracking_seed", args.val_tracking_seed)),
                write_html_report=bool(cfg.get("write_html_report", args.write_html_report)),
                progress_log_interval_pct=int(cfg.get("progress_log_interval_pct", args.progress_log_interval_pct)),
            )
        )
    else:
        trainer = PixelClassifierTrainer()
        result = trainer.train(
            PixelTrainingConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                train_split=str(cfg.get("train_split", "train")),
                max_samples=int(cfg.get("max_samples", args.max_samples)),
                max_iter=int(cfg.get("max_iter", args.max_iter)),
                seed=seed,
            )
        )

    out_root = Path(output_dir)
    cfg_out = dict(cfg)
    cfg_out["_resolved_dataset_dir"] = str(dataset_dir)
    if prepared is not None:
        cfg_out["_dataset_prepare"] = {
            "used_existing_splits": prepared.used_existing_splits,
            "prepared": prepared.prepared,
            "split_counts": prepared.split_counts,
            "source_layout": prepared.source_layout,
            "manifest_path": prepared.manifest_path,
        }
    (out_root / "resolved_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    print(f"training output: {out_root}")
    print(f"dataset: {dataset_dir}")
    if prepared is not None:
        if prepared.used_existing_splits:
            print("dataset layout: existing split folders used")
        else:
            print(f"dataset prepared: {prepared.dataset_dir}")
            print(f"prepare manifest: {prepared.manifest_path}")
    print(f"model: {result['model_path']}")
    if result.get("report_path"):
        print(f"report: {result['report_path']}")
    if result.get("html_report_path"):
        print(f"html: {result['html_report_path']}")
    if str(result.get("status", "")).lower() == "interrupted":
        print("training interrupted; partial artifacts are available")
        return 130
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
    enable_gpu = bool(cfg.get("enable_gpu", args.enable_gpu))
    device_policy = str(cfg.get("device_policy", args.device_policy))
    auto_prepare = bool(cfg.get("auto_prepare_dataset", args.auto_prepare_dataset))
    if auto_prepare:
        prep_out = str(
            cfg.get(
                "prepare_output_dir",
                args.prepare_output_dir or str(Path(output_path).parent / "prepared_dataset"),
            )
        )
        prepared = prepare_training_dataset_layout(
            _build_dataset_prepare_config(
                cfg=cfg,
                args=args,
                dataset_dir=str(dataset_dir),
                output_dir=prep_out,
                fallback_seed=int(args.split_seed),
            )
        )
        dataset_dir = prepared.dataset_dir
    else:
        prepared = None

    evaluator = PixelModelEvaluator()
    payload = evaluator.evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(dataset_dir),
            model_path=str(model_path),
            split=str(split),
            output_path=str(output_path),
            enable_gpu=enable_gpu,
            device_policy=device_policy,
            write_html_report=bool(cfg.get("write_html_report", args.write_html_report)),
            tracking_samples=int(cfg.get("tracking_samples", args.tracking_samples)),
            tracking_seed=int(cfg.get("tracking_seed", args.tracking_seed)),
        )
    )
    cfg_out = dict(cfg)
    cfg_out["_resolved_dataset_dir"] = str(dataset_dir)
    if prepared is not None:
        cfg_out["_dataset_prepare"] = {
            "used_existing_splits": prepared.used_existing_splits,
            "prepared": prepared.prepared,
            "split_counts": prepared.split_counts,
            "source_layout": prepared.source_layout,
            "manifest_path": prepared.manifest_path,
        }
    Path(output_path).with_name("resolved_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    print(f"evaluation report: {output_path}")
    print(f"dataset: {dataset_dir}")
    if prepared is not None and not prepared.used_existing_splits:
        print(f"dataset prepared: {prepared.dataset_dir}")
        print(f"prepare manifest: {prepared.manifest_path}")
    print(f"metrics: {payload['metrics']}")
    if payload.get("html_report_path"):
        print(f"html: {payload['html_report_path']}")
    return 0


def _models(args: argparse.Namespace) -> int:
    mgr = DesktopWorkflowManager()
    specs = mgr.model_specs()
    frozen = {rec.model_id: rec for rec in load_frozen_checkpoint_records()}

    payload: list[dict[str, object]] = []
    for spec in specs:
        model_id = str(spec.get("model_id", ""))
        rec = frozen.get(model_id)
        item = {
            "model_id": model_id,
            "display_name": spec.get("display_name", ""),
            "feature_family": spec.get("feature_family", ""),
            "description": spec.get("description", ""),
            "details": spec.get("details", ""),
        }
        if rec:
            item["frozen_checkpoint"] = {
                "model_nickname": rec.model_nickname,
                "model_type": rec.model_type,
                "framework": rec.framework,
                "input_size": rec.input_size,
                "input_dimensions": rec.input_dimensions,
                "checkpoint_path_hint": rec.checkpoint_path_hint,
                "application_remarks": rec.application_remarks,
                "short_description": rec.short_description,
                "detailed_description": rec.detailed_description,
                "classes": list(rec.classes),
            }
        payload.append(item)

    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0

    for item in payload:
        print(f"- {item['display_name']} ({item['model_id']})")
        print(f"  Family: {item['feature_family']}")
        print(f"  Summary: {item['description']}")
        if args.details:
            print(f"  Details: {item['details']}")
        frozen_meta = item.get("frozen_checkpoint")
        if frozen_meta:
            print(
                "  Frozen model: "
                f"{frozen_meta['model_nickname']} | {frozen_meta['framework']} | "
                f"{frozen_meta['model_type']} | {frozen_meta['input_dimensions']}"
            )
            print(f"  Checkpoint hint: {frozen_meta['checkpoint_path_hint']}")
            if args.details:
                print(f"  Application: {frozen_meta['application_remarks']}")
                print(f"  User tip: {frozen_meta['short_description']}")
        print("")
    return 0


def _validate_registry(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    registry_path = args.registry_path or cfg.get("registry_path")
    output_path = args.output_path or cfg.get("output_path") or "outputs/registry/validation_report.json"
    strict = bool(cfg.get("strict", args.strict))

    report = validate_frozen_registry(registry_path)
    write_registry_validation_report(report, output_path)

    print(f"registry valid: {report.ok}")
    print(f"errors: {len(report.errors)}")
    print(f"warnings: {len(report.warnings)}")
    print(f"report: {output_path}")
    if strict and not report.ok:
        return 2
    return 0


def _dataset_split(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    input_dir = args.input_dir or cfg.get("input_dir")
    if not input_dir:
        raise ValueError("input directory is required (--input-dir or config:input_dir)")
    output_dir = args.output_dir or cfg.get("output_dir") or "outputs/packaged_dataset_v2"
    train_ratio = float(cfg.get("train_ratio", args.train_ratio))
    val_ratio = float(cfg.get("val_ratio", args.val_ratio))
    seed = int(cfg.get("seed", args.seed))
    leakage_group = str(cfg.get("leakage_group", args.leakage_group))

    result = plan_and_materialize_correction_splits(
        CorrectionSplitConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            leakage_group=leakage_group,
        )
    )
    print(f"dataset split output: {result.output_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"split counts: {result.split_counts}")
    return 0


def _dataset_qa(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset directory is required (--dataset-dir or config:dataset_dir)")
    output_path = args.output_path or cfg.get("output_path") or "outputs/dataops/dataset_qa_report.json"
    strict = bool(cfg.get("strict", args.strict))
    imbalance_ratio_warn = float(cfg.get("imbalance_ratio_warn", args.imbalance_ratio_warn))

    report = run_dataset_quality_checks(
        DatasetQualityConfig(
            dataset_dir=str(dataset_dir),
            output_path=str(output_path),
            imbalance_ratio_warn=imbalance_ratio_warn,
            strict=strict,
        )
    )
    print(f"dataset qa ok: {report.ok}")
    print(f"errors: {len(report.errors)} warnings: {len(report.warnings)}")
    print(f"report: {output_path}")
    if strict and not report.ok:
        return 2
    return 0


def _dataset_prepare(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset directory is required (--dataset-dir or config:dataset_dir)")
    output_dir = args.output_dir or cfg.get("output_dir") or "outputs/prepared_dataset"

    result = prepare_training_dataset_layout(
        _build_dataset_prepare_config(
            cfg=cfg,
            args=args,
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            fallback_seed=int(args.split_seed),
        )
    )

    print(f"prepared dataset: {result.dataset_dir}")
    print(f"split counts: {result.split_counts}")
    if result.manifest_path:
        print(f"manifest: {result.manifest_path}")
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


def _phase_gate(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    phase_label = str(args.phase_label or cfg.get("phase_label") or "").strip()
    if not phase_label:
        raise ValueError("phase label is required (--phase-label or config:phase_label)")

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label=phase_label,
            run_tests=not bool(cfg.get("skip_tests", args.skip_tests)),
            output_dir=str(cfg.get("output_dir", args.output_dir)),
            extra_notes=str(cfg.get("notes", args.notes or "")),
            strict=False,
        )
    )
    print(f"phase gate status: {result.status}")
    print(f"json report: {result.artifacts['json_report']}")
    print(f"stocktake: {result.artifacts['markdown_stocktake']}")
    if bool(cfg.get("strict", args.strict)) and result.status != "pass":
        return 2
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
    infer.add_argument("--enable-gpu", action="store_true", help="Enable GPU auto-selection for ML inference")
    infer.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    infer.set_defaults(handler=_infer)

    train = sub.add_parser("train", help="Train segmentation model backend")
    train.add_argument("--config", type=str, help="YAML config path")
    train.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    train.add_argument("--dataset-dir", type=str, help="Dataset directory containing train/images + train/masks")
    train.add_argument("--output-dir", type=str, help="Training output directory")
    train.add_argument("--backend", choices=["unet_binary", "torch_pixel", "sklearn_pixel"], default="unet_binary")
    train.add_argument("--max-samples", type=int, default=250000)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--early-stopping-patience", type=int, default=5)
    train.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    train.add_argument("--checkpoint-every", type=int, default=1)
    train.add_argument("--resume-checkpoint", type=str)
    train.add_argument("--val-tracking-samples", type=int, default=6)
    train.add_argument("--val-tracking-fixed-samples", type=str, default="")
    train.add_argument("--val-tracking-seed", type=int, default=17)
    train.add_argument("--write-html-report", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--progress-log-interval-pct", type=int, default=10)
    train.add_argument("--max-iter", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--auto-prepare-dataset", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--prepare-output-dir", type=str)
    train.add_argument("--split-train-ratio", type=float, default=0.8)
    train.add_argument("--split-val-ratio", type=float, default=0.1)
    train.add_argument("--split-test-ratio", type=float, default=0.1)
    train.add_argument("--split-seed", type=int, default=42)
    train.add_argument("--split-id-width", type=int, default=6)
    train.add_argument("--split-strategy", choices=["leakage_aware", "random"], default="leakage_aware")
    train.add_argument("--leakage-group-mode", choices=["suffix_aware", "stem", "regex"], default="suffix_aware")
    train.add_argument("--leakage-group-regex", type=str, default="")
    train.add_argument("--mask-input-type", choices=["indexed", "rgb_colormap", "auto"], default="indexed")
    train.add_argument("--mask-colormap-json", type=str, default="")
    train.add_argument("--mask-colormap-strict", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--enable-gpu", action="store_true", help="Enable GPU auto-selection for GPU-capable backends")
    train.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    train.set_defaults(handler=_train)

    ev = sub.add_parser("evaluate", help="Evaluate trained model on dataset split")
    ev.add_argument("--config", type=str, help="YAML config path")
    ev.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    ev.add_argument("--dataset-dir", type=str, help="Dataset directory containing split/images + split/masks")
    ev.add_argument("--model-path", type=str, help="Path to trained model artifact")
    ev.add_argument("--split", type=str, default="val", help="Split name to evaluate")
    ev.add_argument("--output-path", type=str, help="Report output path")
    ev.add_argument("--enable-gpu", action="store_true", help="Enable GPU auto-selection for torch model eval")
    ev.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    ev.add_argument("--write-html-report", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--tracking-samples", type=int, default=8)
    ev.add_argument("--tracking-seed", type=int, default=17)
    ev.add_argument("--auto-prepare-dataset", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--prepare-output-dir", type=str)
    ev.add_argument("--split-train-ratio", type=float, default=0.8)
    ev.add_argument("--split-val-ratio", type=float, default=0.1)
    ev.add_argument("--split-test-ratio", type=float, default=0.1)
    ev.add_argument("--split-seed", type=int, default=42)
    ev.add_argument("--split-id-width", type=int, default=6)
    ev.add_argument("--split-strategy", choices=["leakage_aware", "random"], default="leakage_aware")
    ev.add_argument("--leakage-group-mode", choices=["suffix_aware", "stem", "regex"], default="suffix_aware")
    ev.add_argument("--leakage-group-regex", type=str, default="")
    ev.add_argument("--mask-input-type", choices=["indexed", "rgb_colormap", "auto"], default="indexed")
    ev.add_argument("--mask-colormap-json", type=str, default="")
    ev.add_argument("--mask-colormap-strict", action=argparse.BooleanOptionalAction, default=True)
    ev.set_defaults(handler=_evaluate)

    models = sub.add_parser("models", help="List available GUI/CLI models and frozen-checkpoint metadata")
    models.add_argument("--details", action="store_true", help="Show long descriptions and application notes")
    models.add_argument("--as-json", action="store_true", help="Print machine-readable JSON payload")
    models.set_defaults(handler=_models)

    validate_registry = sub.add_parser("validate-registry", help="Validate frozen model registry metadata")
    validate_registry.add_argument("--config", type=str, help="YAML config path")
    validate_registry.add_argument("--set", action="append", default=[], help="Override key=value")
    validate_registry.add_argument("--registry-path", type=str, help="Registry JSON path")
    validate_registry.add_argument("--output-path", type=str, help="Validation report JSON path")
    validate_registry.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    validate_registry.set_defaults(handler=_validate_registry)

    split = sub.add_parser("dataset-split", help="Leakage-aware correction split planner and materializer")
    split.add_argument("--config", type=str, help="YAML config path")
    split.add_argument("--set", action="append", default=[], help="Override key=value")
    split.add_argument("--input-dir", type=str, help="Correction exports root directory")
    split.add_argument("--output-dir", type=str, help="Output packaged dataset directory")
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--leakage-group", choices=["source_stem", "sample_id"], default="source_stem")
    split.set_defaults(handler=_dataset_split)

    qa = sub.add_parser("dataset-qa", help="Run packaged dataset QA checks")
    qa.add_argument("--config", type=str, help="YAML config path")
    qa.add_argument("--set", action="append", default=[], help="Override key=value")
    qa.add_argument("--dataset-dir", type=str, help="Packaged dataset directory")
    qa.add_argument("--output-path", type=str, help="QA report JSON path")
    qa.add_argument("--imbalance-ratio-warn", type=float, default=0.98)
    qa.add_argument("--strict", action="store_true", help="Exit non-zero when QA fails")
    qa.set_defaults(handler=_dataset_qa)

    prep = sub.add_parser("dataset-prepare", help="Prepare split layout from unsplit source/masks dataset")
    prep.add_argument("--config", type=str, help="YAML config path")
    prep.add_argument("--set", action="append", default=[], help="Override key=value")
    prep.add_argument("--dataset-dir", type=str, help="Input dataset root")
    prep.add_argument("--output-dir", type=str, help="Prepared dataset output directory")
    prep.add_argument("--split-train-ratio", type=float, default=0.8)
    prep.add_argument("--split-val-ratio", type=float, default=0.1)
    prep.add_argument("--split-test-ratio", type=float, default=0.1)
    prep.add_argument("--split-seed", type=int, default=42)
    prep.add_argument("--split-id-width", type=int, default=6)
    prep.add_argument("--split-strategy", choices=["leakage_aware", "random"], default="leakage_aware")
    prep.add_argument("--leakage-group-mode", choices=["suffix_aware", "stem", "regex"], default="suffix_aware")
    prep.add_argument("--leakage-group-regex", type=str, default="")
    prep.add_argument("--mask-input-type", choices=["indexed", "rgb_colormap", "auto"], default="indexed")
    prep.add_argument("--mask-colormap-json", type=str, default="")
    prep.add_argument("--mask-colormap-strict", action=argparse.BooleanOptionalAction, default=True)
    prep.set_defaults(handler=_dataset_prepare)

    pack = sub.add_parser("package", help="Package correction exports into dataset splits")
    pack.add_argument("--config", type=str, help="YAML config path")
    pack.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    pack.add_argument("--input-dir", type=str, help="Directory containing corrected sample folders")
    pack.add_argument("--output-dir", type=str, help="Dataset output directory")
    pack.add_argument("--train-ratio", type=float, default=0.8)
    pack.add_argument("--val-ratio", type=float, default=0.1)
    pack.add_argument("--seed", type=int, default=42)
    pack.set_defaults(handler=_package)

    gate = sub.add_parser("phase-gate", help="Run end-of-phase closeout checks and stocktake report")
    gate.add_argument("--config", type=str, help="YAML config path")
    gate.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    gate.add_argument("--phase-label", type=str, help="Phase label (for reports)")
    gate.add_argument("--output-dir", type=str, default="outputs/phase_gates")
    gate.add_argument("--notes", type=str, default="")
    gate.add_argument("--skip-tests", action="store_true")
    gate.add_argument("--strict", action="store_true")
    gate.set_defaults(handler=_phase_gate)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rc = args.handler(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
