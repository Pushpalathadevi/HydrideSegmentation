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
from src.microseg.app.hpc_ga import (
    HpcGaPlanConfig,
    generate_hpc_ga_bundle,
    parse_architectures,
    parse_batch_sizes,
    parse_feedback_sources,
    parse_pretrained_model_map,
    summarize_feedback_sources,
)
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
    validate_pretrained_registry,
    validate_frozen_registry,
    write_pretrained_validation_report,
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


def _normalize_binary_mask_mode(value: object) -> str:
    text = str(value).strip().lower()
    if value is False or text in {"", "0", "false", "off", "none", "null"}:
        return "off"
    if text in {"nonzero_foreground", "nonzero", "all_nonzero_foreground"}:
        return "nonzero_foreground"
    if value is True or text in {"1", "true", "on", "two_value_zero_background"}:
        return "two_value_zero_background"
    raise ValueError(
        "binary_mask_normalization must be one of: off, two_value_zero_background, nonzero_foreground; "
        f"got {value!r}"
    )




def _parse_hw(value: object, *, fallback: str = "512,512") -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value if value is not None else fallback).strip().strip("[]")
    sep = "," if "," in text else "x"
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"input_hw must have two integers, got {value!r}")
    return int(parts[0]), int(parts[1])


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
        binary_mask_normalization=_normalize_binary_mask_mode(
            cfg.get("binary_mask_normalization", args.binary_mask_normalization)
        ),
        class_map_path=str(cfg.get("class_map_path", getattr(args, "class_map_path", ""))),
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
    model_architecture = str(cfg.get("model_architecture", "")).strip()
    if not model_architecture:
        model_architecture = backend
    if (
        backend in {"smp_unet_resnet18", "transunet_tiny", "segformer_mini", "hf_segformer_b0", "hf_segformer_b2", "hf_segformer_b5"}
        and model_architecture == "unet_binary"
    ):
        # Backward-compat guard: if backend was overridden but model_architecture stayed at train.default.yml baseline.
        model_architecture = backend
    enable_gpu = bool(cfg.get("enable_gpu", args.enable_gpu))
    device_policy = str(cfg.get("device_policy", args.device_policy))
    seed = int(cfg.get("seed", args.seed))
    val_tracking_fixed = tuple(_parse_name_list(cfg.get("val_tracking_fixed_samples", args.val_tracking_fixed_samples)))
    auto_prepare = (
        bool(args.auto_prepare_dataset)
        if args.auto_prepare_dataset is not None
        else bool(cfg.get("auto_prepare_dataset", True))
    )
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
                binary_mask_normalization=_normalize_binary_mask_mode(
                    cfg.get("binary_mask_normalization", args.binary_mask_normalization)
                ),
            )
        )
    elif backend in {
        "unet_binary",
        "smp_unet_resnet18",
        "transunet_tiny",
        "segformer_mini",
        "hf_segformer_b0",
        "hf_segformer_b2",
        "hf_segformer_b5",
    }:
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
                model_architecture=model_architecture,
                model_base_channels=int(cfg.get("model_base_channels", args.model_base_channels)),
                transformer_depth=int(cfg.get("transformer_depth", args.transformer_depth)),
                transformer_num_heads=int(cfg.get("transformer_num_heads", args.transformer_num_heads)),
                transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", args.transformer_mlp_ratio)),
                transformer_dropout=float(cfg.get("transformer_dropout", args.transformer_dropout)),
                segformer_patch_size=int(cfg.get("segformer_patch_size", args.segformer_patch_size)),
                backend_label=str(cfg.get("backend_label", backend)),
                pretrained_init_mode=str(cfg.get("pretrained_init_mode", args.pretrained_init_mode)),
                pretrained_model_id=str(cfg.get("pretrained_model_id", args.pretrained_model_id or "")),
                pretrained_bundle_dir=str(cfg.get("pretrained_bundle_dir", args.pretrained_bundle_dir or "")),
                pretrained_registry_path=str(
                    cfg.get("pretrained_registry_path", args.pretrained_registry_path)
                ),
                pretrained_strict=bool(cfg.get("pretrained_strict", args.pretrained_strict)),
                pretrained_ignore_mismatched_sizes=bool(
                    cfg.get(
                        "pretrained_ignore_mismatched_sizes",
                        args.pretrained_ignore_mismatched_sizes,
                    )
                ),
                pretrained_verify_sha256=bool(
                    cfg.get("pretrained_verify_sha256", args.pretrained_verify_sha256)
                ),
                amp_enabled=bool(cfg.get("amp_enabled", args.amp_enabled)),
                grad_accum_steps=int(cfg.get("grad_accum_steps", args.grad_accum_steps)),
                torch_compile=bool(cfg.get("torch_compile", args.torch_compile)),
                num_workers=int(cfg.get("num_workers", args.num_workers)),
                pin_memory=bool(cfg.get("pin_memory", args.pin_memory)),
                persistent_workers=bool(cfg.get("persistent_workers", args.persistent_workers)),
                deterministic=bool(cfg.get("deterministic", args.deterministic)),
                binary_mask_normalization=_normalize_binary_mask_mode(
                    cfg.get("binary_mask_normalization", args.binary_mask_normalization)
                ),
                input_hw=_parse_hw(cfg.get("input_hw", args.input_hw), fallback=args.input_hw),
                input_policy=str(cfg.get("input_policy", args.input_policy)),
                val_input_policy=str(cfg.get("val_input_policy", args.val_input_policy)),
                keep_aspect=bool(cfg.get("keep_aspect", args.keep_aspect)),
                pad_value_image=float(cfg.get("pad_value_image", args.pad_value_image)),
                pad_value_mask=int(cfg.get("pad_value_mask", args.pad_value_mask)),
                image_interpolation=str(cfg.get("image_interpolation", args.image_interpolation)),
                mask_interpolation=str(cfg.get("mask_interpolation", args.mask_interpolation)),
                require_divisible_by=int(cfg.get("require_divisible_by", args.require_divisible_by)),
                dataloader_collate=str(cfg.get("dataloader_collate", args.dataloader_collate)),
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
                binary_mask_normalization=_normalize_binary_mask_mode(
                    cfg.get("binary_mask_normalization", args.binary_mask_normalization)
                ),
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
    auto_prepare = (
        bool(args.auto_prepare_dataset)
        if args.auto_prepare_dataset is not None
        else bool(cfg.get("auto_prepare_dataset", True))
    )
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
            binary_mask_normalization=_normalize_binary_mask_mode(
                cfg.get("binary_mask_normalization", args.binary_mask_normalization)
            ),
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


def _validate_pretrained(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    registry_path = args.registry_path or cfg.get("registry_path") or "pre_trained_weights/registry.json"
    output_path = (
        args.output_path
        or cfg.get("output_path")
        or "outputs/pretrained_weights/validation_report.json"
    )
    strict = bool(cfg.get("strict", args.strict))
    verify_sha256 = bool(cfg.get("verify_sha256", args.verify_sha256))

    report = validate_pretrained_registry(registry_path, verify_sha256=verify_sha256)
    write_pretrained_validation_report(report, output_path)

    print(f"pretrained registry valid: {report.ok}")
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


def _hpc_ga_generate(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset directory is required (--dataset-dir or config:dataset_dir)")
    output_dir = args.output_dir or cfg.get("output_dir") or "outputs/hpc_ga_bundle"

    architectures = parse_architectures(cfg.get("architectures", args.architectures))
    if not architectures:
        architectures = parse_architectures(args.architectures)
    batch_sizes = parse_batch_sizes(cfg.get("batch_size_choices", args.batch_size_choices))
    if not batch_sizes:
        batch_sizes = parse_batch_sizes(args.batch_size_choices)
    feedback_sources = parse_feedback_sources(cfg.get("feedback_sources", args.feedback_sources))
    if not feedback_sources:
        feedback_sources = parse_feedback_sources(args.feedback_sources)
    pretrained_model_map = parse_pretrained_model_map(cfg.get("pretrained_model_map", args.pretrained_model_map))
    if not pretrained_model_map:
        pretrained_model_map = parse_pretrained_model_map(args.pretrained_model_map)

    result = generate_hpc_ga_bundle(
        HpcGaPlanConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            experiment_name=str(cfg.get("experiment_name", args.experiment_name)),
            base_train_config=str(cfg.get("base_train_config", args.base_train_config)),
            base_eval_config=str(cfg.get("base_eval_config", args.base_eval_config)),
            run_mode=str(cfg.get("run_mode", args.run_mode)),
            eval_split=str(cfg.get("eval_split", args.eval_split)),
            architectures=architectures,
            num_candidates=int(cfg.get("num_candidates", args.num_candidates)),
            population_size=int(cfg.get("population_size", args.population_size)),
            generations=int(cfg.get("generations", args.generations)),
            mutation_rate=float(cfg.get("mutation_rate", args.mutation_rate)),
            crossover_rate=float(cfg.get("crossover_rate", args.crossover_rate)),
            seed=int(cfg.get("seed", args.seed)),
            learning_rate_min=float(cfg.get("learning_rate_min", args.learning_rate_min)),
            learning_rate_max=float(cfg.get("learning_rate_max", args.learning_rate_max)),
            batch_size_choices=batch_sizes,
            epochs_min=int(cfg.get("epochs_min", args.epochs_min)),
            epochs_max=int(cfg.get("epochs_max", args.epochs_max)),
            weight_decay_min=float(cfg.get("weight_decay_min", args.weight_decay_min)),
            weight_decay_max=float(cfg.get("weight_decay_max", args.weight_decay_max)),
            max_samples_min=int(cfg.get("max_samples_min", args.max_samples_min)),
            max_samples_max=int(cfg.get("max_samples_max", args.max_samples_max)),
            fitness_mode=str(cfg.get("fitness_mode", args.fitness_mode)),
            feedback_sources=feedback_sources,
            feedback_min_samples=int(cfg.get("feedback_min_samples", args.feedback_min_samples)),
            feedback_k=int(cfg.get("feedback_k", args.feedback_k)),
            exploration_weight=float(cfg.get("exploration_weight", args.exploration_weight)),
            fitness_weight_mean_iou=float(cfg.get("fitness_weight_mean_iou", args.fitness_weight_mean_iou)),
            fitness_weight_macro_f1=float(cfg.get("fitness_weight_macro_f1", args.fitness_weight_macro_f1)),
            fitness_weight_pixel_accuracy=float(cfg.get("fitness_weight_pixel_accuracy", args.fitness_weight_pixel_accuracy)),
            fitness_weight_runtime=float(cfg.get("fitness_weight_runtime", args.fitness_weight_runtime)),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
            scheduler=str(cfg.get("scheduler", args.scheduler)),
            queue=str(cfg.get("queue", args.queue)),
            account=str(cfg.get("account", args.account)),
            qos=str(cfg.get("qos", args.qos)),
            gpus_per_job=int(cfg.get("gpus_per_job", args.gpus_per_job)),
            cpus_per_task=int(cfg.get("cpus_per_task", args.cpus_per_task)),
            mem_gb=int(cfg.get("mem_gb", args.mem_gb)),
            time_limit=str(cfg.get("time_limit", args.time_limit)),
            job_prefix=str(cfg.get("job_prefix", args.job_prefix)),
            python_executable=str(cfg.get("python_executable", args.python_executable)),
            microseg_cli_path=str(cfg.get("microseg_cli_path", args.microseg_cli_path)),
            pretrained_init_mode=str(cfg.get("pretrained_init_mode", args.pretrained_init_mode)),
            pretrained_registry_path=str(cfg.get("pretrained_registry_path", args.pretrained_registry_path)),
            pretrained_model_map=pretrained_model_map,
            pretrained_verify_sha256=bool(cfg.get("pretrained_verify_sha256", args.pretrained_verify_sha256)),
            pretrained_ignore_mismatched_sizes=bool(
                cfg.get("pretrained_ignore_mismatched_sizes", args.pretrained_ignore_mismatched_sizes)
            ),
            pretrained_strict=bool(cfg.get("pretrained_strict", args.pretrained_strict)),
        )
    )

    print(f"hpc-ga bundle: {result.bundle_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"submit script: {result.submit_script}")
    print(f"candidates: {len(result.candidates)}")
    return 0


def _hpc_ga_feedback_report(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    feedback_sources = parse_feedback_sources(cfg.get("feedback_sources", args.feedback_sources))
    if not feedback_sources:
        feedback_sources = parse_feedback_sources(args.feedback_sources)
    if not feedback_sources:
        raise ValueError("feedback sources are required (--feedback-sources or config:feedback_sources)")

    architectures = parse_architectures(cfg.get("architectures", args.architectures))
    if not architectures:
        architectures = parse_architectures(args.architectures)
    batch_sizes = parse_batch_sizes(cfg.get("batch_size_choices", args.batch_size_choices))
    if not batch_sizes:
        batch_sizes = parse_batch_sizes(args.batch_size_choices)

    plan_cfg = HpcGaPlanConfig(
        dataset_dir=str(cfg.get("dataset_dir", args.dataset_dir or "outputs/prepared_dataset")),
        output_dir=str(cfg.get("output_dir", args.output_dir or "outputs/hpc_ga_bundle")),
        architectures=architectures,
        batch_size_choices=batch_sizes,
        fitness_mode=str(cfg.get("fitness_mode", args.fitness_mode)),
        feedback_sources=feedback_sources,
        feedback_min_samples=int(cfg.get("feedback_min_samples", args.feedback_min_samples)),
        feedback_k=int(cfg.get("feedback_k", args.feedback_k)),
        exploration_weight=float(cfg.get("exploration_weight", args.exploration_weight)),
        fitness_weight_mean_iou=float(cfg.get("fitness_weight_mean_iou", args.fitness_weight_mean_iou)),
        fitness_weight_macro_f1=float(cfg.get("fitness_weight_macro_f1", args.fitness_weight_macro_f1)),
        fitness_weight_pixel_accuracy=float(cfg.get("fitness_weight_pixel_accuracy", args.fitness_weight_pixel_accuracy)),
        fitness_weight_runtime=float(cfg.get("fitness_weight_runtime", args.fitness_weight_runtime)),
    )
    report = summarize_feedback_sources(
        feedback_sources,
        cfg=plan_cfg,
        top_k=int(cfg.get("top_k", args.top_k)),
    )
    output_path = str(args.output_path or cfg.get("output_path") or "outputs/hpc_ga_feedback/feedback_report.json")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# HPC GA Feedback Report",
        "",
        f"- samples: {int(report.get('sample_count', 0))}",
        f"- fitness_mode: {report.get('fitness_mode', '')}",
        "",
        "| candidate | backend | fitness | mean_iou | macro_f1 | pixel_acc | runtime_s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("top_candidates", []):
        lines.append(
            "| {cid} | {backend} | {fit:.4f} | {mi:.4f} | {f1:.4f} | {pa:.4f} | {rt:.2f} |".format(
                cid=str(row.get("candidate_id", "")),
                backend=str(row.get("backend", "")),
                fit=float(row.get("fitness_score", 0.0)),
                mi=float(row.get("mean_iou", 0.0)),
                f1=float(row.get("macro_f1", 0.0)),
                pa=float(row.get("pixel_accuracy", 0.0)),
                rt=float(row.get("runtime_seconds", 0.0)),
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"feedback report: {out}")
    print(f"markdown: {md}")
    print(f"samples: {int(report.get('sample_count', 0))}")
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
    train.add_argument(
        "--backend",
        choices=[
            "unet_binary",
            "smp_unet_resnet18",
            "transunet_tiny",
            "segformer_mini",
            "hf_segformer_b0",
            "hf_segformer_b2",
            "hf_segformer_b5",
            "torch_pixel",
            "sklearn_pixel",
        ],
        default="unet_binary",
    )
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
    train.add_argument("--model-base-channels", type=int, default=16)
    train.add_argument("--transformer-depth", type=int, default=2)
    train.add_argument("--transformer-num-heads", type=int, default=4)
    train.add_argument("--transformer-mlp-ratio", type=float, default=2.0)
    train.add_argument("--transformer-dropout", type=float, default=0.0)
    train.add_argument("--segformer-patch-size", type=int, default=4)
    train.add_argument("--pretrained-init-mode", choices=["scratch", "local"], default="scratch")
    train.add_argument("--pretrained-model-id", type=str, default="")
    train.add_argument("--pretrained-bundle-dir", type=str, default="")
    train.add_argument("--pretrained-registry-path", type=str, default="pre_trained_weights/registry.json")
    train.add_argument("--pretrained-strict", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument(
        "--pretrained-ignore-mismatched-sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--pretrained-verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--amp-enabled", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--grad-accum-steps", type=int, default=1)
    train.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--input-hw", type=str, default="512,512", help="Target input H,W")
    train.add_argument(
        "--input-policy",
        choices=["resize", "letterbox", "random_crop", "center_crop"],
        default="random_crop",
    )
    train.add_argument(
        "--val-input-policy",
        choices=["resize", "letterbox", "random_crop", "center_crop"],
        default="letterbox",
    )
    train.add_argument("--keep-aspect", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--pad-value-image", type=float, default=0.0)
    train.add_argument("--pad-value-mask", type=int, default=0)
    train.add_argument("--image-interpolation", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    train.add_argument("--mask-interpolation", choices=["nearest"], default="nearest")
    train.add_argument("--require-divisible-by", type=int, default=32)
    train.add_argument("--dataloader-collate", choices=["default", "pad_to_max"], default="default")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--binary-mask-normalization",
        choices=["off", "two_value_zero_background", "nonzero_foreground"],
        default="off",
    )
    train.add_argument("--max-iter", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--auto-prepare-dataset", action=argparse.BooleanOptionalAction, default=None)
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
    ev.add_argument("--auto-prepare-dataset", action=argparse.BooleanOptionalAction, default=None)
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
    ev.add_argument(
        "--binary-mask-normalization",
        choices=["off", "two_value_zero_background", "nonzero_foreground"],
        default="off",
    )
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

    validate_pretrained = sub.add_parser(
        "validate-pretrained",
        help="Validate local pretrained-weight registry metadata and artifacts",
    )
    validate_pretrained.add_argument("--config", type=str, help="YAML config path")
    validate_pretrained.add_argument("--set", action="append", default=[], help="Override key=value")
    validate_pretrained.add_argument("--registry-path", type=str, help="Pretrained registry JSON path")
    validate_pretrained.add_argument("--output-path", type=str, help="Validation report JSON path")
    validate_pretrained.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    validate_pretrained.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    validate_pretrained.set_defaults(handler=_validate_pretrained)

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
    prep.add_argument(
        "--binary-mask-normalization",
        choices=["off", "two_value_zero_background", "nonzero_foreground"],
        default="off",
    )
    prep.add_argument("--class-map-path", type=str, default="")
    prep.set_defaults(handler=_dataset_prepare)

    hpc_ga = sub.add_parser("hpc-ga-generate", help="Generate GA-planned HPC job bundle")
    hpc_ga.add_argument("--config", type=str, help="YAML config path")
    hpc_ga.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    hpc_ga.add_argument("--dataset-dir", type=str, help="Dataset directory for train/eval jobs")
    hpc_ga.add_argument("--output-dir", type=str, help="Bundle output directory")
    hpc_ga.add_argument("--experiment-name", type=str, default="microseg_hpc_ga_sweep")
    hpc_ga.add_argument("--base-train-config", type=str, default="configs/train.default.yml")
    hpc_ga.add_argument("--base-eval-config", type=str, default="configs/evaluate.default.yml")
    hpc_ga.add_argument("--run-mode", choices=["train_only", "train_eval"], default="train_eval")
    hpc_ga.add_argument("--eval-split", type=str, default="val")
    hpc_ga.add_argument(
        "--architectures",
        type=str,
        default="unet_binary,smp_unet_resnet18,hf_segformer_b0,hf_segformer_b2,hf_segformer_b5,transunet_tiny,segformer_mini,torch_pixel",
    )
    hpc_ga.add_argument("--num-candidates", type=int, default=8)
    hpc_ga.add_argument("--population-size", type=int, default=24)
    hpc_ga.add_argument("--generations", type=int, default=8)
    hpc_ga.add_argument("--mutation-rate", type=float, default=0.2)
    hpc_ga.add_argument("--crossover-rate", type=float, default=0.7)
    hpc_ga.add_argument("--seed", type=int, default=42)
    hpc_ga.add_argument("--learning-rate-min", type=float, default=1e-4)
    hpc_ga.add_argument("--learning-rate-max", type=float, default=1e-2)
    hpc_ga.add_argument("--batch-size-choices", type=str, default="4,8,16,32")
    hpc_ga.add_argument("--epochs-min", type=int, default=8)
    hpc_ga.add_argument("--epochs-max", type=int, default=40)
    hpc_ga.add_argument("--weight-decay-min", type=float, default=1e-6)
    hpc_ga.add_argument("--weight-decay-max", type=float, default=1e-3)
    hpc_ga.add_argument("--max-samples-min", type=int, default=50000)
    hpc_ga.add_argument("--max-samples-max", type=int, default=250000)
    hpc_ga.add_argument("--fitness-mode", choices=["novelty", "feedback_hybrid"], default="novelty")
    hpc_ga.add_argument("--feedback-sources", type=str, default="")
    hpc_ga.add_argument("--feedback-min-samples", type=int, default=3)
    hpc_ga.add_argument("--feedback-k", type=int, default=5)
    hpc_ga.add_argument("--exploration-weight", type=float, default=0.55)
    hpc_ga.add_argument("--fitness-weight-mean-iou", type=float, default=0.50)
    hpc_ga.add_argument("--fitness-weight-macro-f1", type=float, default=0.30)
    hpc_ga.add_argument("--fitness-weight-pixel-accuracy", type=float, default=0.20)
    hpc_ga.add_argument("--fitness-weight-runtime", type=float, default=0.05)
    hpc_ga.add_argument("--enable-gpu", action=argparse.BooleanOptionalAction, default=True)
    hpc_ga.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="auto")
    hpc_ga.add_argument("--scheduler", choices=["slurm", "pbs", "local"], default="slurm")
    hpc_ga.add_argument("--queue", type=str, default="")
    hpc_ga.add_argument("--account", type=str, default="")
    hpc_ga.add_argument("--qos", type=str, default="")
    hpc_ga.add_argument("--gpus-per-job", type=int, default=1)
    hpc_ga.add_argument("--cpus-per-task", type=int, default=8)
    hpc_ga.add_argument("--mem-gb", type=int, default=32)
    hpc_ga.add_argument("--time-limit", type=str, default="08:00:00")
    hpc_ga.add_argument("--job-prefix", type=str, default="microseg")
    hpc_ga.add_argument("--python-executable", type=str, default="python")
    hpc_ga.add_argument("--microseg-cli-path", type=str, default="scripts/microseg_cli.py")
    hpc_ga.add_argument("--pretrained-init-mode", choices=["scratch", "auto", "local"], default="scratch")
    hpc_ga.add_argument("--pretrained-registry-path", type=str, default="pre_trained_weights/registry.json")
    hpc_ga.add_argument(
        "--pretrained-model-map",
        type=str,
        default="",
        help="backend=model_id CSV or JSON object mapping backend->pretrained model_id",
    )
    hpc_ga.add_argument("--pretrained-verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    hpc_ga.add_argument(
        "--pretrained-ignore-mismatched-sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    hpc_ga.add_argument("--pretrained-strict", action=argparse.BooleanOptionalAction, default=False)
    hpc_ga.set_defaults(handler=_hpc_ga_generate)

    hpc_ga_feedback = sub.add_parser("hpc-ga-feedback-report", help="Summarize prior HPC GA run feedback")
    hpc_ga_feedback.add_argument("--config", type=str, help="YAML config path")
    hpc_ga_feedback.add_argument("--set", action="append", default=[], help="Override key=value (supports dotted keys)")
    hpc_ga_feedback.add_argument("--feedback-sources", type=str, default="")
    hpc_ga_feedback.add_argument("--output-path", type=str, help="Output JSON report path")
    hpc_ga_feedback.add_argument("--dataset-dir", type=str, default="outputs/prepared_dataset")
    hpc_ga_feedback.add_argument("--output-dir", type=str, default="outputs/hpc_ga_bundle")
    hpc_ga_feedback.add_argument(
        "--architectures",
        type=str,
        default="unet_binary,smp_unet_resnet18,hf_segformer_b0,hf_segformer_b2,hf_segformer_b5,transunet_tiny,segformer_mini,torch_pixel",
    )
    hpc_ga_feedback.add_argument("--batch-size-choices", type=str, default="4,8,16,32")
    hpc_ga_feedback.add_argument("--fitness-mode", choices=["novelty", "feedback_hybrid"], default="feedback_hybrid")
    hpc_ga_feedback.add_argument("--feedback-min-samples", type=int, default=3)
    hpc_ga_feedback.add_argument("--feedback-k", type=int, default=5)
    hpc_ga_feedback.add_argument("--exploration-weight", type=float, default=0.55)
    hpc_ga_feedback.add_argument("--fitness-weight-mean-iou", type=float, default=0.50)
    hpc_ga_feedback.add_argument("--fitness-weight-macro-f1", type=float, default=0.30)
    hpc_ga_feedback.add_argument("--fitness-weight-pixel-accuracy", type=float, default=0.20)
    hpc_ga_feedback.add_argument("--fitness-weight-runtime", type=float, default=0.05)
    hpc_ga_feedback.add_argument("--top-k", type=int, default=10)
    hpc_ga_feedback.set_defaults(handler=_hpc_ga_feedback_report)

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
