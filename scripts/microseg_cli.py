"""Unified CLI for inference/training/evaluation, dataops, deployment, promotion, and phase gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any

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
from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pipeline import DatasetPreparer
from src.microseg.dataops import (
    CorrectionSplitConfig,
    DatasetPrepareConfig,
    DatasetQualityConfig,
    plan_and_materialize_correction_splits,
    prepare_training_dataset_layout,
    run_dataset_quality_checks,
)
from src.microseg.deployment import (
    CanaryShadowConfig,
    DeploymentPackageConfig,
    DeploymentPerfConfig,
    RuntimeHealthConfig,
    DeploymentSmokeConfig,
    ServiceWorkerConfig,
    create_deployment_package,
    run_canary_shadow_compare,
    run_deployment_perf,
    run_runtime_health,
    run_service_worker_batch,
    run_deployment_smoke,
    validate_deployment_package,
)
from src.microseg.io import resolve_config
from src.microseg.plugins import (
    load_frozen_checkpoint_records,
    validate_pretrained_registry,
    validate_frozen_registry,
    write_pretrained_validation_report,
    write_registry_validation_report,
)
from src.microseg.quality import (
    PhaseGateConfig,
    PreflightConfig,
    SupportBundleConfig,
    create_support_bundle,
    evaluate_and_promote_model,
    load_promotion_policy,
    run_phase_gate,
    run_preflight,
    write_compatibility_matrix,
    write_promotion_decision,
)
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
        backend
        in {
            "smp_unet_resnet18",
            "smp_deeplabv3plus_resnet101",
            "smp_unetplusplus_resnet101",
            "smp_pspnet_resnet101",
            "smp_fpn_resnet101",
            "transunet_tiny",
            "segformer_mini",
            "hf_segformer_b0",
            "hf_segformer_b2",
            "hf_segformer_b5",
            "hf_upernet_swin_large",
        }
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
        "smp_deeplabv3plus_resnet101",
        "smp_unetplusplus_resnet101",
        "smp_pspnet_resnet101",
        "smp_fpn_resnet101",
        "transunet_tiny",
        "segformer_mini",
        "hf_segformer_b0",
        "hf_segformer_b2",
        "hf_segformer_b5",
        "hf_upernet_swin_large",
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



def _prepare_dataset_paired(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    input_dir = args.input_dir or cfg.get("input_dir")
    if not input_dir:
        raise ValueError("input directory is required (--input-dir or config:input_dir)")
    output_root = args.output_root or cfg.get("output_dir") or cfg.get("output_root")
    if not output_root:
        raise ValueError("output root is required (--output-root or config:output_dir)")

    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    max_val_examples_raw = cfg.get("max_val_examples", cfg.get("val_max_examples", args.max_val_examples))
    max_test_examples_raw = cfg.get("max_test_examples", cfg.get("test_max_examples", args.max_test_examples))
    max_val_examples = _optional_int(max_val_examples_raw)
    max_test_examples = _optional_int(max_test_examples_raw)

    resolved = DatasetPrepConfig.from_dict({
        **cfg,
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "styles": [part.strip() for part in str(args.style or cfg.get("style", "mado")).split(",") if part.strip()],
        "train_pct": float(cfg.get("train_pct", args.train_frac)),
        "val_pct": float(cfg.get("val_pct", args.val_frac)),
        "max_val_examples": max_val_examples,
        "max_test_examples": max_test_examples,
        "seed": int(cfg.get("seed", args.seed)),
        "dry_run": bool(cfg.get("dry_run", args.dry_run)),
        "target_size": int(cfg.get("target_size", args.target_size)),
        "resize_policy": str(cfg.get("resize_policy", "short_side_to_target_crop")),
        "crop_mode_train": str(cfg.get("crop_mode_train", args.crop_train)),
        "crop_mode_eval": str(cfg.get("crop_mode_eval", args.crop_eval)),
        "rgb_mask_mode": bool(cfg.get("rgb_mask_mode", True)),
        "mask_r_min": int(cfg.get("mask_r_min", args.mask_r_min)),
        "mask_g_max": int(cfg.get("mask_g_max", args.mask_g_max)),
        "mask_b_max": int(cfg.get("mask_b_max", args.mask_b_max)),
        "allow_red_dominance_fallback": bool(
            cfg.get("allow_red_dominance_fallback", args.allow_red_dominance_fallback)
        ),
        "mask_red_min_fallback": int(cfg.get("mask_red_min_fallback", args.mask_red_min_fallback)),
        "mask_red_dominance_margin": int(cfg.get("mask_red_dominance_margin", args.mask_red_dominance_margin)),
        "mask_red_dominance_ratio": float(cfg.get("mask_red_dominance_ratio", args.mask_red_dominance_ratio)),
        "auto_otsu_for_noisy_grayscale": bool(
            cfg.get("auto_otsu_for_noisy_grayscale", args.auto_otsu_for_noisy_grayscale)
        ),
        "noisy_grayscale_low_max": int(cfg.get("noisy_grayscale_low_max", args.noisy_grayscale_low_max)),
        "noisy_grayscale_high_min": int(cfg.get("noisy_grayscale_high_min", args.noisy_grayscale_high_min)),
        "noisy_grayscale_min_extreme_ratio": float(
            cfg.get("noisy_grayscale_min_extreme_ratio", args.noisy_grayscale_min_extreme_ratio)
        ),
        "empty_mask_action": str(cfg.get("empty_mask_action", args.empty_mask_action)),
        "image_extensions": cfg.get("image_extensions", [".jpg", ".jpeg"]),
        "mask_extensions": cfg.get("mask_extensions", [".png"]),
        "mask_name_patterns": cfg.get("mask_name_patterns", ["{stem}_mask.png", "{stem}.png"]),
        "debug": {
            **(cfg.get("debug", {}) if isinstance(cfg.get("debug"), dict) else {}),
            "enabled": bool(cfg.get("debug", {}).get("enabled", args.debug)) if isinstance(cfg.get("debug"), dict) else bool(args.debug),
            "limit_pairs": int(cfg.get("debug", {}).get("limit_pairs", args.debug_limit)) if isinstance(cfg.get("debug"), dict) else int(args.debug_limit),
            "inspection_limit": int(cfg.get("debug", {}).get("inspection_limit", args.num_debug)) if isinstance(cfg.get("debug"), dict) else int(args.num_debug),
            "show_plots": bool(cfg.get("debug", {}).get("show_plots", args.debug_show_plots)) if isinstance(cfg.get("debug"), dict) else bool(args.debug_show_plots),
            "draw_contours": bool(cfg.get("debug", {}).get("draw_contours", args.debug_draw_contours)) if isinstance(cfg.get("debug"), dict) else bool(args.debug_draw_contours),
        },
    })

    result = DatasetPreparer(resolved).run()
    print(f"prepared dataset: {output_root}")
    print(f"dataset directory for training: {Path(output_root) / 'mado'}")
    print(f"manifest: {result.manifest_path}")
    print(f"split counts: {result.split_counts}")
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


def _preflight(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    mode = str(args.mode or cfg.get("mode") or "train").strip().lower()
    output_path = str(
        args.output_path
        or cfg.get("output_path")
        or f"outputs/preflight/preflight_{mode}.json"
    )
    strict = bool(cfg.get("strict", args.strict))

    if args.train_override:
        train_overrides = [str(v).strip() for v in args.train_override if str(v).strip()]
    else:
        train_overrides = _parse_name_list(cfg.get("train_overrides", []))
    report = run_preflight(
        PreflightConfig(
            mode=mode,
            dataset_dir=str(args.dataset_dir or cfg.get("dataset_dir") or ""),
            model_path=str(args.model_path or cfg.get("model_path") or ""),
            train_config=str(args.train_config or cfg.get("train_config") or ""),
            train_overrides=tuple(train_overrides),
            eval_config=str(args.eval_config or cfg.get("eval_config") or ""),
            benchmark_config=str(args.benchmark_config or cfg.get("benchmark_config") or ""),
            deployment_package_dir=str(args.package_dir or cfg.get("deployment_package_dir") or ""),
            require_dataset_qa=bool(cfg.get("require_dataset_qa", args.require_dataset_qa)),
            dataset_qa_report_path=str(args.dataset_qa_report_path or cfg.get("dataset_qa_report_path") or ""),
            verify_pretrained_sha256=bool(cfg.get("verify_sha256", args.verify_sha256)),
            output_path=output_path,
        )
    )
    print(f"preflight mode: {report.mode}")
    print(f"preflight ok: {report.ok}")
    print(f"issues: errors={report.error_count} warnings={report.warning_count} info={report.info_count}")
    print(f"report: {output_path}")
    if strict and not report.ok:
        return 2
    return 0


def _deploy_package(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    model_path = str(args.model_path or cfg.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("model path is required (--model-path or config:model_path)")
    if args.extra_path:
        extra_paths = tuple(str(v).strip() for v in args.extra_path if str(v).strip())
    else:
        extra_paths = tuple(_parse_name_list(cfg.get("extra_paths", [])))

    result = create_deployment_package(
        DeploymentPackageConfig(
            model_path=model_path,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments"),
            package_name=str(args.package_name or cfg.get("package_name") or ""),
            resolved_config_path=str(args.resolved_config_path or cfg.get("resolved_config_path") or ""),
            training_report_path=str(args.training_report_path or cfg.get("training_report_path") or ""),
            evaluation_report_path=str(args.evaluation_report_path or cfg.get("evaluation_report_path") or ""),
            extra_paths=extra_paths,
            notes=str(args.notes or cfg.get("notes") or ""),
        )
    )
    print(f"deployment package dir: {result.package_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"model artifact: {result.model_artifact_path}")
    print(f"copied files: {result.copied_files}")
    if result.warnings:
        print(f"warnings: {len(result.warnings)}")
    return 0


def _deploy_validate(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    package_dir = str(args.package_dir or cfg.get("package_dir") or "").strip()
    if not package_dir:
        raise ValueError("package directory is required (--package-dir or config:package_dir)")
    output_path = str(args.output_path or cfg.get("output_path") or "").strip()
    strict = bool(cfg.get("strict", args.strict))
    verify_sha256 = bool(cfg.get("verify_sha256", args.verify_sha256))

    report = validate_deployment_package(package_dir, verify_sha256=verify_sha256)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"deployment package valid: {report.ok}")
    print(f"errors: {len(report.errors)} warnings: {len(report.warnings)}")
    print(f"files: {report.file_count}")
    if output_path:
        print(f"report: {output_path}")
    if strict and not report.ok:
        return 2
    return 0


def _deploy_smoke(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    package_dir = str(args.package_dir or cfg.get("package_dir") or "").strip()
    if not package_dir:
        raise ValueError("package directory is required (--package-dir or config:package_dir)")
    image_path = str(args.image_path or cfg.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("smoke image path is required (--image-path or config:image_path)")

    result = run_deployment_smoke(
        DeploymentSmokeConfig(
            package_dir=package_dir,
            image_path=image_path,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments/smoke"),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
        )
    )
    print(f"smoke ok: {result.ok}")
    print(f"runtime_seconds: {result.runtime_seconds:.3f}")
    print(f"mask: {result.output_mask_path}")
    print(f"report: {result.report_path}")
    return 0


def _deploy_health(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    package_dir = str(args.package_dir or cfg.get("package_dir") or "").strip()
    if not package_dir:
        raise ValueError("package directory is required (--package-dir or config:package_dir)")

    if args.image_path:
        image_paths = tuple(str(v).strip() for v in args.image_path if str(v).strip())
    else:
        image_paths = tuple(_parse_name_list(cfg.get("image_paths", [])))
    patterns = _parse_name_list(cfg.get("glob_patterns", args.glob_patterns))
    if not patterns:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    result = run_runtime_health(
        RuntimeHealthConfig(
            package_dir=package_dir,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments/health"),
            image_paths=image_paths,
            image_dir=str(args.image_dir or cfg.get("image_dir") or ""),
            glob_patterns=tuple(patterns),
            max_workers=max(1, int(cfg.get("max_workers", args.max_workers))),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
        ),
        report_path=str(args.report_path or cfg.get("report_path") or ""),
    )
    print(f"runtime health ok: {result.ok}")
    print(f"images: total={result.total_images} failed={result.failed_images}")
    print(f"report: {result.report_path}")
    if bool(cfg.get("strict", args.strict)) and not result.ok:
        return 2
    return 0


def _deploy_worker_run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    package_dir = str(args.package_dir or cfg.get("package_dir") or "").strip()
    if not package_dir:
        raise ValueError("package directory is required (--package-dir or config:package_dir)")
    if args.image_path:
        image_paths = tuple(str(v).strip() for v in args.image_path if str(v).strip())
    else:
        image_paths = tuple(_parse_name_list(cfg.get("image_paths", [])))
    patterns = _parse_name_list(cfg.get("glob_patterns", args.glob_patterns))
    if not patterns:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    result = run_service_worker_batch(
        ServiceWorkerConfig(
            package_dir=package_dir,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments/service"),
            max_workers=max(1, int(cfg.get("max_workers", args.max_workers))),
            max_queue_size=max(1, int(cfg.get("max_queue_size", args.max_queue_size))),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
        ),
        image_paths=image_paths,
        image_dir=str(args.image_dir or cfg.get("image_dir") or ""),
        glob_patterns=tuple(patterns),
        await_completion=bool(cfg.get("await_completion", not args.no_await_completion)),
        timeout_seconds=(
            float(cfg.get("timeout_seconds", args.timeout_seconds))
            if float(cfg.get("timeout_seconds", args.timeout_seconds)) > 0
            else None
        ),
        report_path=str(args.report_path or cfg.get("report_path") or ""),
    )
    print(f"service worker report: {result.report_path}")
    print(
        "jobs: total={total} accepted={accepted} rejected={rejected} completed={completed} failed={failed}".format(
            total=result.total_submitted,
            accepted=result.accepted,
            rejected=result.rejected,
            completed=result.completed,
            failed=result.failed,
        )
    )
    if bool(cfg.get("strict", args.strict)) and (result.failed > 0 or result.rejected > 0):
        return 2
    return 0


def _deploy_canary_shadow(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    baseline_package_dir = str(args.baseline_package_dir or cfg.get("baseline_package_dir") or "").strip()
    candidate_package_dir = str(args.candidate_package_dir or cfg.get("candidate_package_dir") or "").strip()
    if not baseline_package_dir or not candidate_package_dir:
        raise ValueError(
            "both baseline and candidate package dirs are required "
            "(--baseline-package-dir/--candidate-package-dir or config fields)"
        )
    if args.image_path:
        image_paths = tuple(str(v).strip() for v in args.image_path if str(v).strip())
    else:
        image_paths = tuple(_parse_name_list(cfg.get("image_paths", [])))
    patterns = _parse_name_list(cfg.get("glob_patterns", args.glob_patterns))
    if not patterns:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    result = run_canary_shadow_compare(
        CanaryShadowConfig(
            baseline_package_dir=baseline_package_dir,
            candidate_package_dir=candidate_package_dir,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments/canary_shadow"),
            image_paths=image_paths,
            image_dir=str(args.image_dir or cfg.get("image_dir") or ""),
            glob_patterns=tuple(patterns),
            mask_dir=str(args.mask_dir or cfg.get("mask_dir") or ""),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
        ),
        report_path=str(args.report_path or cfg.get("report_path") or ""),
    )
    payload = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    mean_disagreement = float(payload.get("mean_disagreement_fraction", 0.0))
    mean_iou_gain = float(payload.get("mean_candidate_iou_gain", 0.0))
    print(f"canary-shadow report: {result.report_path}")
    print(f"images: total={result.total_images} failed={result.failed_images}")
    print(f"mean disagreement: {mean_disagreement:.6f}")
    print(f"mean candidate IoU gain: {mean_iou_gain:.6f}")
    max_disagree = float(cfg.get("max_mean_disagreement", args.max_mean_disagreement))
    min_iou_gain = float(cfg.get("min_mean_candidate_iou_gain", args.min_mean_candidate_iou_gain))
    strict = bool(cfg.get("strict", args.strict))
    threshold_fail = False
    if max_disagree >= 0 and mean_disagreement > max_disagree:
        threshold_fail = True
        print(
            "threshold failed: mean disagreement {:.6f} > allowed {:.6f}".format(
                mean_disagreement, max_disagree
            )
        )
    if min_iou_gain > -1 and mean_iou_gain < min_iou_gain:
        threshold_fail = True
        print(
            "threshold failed: mean candidate IoU gain {:.6f} < required {:.6f}".format(
                mean_iou_gain, min_iou_gain
            )
        )
    if strict and (result.failed_images > 0 or threshold_fail):
        return 2
    return 0


def _deploy_perf(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    package_dir = str(args.package_dir or cfg.get("package_dir") or "").strip()
    if not package_dir:
        raise ValueError("package directory is required (--package-dir or config:package_dir)")
    if args.image_path:
        image_paths = tuple(str(v).strip() for v in args.image_path if str(v).strip())
    else:
        image_paths = tuple(_parse_name_list(cfg.get("image_paths", [])))
    patterns = _parse_name_list(cfg.get("glob_patterns", args.glob_patterns))
    if not patterns:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    result = run_deployment_perf(
        DeploymentPerfConfig(
            package_dir=package_dir,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/deployments/perf"),
            image_paths=image_paths,
            image_dir=str(args.image_dir or cfg.get("image_dir") or ""),
            glob_patterns=tuple(patterns),
            warmup_runs=max(0, int(cfg.get("warmup_runs", args.warmup_runs))),
            repeat=max(1, int(cfg.get("repeat", args.repeat))),
            max_workers=max(1, int(cfg.get("max_workers", args.max_workers))),
            enable_gpu=bool(cfg.get("enable_gpu", args.enable_gpu)),
            device_policy=str(cfg.get("device_policy", args.device_policy)),
        ),
        report_path=str(args.report_path or cfg.get("report_path") or ""),
    )
    payload = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    p95 = float(payload.get("latency_ms_p95", 0.0))
    throughput = float(payload.get("throughput_images_per_second", 0.0))
    print(f"deploy perf report: {result.report_path}")
    print(f"samples csv: {result.csv_path}")
    print(
        "perf: total={total} failed={failed} p95_ms={p95:.3f} throughput_img_s={tps:.3f}".format(
            total=int(payload.get("total_requests", 0)),
            failed=int(payload.get("failed_requests", 0)),
            p95=p95,
            tps=throughput,
        )
    )
    strict = bool(cfg.get("strict", args.strict))
    max_p95_ms = float(cfg.get("max_p95_ms", args.max_p95_ms))
    min_tps = float(cfg.get("min_throughput_img_s", args.min_throughput_img_s))
    threshold_fail = False
    if max_p95_ms > 0 and p95 > max_p95_ms:
        threshold_fail = True
        print(f"threshold failed: p95_ms {p95:.3f} > allowed {max_p95_ms:.3f}")
    if min_tps > 0 and throughput < min_tps:
        threshold_fail = True
        print(f"threshold failed: throughput {throughput:.3f} < required {min_tps:.3f}")
    if strict and (not result.ok or threshold_fail):
        return 2
    return 0


def _promote_model(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    summary_path = str(args.summary_json or cfg.get("summary_json") or "").strip()
    if not summary_path:
        raise ValueError("summary JSON path is required (--summary-json or config:summary_json)")
    model_name = str(args.model_name or cfg.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("model name is required (--model-name or config:model_name)")
    registry_model_id = str(args.registry_model_id or cfg.get("registry_model_id") or model_name).strip()
    target_stage = str(args.target_stage or cfg.get("target_stage") or "candidate").strip()
    strict = bool(cfg.get("strict", args.strict))
    update_registry = bool(cfg.get("update_registry", args.update_registry))
    create_if_missing = bool(cfg.get("create_if_missing", args.create_if_missing))
    policy = load_promotion_policy(args.policy_config or cfg.get("policy_config"))
    output_path = str(
        args.output_path
        or cfg.get("output_path")
        or f"outputs/promotion/{registry_model_id}.decision.json"
    )

    decision = evaluate_and_promote_model(
        summary_json_path=summary_path,
        model_name=model_name,
        registry_model_id=registry_model_id,
        target_stage=target_stage,
        policy=policy,
        registry_path=str(args.registry_path or cfg.get("registry_path") or "frozen_checkpoints/model_registry.json"),
        update_registry=update_registry,
        create_if_missing=create_if_missing,
    )
    out = write_promotion_decision(decision, output_path=output_path)
    print(f"promotion passed: {decision.passed}")
    print(f"decision report: {out}")
    print(f"registry updated: {decision.registry_updated}")
    if decision.reasons:
        print(f"reasons: {len(decision.reasons)}")
    if strict and not decision.passed:
        return 2
    return 0


def _support_bundle(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    run_root = str(args.run_root or cfg.get("run_root") or "").strip()
    if not run_root:
        raise ValueError("run root is required (--run-root or config:run_root)")
    if args.include_path:
        include_paths = tuple(str(v).strip() for v in args.include_path if str(v).strip())
    else:
        include_paths = tuple(_parse_name_list(cfg.get("include_paths", [])))
    result = create_support_bundle(
        SupportBundleConfig(
            run_root=run_root,
            output_dir=str(args.output_dir or cfg.get("output_dir") or "outputs/support_bundles"),
            bundle_name=str(args.bundle_name or cfg.get("bundle_name") or ""),
            include_paths=include_paths,
        )
    )
    print(f"support bundle dir: {result.bundle_dir}")
    print(f"support bundle zip: {result.zip_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"included paths: {result.included_count}")
    if result.missing_paths:
        print(f"missing paths: {len(result.missing_paths)}")
    return 0


def _compatibility_matrix(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    output_path = str(
        args.output_path
        or cfg.get("output_path")
        or "outputs/support_bundles/compatibility_matrix.json"
    )
    out = write_compatibility_matrix(output_path)
    print(f"compatibility matrix: {out}")
    return 0


def _phase_gate(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config, args.set)
    phase_label = str(args.phase_label or cfg.get("phase_label") or "").strip()
    if not phase_label:
        raise ValueError("phase label is required (--phase-label or config:phase_label)")
    skip_tests = args.skip_tests if args.skip_tests is not None else bool(cfg.get("skip_tests", False))

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label=phase_label,
            run_tests=not bool(skip_tests),
            output_dir=str(cfg.get("output_dir", args.output_dir)),
            extra_notes=str(cfg.get("notes", args.notes or "")),
            strict=False,
            verify_release_policy=bool(cfg.get("verify_release_policy", args.verify_release_policy)),
            release_policy_path=str(cfg.get("release_policy_path", args.release_policy_path)),
            require_rollback_keywords=bool(
                cfg.get("require_rollback_keywords", args.require_rollback_keywords)
            ),
            rollback_keywords=tuple(
                _parse_name_list(cfg.get("rollback_keywords", args.rollback_keywords))
                or ["rollback", "patch", "release"]
            ),
            deployment_package_dirs=tuple(
                _parse_name_list(cfg.get("deployment_package_dirs", args.deployment_package_dirs))
            ),
            verify_deployment_sha256=bool(
                cfg.get("verify_deployment_sha256", args.verify_deployment_sha256)
            ),
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
            "smp_deeplabv3plus_resnet101",
            "smp_unetplusplus_resnet101",
            "smp_pspnet_resnet101",
            "smp_fpn_resnet101",
            "transunet_tiny",
            "segformer_mini",
            "hf_segformer_b0",
            "hf_segformer_b2",
            "hf_segformer_b5",
            "hf_upernet_swin_large",
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

    paired = sub.add_parser("prepare_dataset", help="Prepare paired JPG + RGB PNG masks into MaDo/Oxford layout")
    paired.add_argument("--config", type=str, help="YAML config path")
    paired.add_argument("--set", action="append", default=[], help="Override key=value")
    paired.add_argument("--input-dir", type=str, help="Input paired folder containing {stem}.jpg + {stem}_mask.png (or {stem}.png)")
    paired.add_argument("--output-root", type=str, help="Output dataset root")
    paired.add_argument("--style", type=str, default="mado", help="Comma-separated styles")
    paired.add_argument("--target-size", type=int, default=512)
    paired.add_argument("--crop-train", choices=["center", "random"], default="random")
    paired.add_argument("--crop-eval", choices=["center", "random"], default="center")
    paired.add_argument("--mask-r-min", type=int, default=200)
    paired.add_argument("--mask-g-max", type=int, default=60)
    paired.add_argument("--mask-b-max", type=int, default=60)
    paired.add_argument("--allow-red-dominance-fallback", action=argparse.BooleanOptionalAction, default=True)
    paired.add_argument("--mask-red-min-fallback", type=int, default=16)
    paired.add_argument("--mask-red-dominance-margin", type=int, default=8)
    paired.add_argument("--mask-red-dominance-ratio", type=float, default=1.5)
    paired.add_argument("--auto-otsu-for-noisy-grayscale", action=argparse.BooleanOptionalAction, default=True)
    paired.add_argument("--noisy-grayscale-low-max", type=int, default=5)
    paired.add_argument("--noisy-grayscale-high-min", type=int, default=200)
    paired.add_argument("--noisy-grayscale-min-extreme-ratio", type=float, default=0.98)
    paired.add_argument("--empty-mask-action", choices=["warn", "error"], default="warn")
    paired.add_argument("--debug", action="store_true", help="Write debug inspection panels and masks")
    paired.add_argument("--debug-limit", type=int, default=100, help="Max pairs processed in debug mode")
    paired.add_argument("--num-debug", type=int, default=8, help="Number of debug panels to export")
    paired.add_argument("--debug-show-plots", action="store_true", help="Show matplotlib debug panels during run")
    paired.add_argument("--debug-draw-contours", action="store_true", help="Draw contours on overlay debug output")
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--train-frac", type=float, default=0.8)
    paired.add_argument("--val-frac", type=float, default=0.1)
    paired.add_argument("--max-val-examples", type=int, default=None, help="Optional cap on validation split count")
    paired.add_argument("--max-test-examples", type=int, default=None, help="Optional cap on test split count")
    paired.add_argument("--dry-run", action="store_true")
    paired.set_defaults(handler=_prepare_dataset_paired)

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
        default=(
            "unet_binary,smp_unet_resnet18,smp_deeplabv3plus_resnet101,smp_unetplusplus_resnet101,"
            "smp_pspnet_resnet101,smp_fpn_resnet101,hf_segformer_b0,hf_segformer_b2,hf_segformer_b5,"
            "hf_upernet_swin_large,transunet_tiny,segformer_mini,torch_pixel"
        ),
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
        default=(
            "unet_binary,smp_unet_resnet18,smp_deeplabv3plus_resnet101,smp_unetplusplus_resnet101,"
            "smp_pspnet_resnet101,smp_fpn_resnet101,hf_segformer_b0,hf_segformer_b2,hf_segformer_b5,"
            "hf_upernet_swin_large,transunet_tiny,segformer_mini,torch_pixel"
        ),
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

    preflight = sub.add_parser("preflight", help="Run unified train/eval/benchmark/deploy preflight checks")
    preflight.add_argument("--config", type=str, help="YAML config path")
    preflight.add_argument("--set", action="append", default=[], help="Override key=value")
    preflight.add_argument("--mode", choices=["train", "eval", "benchmark", "deploy"], default="train")
    preflight.add_argument("--dataset-dir", type=str, help="Dataset directory for train/eval preflight")
    preflight.add_argument("--model-path", type=str, help="Model artifact path for eval preflight")
    preflight.add_argument("--train-config", type=str, help="Training config path for pretrained preflight")
    preflight.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Training config override key=value (repeatable)",
    )
    preflight.add_argument("--eval-config", type=str, help="Evaluation config path for benchmark checks")
    preflight.add_argument("--benchmark-config", type=str, help="Benchmark suite YAML path")
    preflight.add_argument("--package-dir", type=str, help="Deployment package directory for deploy mode")
    preflight.add_argument("--require-dataset-qa", action=argparse.BooleanOptionalAction, default=False)
    preflight.add_argument("--dataset-qa-report-path", type=str, default="")
    preflight.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    preflight.add_argument("--output-path", type=str, help="Preflight report JSON output path")
    preflight.add_argument("--strict", action="store_true", help="Exit non-zero when preflight fails")
    preflight.set_defaults(handler=_preflight)

    deploy_package = sub.add_parser("deploy-package", help="Create deployment package bundle with manifest")
    deploy_package.add_argument("--config", type=str, help="YAML config path")
    deploy_package.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_package.add_argument("--model-path", type=str, help="Model artifact path (.pth/.pt/.joblib)")
    deploy_package.add_argument("--output-dir", type=str, default="outputs/deployments")
    deploy_package.add_argument("--package-name", type=str, default="")
    deploy_package.add_argument("--resolved-config-path", type=str, default="")
    deploy_package.add_argument("--training-report-path", type=str, default="")
    deploy_package.add_argument("--evaluation-report-path", type=str, default="")
    deploy_package.add_argument("--extra-path", action="append", default=[], help="Extra file/dir to include")
    deploy_package.add_argument("--notes", type=str, default="")
    deploy_package.set_defaults(handler=_deploy_package)

    deploy_validate = sub.add_parser("deploy-validate", help="Validate deployment package manifest and checksums")
    deploy_validate.add_argument("--config", type=str, help="YAML config path")
    deploy_validate.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_validate.add_argument("--package-dir", type=str, help="Deployment package directory")
    deploy_validate.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    deploy_validate.add_argument("--output-path", type=str, default="", help="Validation report JSON output path")
    deploy_validate.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    deploy_validate.set_defaults(handler=_deploy_validate)

    deploy_smoke = sub.add_parser("deploy-smoke", help="Run one-image inference smoke test from deployment package")
    deploy_smoke.add_argument("--config", type=str, help="YAML config path")
    deploy_smoke.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_smoke.add_argument("--package-dir", type=str, help="Deployment package directory")
    deploy_smoke.add_argument("--image-path", type=str, help="Input image path for smoke inference")
    deploy_smoke.add_argument("--output-dir", type=str, default="outputs/deployments/smoke")
    deploy_smoke.add_argument("--enable-gpu", action="store_true")
    deploy_smoke.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    deploy_smoke.set_defaults(handler=_deploy_smoke)

    deploy_health = sub.add_parser(
        "deploy-health",
        help="Run deployment runtime health checks (supports concurrent batch queue mode)",
    )
    deploy_health.add_argument("--config", type=str, help="YAML config path")
    deploy_health.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_health.add_argument("--package-dir", type=str, help="Deployment package directory")
    deploy_health.add_argument("--image-path", action="append", default=[], help="Input image path (repeatable)")
    deploy_health.add_argument("--image-dir", type=str, default="", help="Directory to scan for input images")
    deploy_health.add_argument(
        "--glob-patterns",
        type=str,
        default="*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
        help="Comma-separated glob patterns used with --image-dir",
    )
    deploy_health.add_argument("--output-dir", type=str, default="outputs/deployments/health")
    deploy_health.add_argument("--report-path", type=str, default="", help="Optional explicit report JSON path")
    deploy_health.add_argument("--max-workers", type=int, default=1, help="Concurrent workers for queue-style processing")
    deploy_health.add_argument("--enable-gpu", action="store_true")
    deploy_health.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    deploy_health.add_argument("--strict", action="store_true", help="Exit non-zero when any health check fails")
    deploy_health.set_defaults(handler=_deploy_health)

    deploy_worker = sub.add_parser(
        "deploy-worker-run",
        help="Run queue-safe deployment worker batch (service-mode core for API/batch pipelines)",
    )
    deploy_worker.add_argument("--config", type=str, help="YAML config path")
    deploy_worker.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_worker.add_argument("--package-dir", type=str, help="Deployment package directory")
    deploy_worker.add_argument("--image-path", action="append", default=[], help="Input image path (repeatable)")
    deploy_worker.add_argument("--image-dir", type=str, default="", help="Directory to scan for input images")
    deploy_worker.add_argument(
        "--glob-patterns",
        type=str,
        default="*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
        help="Comma-separated glob patterns used with --image-dir",
    )
    deploy_worker.add_argument("--output-dir", type=str, default="outputs/deployments/service")
    deploy_worker.add_argument("--report-path", type=str, default="")
    deploy_worker.add_argument("--max-workers", type=int, default=2)
    deploy_worker.add_argument("--max-queue-size", type=int, default=32)
    deploy_worker.add_argument("--timeout-seconds", type=float, default=0.0)
    deploy_worker.add_argument(
        "--no-await-completion",
        action="store_true",
        help="Submit jobs and return queued/rejected statuses without waiting for completion",
    )
    deploy_worker.add_argument("--enable-gpu", action="store_true")
    deploy_worker.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    deploy_worker.add_argument("--strict", action="store_true")
    deploy_worker.set_defaults(handler=_deploy_worker_run)

    deploy_canary = sub.add_parser(
        "deploy-canary-shadow",
        help="Compare candidate vs baseline deployment packages on same images",
    )
    deploy_canary.add_argument("--config", type=str, help="YAML config path")
    deploy_canary.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_canary.add_argument("--baseline-package-dir", type=str, help="Baseline deployment package directory")
    deploy_canary.add_argument("--candidate-package-dir", type=str, help="Candidate deployment package directory")
    deploy_canary.add_argument("--image-path", action="append", default=[], help="Input image path (repeatable)")
    deploy_canary.add_argument("--image-dir", type=str, default="", help="Directory to scan for input images")
    deploy_canary.add_argument("--mask-dir", type=str, default="", help="Optional GT mask directory for quality gains")
    deploy_canary.add_argument(
        "--glob-patterns",
        type=str,
        default="*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
        help="Comma-separated glob patterns used with --image-dir",
    )
    deploy_canary.add_argument("--output-dir", type=str, default="outputs/deployments/canary_shadow")
    deploy_canary.add_argument("--report-path", type=str, default="")
    deploy_canary.add_argument("--max-mean-disagreement", type=float, default=-1.0)
    deploy_canary.add_argument("--min-mean-candidate-iou-gain", type=float, default=-1.0)
    deploy_canary.add_argument("--enable-gpu", action="store_true")
    deploy_canary.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    deploy_canary.add_argument("--strict", action="store_true")
    deploy_canary.set_defaults(handler=_deploy_canary_shadow)

    deploy_perf = sub.add_parser(
        "deploy-perf",
        help="Run deployment latency/throughput benchmark harness",
    )
    deploy_perf.add_argument("--config", type=str, help="YAML config path")
    deploy_perf.add_argument("--set", action="append", default=[], help="Override key=value")
    deploy_perf.add_argument("--package-dir", type=str, help="Deployment package directory")
    deploy_perf.add_argument("--image-path", action="append", default=[], help="Input image path (repeatable)")
    deploy_perf.add_argument("--image-dir", type=str, default="", help="Directory to scan for input images")
    deploy_perf.add_argument(
        "--glob-patterns",
        type=str,
        default="*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
        help="Comma-separated glob patterns used with --image-dir",
    )
    deploy_perf.add_argument("--output-dir", type=str, default="outputs/deployments/perf")
    deploy_perf.add_argument("--report-path", type=str, default="")
    deploy_perf.add_argument("--warmup-runs", type=int, default=1)
    deploy_perf.add_argument("--repeat", type=int, default=1)
    deploy_perf.add_argument("--max-workers", type=int, default=1)
    deploy_perf.add_argument("--max-p95-ms", type=float, default=0.0)
    deploy_perf.add_argument("--min-throughput-img-s", type=float, default=0.0)
    deploy_perf.add_argument("--enable-gpu", action="store_true")
    deploy_perf.add_argument("--device-policy", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    deploy_perf.add_argument("--strict", action="store_true")
    deploy_perf.set_defaults(handler=_deploy_perf)

    promote = sub.add_parser("promote-model", help="Evaluate benchmark summary against policy and update registry stage")
    promote.add_argument("--config", type=str, help="YAML config path")
    promote.add_argument("--set", action="append", default=[], help="Override key=value")
    promote.add_argument("--summary-json", type=str, help="Benchmark summary JSON path")
    promote.add_argument("--model-name", type=str, help="Model name as listed in summary aggregate")
    promote.add_argument("--registry-model-id", type=str, default="", help="Registry model_id to update")
    promote.add_argument("--target-stage", choices=["smoke", "candidate", "promoted", "deprecated"], default="candidate")
    promote.add_argument("--policy-config", type=str, default="", help="Promotion policy YAML path")
    promote.add_argument("--registry-path", type=str, default="frozen_checkpoints/model_registry.json")
    promote.add_argument("--update-registry", action=argparse.BooleanOptionalAction, default=True)
    promote.add_argument("--create-if-missing", action=argparse.BooleanOptionalAction, default=False)
    promote.add_argument("--output-path", type=str, default="")
    promote.add_argument("--strict", action="store_true", help="Exit non-zero when policy check fails")
    promote.set_defaults(handler=_promote_model)

    support = sub.add_parser("support-bundle", help="Collect support diagnostics bundle from a run root")
    support.add_argument("--config", type=str, help="YAML config path")
    support.add_argument("--set", action="append", default=[], help="Override key=value")
    support.add_argument("--run-root", type=str, help="Benchmark/run root directory")
    support.add_argument("--output-dir", type=str, default="outputs/support_bundles")
    support.add_argument("--bundle-name", type=str, default="")
    support.add_argument("--include-path", action="append", default=[], help="Extra file/dir to include")
    support.set_defaults(handler=_support_bundle)

    compat = sub.add_parser("compatibility-matrix", help="Write environment/runtime compatibility fingerprint JSON")
    compat.add_argument("--config", type=str, help="YAML config path")
    compat.add_argument("--set", action="append", default=[], help="Override key=value")
    compat.add_argument("--output-path", type=str, default="outputs/support_bundles/compatibility_matrix.json")
    compat.set_defaults(handler=_compatibility_matrix)

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
    gate.add_argument("--skip-tests", action=argparse.BooleanOptionalAction, default=None)
    gate.add_argument("--verify-release-policy", action=argparse.BooleanOptionalAction, default=False)
    gate.add_argument("--release-policy-path", type=str, default="docs/versioning_and_release_policy.md")
    gate.add_argument("--require-rollback-keywords", action=argparse.BooleanOptionalAction, default=False)
    gate.add_argument(
        "--rollback-keywords",
        type=str,
        default="rollback,patch,release",
        help="Comma-separated keywords required in release policy doc when enabled",
    )
    gate.add_argument(
        "--deployment-package-dirs",
        type=str,
        default="",
        help="Comma-separated deployment package directories to validate",
    )
    gate.add_argument("--verify-deployment-sha256", action=argparse.BooleanOptionalAction, default=True)
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
