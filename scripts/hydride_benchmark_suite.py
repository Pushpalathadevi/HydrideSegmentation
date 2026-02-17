"""Run hydride benchmark suites and produce consolidated comparison dashboards."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"suite config must be a mapping: {path}")
    return data


def _ensure_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_overrides(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    sep = "|" if "|" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


def _run_cmd(cmd: list[str], log_path: Path, *, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(cmd) + "\n[dry-run]\n", encoding="utf-8")
        return 0

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_text = "$ " + " ".join(cmd) + "\n\n" + proc.stdout + ("\n" if proc.stdout else "") + proc.stderr
    log_path.write_text(log_text, encoding="utf-8")
    return int(proc.returncode)


def _resolve_model_path(train_dir: Path) -> Path:
    candidates = [
        train_dir / "best_checkpoint.pt",
        train_dir / "last_checkpoint.pt",
        train_dir / "torch_pixel_classifier.pt",
        train_dir / "pixel_classifier.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no model artifact found in {train_dir}")


def _safe_float(v: object) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        by_model.setdefault(model, []).append(row)

    out: list[dict[str, Any]] = []
    for model, items in sorted(by_model.items()):
        def mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        pa = [float(v) for v in (_safe_float(i.get("pixel_accuracy")) for i in items) if v is not None]
        f1 = [float(v) for v in (_safe_float(i.get("macro_f1")) for i in items) if v is not None]
        mi = [float(v) for v in (_safe_float(i.get("mean_iou")) for i in items) if v is not None]
        rt = [float(v) for v in (_safe_float(i.get("runtime_seconds")) for i in items) if v is not None]
        out.append(
            {
                "model": model,
                "runs": len(items),
                "mean_pixel_accuracy": mean(pa),
                "mean_macro_f1": mean(f1),
                "mean_mean_iou": mean(mi),
                "mean_runtime_seconds": mean(rt),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_dashboard(path: Path, rows: list[dict[str, Any]], agg: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "<html><head><meta charset='utf-8'><title>Hydride Benchmark Dashboard</title></head><body>",
        "<h1>Hydride Benchmark Dashboard</h1>",
        "<h2>Model Summary</h2>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Model</th><th>Runs</th><th>Mean Pixel Acc</th><th>Mean Macro F1</th><th>Mean IoU</th><th>Mean Runtime (s)</th></tr>",
    ]
    for item in agg:
        lines.append(
            "<tr>"
            f"<td>{item['model']}</td>"
            f"<td>{item['runs']}</td>"
            f"<td>{item['mean_pixel_accuracy']:.6f}</td>"
            f"<td>{item['mean_macro_f1']:.6f}</td>"
            f"<td>{item['mean_mean_iou']:.6f}</td>"
            f"<td>{item['mean_runtime_seconds']:.2f}</td>"
            "</tr>"
        )
    lines.extend(
        [
            "</table>",
            "<h2>Run-Level Results</h2>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr><th>Model</th><th>Seed</th><th>Status</th><th>Backend</th><th>Architecture</th><th>Pixel Acc</th><th>Macro F1</th><th>Mean IoU</th><th>Runtime (s)</th><th>Hyperparameters</th><th>Train Config</th><th>Train Dir</th><th>Eval Report</th></tr>",
        ]
    )
    for row in rows:
        hparams = (
            f"ep={row.get('resolved_epochs','')}, "
            f"bs={row.get('resolved_batch_size','')}, "
            f"lr={row.get('resolved_learning_rate','')}, "
            f"wd={row.get('resolved_weight_decay','')}"
        )
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('model','')))}</td>"
            f"<td>{html.escape(str(row.get('seed','')))}</td>"
            f"<td>{html.escape(str(row.get('status','')))}</td>"
            f"<td>{html.escape(str(row.get('resolved_backend') or row.get('backend','')))}</td>"
            f"<td>{html.escape(str(row.get('resolved_model_architecture','')))}</td>"
            f"<td>{_safe_float(row.get('pixel_accuracy')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('macro_f1')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('mean_iou')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('runtime_seconds')) or 0.0:.2f}</td>"
            f"<td>{html.escape(hparams)}</td>"
            f"<td>{html.escape(str(row.get('train_config','')))}</td>"
            f"<td>{html.escape(str(row.get('train_dir','')))}</td>"
            f"<td>{html.escape(str(row.get('eval_report','')))}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(cfg_path: Path, *, dry_run: bool, strict: bool, skip_train: bool, skip_eval: bool) -> int:
    cfg = _load_yaml(cfg_path)
    dataset_dir = str(cfg.get("dataset_dir", "")).strip()
    output_root = Path(str(cfg.get("output_root", "outputs/benchmarks/suite_runs")).strip())
    eval_config = str(cfg.get("eval_config", "configs/hydride/evaluate.hydride.yml")).strip()
    eval_split = str(cfg.get("eval_split", "test")).strip()
    python_exe = str(cfg.get("python_executable", sys.executable)).strip() or sys.executable
    seeds = [int(v) for v in _ensure_list(cfg.get("seeds", [42]))]
    experiments = _ensure_list(cfg.get("experiments", []))
    if not dataset_dir:
        raise ValueError("dataset_dir is required in suite config")
    if not experiments:
        raise ValueError("suite config must include experiments")

    repo_root = Path(__file__).resolve().parents[1]
    cli = repo_root / "scripts" / "microseg_cli.py"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures = 0

    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        model_name = str(exp.get("name", "")).strip() or "unnamed_model"
        train_config = str(exp.get("train_config", "")).strip()
        if not train_config:
            raise ValueError(f"experiment '{model_name}' missing train_config")
        train_overrides = _parse_overrides(exp.get("train_overrides"))
        eval_overrides = _parse_overrides(exp.get("eval_overrides"))

        for seed in seeds:
            run_tag = f"{model_name}_seed{seed}"
            train_dir = output_root / "runs" / run_tag
            eval_report = output_root / "eval" / f"{run_tag}_{eval_split}.json"
            logs_dir = output_root / "logs" / run_tag
            train_log = logs_dir / "train.log"
            eval_log = logs_dir / "eval.log"

            status = "ok"
            model_path = None
            if not skip_train:
                train_cmd = [
                    python_exe,
                    str(cli),
                    "train",
                    "--config",
                    train_config,
                    "--dataset-dir",
                    dataset_dir,
                    "--output-dir",
                    str(train_dir),
                    "--set",
                    f"seed={seed}",
                    "--no-auto-prepare-dataset",
                ]
                for item in train_overrides:
                    train_cmd.extend(["--set", item])
                rc = _run_cmd(train_cmd, train_log, dry_run=dry_run)
                if rc != 0:
                    status = f"train_failed({rc})"
                    failures += 1
            if status == "ok" and not skip_eval:
                try:
                    model_path = _resolve_model_path(train_dir)
                except Exception:
                    if dry_run:
                        model_path = train_dir / "best_checkpoint.pt"
                    else:
                        status = "model_missing"
                        failures += 1
                if status == "ok":
                    eval_cmd = [
                        python_exe,
                        str(cli),
                        "evaluate",
                        "--config",
                        eval_config,
                        "--dataset-dir",
                        dataset_dir,
                        "--model-path",
                        str(model_path),
                        "--split",
                        eval_split,
                        "--output-path",
                        str(eval_report),
                        "--no-auto-prepare-dataset",
                    ]
                    for item in eval_overrides:
                        eval_cmd.extend(["--set", item])
                    rc = _run_cmd(eval_cmd, eval_log, dry_run=dry_run)
                    if rc != 0:
                        status = f"eval_failed({rc})"
                        failures += 1

            eval_payload = _read_json(eval_report)
            metrics = eval_payload.get("metrics", {}) if isinstance(eval_payload, dict) else {}
            train_resolved = _read_json(train_dir / "resolved_config.json")

            row = {
                "model": model_name,
                "seed": seed,
                "status": status,
                "dataset_dir": dataset_dir,
                "train_config": train_config,
                "eval_config": eval_config,
                "train_dir": str(train_dir),
                "eval_report": str(eval_report),
                "pixel_accuracy": metrics.get("pixel_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "mean_iou": metrics.get("mean_iou"),
                "runtime_seconds": eval_payload.get("runtime_seconds") if isinstance(eval_payload, dict) else None,
                "backend": eval_payload.get("backend") if isinstance(eval_payload, dict) else "",
                "model_initialization": eval_payload.get("model_initialization") if isinstance(eval_payload, dict) else "",
                "resolved_backend": train_resolved.get("backend") if isinstance(train_resolved, dict) else "",
                "resolved_model_architecture": train_resolved.get("model_architecture")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_epochs": train_resolved.get("epochs") if isinstance(train_resolved, dict) else "",
                "resolved_batch_size": train_resolved.get("batch_size") if isinstance(train_resolved, dict) else "",
                "resolved_learning_rate": train_resolved.get("learning_rate") if isinstance(train_resolved, dict) else "",
                "resolved_weight_decay": train_resolved.get("weight_decay") if isinstance(train_resolved, dict) else "",
                "resolved_model_base_channels": train_resolved.get("model_base_channels")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_depth": train_resolved.get("transformer_depth")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_num_heads": train_resolved.get("transformer_num_heads")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_mlp_ratio": train_resolved.get("transformer_mlp_ratio")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_dropout": train_resolved.get("transformer_dropout")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_segformer_patch_size": train_resolved.get("segformer_patch_size")
                if isinstance(train_resolved, dict)
                else "",
                "train_overrides": "|".join(train_overrides),
                "eval_overrides": "|".join(eval_overrides),
            }
            rows.append(row)

    agg = _aggregate(rows)
    summary_json = output_root / "benchmark_summary.json"
    summary_csv = output_root / "benchmark_summary.csv"
    aggregate_csv = output_root / "benchmark_aggregate.csv"
    dashboard_html = output_root / "benchmark_dashboard.html"

    summary_payload = {
        "schema_version": "microseg.hydride_benchmark_suite.v1",
        "config_path": str(cfg_path),
        "dataset_dir": dataset_dir,
        "output_root": str(output_root),
        "eval_split": eval_split,
        "run_count": len(rows),
        "failure_count": failures,
        "rows": rows,
        "aggregate": agg,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_csv(
        summary_csv,
        rows,
        [
            "model",
            "seed",
            "status",
            "backend",
            "model_initialization",
            "pixel_accuracy",
            "macro_f1",
            "mean_iou",
            "runtime_seconds",
            "train_config",
            "eval_config",
            "train_dir",
            "eval_report",
            "resolved_backend",
            "resolved_model_architecture",
            "resolved_epochs",
            "resolved_batch_size",
            "resolved_learning_rate",
            "resolved_weight_decay",
            "resolved_model_base_channels",
            "resolved_transformer_depth",
            "resolved_transformer_num_heads",
            "resolved_transformer_mlp_ratio",
            "resolved_transformer_dropout",
            "resolved_segformer_patch_size",
            "train_overrides",
            "eval_overrides",
        ],
    )
    _write_csv(
        aggregate_csv,
        agg,
        [
            "model",
            "runs",
            "mean_pixel_accuracy",
            "mean_macro_f1",
            "mean_mean_iou",
            "mean_runtime_seconds",
        ],
    )
    _write_dashboard(dashboard_html, rows, agg)

    print(f"suite summary json: {summary_json}")
    print(f"suite summary csv: {summary_csv}")
    print(f"suite aggregate csv: {aggregate_csv}")
    print(f"suite dashboard html: {dashboard_html}")
    print(f"runs: {len(rows)} failures: {failures}")

    if strict and failures > 0:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydride benchmark suite runner and summarizer")
    parser.add_argument("--config", required=True, help="Path to benchmark suite YAML")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands but do not execute train/eval")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any run fails")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage")
    args = parser.parse_args()
    rc = run_suite(
        Path(args.config),
        dry_run=bool(args.dry_run),
        strict=bool(args.strict),
        skip_train=bool(args.skip_train),
        skip_eval=bool(args.skip_eval),
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
