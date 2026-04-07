"""Generate a lab-meeting PPTX from benchmark summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark summary must be a JSON object: {path}")
    return payload


def _top_models(aggregate: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(
        aggregate,
        key=lambda item: (
            int(item.get("rank_quality", 10_000_000)),
            -_safe_float(item.get("mean_mean_iou")),
            _safe_float(item.get("mean_total_runtime_seconds"), default=1e18),
        ),
    )[:limit]


def _manifest_from_summary(summary: dict[str, Any], *, deck_title: str) -> dict[str, Any]:
    aggregate = summary.get("aggregate", [])
    rows = summary.get("rows", [])
    if not isinstance(aggregate, list):
        aggregate = []
    if not isinstance(rows, list):
        rows = []
    top = _top_models([item for item in aggregate if isinstance(item, dict)])
    dataset_dir = str(summary.get("dataset_dir", ""))
    eval_split = str(summary.get("eval_split", "test"))
    effective_seeds = summary.get("effective_seeds", [])
    model_names = [str(item.get("model", "")) for item in top if str(item.get("model", "")).strip()]

    quality_rows = [
        {
            "label": str(item.get("model", "")),
            "value": round(_safe_float(item.get("mean_mean_iou")), 4),
        }
        for item in top
    ]
    runtime_rows = [
        {
            "label": str(item.get("model", "")),
            "value": round(_safe_float(item.get("mean_total_runtime_seconds")), 2),
        }
        for item in top
    ]
    comparison_table = [
        [
            str(item.get("model", "")),
            f"{_safe_float(item.get('mean_mean_iou')):.4f}",
            f"{_safe_float(item.get('mean_macro_f1')):.4f}",
            f"{_safe_float(item.get('mean_foreground_dice')):.4f}",
            f"{_safe_float(item.get('mean_total_runtime_seconds')):.1f}",
            str(int(item.get("runs", 0))),
        ]
        for item in top
    ]
    best_model = top[0] if top else {}
    bottom_line = (
        f"Best quality model: {best_model.get('model', 'n/a')} | "
        f"mean IoU={_safe_float(best_model.get('mean_mean_iou')):.4f}"
    )
    return {
        "title": deck_title,
        "slides": [
            {
                "title": "Objective and Scope",
                "bullets": [
                    "Start from raw .oh5 phaseId data with one reproducible command.",
                    "Materialize one frozen dataset split for all compared models.",
                    "Benchmark the configured model suite on the same evaluation policy.",
                    "Publish comparison artifacts as summary JSON, dashboard HTML, and PPTX.",
                ],
                "bottom_line": "The workflow is optimized for repeatable model comparison, not ad hoc one-off training.",
                "content": {
                    "type": "text",
                    "title": "Workflow",
                    "paragraphs": [
                        "raw .oh5 -> extracted PNG pairs -> prepared train/val/test split -> QA -> benchmark suite -> summary dashboard -> lab-meeting PPTX"
                    ],
                },
            },
            {
                "title": "Methodology Overview",
                "bullets": [
                    f"Dataset root: {dataset_dir or 'n/a'}",
                    f"Compared models: {len(model_names)}",
                    f"Evaluation split: {eval_split}",
                    f"Seeds: {', '.join(str(v) for v in effective_seeds) if effective_seeds else 'n/a'}",
                ],
                "bottom_line": "All benchmarked models consume the same prepared dataset and split manifest.",
                "content": {
                    "type": "table",
                    "headers": ["Aspect", "Value"],
                    "rows": [
                        ["Prepared dataset", dataset_dir or "n/a"],
                        ["Effective seeds", ", ".join(str(v) for v in effective_seeds) or "n/a"],
                        ["Runs logged", str(len(rows))],
                        ["Top models", ", ".join(model_names) or "n/a"],
                    ],
                },
            },
            {
                "title": "Core Equations and Setup",
                "bullets": [
                    "Primary ranking metric: mean IoU across benchmark runs.",
                    "Secondary metrics: macro F1, foreground Dice, and total runtime.",
                    "Quality rank favors robust segmentation quality before speed.",
                    "Training and evaluation artifacts remain linked via report.json and summary.json.",
                ],
                "bottom_line": "Rankings stay objective because the suite aggregates multiple quantitative metrics from the same run set.",
                "content": {
                    "type": "table",
                    "headers": ["Metric", "Purpose"],
                    "rows": [
                        ["mean IoU", "Primary segmentation quality score"],
                        ["macro F1", "Class-balanced agreement check"],
                        ["foreground Dice", "Foreground overlap quality"],
                        ["total runtime", "Operational efficiency signal"],
                    ],
                },
            },
            {
                "title": "Results - Quality Leaderboard",
                "bullets": [
                    "Models are ordered by benchmark quality rank.",
                    "Bar length encodes mean IoU over the configured seeds.",
                    "Use this view to identify the shortlist for deeper review.",
                ],
                "bottom_line": bottom_line,
                "content": {
                    "type": "bar_chart",
                    "series_name": "Mean IoU",
                    "categories": [row["label"] for row in quality_rows],
                    "values": [row["value"] for row in quality_rows],
                },
            },
            {
                "title": "Results - Runtime Comparison",
                "bullets": [
                    "Lower total runtime is better when quality remains acceptable.",
                    "This slide helps separate viable lab models from deployment candidates.",
                    "Use alongside quality rank, not as a replacement for it.",
                ],
                "bottom_line": "Runtime should be interpreted jointly with quality and run-success rate.",
                "content": {
                    "type": "bar_chart",
                    "series_name": "Mean Total Runtime (s)",
                    "categories": [row["label"] for row in runtime_rows],
                    "values": [row["value"] for row in runtime_rows],
                },
            },
            {
                "title": "Results - Comparison Matrix",
                "bullets": [
                    "Compact side-by-side view of the main quality and runtime metrics.",
                    "Runs column shows how many completed runs contributed to each mean.",
                    "Use this table to select the next detailed error-analysis target.",
                ],
                "bottom_line": "The best next experiment is the model that balances mean IoU, stability, and runtime cost.",
                "content": {
                    "type": "table",
                    "headers": ["Model", "Mean IoU", "Macro F1", "Fg Dice", "Runtime (s)", "Runs"],
                    "rows": comparison_table,
                },
            },
            {
                "title": "Conclusions and Next Steps",
                "bullets": [
                    f"Current top-quality model: {best_model.get('model', 'n/a')}",
                    "Review the dashboard HTML and per-run artifacts for failure modes.",
                    "Promote only the models that hold up on test split and operator review.",
                    "Extend the phaseId mapping when moving from binary targeting to richer class sets.",
                ],
                "bottom_line": "This deck is generated directly from benchmark artifacts, so it can be regenerated after every sweep.",
                "content": {
                    "type": "text",
                    "title": "Next actions",
                    "paragraphs": [
                        "1. Inspect benchmark_dashboard.html for run-level details.",
                        "2. Review best model checkpoints and tracked validation panels.",
                        "3. Lock the winning config into the next experimental phase.",
                    ],
                },
            },
        ],
    }


def _ensure_node_dependencies(repo_root: Path) -> None:
    node_modules = repo_root / "node_modules" / "pptxgenjs"
    if node_modules.exists():
        return
    proc = subprocess.run(
        ["npm", "install", "--no-fund", "--no-audit"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"npm install failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark lab-meeting PPTX")
    parser.add_argument("--summary-json", required=True, help="Benchmark summary JSON path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--deck-title", required=True, help="Deck title")
    parser.add_argument("--node-executable", default="node", help="Node executable")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(Path(args.summary_json))
    manifest = _manifest_from_summary(summary, deck_title=str(args.deck_title))
    base_name = Path(args.summary_json).stem
    manifest_path = output_dir / f"{base_name}_lab_meeting_manifest.json"
    pptx_path = output_dir / f"{base_name}_lab_meeting.pptx"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _ensure_node_dependencies(repo_root)
    builder = repo_root / "scripts" / "build_benchmark_lab_meeting_ppt.js"
    proc = subprocess.run(
        [str(args.node_executable), str(builder), "--manifest", str(manifest_path), "--output", str(pptx_path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"PPTX build failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    print(f"manifest: {manifest_path}")
    print(f"pptx: {pptx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
