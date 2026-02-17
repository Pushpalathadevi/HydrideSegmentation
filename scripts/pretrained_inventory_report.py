"""Generate pretrained-bundle inventory report for reporting/manuscript traceability."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.plugins import (  # noqa: E402
    load_pretrained_weight_records,
    resolve_bundle_paths,
    validate_pretrained_registry,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_table_row(item: dict[str, object]) -> str:
    return (
        "| {model_id} | {architecture} | {framework} | {source} | {source_revision} | {license} | {citation_key} |"
    ).format(
        model_id=str(item.get("model_id", "")),
        architecture=str(item.get("architecture", "")),
        framework=str(item.get("framework", "")),
        source=str(item.get("source", "")),
        source_revision=str(item.get("source_revision", "")),
        license=str(item.get("license", "")),
        citation_key=str(item.get("citation_key", "")),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pretrained inventory report")
    parser.add_argument("--registry-path", type=str, default="pre_trained_weights/registry.json")
    parser.add_argument("--output-path", type=str, default="outputs/pretrained_weights/inventory_report.json")
    parser.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    registry_path = Path(args.registry_path)
    output_path = Path(args.output_path)

    records = load_pretrained_weight_records(registry_path)
    validation = validate_pretrained_registry(registry_path, verify_sha256=bool(args.verify_sha256))

    model_rows: list[dict[str, object]] = []
    for rec in records:
        bundle, weights, metadata = resolve_bundle_paths(rec, registry_path=registry_path)
        size_bytes = 0
        if rec.files:
            size_bytes = int(sum(int(item.get("size_bytes", 0)) for item in rec.files))
        elif bundle.exists() and bundle.is_dir():
            size_bytes = int(sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()))

        model_rows.append(
            {
                "model_id": rec.model_id,
                "architecture": rec.architecture,
                "framework": rec.framework,
                "source": rec.source,
                "source_url": rec.source_url,
                "source_revision": rec.source_revision,
                "weights_format": rec.weights_format,
                "bundle_dir": str(bundle),
                "weights_path": str(weights),
                "metadata_path": str(metadata) if metadata else "",
                "license": rec.license,
                "citation_key": rec.citation_key,
                "citation": rec.citation,
                "citation_url": rec.citation_url,
                "files_count": len(rec.files),
                "size_bytes": size_bytes,
            }
        )

    payload = {
        "schema_version": "microseg.pretrained_inventory_report.v1",
        "created_utc": _utc_now(),
        "registry_path": str(registry_path),
        "verify_sha256": bool(args.verify_sha256),
        "validation": {
            "ok": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "model_count": validation.model_count,
        },
        "models": model_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    lines = [
        "# Pretrained Inventory Report",
        "",
        f"- registry: `{registry_path}`",
        f"- validation_ok: `{validation.ok}`",
        f"- errors: `{len(validation.errors)}`",
        f"- warnings: `{len(validation.warnings)}`",
        "",
        "| model_id | architecture | framework | source | source_revision | license | citation_key |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in model_rows:
        lines.append(_as_table_row(item))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"inventory report: {output_path}")
    print(f"markdown: {md_path}")
    print(f"models: {len(model_rows)}")
    print(f"validation_ok: {validation.ok}")


if __name__ == "__main__":
    main()
