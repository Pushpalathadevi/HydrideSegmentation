"""Manifest writing utilities for dataset preparation runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManifestWriter:
    """Write deterministic manifest JSON files."""

    def write(self, *, output_root: Path, config: dict[str, Any], input_dir: Path, records: list[dict[str, Any]], split_counts: dict[str, int], warnings: list[str]) -> Path:
        manifest = {
            "schema_version": "microseg.data_preparation_manifest.v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_version": "dataset-prep-v1",
            "git_commit": self._git_commit(),
            "input_dir": str(input_dir),
            "output_dir": str(output_root),
            "resolved_config": config,
            "split_counts": split_counts,
            "records": records,
            "warnings_summary": sorted(warnings),
        }
        path = output_root / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _git_commit() -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"
