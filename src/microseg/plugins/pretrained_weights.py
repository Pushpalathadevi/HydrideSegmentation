"""Local pretrained-weight registry helpers for air-gapped transfer learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .frozen_checkpoints import find_repo_root


PRETRAINED_REGISTRY_SCHEMA = "microseg.pretrained_weights_registry.v1"


@dataclass(frozen=True)
class PretrainedWeightRecord:
    """Metadata record for a local pretrained-weight bundle."""

    model_id: str
    architecture: str
    framework: str
    source: str
    source_revision: str
    bundle_dir: str
    weights_path: str
    weights_format: str
    metadata_path: str = "metadata.json"
    source_url: str = ""
    license: str = ""
    citation_key: str = ""
    citation: str = ""
    citation_url: str = ""
    notes: str = ""
    files: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PretrainedWeightRecord:
        required = (
            "model_id",
            "architecture",
            "framework",
            "source",
            "source_revision",
            "bundle_dir",
            "weights_path",
            "weights_format",
        )
        missing = [name for name in required if not str(payload.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing required pretrained model fields: {', '.join(sorted(missing))}")

        allowed = {f.name for f in fields(cls)}
        obj = {k: v for k, v in dict(payload).items() if k in allowed}
        files = obj.get("files", [])
        if not isinstance(files, list):
            raise ValueError("pretrained record 'files' must be a list when provided")
        obj["files"] = tuple(dict(item) for item in files)
        return cls(**obj)


@dataclass
class PretrainedRegistryValidationReport:
    """Validation report for pretrained-weight registry and local artifacts."""

    schema_version: str
    registry_path: str
    verify_sha256: bool
    ok: bool
    model_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pretrained_weights_root(start: Path | None = None) -> Path:
    """Return canonical local pretrained-weight root path."""

    return find_repo_root(start) / "pre_trained_weights"


def pretrained_registry_path(start: Path | None = None) -> Path:
    """Return canonical pretrained registry JSON path."""

    return pretrained_weights_root(start) / "registry.json"


def _resolve_local_path(path_value: str | Path, *, base: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pretrained_weight_records(path: str | Path | None = None) -> list[PretrainedWeightRecord]:
    """Load pretrained-weight registry records."""

    reg_path = Path(path) if path else pretrained_registry_path()
    if not reg_path.exists():
        return []

    payload = json.loads(reg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pretrained registry payload must be a JSON object")
    if payload.get("schema_version") != PRETRAINED_REGISTRY_SCHEMA:
        raise ValueError(
            f"unsupported pretrained registry schema: {payload.get('schema_version')!r}; "
            f"expected {PRETRAINED_REGISTRY_SCHEMA!r}"
        )

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("pretrained registry 'models' field must be a list")
    return [PretrainedWeightRecord.from_dict(dict(item)) for item in models]


def pretrained_weight_map(path: str | Path | None = None) -> dict[str, PretrainedWeightRecord]:
    """Load pretrained records keyed by model_id."""

    out: dict[str, PretrainedWeightRecord] = {}
    for rec in load_pretrained_weight_records(path):
        out[rec.model_id] = rec
    return out


def resolve_pretrained_record(
    *,
    model_id: str,
    registry_path: str | Path | None = None,
) -> PretrainedWeightRecord:
    """Resolve a pretrained record by model identifier."""

    mid = str(model_id).strip()
    if not mid:
        raise ValueError("pretrained model_id cannot be empty")
    records = pretrained_weight_map(registry_path)
    if mid not in records:
        raise KeyError(f"pretrained model_id not found in registry: {mid}")
    return records[mid]


def resolve_bundle_paths(
    record: PretrainedWeightRecord,
    *,
    registry_path: str | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Resolve absolute bundle, weights, and metadata paths for one record."""

    reg_path = Path(registry_path).resolve() if registry_path else pretrained_registry_path()
    base = reg_path.parent if reg_path.name else reg_path

    bundle_raw = Path(record.bundle_dir)
    bundle = _resolve_local_path(bundle_raw, base=base)
    if not bundle_raw.is_absolute() and bundle_raw.parts and bundle_raw.parts[0] == base.name:
        candidate = (base.parent / bundle_raw).resolve()
        if candidate.exists():
            bundle = candidate
    weights = _resolve_local_path(record.weights_path, base=bundle)
    metadata_raw = str(record.metadata_path).strip()
    metadata = _resolve_local_path(metadata_raw, base=bundle) if metadata_raw else None
    return bundle, weights, metadata


def validate_pretrained_registry(
    path: str | Path | None = None,
    *,
    verify_sha256: bool = True,
) -> PretrainedRegistryValidationReport:
    """Validate pretrained registry metadata and local artifact presence."""

    reg_path = Path(path) if path else pretrained_registry_path()
    report = PretrainedRegistryValidationReport(
        schema_version="microseg.pretrained_registry_validation.v1",
        registry_path=str(reg_path),
        verify_sha256=bool(verify_sha256),
        ok=False,
    )
    if not reg_path.exists():
        report.errors.append(f"pretrained registry file does not exist: {reg_path}")
        return report

    try:
        payload = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.errors.append(f"pretrained registry is not valid JSON: {exc}")
        return report

    if not isinstance(payload, dict):
        report.errors.append("pretrained registry root must be an object")
        return report
    if payload.get("schema_version") != PRETRAINED_REGISTRY_SCHEMA:
        report.errors.append(
            f"unsupported schema_version: {payload.get('schema_version')!r}; expected {PRETRAINED_REGISTRY_SCHEMA!r}"
        )

    models = payload.get("models")
    if not isinstance(models, list):
        report.errors.append("pretrained registry 'models' must be a list")
        return report

    seen_ids: set[str] = set()
    report.model_count = len(models)
    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            report.errors.append(f"models[{idx}] must be an object")
            continue
        try:
            rec = PretrainedWeightRecord.from_dict(dict(item))
        except Exception as exc:
            report.errors.append(f"models[{idx}] invalid metadata: {exc}")
            continue

        if rec.model_id in seen_ids:
            report.errors.append(f"duplicate pretrained model_id: {rec.model_id}")
        seen_ids.add(rec.model_id)

        bundle, weights, metadata = resolve_bundle_paths(rec, registry_path=reg_path)
        if not bundle.exists():
            report.errors.append(f"model '{rec.model_id}' bundle_dir not found: {bundle}")
            continue
        if rec.weights_format == "hf_model_dir":
            if not weights.exists() or not weights.is_dir():
                report.errors.append(
                    f"model '{rec.model_id}' weights_path must be an existing directory for hf_model_dir: {weights}"
                )
        else:
            if not weights.exists() or not weights.is_file():
                report.errors.append(
                    f"model '{rec.model_id}' weights_path must be an existing file for {rec.weights_format}: {weights}"
                )
        if metadata is not None and not metadata.exists():
            report.warnings.append(f"model '{rec.model_id}' metadata_path missing: {metadata}")
        if not str(rec.source_url).strip():
            report.warnings.append(f"model '{rec.model_id}' source_url is empty")
        if not str(rec.license).strip():
            report.warnings.append(f"model '{rec.model_id}' license field is empty")
        if not (str(rec.citation_key).strip() or str(rec.citation).strip()):
            report.warnings.append(
                f"model '{rec.model_id}' citation metadata is empty (citation_key/citation)"
            )

        if bool(verify_sha256):
            for file_idx, file_item in enumerate(rec.files):
                rel = str(file_item.get("path", "")).strip()
                expected = str(file_item.get("sha256", "")).strip().lower()
                if not rel:
                    report.errors.append(
                        f"model '{rec.model_id}' files[{file_idx}] missing path for sha256 validation"
                    )
                    continue
                if not expected:
                    report.errors.append(
                        f"model '{rec.model_id}' files[{file_idx}] missing sha256 for validation"
                    )
                    continue
                fpath = _resolve_local_path(rel, base=bundle)
                if not fpath.exists() or not fpath.is_file():
                    report.errors.append(f"model '{rec.model_id}' listed file missing: {fpath}")
                    continue
                actual = _sha256(fpath).lower()
                if actual != expected:
                    report.errors.append(
                        f"model '{rec.model_id}' checksum mismatch for {rel}: expected {expected}, got {actual}"
                    )

    report.ok = len(report.errors) == 0
    return report


def write_pretrained_validation_report(
    report: PretrainedRegistryValidationReport,
    output_path: str | Path,
) -> Path:
    """Write pretrained-registry validation report to disk."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out
