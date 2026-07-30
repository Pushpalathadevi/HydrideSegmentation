"""Install, verify, and remove locally supplied trained checkpoints.

Trained checkpoint binaries are never tracked in git, so a fresh clone or a
freshly deployed air-gapped machine has no ``.pt``/``.pth`` files at all. This
module turns "make my checkpoint usable in the GUI and CLI" into a single
mechanical operation:

1. introspect the checkpoint, which is self-describing about its architecture,
2. copy it into the lifecycle folder matching its artifact stage,
3. prove it loads and runs one forward pass,
4. record it in the local registry overlay ``model_registry.local.json``,
5. validate the merged registry, rolling back completely if anything fails.

Discovery is already registry-driven end to end, so a successfully installed
model appears in the GUI selector and the CLI without any code change.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frozen_checkpoints import (
    REGISTRY_SCHEMA,
    FrozenCheckpointRecord,
    find_repo_root,
    load_frozen_checkpoint_records,
    local_registry_path,
    registry_path,
)
from .registry_validation import validate_frozen_registry

INSTALL_REPORT_SCHEMA = "microseg.model_install_report.v1"

#: Artifact stages that accept an installed checkpoint, mapped to their
#: repository-relative lifecycle folder. Mirrors the validator's stage rules.
STAGE_DIRECTORIES: dict[str, str] = {
    "smoke": "frozen_checkpoints/smoke",
    "candidate": "frozen_checkpoints/candidates",
    "promoted": "frozen_checkpoints/promoted",
}

#: Model identifiers that may never be created or replaced by an install.
RESERVED_MODEL_IDS: frozenset[str] = frozenset({"hydride_conventional"})

#: Checkpoint file extensions accepted by the installer.
SUPPORTED_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".pt", ".pth", ".ckpt")

#: Class map applied when the caller does not supply one.
DEFAULT_CLASSES: tuple[dict[str, Any], ...] = (
    {"index": 0, "name": "background", "color_hex": "#000000"},
    {"index": 1, "name": "hydride", "color_hex": "#00FFFF"},
)

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_VERIFICATION_IMAGE_SIZE = 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory.

    Parameters
    ----------
    path:
        File to digest.
    chunk_size:
        Read block size in bytes.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointIntrospection:
    """Metadata recovered by reading a trained checkpoint file."""

    checkpoint_path: str
    file_size_bytes: int
    file_sha256: str
    schema_version: str = ""
    architecture: str = ""
    backend: str = ""
    model_initialization: str = ""
    created_utc: str = ""
    epoch: int | None = None
    best_val_loss: float | None = None
    parameter_count: int | None = None
    input_dimensions: str = "H x W x 3"
    input_size: str = "variable"
    framework: str = "pytorch"
    training_config: dict[str, Any] = field(default_factory=dict)
    architecture_supported: bool = False
    warnings: tuple[str, ...] = ()

    def suggested_model_id(self) -> str:
        """Return a filesystem-derived identifier suggestion for this checkpoint."""

        stem = Path(self.checkpoint_path).stem.strip().lower()
        slug = re.sub(r"[^a-z0-9_.-]+", "_", stem).strip("_.-")
        return slug or "installed_model"


@dataclass(frozen=True)
class ModelInstallRequest:
    """User-supplied parameters for one checkpoint installation."""

    checkpoint_path: str
    model_id: str
    model_nickname: str = ""
    artifact_stage: str = "candidate"
    architecture: str = ""
    framework: str = ""
    input_size: str = ""
    input_dimensions: str = ""
    application_remarks: str = ""
    short_description: str = ""
    detailed_description: str = ""
    source_run_manifest: str = ""
    quality_report_path: str = ""
    classes: tuple[dict[str, Any], ...] = DEFAULT_CLASSES
    verify_forward_pass: bool = True
    overwrite: bool = False


@dataclass
class ModelInstallResult:
    """Outcome of one checkpoint installation attempt."""

    schema_version: str
    ok: bool
    created_utc: str
    model_id: str
    model_nickname: str = ""
    architecture: str = ""
    artifact_stage: str = ""
    checkpoint_path_hint: str = ""
    installed_checkpoint_path: str = ""
    registry_path: str = ""
    file_sha256: str = ""
    file_size_bytes: int = 0
    verification: dict[str, Any] = field(default_factory=dict)
    introspection: dict[str, Any] = field(default_factory=dict)
    registry_entry: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelUninstallResult:
    """Outcome of removing one locally installed model."""

    schema_version: str
    ok: bool
    created_utc: str
    model_id: str
    registry_path: str = ""
    removed_registry_entry: bool = False
    removed_checkpoint_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InstalledModelStatus:
    """Availability of one registry model, for catalog listings and GUI tables."""

    model_id: str
    model_nickname: str
    model_type: str
    artifact_stage: str
    checkpoint_path_hint: str
    resolved_checkpoint_path: str
    status: str
    message: str
    locally_installed: bool
    file_size_bytes: int | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"registry payload must be a JSON object: {path}")
    return payload


def _repo_relative_posix(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"path is outside the repository root and cannot be registered: {resolved}"
        ) from exc


def _supported_architectures() -> tuple[str, ...]:
    from src.microseg.inference.trained_model_loader import supported_trainable_architectures

    return supported_trainable_architectures()


def inspect_checkpoint(checkpoint_path: str | Path) -> CheckpointIntrospection:
    """Recover architecture and provenance metadata from a checkpoint file.

    Checkpoints written by this repository embed ``model_architecture``,
    ``backend`` and the resolved training ``config``, which is everything the
    runtime loader needs to rebuild the network. Reading them here is what lets
    an install avoid asking the user for architecture tokens.

    Parameters
    ----------
    checkpoint_path:
        Path of a ``.pt``/``.pth``/``.ckpt`` file.

    Returns
    -------
    CheckpointIntrospection
        Recovered metadata, including whether the architecture token is
        supported by the inference loader.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    ValueError
        If the payload is not a checkpoint mapping this repository understands.
    """

    import torch

    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {path}")

    warnings: list[str] = []
    if path.suffix.lower() not in SUPPORTED_CHECKPOINT_SUFFIXES:
        warnings.append(
            f"unexpected checkpoint extension {path.suffix!r}; "
            f"expected one of {', '.join(SUPPORTED_CHECKPOINT_SUFFIXES)}"
        )

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint payload must be a mapping produced by this repository's training loop: {path}"
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}

    schema_version = str(payload.get("schema_version", "")).strip()
    architecture = str(
        payload.get(
            "model_architecture",
            config.get("model_architecture", config.get("backend", "")),
        )
    ).strip().lower()
    if schema_version == "microseg.torch_unet_binary.v1":
        architecture = "unet_binary"
    if not architecture:
        warnings.append(
            "checkpoint does not declare model_architecture; the architecture must be selected manually"
        )

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = payload.get("state_dict") if isinstance(payload.get("state_dict"), dict) else None
    parameter_count: int | None = None
    if isinstance(state_dict, dict):
        total = 0
        for tensor in state_dict.values():
            numel = getattr(tensor, "numel", None)
            if callable(numel):
                total += int(numel())
        parameter_count = total
    else:
        warnings.append("checkpoint has no model_state_dict; verification will fail")

    input_hw = config.get("input_hw")
    if isinstance(input_hw, (list, tuple)) and len(input_hw) == 2:
        input_size = f"{int(input_hw[0])}x{int(input_hw[1])}"
    else:
        input_size = "variable"

    supported = architecture in _supported_architectures()
    if architecture and not supported:
        warnings.append(
            f"architecture {architecture!r} is not supported by the inference loader; "
            f"supported tokens: {', '.join(_supported_architectures())}"
        )

    epoch_value = payload.get("epoch")
    loss_value = payload.get("best_val_loss")

    return CheckpointIntrospection(
        checkpoint_path=str(path.resolve()),
        file_size_bytes=int(path.stat().st_size),
        file_sha256=sha256_file(path),
        schema_version=schema_version,
        architecture=architecture,
        backend=str(payload.get("backend", config.get("backend_label", architecture))).strip().lower(),
        model_initialization=str(payload.get("model_initialization", "")).strip(),
        created_utc=str(payload.get("created_utc", "")).strip(),
        epoch=int(epoch_value) if isinstance(epoch_value, (int, float)) else None,
        best_val_loss=float(loss_value) if isinstance(loss_value, (int, float)) else None,
        parameter_count=parameter_count,
        input_size=input_size,
        training_config=dict(config),
        architecture_supported=bool(supported),
        warnings=tuple(warnings),
    )


def verify_checkpoint_runtime(
    checkpoint_path: str | Path,
    *,
    run_forward_pass: bool = True,
) -> dict[str, Any]:
    """Prove a checkpoint loads, and optionally that it runs one forward pass.

    Metadata alone cannot detect a state-dict that disagrees with the
    architecture it claims, so this performs a real CPU load and a synthetic
    inference call.

    Parameters
    ----------
    checkpoint_path:
        Path of the checkpoint to verify.
    run_forward_pass:
        When ``True``, run one synthetic forward pass and check the mask shape.

    Returns
    -------
    dict
        Verification outcome with ``ok``, ``loaded``, ``forward_pass``,
        ``architecture``, ``device`` and ``error`` keys.
    """

    import numpy as np

    from src.microseg.training.unet_binary import (
        load_unet_binary_model,
        predict_unet_binary_mask,
    )

    outcome: dict[str, Any] = {
        "ok": False,
        "loaded": False,
        "forward_pass": False,
        "architecture": "",
        "device": "",
        "error": "",
    }
    try:
        bundle = load_unet_binary_model(checkpoint_path, enable_gpu=False, device_policy="cpu")
    except Exception as exc:
        outcome["error"] = f"checkpoint failed to load: {exc}"
        return outcome

    outcome["loaded"] = True
    outcome["architecture"] = str(bundle.get("architecture", ""))
    outcome["device"] = str(bundle.get("device", "cpu"))

    if not run_forward_pass:
        outcome["ok"] = True
        return outcome

    size = _VERIFICATION_IMAGE_SIZE
    probe = np.zeros((size, size, 3), dtype=np.uint8)
    probe[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    try:
        mask = predict_unet_binary_mask(probe, bundle)
    except Exception as exc:
        outcome["error"] = f"checkpoint loaded but inference failed: {exc}"
        return outcome

    if tuple(mask.shape[:2]) != (size, size):
        outcome["error"] = (
            f"forward pass produced mask of shape {tuple(mask.shape)}; expected ({size}, {size})"
        )
        return outcome

    outcome["forward_pass"] = True
    outcome["ok"] = True
    return outcome


def _validate_request(
    request: ModelInstallRequest,
    *,
    existing_local_ids: set[str],
) -> list[str]:
    errors: list[str] = []

    model_id = str(request.model_id).strip()
    if not model_id:
        errors.append("model_id is required")
    elif not _MODEL_ID_PATTERN.match(model_id):
        errors.append(
            "model_id must start with a lowercase letter or digit and use only "
            "lowercase letters, digits, '_', '-' or '.'"
        )
    elif model_id in RESERVED_MODEL_IDS:
        errors.append(f"model_id {model_id!r} is reserved and cannot be installed")
    elif model_id in existing_local_ids and not request.overwrite:
        errors.append(
            f"model_id {model_id!r} is already installed locally; enable overwrite to replace it"
        )

    stage = str(request.artifact_stage).strip().lower()
    if stage not in STAGE_DIRECTORIES:
        errors.append(
            f"artifact_stage {request.artifact_stage!r} is not installable; "
            f"choose one of {', '.join(sorted(STAGE_DIRECTORIES))}"
        )

    if not str(request.model_nickname).strip():
        errors.append("model_nickname is required")

    if not isinstance(request.classes, (list, tuple)) or not request.classes:
        errors.append("at least one class definition is required")

    return errors


def _build_registry_entry(
    request: ModelInstallRequest,
    introspection: CheckpointIntrospection,
    *,
    architecture: str,
    checkpoint_hint: str,
    installed_sha256: str,
    installed_size: int,
) -> dict[str, Any]:
    detailed = str(request.detailed_description).strip()
    if not detailed:
        provenance = []
        if introspection.created_utc:
            provenance.append(f"trained {introspection.created_utc}")
        if introspection.epoch is not None:
            provenance.append(f"epoch {introspection.epoch}")
        if introspection.best_val_loss is not None:
            provenance.append(f"best validation loss {introspection.best_val_loss:.6f}")
        suffix = f" Source checkpoint: {', '.join(provenance)}." if provenance else ""
        detailed = (
            f"Locally installed {architecture} checkpoint registered through the model installer."
            f"{suffix}"
        )

    return {
        "model_id": str(request.model_id).strip(),
        "model_nickname": str(request.model_nickname).strip(),
        "model_type": architecture,
        "framework": str(request.framework).strip() or introspection.framework or "pytorch",
        "input_size": str(request.input_size).strip() or introspection.input_size or "variable",
        "input_dimensions": str(request.input_dimensions).strip()
        or introspection.input_dimensions
        or "H x W x 3",
        "checkpoint_path_hint": checkpoint_hint,
        "application_remarks": str(request.application_remarks).strip()
        or "Locally installed checkpoint for desktop and CLI inference.",
        "short_description": str(request.short_description).strip()
        or "Installed locally; confirm segmentation quality on a known image before routine use.",
        "detailed_description": detailed,
        "artifact_stage": str(request.artifact_stage).strip().lower(),
        "source_run_manifest": str(request.source_run_manifest).strip(),
        "quality_report_path": str(request.quality_report_path).strip(),
        "file_sha256": installed_sha256,
        "file_size_bytes": int(installed_size),
        "classes": [dict(item) for item in request.classes],
    }


def _write_overlay(
    overlay_path: Path,
    entry: dict[str, Any],
) -> None:
    payload = _read_json_object(overlay_path)
    if not payload:
        payload = {"schema_version": REGISTRY_SCHEMA, "models": []}
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError(
            f"local registry overlay has unsupported schema_version "
            f"{payload.get('schema_version')!r}; expected {REGISTRY_SCHEMA!r}"
        )
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"local registry overlay 'models' must be a list: {overlay_path}")

    model_id = str(entry["model_id"])
    retained = [
        item
        for item in models
        if not (isinstance(item, dict) and str(item.get("model_id", "")).strip() == model_id)
    ]
    retained.append(entry)
    payload["models"] = retained
    payload["updated_utc"] = _utc_now()
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def local_registry_entries(overlay_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the raw model entries recorded in the local registry overlay.

    Parameters
    ----------
    overlay_path:
        Optional explicit overlay path. Defaults to the repository overlay.

    Returns
    -------
    list of dict
        Overlay entries, empty when no local model has been installed.
    """

    path = Path(overlay_path) if overlay_path else local_registry_path()
    payload = _read_json_object(path)
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    return [dict(item) for item in models if isinstance(item, dict)]


def install_model(
    request: ModelInstallRequest,
    *,
    repo_root: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> ModelInstallResult:
    """Install a trained checkpoint so the GUI and CLI can discover it.

    The checkpoint is copied into the lifecycle folder for its artifact stage,
    verified by a real load, and recorded in the local registry overlay. Any
    failure rolls the filesystem and registry back to their previous state, so a
    failed install never leaves a half-registered model behind.

    Parameters
    ----------
    request:
        Installation parameters. Fields left empty are filled from checkpoint
        introspection.
    repo_root:
        Optional repository root override, primarily for tests.
    overlay_path:
        Optional local registry overlay path override, primarily for tests.

    Returns
    -------
    ModelInstallResult
        Structured outcome. ``ok`` is ``False`` when ``errors`` is non-empty.
    """

    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    overlay = Path(overlay_path) if overlay_path else local_registry_path(root)
    result = ModelInstallResult(
        schema_version=INSTALL_REPORT_SCHEMA,
        ok=False,
        created_utc=_utc_now(),
        model_id=str(request.model_id).strip(),
        model_nickname=str(request.model_nickname).strip(),
        artifact_stage=str(request.artifact_stage).strip().lower(),
        registry_path=str(overlay),
    )

    existing_local_ids = {
        str(item.get("model_id", "")).strip() for item in local_registry_entries(overlay)
    }
    result.errors.extend(_validate_request(request, existing_local_ids=existing_local_ids))

    try:
        introspection = inspect_checkpoint(request.checkpoint_path)
    except Exception as exc:
        result.errors.append(str(exc))
        return result

    result.introspection = asdict(introspection)
    result.warnings.extend(introspection.warnings)

    architecture = str(request.architecture).strip().lower() or introspection.architecture
    if not architecture:
        result.errors.append(
            "architecture could not be determined from the checkpoint; specify it explicitly"
        )
    elif architecture not in _supported_architectures():
        result.errors.append(
            f"architecture {architecture!r} is not supported by the inference loader; "
            f"supported tokens: {', '.join(_supported_architectures())}"
        )
    result.architecture = architecture

    if result.errors:
        return result

    stage_dir = root / STAGE_DIRECTORIES[result.artifact_stage] / result.model_id
    source = Path(introspection.checkpoint_path)
    destination = stage_dir / source.name
    destination_existed = destination.exists()
    created_stage_dir = not stage_dir.exists()

    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        checkpoint_hint = _repo_relative_posix(destination, repo_root=root)
    except Exception as exc:
        result.errors.append(f"failed to copy checkpoint into the repository: {exc}")
        return result

    def _rollback() -> None:
        if not destination_existed and destination.exists():
            destination.unlink(missing_ok=True)
        if created_stage_dir and stage_dir.exists() and not any(stage_dir.iterdir()):
            stage_dir.rmdir()

    verification = verify_checkpoint_runtime(
        destination,
        run_forward_pass=bool(request.verify_forward_pass),
    )
    result.verification = verification
    if not verification.get("ok", False):
        result.errors.append(str(verification.get("error", "checkpoint verification failed")))
        _rollback()
        return result

    loaded_architecture = str(verification.get("architecture", "")).strip().lower()
    if loaded_architecture and loaded_architecture != architecture:
        result.warnings.append(
            f"loader resolved architecture {loaded_architecture!r} instead of declared {architecture!r}; "
            "registering the loader-resolved value"
        )
        architecture = loaded_architecture
        result.architecture = architecture

    entry = _build_registry_entry(
        request,
        introspection,
        architecture=architecture,
        checkpoint_hint=checkpoint_hint,
        installed_sha256=sha256_file(destination),
        installed_size=int(destination.stat().st_size),
    )

    overlay_backup = overlay.read_text(encoding="utf-8") if overlay.exists() else None
    try:
        _write_overlay(overlay, entry)
    except Exception as exc:
        result.errors.append(f"failed to update the local registry overlay: {exc}")
        _rollback()
        return result

    validation = validate_frozen_registry(overlay)
    if not validation.ok:
        result.errors.extend(validation.errors)
        if overlay_backup is None:
            overlay.unlink(missing_ok=True)
        else:
            overlay.write_text(overlay_backup, encoding="utf-8")
        _rollback()
        return result
    result.warnings.extend(validation.warnings)

    result.ok = True
    result.registry_entry = entry
    result.checkpoint_path_hint = checkpoint_hint
    result.installed_checkpoint_path = str(destination)
    result.file_sha256 = str(entry["file_sha256"])
    result.file_size_bytes = int(entry["file_size_bytes"])
    return result


def uninstall_model(
    model_id: str,
    *,
    delete_checkpoint: bool = False,
    repo_root: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> ModelUninstallResult:
    """Remove a locally installed model from the registry overlay.

    Only overlay entries can be removed; the canonical registry tracked in git is
    never modified.

    Parameters
    ----------
    model_id:
        Identifier of the locally installed model.
    delete_checkpoint:
        When ``True``, also delete the installed checkpoint binary.
    repo_root:
        Optional repository root override, primarily for tests.
    overlay_path:
        Optional local registry overlay path override, primarily for tests.

    Returns
    -------
    ModelUninstallResult
        Structured outcome. ``ok`` is ``False`` when ``errors`` is non-empty.
    """

    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    overlay = Path(overlay_path) if overlay_path else local_registry_path(root)
    target = str(model_id).strip()
    result = ModelUninstallResult(
        schema_version=INSTALL_REPORT_SCHEMA,
        ok=False,
        created_utc=_utc_now(),
        model_id=target,
        registry_path=str(overlay),
    )

    entries = local_registry_entries(overlay)
    match = next((item for item in entries if str(item.get("model_id", "")).strip() == target), None)
    if match is None:
        result.errors.append(f"model_id {target!r} is not installed locally")
        return result

    retained = [item for item in entries if str(item.get("model_id", "")).strip() != target]
    payload = _read_json_object(overlay)
    payload["schema_version"] = REGISTRY_SCHEMA
    payload["models"] = retained
    payload["updated_utc"] = _utc_now()
    overlay.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result.removed_registry_entry = True

    if delete_checkpoint:
        hint = str(match.get("checkpoint_path_hint", "")).strip()
        candidate = Path(hint)
        if hint and not candidate.is_absolute():
            candidate = root / candidate
        if hint and candidate.exists() and candidate.is_file():
            candidate.unlink()
            result.removed_checkpoint_path = str(candidate)
            parent = candidate.parent
            if parent != root and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        else:
            result.warnings.append(f"checkpoint file was not found for deletion: {hint or '<empty hint>'}")

    result.ok = True
    return result


def _resolve_hint(hint: str, *, repo_root: Path) -> Path | None:
    text = str(hint).strip()
    if not text or text.lower().startswith("n/a"):
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def model_catalog_status(
    *,
    repo_root: str | Path | None = None,
    overlay_path: str | Path | None = None,
    records: list[FrozenCheckpointRecord] | None = None,
) -> list[InstalledModelStatus]:
    """Report availability for every model known to the merged registry.

    Parameters
    ----------
    repo_root:
        Optional repository root override, primarily for tests.
    overlay_path:
        Optional local registry overlay path override, primarily for tests.
    records:
        Optional pre-loaded registry records, primarily for tests.

    Returns
    -------
    list of InstalledModelStatus
        One entry per registered model, sorted by identifier. ``status`` is one
        of ``ready``, ``no_checkpoint_required``, ``checkpoint_missing`` or
        ``unsupported_architecture``.
    """

    try:
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
    except FileNotFoundError:
        root = Path.cwd()
    overlay = Path(overlay_path) if overlay_path else None
    local_ids = {
        str(item.get("model_id", "")).strip()
        for item in local_registry_entries(overlay)
    }

    if records is None:
        try:
            merged = {
                record.model_id: record
                for record in load_frozen_checkpoint_records(registry_path(root))
            }
        except Exception:
            merged = {}
        if overlay is not None:
            for item in local_registry_entries(overlay):
                try:
                    record = FrozenCheckpointRecord.from_dict(dict(item))
                except Exception:
                    continue
                merged[record.model_id] = record
        records = list(merged.values())

    supported = _supported_architectures()
    catalog: list[InstalledModelStatus] = []
    for record in records:
        resolved = _resolve_hint(record.checkpoint_path_hint, repo_root=root)
        architecture = str(record.model_type or "").strip().lower()
        size: int | None = None

        if resolved is None:
            status = "no_checkpoint_required"
            message = "Built-in pipeline; no checkpoint file is needed."
        elif not resolved.exists():
            status = "checkpoint_missing"
            message = (
                f"Checkpoint file not found at {record.checkpoint_path_hint}. "
                "Install a checkpoint for this entry to make it selectable."
            )
        elif architecture not in supported:
            status = "unsupported_architecture"
            message = (
                f"Architecture {architecture!r} is not supported by the inference loader; "
                f"supported tokens: {', '.join(supported)}."
            )
            size = int(resolved.stat().st_size)
        else:
            status = "ready"
            message = "Checkpoint present and architecture supported."
            size = int(resolved.stat().st_size)

        catalog.append(
            InstalledModelStatus(
                model_id=record.model_id,
                model_nickname=record.model_nickname,
                model_type=architecture,
                artifact_stage=record.artifact_stage,
                checkpoint_path_hint=record.checkpoint_path_hint,
                resolved_checkpoint_path=str(resolved) if resolved is not None else "",
                status=status,
                message=message,
                locally_installed=record.model_id in local_ids,
                file_size_bytes=size,
            )
        )
    return sorted(catalog, key=lambda item: item.model_id)


def write_install_report(
    result: ModelInstallResult | ModelUninstallResult,
    output_path: str | Path,
) -> Path:
    """Write an install or uninstall outcome to a JSON report file.

    Parameters
    ----------
    result:
        Outcome returned by :func:`install_model` or :func:`uninstall_model`.
    output_path:
        Destination JSON path; parent directories are created.

    Returns
    -------
    Path
        The written report path.
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out
