"""Configuration loading for the intranet web application.

Values come from a YAML file, then from ``MICROSEG_WEB_*`` environment
variables. Nothing is fetched from the network, and every default is safe for an
air-gapped host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from src.microseg.plugins import find_repo_root

#: Environment variable prefix for configuration overrides.
ENV_PREFIX = "MICROSEG_WEB_"

#: Repository-relative path of the packaged default configuration.
DEFAULT_CONFIG_RELATIVE_PATH = "configs/app/web_server.default.yml"


@dataclass
class SampleImage:
    """One example image offered to users who have nothing to upload."""

    path: str
    label: str

    @property
    def sample_id(self) -> str:
        """Return the stable identifier used in URLs for this sample."""

        return Path(self.path).stem


@dataclass
class WebServerConfig:
    """Resolved runtime configuration for the browser application."""

    site_name: str = "MicroSeg Hydride Segmentation"
    site_description: str = "Upload a micrograph and segment hydrides in your browser."
    maintainer: str = ""

    host: str = "0.0.0.0"
    port: int = 5005
    threads: int = 4
    request_timeout_seconds: int = 300

    max_upload_mb: int = 5
    max_long_side_px: int = 2048
    max_image_pixels: int = 40_000_000
    max_concurrent_jobs: int = 2
    max_retained_jobs: int = 32
    job_retention_seconds: int = 1800

    preload_on_startup: bool = True
    preload_in_background: bool = True
    preload_model_ids: tuple[str, ...] = ()
    default_model_id: str = "auto"
    enable_gpu: bool = False
    device_policy: str = "cpu"

    include_analysis: bool = True

    sample_images: tuple[SampleImage, ...] = ()

    #: Absolute path of the browsable image library, or "" when none is configured.
    #: The folder is deliberately not enumerated at startup so that images added
    #: on the server appear without restarting the application.
    library_dir: str = ""
    library_max_images: int = 50

    repo_root: str = ""
    source_path: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def max_upload_bytes(self) -> int:
        """Return the upload ceiling in bytes."""

        return int(self.max_upload_mb) * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this configuration."""

        payload = asdict(self)
        payload["sample_images"] = [asdict(item) for item in self.sample_images]
        payload["preload_model_ids"] = list(self.preload_model_ids)
        payload["warnings"] = list(self.warnings)
        return payload


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    node = payload.get(name)
    return node if isinstance(node, dict) else {}


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``MICROSEG_WEB_*`` environment variables onto a config mapping."""

    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower().split("__")
        if not path or not path[0]:
            continue
        if len(path) == 1:
            # Flat aliases such as MICROSEG_WEB_PORT map onto their natural section.
            flat_aliases = {
                "host": ("server", "host"),
                "port": ("server", "port"),
                "threads": ("server", "threads"),
                "name": ("site", "name"),
                "max_upload_mb": ("limits", "max_upload_mb"),
                "default_model_id": ("models", "default_model_id"),
                "preload_on_startup": ("models", "preload_on_startup"),
                "enable_gpu": ("models", "enable_gpu"),
                "device_policy": ("models", "device_policy"),
            }
            path = list(flat_aliases.get(path[0], ("site", path[0])))
        node = payload
        for part in path[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[path[-1]] = raw_value
    return payload


def default_config_path(repo_root: Path | None = None) -> Path:
    """Return the packaged default configuration path.

    Parameters
    ----------
    repo_root:
        Optional repository root override.

    Returns
    -------
    Path
        Path of ``configs/app/web_server.default.yml``.
    """

    root = Path(repo_root) if repo_root else find_repo_root()
    return root / DEFAULT_CONFIG_RELATIVE_PATH


def load_web_config(
    config_path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> WebServerConfig:
    """Load web server configuration from YAML plus environment overrides.

    Parameters
    ----------
    config_path:
        Optional YAML path. Defaults to the packaged configuration.
    repo_root:
        Optional repository root override, primarily for tests.

    Returns
    -------
    WebServerConfig
        Resolved configuration. Missing files fall back to built-in defaults and
        are reported through ``warnings`` rather than raising, so the server
        still starts on a partially configured host.
    """

    warnings: list[str] = []
    try:
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
    except FileNotFoundError:
        root = Path.cwd()
        warnings.append("repository root could not be located; using the current working directory")

    if config_path:
        path = Path(config_path)
    else:
        env_path = os.environ.get(f"{ENV_PREFIX}CONFIG", "").strip()
        path = Path(env_path) if env_path else default_config_path(root)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
            else:
                warnings.append(f"configuration file is not a mapping; using defaults: {path}")
        except Exception as exc:
            warnings.append(f"configuration file could not be parsed; using defaults: {exc}")
    else:
        warnings.append(f"configuration file not found; using built-in defaults: {path}")

    payload = _apply_env_overrides(payload)

    site = _section(payload, "site")
    server = _section(payload, "server")
    limits = _section(payload, "limits")
    models = _section(payload, "models")
    analysis = _section(payload, "analysis")
    demo = _section(payload, "demo")

    raw_samples = demo.get("sample_images", [])
    samples: list[SampleImage] = []
    if isinstance(raw_samples, list):
        for item in raw_samples:
            if not isinstance(item, dict):
                continue
            sample_path = str(item.get("path", "")).strip()
            if not sample_path:
                continue
            candidate = Path(sample_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.exists():
                warnings.append(f"example image is missing and will not be offered: {sample_path}")
                continue
            samples.append(SampleImage(path=sample_path, label=str(item.get("label", "")) or candidate.name))

    # The library folder is resolved but never listed here. Deployments drop new
    # micrographs into it while the server is running, so listing happens per
    # request instead. A missing folder is not a warning: falling back to the
    # configured example images is the documented behaviour.
    raw_library_dir = str(demo.get("library_dir", "test_library")).strip()
    library_dir = ""
    if raw_library_dir:
        library_candidate = Path(raw_library_dir)
        if not library_candidate.is_absolute():
            library_candidate = root / library_candidate
        library_dir = str(library_candidate)

    preload_ids = models.get("preload_model_ids", [])
    if isinstance(preload_ids, str):
        preload_ids = [part.strip() for part in preload_ids.split(",") if part.strip()]
    if not isinstance(preload_ids, list):
        preload_ids = []

    return WebServerConfig(
        site_name=str(site.get("name", "MicroSeg Hydride Segmentation")),
        site_description=str(
            site.get("description", "Upload a micrograph and segment hydrides in your browser.")
        ),
        maintainer=str(site.get("maintainer", "")),
        host=str(server.get("host", "0.0.0.0")),
        port=_coerce_int(server.get("port", 5005), 5005),
        threads=max(1, _coerce_int(server.get("threads", 4), 4)),
        request_timeout_seconds=_coerce_int(server.get("request_timeout_seconds", 300), 300),
        max_upload_mb=max(1, _coerce_int(limits.get("max_upload_mb", 5), 5)),
        max_long_side_px=max(0, _coerce_int(limits.get("max_long_side_px", 2048), 2048)),
        max_image_pixels=max(
            1_000_000,
            _coerce_int(limits.get("max_image_pixels", 40_000_000), 40_000_000),
        ),
        max_concurrent_jobs=max(1, _coerce_int(limits.get("max_concurrent_jobs", 2), 2)),
        max_retained_jobs=max(2, _coerce_int(limits.get("max_retained_jobs", 32), 32)),
        job_retention_seconds=max(
            60,
            _coerce_int(limits.get("job_retention_seconds", 1800), 1800),
        ),
        preload_on_startup=_coerce_bool(models.get("preload_on_startup", True), True),
        preload_in_background=_coerce_bool(models.get("preload_in_background", True), True),
        preload_model_ids=tuple(str(item).strip() for item in preload_ids if str(item).strip()),
        default_model_id=str(models.get("default_model_id", "auto")).strip() or "auto",
        enable_gpu=_coerce_bool(models.get("enable_gpu", False), False),
        device_policy=str(models.get("device_policy", "cpu")).strip() or "cpu",
        include_analysis=_coerce_bool(analysis.get("include_analysis", True), True),
        sample_images=tuple(samples),
        library_dir=library_dir,
        library_max_images=max(1, _coerce_int(demo.get("library_max_images", 50), 50)),
        repo_root=str(root),
        source_path=str(path),
        warnings=tuple(warnings),
    )
