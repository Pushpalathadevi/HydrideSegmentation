"""YAML-backed configuration loading with --set style overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Configuration loading or override parsing error."""


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON override value: {text!r}") from exc
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ConfigError("override key cannot be empty")
    node = config
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ConfigError(f"cannot set nested key under non-dict path: {part}")
        node = node[part]
    node[parts[-1]] = value


def parse_set_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=value`` style overrides into nested dictionary."""

    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"invalid override '{item}', expected key=value")
        key, raw_val = item.split("=", 1)
        _set_nested(out, key.strip(), _parse_scalar(raw_val))
    return out


def merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge dictionaries, returning a new dictionary."""

    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency import
        raise RuntimeError("PyYAML is required for YAML config support. Install with `pip install pyyaml`.") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")

    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError("top-level YAML config must be a mapping")
    return payload


def resolve_config(config_path: str | Path | None, set_items: list[str] | None = None) -> dict[str, Any]:
    """Resolve final config from YAML file plus ``--set`` overrides."""

    base = load_yaml_config(config_path) if config_path else {}
    overrides = parse_set_overrides(set_items)
    return merge_dicts(base, overrides)
