"""Tests for class-map JSON resolution and fallback behavior."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.corrections import classes as classes_module


def _write_class_map(path: Path, *, class1_name: str) -> None:
    payload = {
        "classes": [
            {
                "index": 0,
                "name": "background",
                "color_rgb": [0, 0, 0],
                "description": "bg",
            },
            {
                "index": 1,
                "name": class1_name,
                "color_hex": "#FF0000",
                "description": "fg",
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_class_map_resolution_uses_explicit_over_env(tmp_path: Path, monkeypatch) -> None:
    env_map = tmp_path / "env_classes.json"
    explicit_map = tmp_path / "explicit_classes.json"
    _write_class_map(env_map, class1_name="env_foreground")
    _write_class_map(explicit_map, class1_name="explicit_foreground")
    monkeypatch.setenv("MICROSEG_CLASS_MAP_PATH", str(env_map))

    class_map, source = classes_module.resolve_class_map(str(explicit_map))

    assert source.startswith("explicit:")
    assert class_map.class_for_index(1).name == "explicit_foreground"


def test_class_map_resolution_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    env_map = tmp_path / "env_classes.json"
    _write_class_map(env_map, class1_name="env_foreground")
    monkeypatch.setenv("MICROSEG_CLASS_MAP_PATH", str(env_map))

    class_map, source = classes_module.resolve_class_map()

    assert source.startswith("env:MICROSEG_CLASS_MAP_PATH:")
    assert class_map.class_for_index(1).name == "env_foreground"


def test_class_map_resolution_falls_back_to_builtin_when_repo_default_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MICROSEG_CLASS_MAP_PATH", raising=False)
    monkeypatch.setattr(classes_module, "REPO_DEFAULT_CLASS_MAP_PATH", tmp_path / "missing_default.json")

    class_map, source = classes_module.resolve_class_map()

    assert source == "builtin_default"
    assert class_map.class_for_index(1).name == "feature"
