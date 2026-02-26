#!/usr/bin/env python3
"""Backward-compatible wrapper for dataset preparation CLI."""

from __future__ import annotations

from src.microseg.data_preparation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
