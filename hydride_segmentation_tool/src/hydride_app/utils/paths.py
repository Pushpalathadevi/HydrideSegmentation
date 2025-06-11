"""Path utilities for Hydride Segmentation Tool."""
from pathlib import Path


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[3]


def ensure_directory(path: Path) -> Path:
    """Ensure that ``path`` exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

