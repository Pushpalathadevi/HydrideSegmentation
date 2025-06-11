"""I/O utilities for images."""
from pathlib import Path
from typing import Union
from PIL import Image

from ..utils.paths import ensure_directory


PathLike = Union[str, Path]


def load_image(path: PathLike) -> Image.Image:
    """Load an image from ``path``.

    Args:
        path: Path to the image.

    Returns:
        Loaded ``PIL.Image`` instance.
    """
    return Image.open(path)


def save_image(image: Image.Image, path: PathLike) -> None:
    """Save ``image`` to ``path``.

    Args:
        image: Image to save.
        path: Target file path.
    """
    path = Path(path)
    ensure_directory(path.parent)
    image.save(path)

