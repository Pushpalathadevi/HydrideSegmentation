"""Placeholder hydride segmentation algorithms."""

from PIL import Image


def segment_hydrides(image: Image.Image) -> Image.Image:
    """Rotate ``image`` by 90 degrees as a placeholder segmentation step.

    Args:
        image: Input micrograph.

    Returns:
        Rotated image simulating a segmentation output.
    """
    return image.rotate(90, expand=True)


__all__ = ["segment_hydrides"]

