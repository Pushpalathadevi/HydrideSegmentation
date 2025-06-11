"""Command-line pipeline for the Hydride Segmentation Tool."""
from pathlib import Path
from typing import Union

from .core.image_io import load_image, save_image
from .core.segmentation import segment_hydrides
from .utils.paths import project_root

PathLike = Union[str, Path]


def run_pipeline(input_path: PathLike) -> Path:
    """Run the segmentation pipeline on ``input_path``.

    The image is rotated by 90 degrees and saved with ``_processed`` appended to
    the base filename.

    Args:
        input_path: Path to the input image.

    Returns:
        Path to the generated output image.
    """
    input_path = Path(input_path)
    img = load_image(input_path)
    result = segment_hydrides(img)
    output_path = input_path.with_name(f"{input_path.stem}_processed.png")
    save_image(result, output_path)
    return output_path


def main() -> None:
    """Entry point for running the pipeline."""
    default_input = project_root() / "hydride.png"
    output_path = run_pipeline(default_input)
    if output_path.exists():
        print(f"Processed image saved to: {output_path}")
    else:
        print("Processing failed: output not found")


if __name__ == "__main__":
    main()
