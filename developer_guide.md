# Developer Guide

This document provides quick examples for using the package from your own projects.

## Installation

```bash
pip install -e .
```

or add the repository as a Git submodule and run the same command in the submodule directory.

## Command Line Interfaces

The package exposes a few convenience entry points once installed:

```bash
hydride-gui           # Launch the Tkinter application
hydride-orientation   # Analyse hydride orientations from an image
segmentation-eval     # Evaluate segmentation quality on sample data
```

Append `--help` to any command for detailed options.

## Using as a Library

```python
from hydride_segmentation import run_model, orientation_analysis

mask_image, mask = run_model("image.png", params)
orient, size_plot, angle_plot = orientation_analysis(mask)
```

Each module contains docstrings describing expected parameters and return values.
