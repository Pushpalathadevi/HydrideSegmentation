# Hydride Segmentation Tool

An offline-ready Python application for basic hydride segmentation using image thresholding. The tool provides a simple GUI for loading images, running a placeholder segmentation algorithm, visualizing results, and saving outputs.

## Installation

```bash
python -m pip install .
```

Or install dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

```bash
python -m hydride_app.gui.app
```

To run the command-line pipeline that processes ``hydride.png`` and outputs ``hydride_processed.png``:

```bash
python -m hydride_app.pipeline
```

Refer to `USER_GUIDE.md` for a detailed walk-through.
