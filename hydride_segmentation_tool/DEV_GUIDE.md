# Developer Guide

## Development Environment
Install dependencies in a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

## Running Tests
```bash
pytest
```

## Building the Executable
Use PyInstaller to package the app for offline use.
```bash
pyinstaller hydride_app/gui/app.py --onefile --name HydrideSegTool
```

[tool.pyinstaller]
command = "pyinstaller hydride_app/gui/app.py --onefile --name HydrideSegTool"
