# Deploying HydrideSegmentation on an Intranet

There are two ways to make segmentation available to colleagues on an internal network. Both are CPU-only and need no CUDA and no internet access at run time.

## 1. Browser App (recommended)

One host runs a small web server; everyone else opens a URL. Nobody except the host operator installs anything.

```bash
pip install -r requirements-web.txt
pip install -e .
python scripts/run_web_server.py
```

The launcher prints the intranet links to share. Open the host firewall for the port and you are done.

Users can upload a micrograph or run a bundled example image, choose between the conventional pipeline and any installed trained model, tune conventional parameters with in-app help, view the overlay, mask, orientation map and distributions, and download results. Uploaded images are processed in memory and never stored.

**Complete guide, including the air-gapped wheelhouse install, configuration, systemd and Windows service setup, performance tuning, and the JSON API: [`docs/intranet_web_app.md`](docs/intranet_web_app.md).**

## 2. Library Integration

To call segmentation from another internal service rather than serving a UI, install this package into that service's environment and import it directly:

```bash
pip install -e /path/to/HydrideSegmentation
```

```python
from hydride_segmentation.microseg_adapter import run_pipeline

result = run_pipeline("micrograph.png", model_id="hydride_conventional", include_analysis=True)
```

`run_pipeline` is the same entry point the desktop app, the CLI, and the web app use, so every surface produces identical results for identical inputs.

## Model Weights

Checkpoint binaries are never tracked in git, so a fresh copy of the repository has no `.pt` file. The conventional pipeline works immediately; trained models are reported as unavailable until a checkpoint is installed.

Install one on the host with:

```bash
microseg-cli install-model --checkpoint path/to/best_checkpoint.pth --model-id site_unet_v1 --nickname site_unet_v1_optical
```

The installer reads the architecture and provenance from the checkpoint, copies it into the repository's lifecycle folder, verifies it loads and runs, and registers it locally. See [`docs/gui_model_integration_guide.md`](docs/gui_model_integration_guide.md).

Check what a host currently has:

```bash
microseg-cli models --details
```

## Repository Hygiene

No binary model files are tracked in this repository. Installed checkpoints live under `frozen_checkpoints/`, which is excluded from git, together with the untracked `frozen_checkpoints/model_registry.local.json` overlay that records them.
