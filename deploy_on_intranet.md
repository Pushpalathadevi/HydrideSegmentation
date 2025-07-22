# Deploying HydrideSegmentation on an Intranet

This guide covers installing the CPU‑only version of HydrideSegmentation and integrating it with the `ml_server` service.

## Installation

1. Create a virtual environment and activate it.
2. Install the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

All dependencies are CPU compatible and require no CUDA libraries.

## Model Weights

Model weights are loaded from the directory specified by the environment variable `HYDRIDE_MODEL_PATH`. If not set, it defaults to `/opt/models/hydride_segmentation/`.

Place `model.pt` inside this folder or adjust the path when starting the service:

```bash
export HYDRIDE_MODEL_PATH=/path/to/weights
```

## Integrating with `ml_server`

The `ml_server` repository provides a REST endpoint that invokes `hydride_segmentation.inference.run_model`.
To enable it:

1. Install HydrideSegmentation inside the same environment used by `ml_server`:

```bash
pip install -e /path/to/HydrideSegmentation
```

2. Ensure the model weights are accessible as described above.
3. Start `ml_server` and send a `POST` request to `/hydride_segmentation` with an image file.

A sample image is available in the `ml_server` repository under `sample_data/sample_hydride_image.png`.

## Testing the Endpoint

After starting `ml_server`, you can verify the new endpoint using `curl`:

```bash
curl -F "image=@sample_hydride_image.png" http://localhost:8000/hydride_segmentation
```

A JSON response with the segmentation mask will be returned if everything is configured correctly.

## Repository Hygiene

No binary model files are tracked in this repository. Ensure that any weight files remain outside the repository, typically under `/opt/models/hydride_segmentation/`.
