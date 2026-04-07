# Usage Commands

This page is intentionally exact. Commands here should be copy-pastable.

## Desktop GUI

Launch the Qt desktop app:

```bash
hydride-gui
```

Launch Qt with an explicit UI config:

```bash
hydride-gui --ui-config configs/app/desktop_ui.default.yml
```

Launch the legacy Tk fallback:

```bash
hydride-gui --framework tk
```

Launch the Qt entry point directly:

```bash
hydride-gui-qt --ui-config configs/app/desktop_ui.default.yml
```

## Student Notebooks

Open the hands-on tutorial notebooks in JupyterLab:

```bash
jupyter lab docs/notebooks/01_data_preparation_and_dataset_planning.ipynb
```

Or open the notebook directory as a workspace:

```bash
jupyter lab docs/notebooks/
```

## Model Discovery

Inspect available GUI/CLI models and frozen-checkpoint hints:

```bash
microseg-cli models --details
```

Validate the frozen registry:

```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

Validate local pretrained bundles:

```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

## Inference

Run a single-image inference job:

```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120
```

Run inference with explicit architecture-aware loading:

```bash
microseg-cli infer \
  --config configs/inference.default.yml \
  --set params.run_dir=outputs/runs/example_run \
  --set params.enable_gpu=false \
  --set params.device_policy=cpu
```

Outputs are written under a dedicated run folder and include the input image, predicted mask, overlays, metrics, and a manifest.

If you are integrating a new trained model into the GUI, review:

- [`docs/gui_model_integration_guide.md`](gui_model_integration_guide.md)
- [`docs/model_selection_decision_tree.md`](model_selection_decision_tree.md)

## Training

Train the default UNet path:

```bash
microseg-cli train --config configs/train.default.yml --set epochs=20
```

Train with fixed input size, AMP, and gradient accumulation:

```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set input_hw=[512,512] \
  --set input_policy=random_crop \
  --set val_input_policy=letterbox \
  --set amp_enabled=true \
  --set grad_accum_steps=2 \
  --set deterministic=true
```

Track fixed validation exemplars:

```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set val_tracking_samples=8 \
  --set "val_tracking_fixed_samples=val_000.png|val_123.png"
```

Key output roots:

- `outputs/training/<run_name>/report.json`
- `outputs/training/<run_name>/report.html`
- `outputs/training/<run_name>/checkpoint_epoch_*.pth`
- `outputs/training/<run_name>/training.log`

## Evaluation

Evaluate on the test split:

```bash
microseg-cli evaluate --config configs/evaluate.default.yml --set split=test
```

Evaluation reports are written under `outputs/evaluation/` by default and include a JSON summary plus optional HTML panels.

## Dataset Preparation

Leakage-aware split planning:

```bash
microseg-cli dataset-split --config configs/dataset_split.default.yml
```

Prepare an unsplit `source/masks` dataset:

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

Prepare paired JPG + RGB PNG masks into MaDo/Oxford layout:

```bash
python scripts/microseg_cli.py prepare_dataset \
  --config configs/hydride/prepare_dataset.paired_rgb_mask.mado.yml \
  --input-dir D:/data/hydride_pairs \
  --output-root D:/data/HydrideData7.0 \
  --style mado \
  --target-size 512 \
  --crop-train random \
  --crop-eval center \
  --mask-r-min 200 --mask-g-max 60 --mask-b-max 60 \
  --allow-red-dominance-fallback \
  --auto-otsu-for-noisy-grayscale \
  --empty-mask-action warn \
  --seed 42 --train-frac 0.8 --val-frac 0.1 \
  --max-val-examples 200 \
  --max-test-examples 200
```

## Deployment and Quality Gates

Run the phase gate:

```bash
microseg-cli phase-gate --phase-label "Phase N" --strict
```

Create and validate a deployment package:

```bash
microseg-cli deploy-package --config configs/deploy.default.yml
microseg-cli deploy-validate --package-dir outputs/deployment/package_name --strict
```

Run a deployment smoke test:

```bash
microseg-cli deploy-smoke --package-dir outputs/deployment/package_name --image-path path/to/image.png
```

## Benchmark and HPC Planning

Generate a GA-planned HPC bundle:

```bash
microseg-cli hpc-ga-generate --config configs/hpc_ga.default.yml
```

Summarize prior GA feedback:

```bash
microseg-cli hpc-ga-feedback-report --config configs/hpc_ga_feedback.default.yml
```

Run the phase-ID benchmark workflow:

```bash
microseg-cli phaseid-benchmark \
  --config configs/phaseid_oh5_benchmark.default.yml \
  --raw-input-dir D:/phaseid/raw_oh5 \
  --working-dir D:/phaseid/run_001
```

## Docs Build

Build HTML + PDF documentation:

```bash
python scripts/build_docs.py
```
