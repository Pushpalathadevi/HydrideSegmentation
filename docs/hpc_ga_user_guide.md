# HPC GA User Guide

## What This Feature Does

The HPC GA Planner helps you generate a complete multi-experiment job bundle for HPC execution.
You select model backends and hyperparameter ranges in GUI or CLI, then the tool writes:

- candidate parameter files
- per-candidate job scripts
- one `submit_all.sh` script to launch everything
- one machine-readable manifest JSON

This is designed for:
- architecture comparison (for example `unet_binary` vs `hf_segformer_b0` vs `hf_segformer_b2` vs `hf_segformer_b5`)
- hyperparameter sweep initialization
- repeatable GPU training/evaluation batch runs on Slurm/PBS/local schedulers

## Planning Modes

- `novelty`:
  - diversity-first search in parameter space
  - recommended for first sweep when no prior run metrics exist
The desktop application is intentionally focused on inference, correction, quantification, and results review. HPC planning is a CLI workflow.

## Quick Start (CLI)

Environment/bootstrap (recommended before running GA commands):

```bash
python -m pip install -e .
microseg-cli models --details
```

```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle
```

Air-gapped local-pretrained profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle_airgap_pretrained
```

Top-5 scratch profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_scratch.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_scratch
```

Top-5 air-gapped local-pretrained profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_airgap_pretrained
```

If CLI import errors occur (`No module named src`), run from repo root using module form:

```bash
python -m scripts.microseg_cli hpc-ga-generate \
  --config configs/hpc_ga.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle
```

## Upload And Run On HPC

1. Copy bundle folder to HPC workspace.
2. Ensure project repository and dependencies are available on HPC.
3. Run:
```bash
cd outputs/hpc_ga_bundle
REPO_ROOT=/path/to/HydrideSegmentation ./submit_all.sh
```

Scheduler behavior:
- `slurm`: `submit_all.sh` uses `sbatch`
- `pbs`: `submit_all.sh` uses `qsub`
- `local`: `submit_all.sh` runs each job script via `bash`

## Output Structure

Example:
```text
outputs/hpc_ga_bundle/
  ga_plan_manifest.json
  submit_all.sh
  README.txt
  jobs/
    cand_001.sh
    cand_002.sh
  candidates/
    cand_001.json
    cand_001.yml
  runs/
    (created when jobs execute)
```

## Recommended Beginner Workflow

1. Start with 4 candidates, 2 backends, small epochs.
2. Run `train_eval` mode to get both train and eval artifacts.
3. Review the generated training/evaluation reports and narrow ranges before regenerating the next bundle.

## Common Problems And Fixes

`Dataset Dir missing`
- Set `Dataset Dir` in GUI or pass `--dataset-dir`.

`No GPU available on HPC node`
- Keep `enable_gpu=true` but ensure scheduler resources request GPU.
- Or set `enable_gpu=false` for CPU-only nodes.

`Config path errors on HPC`
- Use repository-relative config paths.
- Set `REPO_ROOT` before running `submit_all.sh`.

`Expected local pretrained init but candidate ran scratch`
- Set `pretrained_init_mode=local` to force mapping completeness, or use `auto` with an explicit `pretrained_model_map`.
- Validate local bundles first:
  - `microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict`

`Model checkpoint not found during evaluate step`
- Job script already tries `best_checkpoint.pt`, then `last_checkpoint.pt`, then `torch_pixel_classifier.pt`.
- Inspect training run folder under `runs/cand_xxx/`.

## Related Docs

- `docs/hpc_ga_developer_guide.md`
- `docs/configuration_workflow.md`
- `docs/gui_user_guide.md`
- `docs/phase15_hpc_ga_hpc_bundle.md`
- `docs/offline_pretrained_transfer_workflow.md`
