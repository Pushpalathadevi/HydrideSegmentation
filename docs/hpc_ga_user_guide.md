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
- `feedback_hybrid`:
  - combines novelty with predicted fitness from previous run reports
  - fitness uses `mean_iou`, `macro_f1`, `pixel_accuracy`, and runtime penalty weights
  - recommended for second+ sweeps after you have candidate `eval_report.json` artifacts

## Quick Start (GUI)

1. Launch GUI:
```bash
hydride-gui
```
2. Open `Workflow Hub` -> `HPC GA Planner`.
3. Fill minimum required fields:
- `Dataset Dir`
- `Bundle Output Dir`
- `Architectures` (comma-separated)
4. Choose scheduler:
- `slurm`, `pbs`, or `local`
5. Click `Generate HPC GA Bundle`.
6. Check `Orchestration Log` for command and completion status.
7. Open generated bundle directory and inspect:
- `ga_plan_manifest.json`
- `submit_all.sh`
- `jobs/*.sh`
- `candidates/*.json` and `candidates/*.yml`

Feedback analysis in GUI:
1. Set `Feedback Sources` to one or more prior bundle directories (or `ga_plan_manifest.json` paths), comma-separated.
2. Set `Fitness Mode` to `feedback_hybrid`.
3. Optionally tune:
- `Feedback Min Samples`
- `Feedback K (kNN)`
- metric/runtime fitness weights
4. Click `Analyze Feedback` to generate:
- JSON report at `Feedback Report Output`
- markdown summary table next to it (`.md` suffix)

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

Feedback summary from prior bundles:
```bash
microseg-cli hpc-ga-feedback-report \
  --config configs/hpc_ga.default.yml \
  --feedback-sources outputs/hpc_ga_bundle_a,outputs/hpc_ga_bundle_b \
  --output-path outputs/hpc_ga_feedback/feedback_report.json
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

Feedback report outputs:
```text
outputs/hpc_ga_feedback/
  feedback_report.json
  feedback_report.md
```

## Recommended Beginner Workflow

1. Start with 4 candidates, 2 backends, small epochs.
2. Run `train_eval` mode to get both train and eval artifacts.
3. Use GUI `Run Review` tab to compare reports.
4. Add previous bundle paths into `Feedback Sources`.
5. Switch planner to `feedback_hybrid` and click `Analyze Feedback`.
6. Narrow ranges/weights and regenerate the next bundle.

## Common Problems And Fixes

`Dataset Dir missing`
- Set `Dataset Dir` in GUI or pass `--dataset-dir`.

`feedback_hybrid behaves like novelty`
- Ensure enough valid samples are found in feedback sources.
- Increase feedback coverage or lower `feedback_min_samples`.

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
- `docs/phase17_hpc_ga_feedback.md`
- `docs/offline_pretrained_transfer_workflow.md`
