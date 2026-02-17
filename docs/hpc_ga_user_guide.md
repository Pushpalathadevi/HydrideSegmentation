# HPC GA User Guide

## What This Feature Does

The HPC GA Planner helps you generate a complete multi-experiment job bundle for HPC execution.
You select model backends and hyperparameter ranges in GUI or CLI, then the tool writes:

- candidate parameter files
- per-candidate job scripts
- one `submit_all.sh` script to launch everything
- one machine-readable manifest JSON

This is designed for:
- architecture comparison (for example `unet_binary` vs `torch_pixel`)
- hyperparameter sweep initialization
- repeatable GPU training/evaluation batch runs on Slurm/PBS/local schedulers

## Important Scientific Note

The current GA planner is novelty-oriented (diversity-first).  
It generates diverse candidate settings before you have model performance feedback.
Treat this as phase-1 search orchestration, then refine from evaluation reports.

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

## Quick Start (CLI)

```bash
microseg-cli hpc-ga-generate \
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
3. Use GUI `Run Review` tab to compare reports.
4. Narrow hyperparameter ranges based on first run.
5. Regenerate another bundle with updated ranges.

## Common Problems And Fixes

`Dataset Dir missing`
- Set `Dataset Dir` in GUI or pass `--dataset-dir`.

`No GPU available on HPC node`
- Keep `enable_gpu=true` but ensure scheduler resources request GPU.
- Or set `enable_gpu=false` for CPU-only nodes.

`Config path errors on HPC`
- Use repository-relative config paths.
- Set `REPO_ROOT` before running `submit_all.sh`.

`Model checkpoint not found during evaluate step`
- Job script already tries `best_checkpoint.pt`, then `last_checkpoint.pt`, then `torch_pixel_classifier.pt`.
- Inspect training run folder under `runs/cand_xxx/`.

## Related Docs

- `docs/hpc_ga_developer_guide.md`
- `docs/configuration_workflow.md`
- `docs/gui_user_guide.md`
- `docs/phase15_hpc_ga_hpc_bundle.md`
