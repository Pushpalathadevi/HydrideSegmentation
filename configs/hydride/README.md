# Hydride Benchmark Configs

These templates are for fair model-comparison studies on a fixed hydride dataset split.

Hydride defaults in this folder:
- `binary_mask_normalization: nonzero_foreground` for train/evaluate presets (`0`=background, any non-zero pixel=foreground).
- Benchmark suite templates support `benchmark_mode: true`; if `dataset_manifest.json` is missing, it is auto-generated from split folders.

Available scratch training presets:
- `train.unet_binary.baseline.yml`
- `train.smp_deeplabv3plus_resnet101_scratch.yml`
- `train.smp_unetplusplus_resnet101_scratch.yml`
- `train.smp_pspnet_resnet101_scratch.yml`
- `train.smp_fpn_resnet101_scratch.yml`
- `train.hf_segformer_b0_scratch.yml`
- `train.hf_segformer_b2_scratch.yml`
- `train.hf_segformer_b5_scratch.yml`
- `train.hf_upernet_swin_large_scratch.yml`
- `train.transunet_tiny.yml`
- `train.transunet_tiny_deep.yml`
- `train.segformer_mini.yml`
- `train.segformer_mini_wide.yml`

Available local-pretrained training presets (real runs):
- `train.unet_binary_local_pretrained.yml`
- `train.smp_unet_resnet18_local_pretrained.yml`
- `train.smp_deeplabv3plus_resnet101_local_pretrained.yml`
- `train.smp_unetplusplus_resnet101_local_pretrained.yml`
- `train.smp_pspnet_resnet101_local_pretrained.yml`
- `train.smp_fpn_resnet101_local_pretrained.yml`
- `train.hf_segformer_b0_local_pretrained.yml`
- `train.hf_segformer_b2_local_pretrained.yml`
- `train.hf_segformer_b5_local_pretrained.yml`
- `train.hf_upernet_swin_large_local_pretrained.yml`
- `train.transunet_tiny_local_pretrained.yml`
- `train.segformer_mini_local_pretrained.yml`

Available local-pretrained debug presets:
- `train.unet_binary_local_pretrained.debug.yml`
- `train.smp_unet_resnet18_local_pretrained.debug.yml`
- `train.hf_segformer_b0_local_pretrained.debug.yml`
- `train.hf_segformer_b2_local_pretrained.debug.yml`
- `train.hf_segformer_b5_local_pretrained.debug.yml`
- `train.transunet_tiny_local_pretrained.debug.yml`
- `train.segformer_mini_local_pretrained.debug.yml`

Benchmark suite presets:
- Scratch top-10 baseline: `benchmark_suite.top5.yml`
- Local-pretrained top-10 baseline: `benchmark_suite.top5_local_pretrained.yml`
- Scratch top-5 debug integrity run: `benchmark_suite.top5_scratch.debug.yml`
- Local-pretrained top-5 debug integrity run: `benchmark_suite.top5_local_pretrained.debug.yml`
- Combined scratch vs pretrained debug run (single dashboard): `benchmark_suite.top5_scratch_vs_pretrained.debug.yml`
- Real-data templates:
  - `benchmark_suite.top5_scratch.realdata.template.yml`
  - `benchmark_suite.top5_local_pretrained.realdata.template.yml`
  - `benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml`

Suite execution hardening knobs (optional, in benchmark suite YAML):
- `command_idle_timeout_seconds`: kill a run if its log file has no new bytes for this many seconds.
- `command_wall_timeout_seconds`: kill a run after this total runtime budget in seconds.
- `command_terminate_grace_seconds`: grace period before force-kill after timeout (default `30`).
- `command_poll_interval_seconds`: watchdog polling cadence in seconds (default `1`).
- Structured suite timeline is written to `output_root/logs/suite_events.jsonl`.
- Per-run logs are written continuously to `output_root/logs/<run_tag>/{train,eval}.log`.
- Per-run stage/timing events are written to `output_root/logs/<run_tag>/run_events.jsonl`.

Operational runbook:
- `docs/hpc_airgap_top5_realdata_runbook.md`

Run combined debug suite + dashboard:
```bash
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.debug.yml \
  --strict
```

Run canonical scratch suite + dashboard:
```bash
python scripts/hydride_benchmark_suite.py --config configs/hydride/benchmark_suite.top5.yml --strict
```

Run canonical local-pretrained suite + dashboard:
```bash
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_local_pretrained.yml \
  --strict
```

Local-pretrained notes:
- `unet_binary` is warm-started from a ResNet18-derived partial bootstrap bundle.
- `smp_deeplabv3plus_resnet101`, `smp_unetplusplus_resnet101`, `smp_pspnet_resnet101`, and `smp_fpn_resnet101` initialize from local ImageNet-initialized SMP state dict bundles.
- `hf_segformer_b0`, `hf_segformer_b2`, and `hf_segformer_b5` initialize from local Hugging Face model directories.
- `hf_upernet_swin_large` initializes from a local Hugging Face UPerNet-Swin-Large directory.
- `transunet_tiny` and `segformer_mini` initialize from local ViT-tiny-derived partial warm-start state dict bundles.
- See `docs/pretrained_model_catalog.md` and `pre_trained_weights/metadata/*.meta.json` for provenance/citation details.
