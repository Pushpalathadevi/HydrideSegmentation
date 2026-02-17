# Hydride Benchmark Configs

These templates are for fair model-comparison studies on a fixed hydride dataset split.

Available scratch training presets:
- `train.unet_binary.baseline.yml`
- `train.hf_segformer_b0_scratch.yml`
- `train.hf_segformer_b2_scratch.yml`
- `train.hf_segformer_b5_scratch.yml`
- `train.transunet_tiny.yml`
- `train.transunet_tiny_deep.yml`
- `train.segformer_mini.yml`
- `train.segformer_mini_wide.yml`

Available local-pretrained training presets (real runs):
- `train.unet_binary_local_pretrained.yml`
- `train.smp_unet_resnet18_local_pretrained.yml`
- `train.hf_segformer_b0_local_pretrained.yml`
- `train.hf_segformer_b2_local_pretrained.yml`
- `train.hf_segformer_b5_local_pretrained.yml`
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
- Scratch top-5 baseline: `benchmark_suite.top5.yml`
- Local-pretrained top-5 baseline: `benchmark_suite.top5_local_pretrained.yml`
- Scratch top-5 debug integrity run: `benchmark_suite.top5_scratch.debug.yml`
- Local-pretrained top-5 debug integrity run: `benchmark_suite.top5_local_pretrained.debug.yml`
- Combined scratch vs pretrained debug run (single dashboard): `benchmark_suite.top5_scratch_vs_pretrained.debug.yml`
- Real-data templates:
  - `benchmark_suite.top5_scratch.realdata.template.yml`
  - `benchmark_suite.top5_local_pretrained.realdata.template.yml`
  - `benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml`

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
- `hf_segformer_b0`, `hf_segformer_b2`, and `hf_segformer_b5` initialize from local Hugging Face model directories.
- `transunet_tiny` and `segformer_mini` initialize from local ViT-tiny-derived partial warm-start state dict bundles.
- See `docs/pretrained_model_catalog.md` and `pre_trained_weights/metadata/*.meta.json` for provenance/citation details.
