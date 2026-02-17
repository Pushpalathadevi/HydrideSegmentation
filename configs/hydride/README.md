# Hydride Benchmark Configs

These templates are for fair model-comparison studies on a fixed hydride dataset split.

Available training presets:
- `train.unet_binary.baseline.yml`
- `train.hf_segformer_b0_scratch.yml`
- `train.hf_segformer_b2_scratch.yml`
- `train.hf_segformer_b5_scratch.yml`
- `train.transunet_tiny.yml`
- `train.transunet_tiny_deep.yml`
- `train.segformer_mini.yml`
- `train.segformer_mini_wide.yml`
- `train.torch_pixel.yml`
- `train.sklearn_pixel.yml`
- `train.smp_unet_resnet18_local_pretrained.debug.yml`
- `train.hf_segformer_b0_local_pretrained.debug.yml`
- `train.hf_segformer_b2_local_pretrained.debug.yml`
- `train.hf_segformer_b5_local_pretrained.debug.yml`

Transformer presets marked with `_scratch` are initialized from architecture configs only.
They do not load pretrained weights and are suitable for offline HPC runs.

Debug presets ending with `_local_pretrained.debug` are for validating air-gapped local transfer-learning:
- `smp_unet_resnet18` initializes from local pretrained state dict.
- `hf_segformer_b0`, `hf_segformer_b2`, and `hf_segformer_b5` initialize from local Hugging Face model directories.

Top-5 benchmark suite config:
- `benchmark_suite.top5.yml`

Run full suite + dashboard:
```bash
python scripts/hydride_benchmark_suite.py --config configs/hydride/benchmark_suite.top5.yml --strict
```

Usage pattern:

```bash
microseg-cli train \
  --config configs/hydride/train.transunet_tiny.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/benchmarks/transunet_tiny_seed42 \
  --set seed=42 \
  --no-auto-prepare-dataset
```

Evaluate with:

```bash
microseg-cli evaluate \
  --config configs/hydride/evaluate.hydride.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --model-path outputs/benchmarks/transunet_tiny_seed42/best_checkpoint.pt \
  --split test \
  --output-path outputs/eval/transunet_tiny_seed42_test.json \
  --no-auto-prepare-dataset
```
