# Local Pretrained Weights (Air-Gap Staging)

This folder is reserved for local transfer-learning bundles copied into offline systems.

- Use `python scripts/download_pretrained_weights.py` on a connected machine.
- Default `--targets=all` stages all currently supported local-pretrained bundles:
  - `hf_segformer_b0_ade20k`
  - `hf_segformer_b2_ade20k`
  - `hf_segformer_b5_ade20k`
  - `smp_unet_resnet18_imagenet`
- Copy the whole `pre_trained_weights/` folder to air-gapped systems.
- Validate before training:
  - `microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict`

Registry metadata standards:
- `source` + `source_url`
- `source_revision`
- `license`
- `citation_key` + `citation` (+ optional `citation_url`)
- `files` checksums for integrity verification

Git policy:
- Binary model artifacts and downloaded model directories in this folder are ignored by git.
- Keep only lightweight templates/docs tracked in git.
