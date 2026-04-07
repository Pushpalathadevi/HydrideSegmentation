# Local Pretrained Weights (Air-Gap Staging)

This folder is reserved for local transfer-learning bundles copied into offline systems.

- Use `python scripts/download_pretrained_weights.py` on a connected machine.
- Default `--targets=all` stages all currently supported local-pretrained bundles:
  - `unet_binary_resnet18_imagenet_partial`
  - `hf_segformer_b0_ade20k`
  - `hf_segformer_b2_ade20k`
  - `hf_segformer_b5_ade20k`
  - `hf_upernet_swin_large_ade20k`
  - `smp_unet_resnet18_imagenet`
  - `smp_deeplabv3plus_resnet101_imagenet`
  - `smp_unetplusplus_resnet101_imagenet`
  - `smp_pspnet_resnet101_imagenet`
  - `smp_fpn_resnet101_imagenet`
  - `transunet_tiny_vit_tiny_patch16_imagenet`
  - `segformer_mini_vit_tiny_patch16_imagenet`
- Copy the whole `pre_trained_weights/` folder to air-gapped systems.
- Validate before training:
  - `microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict`
- HPC copy-paste runbook with exact top-10 commands + manual direct download URLs:
  - `docs/hpc_airgap_top5_realdata_runbook.md`

Registry metadata standards:
- `source` + `source_url`
- `source_revision`
- `license`
- `citation_key` + `citation` (+ optional `citation_url`)
- `files` checksums for integrity verification
- curated tracked metadata records live under `pre_trained_weights/metadata/*.meta.json`

Git policy:
- Binary model artifacts and downloaded model directories in this folder are ignored by git.
- Keep only lightweight templates/docs/metadata tracked in git.
