# Pretrained Model Catalog (Air-Gapped Transfer)

## Purpose

This catalog defines the canonical pretrained bundles supported for offline transfer learning,
with provenance and citation metadata for reports and manuscript preparation.

## Canonical Registry

- Runtime registry: `pre_trained_weights/registry.json` (generated on connected machine)
- Template/schema example: `pre_trained_weights/registry.template.json`
- Validation command:

```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

## Supported Bundles

| Model ID | Backend / Architecture | Upstream Source | License Field | Citation Key | Notes |
|---|---|---|---|---|---|
| `smp_unet_resnet18_imagenet` | `smp_unet_resnet18` | `segmentation_models_pytorch:Unet(resnet18, encoder_weights=imagenet)` | `See upstream package licenses` | `ronneberger2015unet;he2016resnet` | Full U-Net state dict with ImageNet-initialized ResNet18 encoder. |
| `unet_binary_resnet18_imagenet_partial` | `unet_binary` | `timm:resnet18` | `See upstream timm model card/license terms` | `he2016resnet;ronneberger2015unet` | Partial warm-start bootstrap mapped into internal `unet_binary` encoder tensors. |
| `hf_segformer_b0_ade20k` | `hf_segformer_b0` | `huggingface:nvidia/segformer-b0-finetuned-ade-512-512` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B0 snapshot for offline local init. |
| `hf_segformer_b2_ade20k` | `hf_segformer_b2` | `huggingface:nvidia/segformer-b2-finetuned-ade-512-512` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B2 snapshot for offline local init. |
| `hf_segformer_b5_ade20k` | `hf_segformer_b5` | `huggingface:nvidia/segformer-b5-finetuned-ade-640-640` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B5 snapshot for offline local init. |
| `transunet_tiny_vit_tiny_patch16_imagenet` | `transunet_tiny` | `timm:vit_tiny_patch16_224` | `See upstream timm model card/license terms` | `dosovitskiy2020vit` | Partial warm-start bootstrap mapped into internal `transunet_tiny` tensor shapes. |
| `segformer_mini_vit_tiny_patch16_imagenet` | `segformer_mini` | `timm:vit_tiny_patch16_224` | `See upstream timm model card/license terms` | `dosovitskiy2020vit` | Partial warm-start bootstrap mapped into internal `segformer_mini` tensor shapes. |

## Manuscript Citation Source

- BibTeX file: `docs/pretrained_model_citations.bib`
- Recommended practice:
  - cite architecture papers (SegFormer, U-Net, ResNet)
  - cite ViT paper for `transunet_tiny`/`segformer_mini` bootstrap bundles
  - additionally cite dataset/model cards where required by upstream terms
  - include exact `source_revision` commit hash from `registry.json`

## Minimum Reporting Fields

Each run report/manuscript table should capture at least:

- `model_id`
- `architecture`
- `source`
- `source_url`
- `source_revision`
- `weights_format`
- `license`
- `citation_key`
- `pretrained_registry_path`
- checksum validation status (`pretrained_verify_sha256`)

For internal bootstrap bundles (`unet_binary_resnet18_imagenet_partial`, `transunet_tiny_vit_tiny_patch16_imagenet`, `segformer_mini_vit_tiny_patch16_imagenet`), include:
- `bootstrap_type=partial_warm_start`
- mapped vs unmapped component list (from `pre_trained_weights/metadata/*.meta.json`)
- note that these are not official external checkpoints for the internal architectures

## Known Limits

- Internal research backends `transunet_tiny` and `segformer_mini` now support local-pretrained bootstrap bundles derived from ViT-tiny mappings.
- Local-pretrained support is implemented for `unet_binary`, `smp_unet_resnet18`, HF SegFormer (`b0/b2/b5`), `transunet_tiny`, and `segformer_mini`.
