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
| `smp_deeplabv3plus_resnet101_imagenet` | `smp_deeplabv3plus_resnet101` | `segmentation_models_pytorch:DeepLabV3Plus(resnet101, encoder_weights=imagenet)` | `See upstream package licenses` | `chen2018encoderdecoder;he2016resnet` | Full DeepLabV3+ state dict with ImageNet-initialized ResNet101 encoder. |
| `smp_unetplusplus_resnet101_imagenet` | `smp_unetplusplus_resnet101` | `segmentation_models_pytorch:UnetPlusPlus(resnet101, encoder_weights=imagenet)` | `See upstream package licenses` | `zhou2018unetplusplus;he2016resnet` | Full U-Net++ state dict with ImageNet-initialized ResNet101 encoder. |
| `smp_pspnet_resnet101_imagenet` | `smp_pspnet_resnet101` | `segmentation_models_pytorch:PSPNet(resnet101, encoder_weights=imagenet)` | `See upstream package licenses` | `zhao2017pspnet;he2016resnet` | Full PSPNet state dict with ImageNet-initialized ResNet101 encoder. |
| `smp_fpn_resnet101_imagenet` | `smp_fpn_resnet101` | `segmentation_models_pytorch:FPN(resnet101, encoder_weights=imagenet)` | `See upstream package licenses` | `lin2017fpn;he2016resnet` | Full FPN state dict with ImageNet-initialized ResNet101 encoder. |
| `unet_binary_resnet18_imagenet_partial` | `unet_binary` | `timm:resnet18` | `See upstream timm model card/license terms` | `he2016resnet;ronneberger2015unet` | Partial warm-start bootstrap mapped into internal `unet_binary` encoder tensors. |
| `hf_segformer_b0_ade20k` | `hf_segformer_b0` | `huggingface:nvidia/segformer-b0-finetuned-ade-512-512` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B0 snapshot for offline local init. |
| `hf_segformer_b2_ade20k` | `hf_segformer_b2` | `huggingface:nvidia/segformer-b2-finetuned-ade-512-512` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B2 snapshot for offline local init. |
| `hf_segformer_b5_ade20k` | `hf_segformer_b5` | `huggingface:nvidia/segformer-b5-finetuned-ade-640-640` | `See upstream model card license metadata` | `xie2021segformer` | ADE20K-finetuned SegFormer B5 snapshot for offline local init. |
| `hf_upernet_swin_large_ade20k` | `hf_upernet_swin_large` | `huggingface:openmmlab/upernet-swin-large` | `See upstream model card license metadata` | `xiao2018upernet;liu2021swin` | ADE20K-finetuned UPerNet Swin-Large snapshot for offline local init. |
| `transunet_tiny_vit_tiny_patch16_imagenet` | `transunet_tiny` | `timm:vit_tiny_patch16_224` | `See upstream timm model card/license terms` | `dosovitskiy2020vit` | Partial warm-start bootstrap mapped into internal `transunet_tiny` tensor shapes. |
| `segformer_mini_vit_tiny_patch16_imagenet` | `segformer_mini` | `timm:vit_tiny_patch16_224` | `See upstream timm model card/license terms` | `dosovitskiy2020vit` | Partial warm-start bootstrap mapped into internal `segformer_mini` tensor shapes. |

## Manual Upstream Download URLs

Use these URLs on a connected machine when manual download is required.

| Model ID | Direct URL(s) |
|---|---|
| `hf_segformer_b0_ade20k` | `https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/pytorch_model.bin` ; `https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors` |
| `hf_segformer_b2_ade20k` | `https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/main/pytorch_model.bin` |
| `hf_segformer_b5_ade20k` | `https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640/resolve/main/pytorch_model.bin` |
| `hf_upernet_swin_large_ade20k` | `https://huggingface.co/openmmlab/upernet-swin-large/resolve/main/pytorch_model.bin` ; `https://huggingface.co/openmmlab/upernet-swin-large/resolve/main/model.safetensors` |
| `smp_unet_resnet18_imagenet` | encoder source: `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/pytorch_model.bin` |
| `smp_deeplabv3plus_resnet101_imagenet` | encoder source: `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` |
| `smp_unetplusplus_resnet101_imagenet` | encoder source: `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` |
| `smp_pspnet_resnet101_imagenet` | encoder source: `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` |
| `smp_fpn_resnet101_imagenet` | encoder source: `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` |
| `unet_binary_resnet18_imagenet_partial` | `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/pytorch_model.bin` |
| `transunet_tiny_vit_tiny_patch16_imagenet` | `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin` |
| `segformer_mini_vit_tiny_patch16_imagenet` | `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin` |

## Architecture And Manuscript Discussion Companion

For architecture diagrams, critical model comparison, and manuscript-focused discussion prompts:
- `docs/model_architecture_manuscript_foundation.md`

## Manuscript Citation Source

- BibTeX file: `docs/pretrained_model_citations.bib`
- Recommended practice:
  - cite architecture papers (SegFormer, U-Net, ResNet)
  - cite ViT and TransUNet papers for `transunet_tiny` bootstrap discussions
  - cite ViT and SegFormer papers for `segformer_mini` bootstrap discussions
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
- Local-pretrained support is implemented for `unet_binary`, SMP backends (`smp_unet_resnet18`, `smp_deeplabv3plus_resnet101`, `smp_unetplusplus_resnet101`, `smp_pspnet_resnet101`, `smp_fpn_resnet101`), HF transformer backends (`hf_segformer_b0/b2/b5`, `hf_upernet_swin_large`), `transunet_tiny`, and `segformer_mini`.
