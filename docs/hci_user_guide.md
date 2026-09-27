# Measuring Hydride Connectivity (HCI): User Guide

The Hydride Connectivity Index tells you how far the hydrides in a segmented micrograph link into
long connected paths. It is independent of how much hydride there is. This guide shows how to
obtain it and how to read it. The theory is in [`hci_theory.md`](hci_theory.md).

## 1. In the Browser (Enabled by Default)

1. Choose an image and a segmentation method as usual.
2. Under **5. Connectivity (HCI)**, keep *Compute the Hydride Connectivity Index* ticked. Then:
   - **Bridging range δ_max**. The default is automatic: one fifth of the smallest accepted hydride.
     To compare several specimens, choose **Fixed** and enter the same value for all of them,
     for example 10 µm.
   - **Pixel size** (optional). Enter µm per pixel so that lengths are in micrometres and results
     compare across magnifications. Read it from the scale bar: 100 µm divided by the bar length in
     pixels.
   - **Radial reference length** (optional). Enter the wall thickness, so that HCI radial = 1 means
     a through-wall network.
3. Run the segmentation. The **HCI panel** shows three values:
   - **radial**, which matters for through-wall cracking;
   - **circumferential**;
   - **isotropic**.

   It also shows δ½, the bridging range used, and the counts and flags.
4. Inspect the four HCI views:

   | View | What it shows |
   |---|---|
   | HCI clusters | One colour per linked cluster; red lines are synthetic bridges, not hydride |
   | HCI curve | The connectivity function C(δ); the shaded area divided by δ_max is HCI radial |
   | HCI topology | Free ends and junctions |
   | HCI path | The weakest radial path |

5. **Download detailed report** adds an HCI page. The ZIP contains a *Connectivity* sheet with the
   curve data and cluster table.

## 2. From the Command Line

```bash
python scripts/microseg_cli.py hci --mask outputs/masks --pixel-size-um 0.5 --max-bridge-distance 10 --output-dir outputs/hci
```

- `--mask` accepts a single mask or a folder. The masks are 0 for matrix and 1 for hydride; 0/255
  and red-on-black masks are also accepted.
- Each image gets `hci_result.json`, `clusters.csv` and four PNG views. The run writes
  `report.json` and `summary.csv`.
- Omit `--max-bridge-distance` to use the automatic rule. Every other setting is in
  `configs/hci.default.yml` and can be overridden with `--set hci.<key>=<value>`.

## 3. From Python

```python
import numpy as np
from PIL import Image
from src.microseg.evaluation.continuity import Calibration, ContinuityAnalysisConfig, analyze_continuity

mask = np.array(Image.open("mask.png")) > 0
res = analyze_continuity(mask, ContinuityAnalysisConfig(max_bridge_distance=10.0), Calibration(0.5, "scale bar"))
print(res.status, res.specimen["HCI"], res.specimen["critical_linking_distance"])
```

## 4. Reading the Numbers

| Value | Meaning | Typical use |
|---|---|---|
| HCI radial | 0 → isolated specks; 1 → every hydride in a cluster spanning the wall | Headline for tubes |
| HCI circumferential / isotropic | The same along the tube circumference / in any direction | Circumferential hydrides; unoriented specimens |
| δ½ | Matrix gap at which half the hydride joins half-field clusters (µm) | Physical spacing; lower means more closely spaced |
| Π radial | Mean hydride fraction of the best through-wall path through each hydride | Comparison with RHCP-type metrics |
| κ | Fraction of potential free ends closed into loops | Network character, reported beside the HCI |

## 5. Rules for Reporting

- State δ_max and whether it was automatic or fixed. Also state the pixel size, the reference
  length and the segmentation method. All four are in the JSON and the PDF.
- Compare specimens only at the **same fixed δ_max**, pixel size and reference length.
- Report the HCI together with the area fraction and Fn; the HCI does not replace them.
- Check the flags. For example, *largest cluster touches border* means the value is a lower
  bound, and *thickness below 3 px* means the topology is unreliable.
- The HCI is experimental: validated on synthetic masks, and not yet correlated with mechanical
  properties.
