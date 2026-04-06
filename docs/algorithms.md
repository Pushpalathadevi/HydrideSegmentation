# Algorithms And Mathematics

This page explains the mathematics behind the main segmentation and analysis pathways.

For the end-to-end classical segmentation flow sheet, parameter meaning, and tuning guidance, see [`docs/conventional_segmentation_pipeline.md`](conventional_segmentation_pipeline.md).

## Binary Segmentation Core

Let a binary mask be $M \in \{0,1\}^{H \times W}$.

The area fraction is:

$$
f = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} \mathbb{1}[M_{ij} > 0]
$$

The connected-feature count is the number of connected components in $M > 0$, computed by connected-component labeling.

For component $k$, the area is:

$$
a_k = \sum_{i,j}\mathbb{1}[L_{ij} = k]
$$

and the equivalent diameter is:

$$
d_k = 2 \sqrt{\frac{a_k}{\pi}}
$$

These are the core quantities used in desktop reporting and validation summaries.

## Orientation Estimation

For each connected component, the implementation:

1. fills holes,
2. dilates slightly,
3. skeletonizes the feature,
4. computes the covariance of skeleton coordinates,
5. extracts the dominant eigenvector,
6. converts that vector to an angle in degrees.

If the skeleton coordinates are $x_1, \ldots, x_n$, then the covariance matrix is:

$$
\Sigma = \frac{1}{n-1}\sum_{k=1}^{n}(x_k - \bar{x})(x_k - \bar{x})^\top
$$

Let $v_1$ be the eigenvector corresponding to the largest eigenvalue of $\Sigma$. The component orientation is:

$$
\theta = \operatorname{atan2}(v_{1y}, v_{1x})
$$

The code maps $\theta$ into a 0-90 degree line-orientation interval, because hydride platelets are directionally symmetric under 180-degree rotation.

## Orientation Alignment Index

The orientation alignment index summarizes how concentrated the component orientations are:

$$
A = \left|\frac{1}{N}\sum_{k=1}^{N} e^{2 i \theta_k}\right|
$$

where $\theta_k$ is in radians. Values near 1 indicate strong alignment; values near 0 indicate broad orientation spread.

## Entropy Of Orientation Histogram

Given orientation-bin probabilities $p_b$, the entropy in bits is:

$$
H = -\sum_b p_b \log_2(p_b)
$$

This is used as a compact diversity measure for orientation distributions.

## Evaluation Metrics

For binary segmentation, the common contingency counts are:

$$
TP,\ FP,\ TN,\ FN
$$

Foreground precision, recall, specificity, IoU, and Dice are:

$$
\text{precision} = \frac{TP}{TP + FP}
$$

$$
\text{recall} = \frac{TP}{TP + FN}
$$

$$
\text{IoU} = \frac{TP}{TP + FP + FN}
$$

$$
\text{Dice} = \frac{2TP}{2TP + FP + FN}
$$

The Matthews correlation coefficient is:

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

when the denominator is nonzero.

The evaluation layer also computes Cohen's kappa, macro precision, macro recall, weighted F1, balanced accuracy, and frequency-weighted IoU.

## Conventional Pipeline Summary

The classical baseline used in this repository follows:

1. local contrast normalization,
2. local thresholding,
3. morphology-based cleanup,
4. connected-component filtering,
5. optional export and analysis.

The implementation is deliberately simple so that the resulting mask can be inspected and reasoned about without interpreting a learned feature hierarchy.

## How To Read The Classical Baseline

- CLAHE controls how much local contrast is exposed.
- Adaptive thresholding turns grayscale intensity into a candidate foreground mask.
- Morphological closing repairs small gaps.
- Connected components separate true plates from isolated noise.
- Area filtering removes objects that are too small to be scientifically meaningful.

## Scientific Distance Metrics

The hydride-specific scientific metrics compare predicted and ground-truth distributions of component sizes and orientations.

The implementation uses:

- Wasserstein distance $W_1$, a transport-based distance
- Kolmogorov-Smirnov statistic $D_{KS}$, a maximum CDF deviation

These metrics are reported for both size distributions and orientation distributions so users can tell whether a model is matching the morphology distribution, not only the pixelwise overlap.

## Baseline Pixel Classifier

The classical pixel classifier samples RGB pixels and trains a logistic-loss stochastic gradient descent model.

Input features are:

$$
x = [R, G, B] / 255
$$

The classifier learns a decision boundary over per-pixel color space and is therefore fast and CPU-friendly, but it has limited spatial context.

## UNet Binary Trainer

The binary UNet trainer uses a segmentation model and optimizes:

$$
\mathcal{L} = \text{BCEWithLogitsLoss}(z, y)
$$

where $z$ are raw logits and $y$ are binary targets.

The trainer reports:

- batch and epoch loss
- training IoU
- validation loss and validation IoU
- tracked validation exemplar panels

Optimization is Adam-based with checkpointing, resume support, gradient accumulation, and optional AMP.

## Why These Metrics Matter Scientifically

Pixel overlap alone is not enough for hydride and microstructural studies.

The documentation therefore separates:

- overlap quality
- count fidelity
- orientation fidelity
- size-distribution fidelity
- deployment/runtime evidence

That separation is the basis for the repository's scientific traceability standard.
