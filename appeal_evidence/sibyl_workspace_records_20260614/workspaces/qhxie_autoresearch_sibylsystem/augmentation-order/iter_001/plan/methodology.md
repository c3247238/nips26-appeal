# Methodology: Does Augmentation Operation Ordering Matter?

## Overview

This study isolates augmentation operation ordering as the **sole independent variable** in image
classification pipelines. We combine (1) a Wasserstein Non-Commutativity (NC) theoretical bound,
(2) a DPI-based reversibility principle, and (3) a controlled factorial empirical study across
permutations, architectures (ResNet-18 vs. ViT-Small), datasets (CIFAR-10, CIFAR-100), and
augmentation magnitudes.

The active candidate is `cand_a`: Theory-Grounded Augmentation Ordering Study with revised
hypotheses H1–H4, new H5b (category-level interleaved ordering) replacing falsified H5, and new
H6 (inverted-U magnitude interaction).

**Pilot status (as of 2026-04-03)**: All Tier 0 and Tier 1 pilots completed at small scale (1 seed,
10 epochs, 100-sample subsets). Full-scale runs (5 seeds, 200 epochs, full datasets) are the
immediate next step.

---

## Experimental Setup

### Hardware and Environment

- Framework: PyTorch + torchvision + timm
- Models: ResNet-18 (11M params, SGD + cosine), ViT-Small patch=4 (22M params, AdamW + warmup)
- Datasets: CIFAR-10, CIFAR-100 (standard 50k train / 10k test splits)
- Optimizer (ResNet-18): SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, cosine annealing
- Optimizer (ViT-Small): AdamW, lr=1e-3, weight_decay=0.05, 5-epoch linear warmup + cosine decay
- Seeds: 42 (pilot); seeds [42, 43, 44, 45, 46] for full experiments (paired across orderings)
- GPU: Single RTX PRO 6000 per run (24 GB VRAM); runs parallelized across available GPUs

### Augmentation Operations

**3-Operation set (Tier 0 and Tier 1):**
- `Crop`: `transforms.RandomCrop(32, padding=4)` — lowest reversibility (discards pixels)
- `Flip`: `transforms.RandomHorizontalFlip(p=0.5)` — highest reversibility (strictly bijective, eta=1)
- `ColorJitter`: `transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)` — medium reversibility (approximately bijective)

**Corrected reversibility ranking** (revised from prior proposal):
RandomHorizontalFlip is strictly bijective — it is its own inverse. Correct order from highest to lowest reversibility: Flip > ColorJitter > Crop.
DPI-corrected prediction: Flip-first orderings (Flip→X→Y) should outperform Crop-first orderings.
Empirically consistent: Flip→Crop→CJ wins 2/4 blocks in pilot.

**6-Operation set (Tier 2):**
- Geometric group: `RandomCrop(32, pad=4)`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`
- Photometric group: `ColorJitter(0.4,0.4,0.4,0.1)`, `RandomGrayscale(p=0.1)`, `GaussianBlur(kernel_size=3)`

All pipelines end with `ToTensor()` and `Normalize(mean, std)` (never permuted).

---

## Experimental Tiers

### Tier 0: Pilot (COMPLETED)

- 6 orderings of {Crop, Flip, CJ}, ResNet-18, CIFAR-10 5k subset, 10 epochs, seeds [42,43]
- Result: spread = 2.68%; CJ→Crop→Flip best (28.04%), Flip→CJ→Crop worst (25.36%)
- Decision: GO (high confidence)

### Tier 1: Full Factorial on 3 Operations (FULL-SCALE REQUIRED)

**Purpose**: Test H1 (ordering effect) and H2 (architecture differential) with proper statistical inference.

- Orderings: All 6 permutations of {Crop, Flip, CJ}
  - order_0: Crop→Flip→CJ (conventional torchvision default)
  - order_1: Crop→CJ→Flip
  - order_2: Flip→Crop→CJ
  - order_3: Flip→CJ→Crop
  - order_4: CJ→Crop→Flip
  - order_5: CJ→Flip→Crop (reversibility-sorted)
- Architectures: ResNet-18, ViT-Small/4
- Datasets: CIFAR-10, CIFAR-100
- Seeds: 5 (paired — same seed, different ordering per run)
- Epochs: 200
- Total runs: 6 × 2 × 2 × 5 = **120 runs**
- Estimated wall-clock: ~10h on 4 parallel GPUs

**Primary output**: Per-ordering final accuracy (mean ± std over 5 seeds) per (arch, dataset) block.

**Pilot status**: Completed at 1 seed, 10 epochs, 100-sample subsets. Spread confirmed in 3/4 blocks.
Full-scale is blocking for all downstream statistical analyses.

### Tier 2: Category-Level Ordering Confirmation Pilot (REQUIRED BEFORE FULL TIER 2)

**Purpose**: Validate the 9.01% pilot effect (1 seed, 5k samples, 10 epochs) before committing to
full 100-run Tier 2 investment.

- 5 orderings: all-geo-first, all-photo-first, interleaved G-P, interleaved P-G, random-per-image
- Architecture: ResNet-18 and ViT-Small
- Dataset: CIFAR-10 and CIFAR-100 (2 arch-dataset combinations for cross-check)
- Seeds: 2 (seeds 42, 43)
- Epochs: 30
- Total runs: 5 × 2 × 1 × 2 = **20 runs** (~1.5 GPU-hours)
- Pass criterion: interleaved_PG spread > 1% on CIFAR-10/ResNet-18 at 30 epochs (softer than pilot 9%)

### Tier 2: Category-Level Ordering Full (CONDITIONAL ON CONFIRMATION PILOT)

**Purpose**: Test H5b (interleaved photometric-geometric ordering dominates) at full scale.

- 5 canonical orderings × 2 architectures × 2 datasets × 5 seeds = **100 runs**
- Same operations as confirmation pilot
- 200 epochs
- Estimated wall-clock: ~9h on 4 GPUs

### Tier 3: Magnitude Interaction Full (REQUIRED — Test H6)

**Purpose**: Test H6 (inverted-U relationship between magnitude and ordering spread).

- Best and worst orderings from Tier 1 full analysis (determined post-Tier 1)
- Magnitudes: M=5 (low), M=9 (medium), M=14 (high)
  - M=5: ColorJitter(0.15,0.15,0.15), pad=1, rotation=5°
  - M=9: standard (same as Tier 1)
  - M=14: ColorJitter(0.8,0.8,0.8), pad=8, rotation=30°
- Architecture: ResNet-18 only
- Dataset: CIFAR-100 only (harder task)
- Seeds: 5
- Total: 2 × 3 × 1 × 1 × 5 = **30 runs** (~6 GPU-hours)

### Tier 4: Feature-Space NC_2 Computation (REVISED — Test H3)

**Purpose**: Test H3 revised — feature-space NC_2 predicts accuracy ranking better than pixel-space.

**4a — Feature-Space NC_2** (10k samples, 512-D penultimate-layer embeddings from trained ResNet-18):
- Use Tier 1 trained ResNet-18/CIFAR-10 checkpoint (200 epochs)
- Extract penultimate-layer features (512-D) for 10k images under each ordering
- Compute SWD (Sliced Wasserstein Distance, 1000 projections) between P_AB and P_BA in feature space
- Also compute pixel-space NC_2 with proper sample size (10k, 1000 projections) for direct comparison
- Spearman rho between NC_2 ranking and accuracy-difference ranking; permutation test
- Expected time: 30 min (CPU + single GPU for feature extraction)

**4b — InfoNCE MI Estimation with Proper Encoder** (revised from pilot):
- Use properly trained ResNet-18 checkpoints (from Tier 1 full, 200 epochs)
- Estimate I(y; augmented_x) via InfoNCE lower bound with frozen pretrained features
- 10k samples, proper encoder (not the underpowered 10-epoch pilot encoder)
- Expected time: 1h (CPU/single GPU)

---

## Baselines (Completed at Pilot Scale — Rerun at Full Scale with Tier 1)

| Baseline | Description |
|---|---|
| Conventional (Crop→Flip→CJ) | torchvision default; one of the 6 Tier 1 orderings |
| Random-per-image | Operations shuffled each sample |
| TrivialAugment | Single operation per image (timm) |
| No augmentation | Only ToTensor + Normalize |
| RandAugment (N=2, M=9) | State-of-the-art reference |

Baseline training completed at pilot scale: RandAugment N=2 M=9 achieves best pilot performance
(CIFAR-100 ResNet-18 pilot: 0.7283; CIFAR-100 ViT pilot: 0.5985).
Full-scale baselines run jointly with Tier 1 (conventional ordering is one of the 6 orderings).

---

## Statistical Analysis Plan

- **H1 (primary)**: Paired t-test (same seed, different ordering) between best and worst ordering per
  (arch, dataset) block. Bonferroni correction across 4 blocks. Threshold: p < 0.05, spread > 0.3%, |d| > 0.2
- **H2 (interaction)**: Two-way ANOVA ordering × architecture. Partial eta-squared for interaction.
  Threshold: p < 0.05, eta-sq > 0.05
- **H3 (feature-space NC_2)**: Spearman rho + permutation test (10k permutations). Threshold: rho > 0.5, p < 0.05
- **H4 (DPI reversibility)**: Paired comparison Flip-first vs. conventional (Crop-first) across 5 seeds.
  Threshold: Flip-first better in ≥2/4 blocks; InfoNCE MI Spearman rho > 0.4
- **H5b (category interleaved)**: Paired t-test (interleaved P→G vs. all-geometric-first) per block.
  Threshold: interleaved better in ≥3/4 blocks, p < 0.05, |d| > 0.2
- **H6 (magnitude inverted-U)**: Quadratic regression of spread on magnitude level (M=5,9,14).
  Threshold: negative quadratic term (beta_2 < 0), p < 0.05; OR M9 spread > M5 and M9 spread > M14
- **Null result**: If all spreads < 0.2% in 5-seed full experiment, N0 confirmed and published as
  rigorous negative result

---

## Expected Visualizations

- **Table 1** (main results): 6 orderings + 5 baselines × 4 arch-dataset blocks; mean ± std; best bolded; spread in final row
- **Figure 1**: Architecture diagram — schematic showing 3 transforms composing in different orders through distribution space
- **Figure 2**: Ordering × architecture spread comparison — violin plots of accuracy distribution by ordering, ResNet-18 vs. ViT-Small side by side (H2)
- **Figure 3**: NC_2 vs. accuracy-diff scatter — pixel-space vs. feature-space NC_2, showing improvement in predictive power (H3)
- **Figure 4**: DPI reversibility bar chart — InfoNCE MI per ordering with accuracy overlay (H4)
- **Figure 5**: Category-level ordering results — bar chart for 5 Tier 2 orderings × 4 arch-dataset combos (H5b)
- **Figure 6**: Magnitude interaction — spread vs. magnitude (M=5,9,14) with inverted-U fit (H6)
- **Appendix Table A1**: Per-class accuracy breakdown for CIFAR-100 (zero-cost secondary analysis from cand_c)
- **Appendix Figure A1**: Learning curves per ordering (Tier 1, all 4 blocks)

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Effect shrinks at full scale (200ep) | Paired seed design reduces noise; pilot shows effects at 10ep/100-sample scale |
| ViT training instability on CIFAR | Established timm recipe (patch=4); verified in pilot |
| Category ordering effect (9.01%) is noise | Tier 2 confirmation pilot (30 epochs, 2 seeds) required before full Tier 2 investment |
| Feature-space NC_2 also fails to predict | Would be a publishable negative finding; update theory section accordingly |
| Theorem 1 proof gap (bubble-sort decomposition) | Acknowledged; weakened claim or deferred to appendix |
| DPI contraction coefficient input-distribution dependence | Weakened to directional claim; formal revision in camera-ready |
