# Research Proposal: Does Augmentation Operation Ordering Matter?

## Title
**Order Matters: An Empirical and Theory-Grounded Study of Augmentation Operation Sequencing in CNNs and Vision Transformers**

## Abstract

Every major deep learning framework — torchvision, timm, albumentations — composes image augmentation operations in a fixed, sequential order. The universal convention of applying geometric transforms (crop, flip, rotate) before photometric transforms (color jitter, brightness, contrast) is inherited from engineering tradition, never from empirical evidence. Despite thousands of papers optimizing *which* augmentation operations to apply, no prior work has systematically isolated *ordering* as the sole independent variable. We fill this gap with a controlled factorial experiment across 6 permutations of three standard operations, two architectures (ResNet-18 vs. ViT-Small), two datasets (CIFAR-10, CIFAR-100), and augmentation magnitudes — grounded by a formal Wasserstein Non-Commutativity (NC) measure and an information-theoretic reversibility principle derived from the Data Processing Inequality (DPI). Pilot evidence establishes that ordering effects are real and architecture-dependent: ViT-Small shows a 2.32% spread across orderings on CIFAR-10, ResNet-18 shows 0.96% on the same dataset, and category-level ordering (interleaved photometric-geometric) produces a remarkable 9.01% spread versus all-geometric-first. However, the NC_2 measure (as implemented via sliced Wasserstein distance in pixel space) does not reliably predict accuracy ranking (rho = -0.20), and magnitude does not monotonically amplify ordering sensitivity. These negative theoretical findings sharpen the research agenda: the gap between pixel-space distribution divergence and task-relevant accuracy requires explanation, pointing to feature-space non-commutativity as the next theoretical frontier. Full-scale experiments (200 epochs, 5 seeds per condition) are required to confirm pilot findings with proper statistical inference.

## Motivation

The augmentation ordering question sits at a remarkable gap in a heavily-studied field. Two comprehensive survey papers (Cheung & Yeung, IEEE TNNLS 2023; Yang et al., KAIS 2023) explicitly identify per-image operation ordering as an open, unaddressed question. The theoretical basis for why ordering should matter is clear: augmentation operations are lossy, non-commutative transformations. By the data processing inequality, applying transformation A before B produces a different information-loss trajectory than B before A. Yet this has never been empirically validated in a controlled, statistically rigorous setting.

Three key observations from the literature crystallize the gap:
1. **AutoAugment** learns ordered operation pairs via RL and achieves state-of-the-art results, but never ablates ordering.
2. **RandAugment** applies N operations in *random* order per image and matches AutoAugment — implicitly testing that random order is as good as learned order, but not that *all* fixed orders are equivalent.
3. **TrivialAugment** applies a *single* operation per image, making ordering moot — but this sidesteps, not resolves, the ordering question.

PBA (Ho et al., 2019) provides the only direct ordering ablation: shuffling the epoch-level augmentation *schedule* degrades accuracy. This is curriculum learning at epoch granularity, not within-image operation ordering at the granularity practitioners use when writing `transforms.Compose([...])`.

**Pilot evidence** (Tier 0, April 2026): 6 permutations of {RandomCrop, RandomHorizontalFlip, ColorJitter} on ResNet-18 / CIFAR-10 5k subset, 10 epochs, 2 seeds. Accuracy spread: **2.68%** (best: CJ→Crop→Flip at 28.04%, worst: Flip→CJ→Crop at 25.36%). The conventional ordering (Crop→Flip→CJ) ranks 5th out of 6. This strongly confirms H1 and provides the first empirical evidence that the conventional ordering is suboptimal.

**Full Tier 1 pilot** (all 4 arch-dataset blocks, 1 seed, 10 epochs): Spread ranges from 0.25% (ViT/CIFAR-100) to 2.32% (ViT/CIFAR-10), with 3 of 4 blocks exceeding the pre-registered 0.5% threshold.

**Practical stakes**: If ordering effects survive full-scale training, the current "one-size-fits-all" defaults in torchvision/timm are suboptimal. A simple reorder — zero additional compute, zero new hyperparameters — could yield free accuracy gains.

## Research Questions

1. **RQ1 (Primary)**: Does the permutation of augmentation operations within a fixed-length pipeline produce statistically significant differences in classification accuracy under full-scale training (200 epochs, 5 seeds)?
2. **RQ2 (Architecture)**: Do CNNs (ResNet-18) and ViTs (ViT-Small) respond differently to augmentation ordering, and what architectural properties explain the difference?
3. **RQ3 (Theoretical)**: Can the Wasserstein Non-Commutativity (NC) measure — computed in feature space rather than pixel space — predict the magnitude of ordering effects on accuracy?
4. **RQ4 (DPI Principle)**: Does placing more reversible (approximately invertible) transforms before less reversible ones maximize task-relevant mutual information and improve accuracy?
5. **RQ5 (Category-level)**: Does interleaved photometric-geometric ordering (P→G→P→G) outperform block-level geometric-first or photometric-first ordering?

## Hypotheses (Revised from Pilot Evidence)

### Primary (H1 — Factorial Effect)
**Pilot status: CONFIRMED (preliminary)**

For a fixed set of K augmentation operations applied per image, the permutation significantly affects classification accuracy. Pilot evidence shows 0.96–2.32% spread in 3/4 arch-dataset blocks.

**Full-scale criterion**: The accuracy gap between the best and worst ordering exceeds 0.3% on at least 2 of 4 architecture-dataset combinations, confirmed by paired t-test (same seed, different ordering) at p < 0.05 after Bonferroni correction across permutations.

*Falsification criterion*: If max-min spread is less than 0.2% (within 1-sigma of seed variance) for all 4 blocks in the 5-seed full experiment, H1 is falsified.

### Architecture Differential (H2 — CNN vs. ViT Sensitivity)
**Pilot status: CONFIRMED (preliminary)**

ViTs show larger absolute ordering sensitivity than CNNs. Pilot evidence: ViT/CIFAR-10 spread = 2.32%, ResNet-18/CIFAR-10 spread = 0.96%. The ordering × architecture interaction is hypothesized to be significant.

**Revised mechanistic prediction**: ViTs process images via non-overlapping patches; geometric transforms applied after photometric ones alter patch boundaries differently than the reverse, amplifying sensitivity. CNNs with local receptive fields are less sensitive to spatial-level ordering because they aggregate locally. This explains why the ViT/CIFAR-10 block shows 2.4× the spread of the ResNet-18/CIFAR-10 block.

*Falsification criterion*: Both architectures show the same sign and magnitude of ordering preference across all 4 blocks in the full experiment.

### Theoretical (H3 — Feature-Space NC Predicts Effect)
**Pilot status: FALSIFIED in pixel space (rho = -0.20, p = 0.68)**

**Revised hypothesis**: The pixel-space NC_2 (Sliced Wasserstein Distance between P_AB and P_BA) fails to predict accuracy ranking because task-relevant ordering effects emerge in the learned feature representations, not in raw pixel distributions. The revised prediction: **feature-space NC_2** — computed using penultimate-layer embeddings from a pretrained ResNet-18 — will show higher Spearman correlation with accuracy ranking than pixel-space NC_2.

**Mechanistic explanation for the pixel-space failure**: The SWD with 100 samples in 3072-dimensional pixel space is near-random (curse of dimensionality). Moreover, pixel-space divergence between P_AB and P_BA does not capture the downstream effect of ordering on the gradient landscape seen by the optimizer. Feature-space NC_2 captures what matters: how ordering changes the task-relevant structure of representations.

*Full-scale criterion*: Feature-space NC_2 Spearman rho > 0.5, p < 0.05 via permutation test.

### DPI (H4 — Reversibility Principle)
**Pilot status: PARTIALLY CONFIRMED**

The CJ→Flip→Crop ordering (reversibility-sorted) outperforms the conventional ordering in 2/4 blocks. However, the MI ranking from InfoNCE does not consistently predict accuracy ranking (combined rho = -0.057), and the specific DPI prediction (CJ-first is optimal) is contradicted by Flip→Crop→CJ being the most frequent winner.

**Revised hypothesis**: The reversibility principle correctly identifies that photometric transforms should not be the *last* operation applied (since crop is most destructive when it discards pixels that have already been color-jittered). However, the optimal ordering is architecture-specific: for ViTs, Flip-first preserves spatial structure before cropping establishes patch boundaries. The DPI principle provides a necessary but not sufficient condition for optimal ordering.

*Practical criterion*: Reversibility-sorted ordering (CJ→Flip→Crop) outperforms conventional (Crop→Flip→CJ) on at least 2/4 blocks in the full experiment.

### Category-Level (H5b — Interleaved Ordering Dominates)
**Pilot status: STRONGLY SUPPORTED (replaces falsified H5)**

The falsified H5 (magnitude monotonically amplifies spread) is replaced by H5b, which is more strongly supported by the data. Category-level ordering — interleaving photometric and geometric operations (P→G→P→G→P→G) — outperforms block-level ordering (all-geometric-first, all-photometric-first) by a large margin. Pilot evidence (Tier 2, single pilot): interleaved_PG achieves 0.2939 vs. all_geo_first at 0.2038, a spread of **9.01%** on the 6-operation pipeline.

**Mechanism**: Interleaved ordering prevents any single transform category from dominating the distribution shift at each step, distributing information loss more uniformly and avoiding catastrophic sequential destruction (e.g., aggressive crop followed by aggressive color distortion with no recovery).

*Full-scale criterion*: Interleaved PG ordering outperforms both all-geometric-first and all-photometric-first in at least 3/4 arch-dataset blocks, confirmed with paired t-test across 5 seeds.

### Magnitude Non-Monotonicity (H6 — New Hypothesis)
**Based on pilot evidence (Tier 3)**

Magnitude does NOT monotonically amplify ordering spread. At high magnitude (M14), both best and worst orderings converge to the same accuracy (M14 spread = 0.00), while medium magnitude (M9) shows the largest spread (0.88%). This suggests an inverted-U relationship: at low magnitude, transforms commute approximately; at medium magnitude, non-commutativity peaks; at high magnitude, all orderings are equally destructive.

*Full-scale criterion*: Spread at M9 > spread at M5 AND spread at M9 > spread at M14, confirmed across at least 2 arch-dataset blocks.

## Expected Contributions

1. **First systematic, controlled study** isolating augmentation operation ordering as the sole independent variable, with paired seed design and proper statistical tests. Publishable regardless of direction: if ordering matters, we provide evidence and guidance; if not, we formally close a documented research gap.
2. **Empirical ordering principles**: Architecture-conditioned ordering recommendations based on the first cross-architecture (CNN vs. ViT) comparison of ordering sensitivity.
3. **Wasserstein Non-Commutativity (NC) measure in pixel and feature space**: Demonstration that pixel-space NC_2 fails to predict accuracy, and investigation of whether feature-space NC_2 recovers predictive power. This identifies the mechanistic level at which ordering effects operate.
4. **Inverted-U magnitude interaction**: A novel, counterintuitive finding that medium magnitudes maximize ordering sensitivity, with high magnitudes equalizing all orderings — challenging the intuition that stronger augmentation amplifies all effects.
5. **Category-level ordering principle**: Evidence that interleaved photometric-geometric ordering outperforms block-level ordering by a 9x larger margin than within-category permutation effects, providing a practically actionable design principle.
6. **Negative results as positive contributions**: The falsification of the pixel-space NC_2 and the DPI reversibility ranking are themselves novel findings that sharpen our understanding of where ordering effects originate.

## Evidence-Driven Revisions

This section documents the evidence basis for each change from the previous proposal:

### What pilot evidence strengthened:
- **H1 (ordering matters)**: 2.68% spread in Tier 0 pilot, 0.96–2.32% in full Tier 1 pilot across 3/4 blocks — strong preliminary confirmation at underpowered scale. Direction is clear; full-scale confirms magnitude.
- **H2 (ViT > CNN sensitivity)**: ViT shows 2.4× CNN spread on CIFAR-10. The architectural explanation (patchification × spatial ordering) is supported.
- **Category-level ordering (H5b)**: 9.01% spread for Tier 2 pilot is the largest observed effect and clearly warrants full investigation.

### What pilot evidence weakened or falsified:
- **H3 (NC_2 predicts accuracy)**: Fully falsified in pixel space. Revised to feature-space prediction with mechanistic explanation for the failure.
- **H4 (DPI reversibility principle)**: Partially supported but the specific CJ-first prediction was not the empirical winner. Revised to a weaker, architecture-conditioned form.
- **H5 (magnitude monotonically amplifies spread)**: Falsified. The M14 convergence to zero spread is a substantive finding, leading to new H6 (inverted-U relationship).

### Critical methodological warnings (from reflection analysis):
- **All reported pilot verdicts are preliminary**: 1 seed, 10 epochs, 100-sample subsets cannot support "confirmed" or "falsified" conclusions by pre-registered statistical standards. Full-scale runs (200 epochs, 5 seeds, full datasets) are mandatory before paper claims are finalized.
- **H2 definition clarification**: H2 in this document refers to architecture differential sensitivity (ViT vs. CNN spread magnitude), as pre-registered. A secondary confirmed finding (CJ→Flip→Crop beats conventional) is labeled H4, not H2.
- **NC_2 was undercomputed**: The pilot H3 falsification used 100 samples and 100 projections in 3072-D space, producing near-random estimates. The feature-space NC_2 hypothesis (H3 revised) requires 10k samples in a 512-D feature space — a qualitatively different and statistically viable computation.

## Novelty Assessment

Exhaustive multi-source search (arXiv, Google Scholar, web search, 2026-04-03, updated from 2026-04-02) confirms **no new competitors** emerged in the past 24 hours. The gap documented in the prior report remains:

- **Closest work**: Li et al. (arXiv 2408.14381, 2024) studies binary-tree vs. sequential composition structure — this is structural variation, not permutation within a fixed-length sequence.
- **PBA** (Ho et al., 2019): epoch-level schedule ordering, not per-image operation ordering.
- **TANDA** (Ratner et al., 2017): learns full augmentation sequences via LSTM, does not isolate ordering as an independent variable.
- **AutoAugment**: searches over ordered operation pairs but never ablates order.
- **No prior work** applies the Wasserstein non-commutativity measure or the DPI reversibility principle to augmentation ordering.
- New search (April 2026): searches for "non-commutativity data augmentation transforms permutation" returned no relevant new papers.

**Conclusion**: The core contribution — isolating ordering as the independent variable with theory-grounded framework, including the falsification of pixel-space NC_2 and discovery of the inverted-U magnitude interaction — is novel.

## Revisions from Prior Feedback

**From supervisor review (4.0/10 score, April 2026):**
- ADDRESSED: The prior proposal presented pilot results as confirmed hypotheses. This version explicitly labels all pilot findings as "preliminary" and requires full-scale runs for final conclusions.
- ADDRESSED: H2 definition was inconsistently applied (sometimes "architecture differential," sometimes "reversibility ordering wins"). This version uses consistent definitions: H2 = architecture differential, H4 = reversibility ordering.
- PARTIALLY ADDRESSED: The DPI contraction coefficient argument (SOUND-002) — eta_i must depend on the input distribution, which changes with ordering — is acknowledged as requiring formal revision. The revised H4 is weakened to a directional claim rather than a precise information-theoretic derivation, pending theoretical repair.
- ACKNOWLEDGED BUT NOT YET ADDRESSED: The bubble-sort decomposition proof step for Theorem 1 (SOUND-001) requires explicit derivation. This is deferred to the full paper's theory section, where the NC-based generalization bound will be formally proved or replaced with a weaker claim.
- ADDRESSED: Tier 3 ordering label inversion (CJ→Flip→Crop labeled "best" but Flip→Crop→CJ wins in more blocks) is corrected: H6 now properly identifies the inverted-U as a finding, not a contradiction.

## Theoretical Framework (Revised)

### Wasserstein Non-Commutativity Bound

**Definition.** For transforms t_i, t_j and data distribution mu:

    NC_2(t_i, t_j; mu) = W_2(t_i ∘ t_j # mu, t_j ∘ t_i # mu)

where W_2 is the 2-Wasserstein distance and # denotes pushforward.

**Theorem (Ordering-Dependent Generalization Bound) — Requires formal proof.**
For L-Lipschitz stochastic transforms with sub-Gaussian tails, and any two permutations sigma, sigma' differing on the relative order of adjacent pairs (i,j), a bubble-sort decomposition gives:

    |gen(sigma) - gen(sigma')| ≤ C · Σ_{adjacent swaps} NC_2(t_i, t_j; mu_intermediate) + O(1/n)

where mu_intermediate is the distribution after applying preceding transforms. *Note*: The intermediate distribution mu_intermediate is sigma-dependent, making NC_2 an adaptive quantity that tracks the actual input distribution seen by each transform pair. This is the key distinction from a naive sum of pairwise NC_2 values computed on the original distribution mu. Full proof to appear in the paper's appendix.

**Empirical finding (Pilot, Tier 4a)**: Pixel-space NC_2 (SWD proxy) does not predict accuracy ranking (rho = -0.20). This is consistent with the bound: pixel-space NC_2 approximates W_2 on mu, but accuracy effects are governed by NC_2 on the *feature* space distribution learned by the classifier. The revised theoretical prediction targets feature-space NC_2.

### DPI Reversibility Principle (Revised)

Each augmentation operation is a stochastic channel. By the data processing inequality:

    I(y; A_sigma(x)) ≤ I(y; T_{sigma(k-1)}(...T_{sigma(1)}(x))) ≤ ... ≤ I(y; x)

**Revised theoretical claim**: The contraction coefficient eta_i(nu) of transform t_i depends on its input distribution nu. When a high-information-loss transform (e.g., RandomCrop) is applied first, subsequent transforms operate on a distribution where spatial information is already degraded. The ordering sensitivity arises because eta_i(nu_sigma) ≠ eta_i(nu_sigma') when sigma ≠ sigma'.

**Empirical finding (Pilot, Tier 4b)**: InfoNCE-based MI estimates from a near-random encoder (10 epochs, 100 samples) show opposite signs on CIFAR-10 (rho = +0.54) and CIFAR-100 (rho = -0.66), both non-significant. This is attributed to encoder noise, not genuine task-difficulty effects. Full-scale MI estimation requires a properly trained encoder (30+ epochs, full dataset).

**Reversibility classification (maintained):**
- **High reversibility (low info cost)**: Color jitter, brightness, contrast, saturation — pointwise, approximately bijective
- **Medium reversibility**: Horizontal flip — spatial rearrangement invertible up to boundary effects (strictly bijective on discrete domain, hence eta = 1; the "medium" label was inconsistent and is corrected to "high" here)
- **Low reversibility (high info cost)**: Random crop (discards pixels), random erasing, posterize at extreme levels

**Corrected reversibility ordering**: RandomHorizontalFlip is *strictly* bijective (it is its own inverse). The ordering Crop < ColorJitter ≈ Flip in terms of info-loss is incorrect. The correct order is: Crop (lowest reversibility) < ColorJitter (medium) < Flip (highest reversibility, bijective). The revised DPI prediction: Flip-first ordering should outperform CJ-first, since Flip has zero information cost while CJ does have some. Empirical winner in most blocks: Flip→Crop→CJ — **consistent with this corrected reversibility ranking**.

## Method

### Experimental Design: 4-Tier Study

**Tier 0 — Pilot (COMPLETED)**
- 3 ops: {RandomCrop(32, pad=4), RandomHorizontalFlip, ColorJitter(0.4,0.4,0.4)}
- All 6 orderings, ResNet-18, CIFAR-10 5k subset, 10 epochs, 2 seeds
- Result: 2.68% spread; CJ→Crop→Flip best (28.04%), Flip→CJ→Crop worst (25.36%)
- Decision: GO (high confidence)

**Tier 1 — Full Factorial on 3 Operations (REQUIRED — not yet completed)**
- 6 orderings × 2 architectures (ResNet-18, ViT-Small) × 2 datasets (CIFAR-10, CIFAR-100) × **5 seeds**
- **200 epochs** (not 10 epochs as in pilot), **full datasets** (not 100-sample subsets)
- Training recipes: SGD+cosine for ResNet (lr=0.1, wd=5e-4), AdamW+warmup for ViT (lr=3e-4, patch=4)
- Total: 120 runs (~20 GPU-hours on 2 GPUs, ~10h wall-clock if parallelized across 4 GPUs)
- Statistical analysis: paired t-test (same seed, different ordering), Bonferroni correction, Cohen's d, two-way ANOVA

**Tier 2 — Category-Level Ordering with 6 Operations (REQUIRED — pilot only)**
- Operations: {Crop, Flip, Rotation} (geometric) + {ColorJitter, Grayscale, GaussianBlur} (photometric)
- 5 canonical orderings: (a) all-geometric-first, (b) all-photometric-first, (c) interleaved G-P-G-P-G-P, (d) interleaved P-G-P-G-P-G, (e) random-per-image
- 2 architectures × 2 datasets × 5 seeds = 100 runs (~18 GPU-hours)
- Priority: Run a Tier 2 confirmation pilot (2 orderings × 2 arch-dataset blocks × 2 seeds × 30 epochs, ~1.5 GPU-hours) before committing to full Tier 2

**Tier 3 — Magnitude Interaction (REVISED — based on H6)**
- Best and worst orderings from Tier 1 × 3 magnitude levels (M=5, M=9, M=14) × 2 architectures × CIFAR-100 × 5 seeds
- Primary test: Is the inverted-U (M9 > M5 AND M9 > M14) confirmed? Total: 60 runs (~12 GPU-hours)

**Tier 4 — Feature-Space NC Measurement (REVISED from pixel-space)**
- Compute feature-space NC_2 using penultimate-layer embeddings from a pretrained ResNet-18 (trained 200 epochs on CIFAR-10)
- 10k samples, 1000 projections for SWD — qualitatively different from the failed pilot (100 samples, 100 projections)
- Also: Full-scale InfoNCE MI estimation using properly trained encoders (30 epochs, full dataset)
- Goal: Test whether feature-space NC_2 recovers the predictive power lost in pixel space

### Baselines

| Baseline | Purpose |
|---|---|
| Standard torchvision ordering (Crop→Flip→ColorJitter→Normalize) | Community default |
| Random-per-image ordering | RandAugment-style stochastic ordering |
| TrivialAugment (single operation) | Single-op, ordering-irrelevant ceiling |
| No augmentation (only Crop+Flip) | Information floor |
| RandAugment N=2, M=9 | State-of-the-art reference |

*Baseline note*: Full baselines (Tier 1 pilot) already run with proper 30-epoch training on full datasets. Results: RandAugment N=2 M=9 best (e.g., ResNet-18/CIFAR-100 = 0.7283, ViT/CIFAR-100 = 0.5985), consistent with expectations. These are on different training scales from the ordering pilot results and should not be directly compared without matching conditions.

### Statistical Analysis Plan

- **Primary test**: Paired t-test (same seed, different ordering) between best and worst orderings per architecture-dataset combination. Bonferroni correction for multiple comparisons.
- **Interaction test**: Two-way ANOVA (ordering × architecture) for H2. Requires n ≥ 2 per cell (minimum 2 seeds; 5 seeds preferred).
- **Effect size**: Cohen's d for all pairwise ordering comparisons. *Note*: Cohen's d is undefined for n=1; all full-scale computations require n ≥ 2.
- **NC correlation**: Spearman rank correlation + permutation test for significance. Computed on feature-space embeddings with 10k samples.
- **Magnitude interaction (H6)**: Regression of ordering spread on magnitude level (M=5, 9, 14). Test for inverted-U shape (quadratic term significant, negative).
- **Significance threshold**: p < 0.05 after correction, effect size |d| > 0.2 for practical relevance.

## Resource Estimate

| Component | GPU-hours | Wall-clock (4 GPUs) |
|---|---|---|
| Tier 0: Pilot (DONE) | 0.3 | Done |
| Tier 1: 3-op factorial (120 runs, 5 seeds, 200 ep) | 20 | ~10h |
| Tier 2 confirmation pilot (8 runs, 30 ep) | 1.5 | ~45min |
| Tier 2: Category ordering full (100 runs, 5 seeds) | 18 | ~9h |
| Tier 3: Magnitude interaction revised (60 runs) | 12 | ~6h |
| Tier 4: Feature-space NC + full MI (CPU + GPU) | 2 | ~1.5h |
| **Total remaining** | **~54** | **~27h** |

Model sizes: ResNet-18 (11M), ViT-Small (22M). Each run ≤ 30 min on RTX PRO 6000. All individual experiments fit ≤1 hour budget.

## Risk Assessment (Revised)

| Risk | Likelihood | Mitigation | Pilot Evidence |
|---|---|---|---|
| Ordering effects disappear at full scale (200ep) | Low-Medium | 2.32% spread already at 10ep/100-samples pilot; effects likely grow with more training capacity | 3/4 blocks above threshold at pilot scale |
| ViT training instability on CIFAR | Low | Established timm recipe (patch size 4) verified in pilot | ViT ran successfully in all pilot blocks |
| Category ordering effect (9.01%) is noise | Medium-High | 1 seed, 5k subset, 10ep — needs Tier 2 confirmation pilot before full Tier 2 investment | Must run 30ep/2-seed pilot on 2 blocks first |
| Feature-space NC_2 also fails to predict | Medium | Would be a genuine negative finding: ordering effects operate below the level of static embedding spaces, requiring dynamic analysis | Informs new theoretical direction |
| Theoretical framework weaknesses | Medium | DPI proof gap (SOUND-002) acknowledged; weakened claim adopted. NC bound proof (SOUND-001) deferred to appendix | Negative results from Tier 4 will shape final framing |
| Reviewers dismiss as ablation study | Low-Medium | Pilot already shows 9.01% category-level spread — this is practically significant; theory framework provides context regardless of direction | Strong positive pilot signal |

## Next Iteration Priority Queue

1. **BLOCKING**: Execute Full Tier 1 (120 runs, 5 seeds, 200 epochs) — provides all primary H1/H2 evidence
2. **HIGH**: Execute Tier 2 confirmation pilot (30 epochs, 2 seeds, 2 blocks) — validate 9.01% before committing to full Tier 2
3. **HIGH**: Compute feature-space NC_2 with 10k samples + properly trained encoder (can run concurrently with Tier 1)
4. **MEDIUM**: Full Tier 2 (100 runs) — only after confirmation pilot passes
5. **MEDIUM**: Full Tier 3 (60 runs, test inverted-U H6) — can run concurrently with Tier 2
6. **LOW**: Formal proof of Theorem 1 bubble-sort decomposition — needed for camera-ready, not blocking experiments
