# Optimist Analysis

## Evidence Map

### Primary Metric: Absorption Rate (Ground-Truth, No Probes)

| Variant | Absorption Rate | Baseline | Delta | Reduction % | Cohen's d | Signal Strength |
|---------|-----------------|----------|-------|-------------|-----------|-----------------|
| TopK (k=50) | 0.056 +/- 0.021 | 0.252 | -0.197 | 78.0% | 4.93 | **Strong** |
| MultiScale | 0.055 +/- 0.024 | 0.252 | -0.197 | 78.3% | 4.81 | **Strong** |
| Full Matryoshka | 0.066 +/- 0.029 | 0.252 | -0.186 | 73.7% | 4.31 | **Strong** |
| Orthogonality | 0.245 +/- 0.050 | 0.252 | -0.007 | 2.7% | 0.13 | Negligible |
| Gated | 0.261 +/- 0.050 | 0.252 | +0.009 | -3.6% | -0.17 | None |
| Random Control | 0.534 +/- 0.050 | 0.252 | +0.282 | -111.8% | -5.24 | Validates metric |

**Key numbers:**
- TopK reduces absorption by **78.0%** with Cohen's d = **4.93** (5 replicates, seeds 42/123/456/789/1011)
- MultiScale matches TopK almost exactly (0.055 vs 0.056) --- the effect is mediated by shared L0=50 sparsity
- The effect is **clearly beyond noise**: all 5 TopK replicates (0.032, 0.087, 0.070, 0.035, 0.054) are below the lowest baseline replicate (0.178) -- **zero overlap in ranges**
- Orthogonality achieves only 2.7% reduction (d = 0.13, p = 0.845), directly contradicting OrtSAE's claimed 65% reduction

### Secondary Metrics

| Variant | MSE | L0 | Hedging | Dead Latents | Uniqueness |
|---------|-----|-----|---------|--------------|------------|
| Baseline | 0.0104 +/- 0.0008 | 964 +/- 75 | 0.240 +/- 0.007 | 0 | 0.382 +/- 0.002 |
| TopK | 0.0077 +/- 0.0003 | 50.0 (exact) | 0.237 +/- 0.025 | 1672 (81.6%) | 0.354 +/- 0.010 |
| MultiScale | 0.0071 +/- 0.0004 | 50.0 (exact) | 0.235 +/- 0.009 | 1155 (56.4%) | 0.416 +/- 0.003 |
| Orthogonality | 0.00003 +/- 0.000002 | 550 +/- 4 | 0.240 +/- 0.009 | 11 (0.5%) | 0.437 +/- 0.005 |
| Gated | 0.0082 +/- 0.0006 | 966 +/- 101 | 0.233 +/- 0.008 | 0 | 0.396 +/- 0.003 |
| Full Matryoshka | 0.0071 +/- 0.0004 | 50.0 (exact) | 0.233 +/- 0.009 | 1162 (56.7%) | 0.415 +/- 0.005 |
| Random Control | 0.649 +/- 0.059 | 1029 +/- 10 | 0.236 +/- 0.005 | 0 | 0.428 +/- 0.004 |

**Notable secondary findings:**
- TopK improves reconstruction MSE by **26.4%** (0.0104 -> 0.0077) while reducing absorption -- this is a **double win**, not a trade-off
- Orthogonality achieves near-perfect reconstruction (MSE ~3e-5, explained variance ~0.994) but does not reduce absorption -- this is an important **negative result** that constrains theory
- L0 sparsity strongly correlates with absorption: Pearson r = 0.865, p = 0.012 across all 7 variant means
- Hedging is invariant across all variants (~0.235-0.240), refuting the predicted absorption-hedging trade-off
- MCC is degenerate across all variants (~0.21-0.22, including Random), confirming it is not a discriminating metric in overcomplete settings

### Statistical Validation

- **ANOVA across 7 variants**: F = 73.36, p = 8.02e-16 (highly significant)
- **TopK vs Baseline**: t = 7.79, p = 5.28e-05
- **MultiScale vs Baseline**: t = 7.60, p = 6.31e-05
- **Full Matryoshka vs Baseline**: t = 6.82, p = 1.35e-04
- **Orthogonality vs Baseline**: t = 0.20, p = 0.845 (not significant)
- **Gated vs Baseline**: t = -0.27, p = 0.797 (not significant)
- **L0-Absorption correlation**: Pearson r = 0.865, p = 0.012

---

## Root Cause Analysis

### Finding 1: TopK Sparsity is the Dominant Driver of Absorption Reduction

**Mechanism**: TopK enforces exact k-sparsity (k=50 latents active per token), which directly limits the number of features that can fire simultaneously. Since absorption requires both parent and child features to co-fire (parent fires, child fires with higher magnitude), restricting the total number of active latents mechanically reduces the opportunity for parent-child co-activation.

**Design decision**: Replacing L1 sparsity penalty with explicit TopK selection (keep top-50 activations, zero others).

**Expected or surprising**: This was **predicted but underestimated**. The proposal hypothesized TopK would have a moderate effect. The observed effect (d = 4.93) is **an order of magnitude larger than orthogonality (d = 0.13)** and makes TopK the dominant component by a wide margin.

**Why it was underestimated**: The hypothesis formulation was influenced by the literature's emphasis on multi-scale decomposition (Matryoshka) and orthogonality (OrtSAE) as "sophisticated" solutions. The simplicity and power of explicit sparsity control was overlooked.

### Finding 2: MultiScale Matches TopK (d = 4.81, 78.3% reduction)

**Mechanism**: MultiScale (Matryoshka BatchTopK) enforces the same L0=50 constraint as TopK, but through a nested dictionary structure. The absorption reduction appears to be mediated entirely by the shared sparsity level, not by the multi-scale decomposition itself.

**Design decision**: Nested dictionaries with 3 levels, but critically, the BatchTopK component enforces the same k=50 sparsity.

**Expected or surprising**: Surprising in that MultiScale does NOT outperform TopK. If multi-scale decomposition provided additional architectural benefit beyond sparsity, we would expect MultiScale absorption < TopK absorption. Instead, they are statistically indistinguishable (0.055 vs 0.056).

**Why it matters**: This directly challenges Bussmann et al.'s attribution. Matryoshka SAEs' absorption reduction may be entirely due to their TopK component, not their multi-scale structure.

### Finding 3: Orthogonality Penalty Has Negligible Effect on Absorption

**Mechanism**: The orthogonality penalty (lambda=0.001 on decoder chunks) successfully enforces decoder orthogonality (uniqueness = 0.437 vs. baseline 0.382) and achieves excellent reconstruction (MSE ~3e-5), but does not change the encoder's activation pattern enough to reduce absorption.

**Design decision**: Adding chunk-wise decoder orthogonality penalty to baseline ReLU SAE.

**Expected or surprising**: Expected per H3 (orthogonality null hypothesis), but the near-zero effect is still striking. OrtSAE claims 65% reduction; our data shows 2.7%.

**Isolation insight**: This is exactly why component-isolated studies matter. Without this experiment, one might attribute OrtSAE's absorption reduction to orthogonality when the actual driver could be co-occurring design choices (e.g., TopK sparsity).

### Finding 4: Full Matryoshka Shows Antagonistic Interaction

**Mechanism**: Full Matryoshka (TopK + MultiScale + hierarchical loss) achieves absorption = 0.066, which is HIGHER than both TopK (0.056) and MultiScale (0.055) individually. The component interaction analysis shows antagonism: observed (0.066) > predicted additive (-0.142).

**Design decision**: Combining all three Matryoshka components (TopK + multi-scale + hierarchical loss).

**Expected or surprising**: Highly surprising. One would expect synergy or at least additivity. The antagonism suggests that architectural complexity beyond sparsity control may be counterproductive.

**Why it matters**: This undermines the "more components = better" intuition and challenges the field's tendency toward increasingly complex SAE architectures.

---

## Unexpected Signals

### Unexpected Finding 1: TopK Improves BOTH Absorption AND Reconstruction

**Observation**: TopK achieves 78% absorption reduction AND 26% MSE improvement simultaneously (0.0104 -> 0.0077). There is no reconstruction-absorption trade-off for this component.

**Mini-hypothesis**: Explicit k-sparsity is a better optimization target than L1 for this synthetic data because (a) the ground-truth features are truly sparse (true L0 ~ 1.58 features per token), and (b) L1's soft thresholding allows many small activations that add noise without improving reconstruction. TopK's hard thresholding removes this noise.

**Significance**: If this transfers to real LLMs, it means TopK is not just an absorption-reduction technique but a general SAE improvement. This would make it the default recommendation for SAE architecture.

### Unexpected Finding 2: Orthogonality Achieves Near-Perfect Reconstruction Without Absorption Benefit

**Observation**: Orthogonality SAE achieves MSE = 3.2e-5 (99.4% explained variance) -- the best reconstruction of any variant -- but absorption remains at 0.245 (virtually unchanged from baseline 0.252).

**Mini-hypothesis**: Decoder orthogonality improves the quality of individual feature representations (hence low MSE) but does not change the encoder's activation patterns. Absorption is an **encoder-side phenomenon**: the encoder chooses which latents to activate, and orthogonality constraints on the decoder do not strongly influence this choice.

**Significance**: This decouples two previously conflated goals: (1) learning orthogonal features (decoder quality) and (2) preventing feature absorption (encoder behavior). Practitioners may need to optimize both separately.

### Unexpected Finding 3: No Absorption-Hedging Trade-off Across All 7 Variants

**Observation**: Hedging scores are virtually identical across all variants: Baseline 0.240, TopK 0.237, Orthogonality 0.240, Gated 0.233, Full Matryoshka 0.233, Random 0.236. No statistically significant differences.

**Mini-hypothesis**: Hedging may be a property of the data distribution or dictionary overcompleteness (2048 latents for 1024 features) rather than an architecture-dependent pathology. With 2x overcomplete dictionaries, some latents will inevitably mix correlated features regardless of architecture.

**Significance**: The Chanin et al. (2025) absorption-hedging trade-off may be specific to their experimental setup (real LLM features, different dictionary sizes) or may require conditions not met here. This is a testable prediction for future work.

### Unexpected Finding 4: MCC is Degenerate Across All Variants (Including Random)

**Observation**: All variants show MCC ~ 0.21-0.22, including the random control (0.221). The Hungarian matching algorithm on overcomplete dictionaries finds ~0.21 correlation by chance.

**Mini-hypothesis**: MCC computed via Hungarian matching on overcomplete dictionaries (2048 latents, 1024 features) has a **floor effect** -- the algorithm can always find non-trivial matches between random vectors and structured ground truth when there are more latents than features.

**Significance**: This validates the decision to use absorption rate as the primary metric. It also suggests the community needs better feature recovery metrics for overcomplete SAE evaluation.

### Unexpected Finding 5: TopK and MultiScale Have Massive Dead Latent Rates (82% and 56%)

**Observation**: TopK has 1672 dead latents out of 2048 (81.6%). MultiScale has 1155 dead latents (56.4%). Full Matryoshka has 1162 (56.7%). Baseline, Orthogonality, Gated, and Random have 0 dead latents.

**Mini-hypothesis**: With k=50 and 2048 latents, only a small subset of latents ever activate; the remaining receive no gradient signal and die. The dictionary effectively shrinks to a much smaller active set.

**Significance**: While TopK dominates on absorption, the dead latent problem raises a critical question: is the absorption reduction artifactual --- i.e., would absorption be higher if measured only on active latents? This must be tested in follow-up.

---

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| TopK dead latent artifact | Recompute absorption using only active latents for TopK, MultiScale, Full Matryoshka | If active-only absorption << baseline: signal is real. If comparable: artifactual. | 0.1 (CPU) | **High** |
| L0-matched baseline | Train L1 SAE with tuned lambda to achieve L0=50 and L0=550; compare absorption | If L0=50 matches TopK: sparsity mediation confirmed. If not: TopK has additional benefit. | 0.5 | **High** |
| TopK dose-response | Train TopK with k in {10, 25, 50, 100, 200, 500} | Absorption increases monotonically with k; curve fit to characterize relationship | 0.5 | **High** |
| Full Matryoshka antagonism | Ablation: Full Matryoshka without hierarchical loss, without multi-scale | Isolate which component causes the antagonistic interaction | 0.5 | Medium |
| Real-LLM validation | Load SAELens pretrained SAEs (varying L0); measure SAEBench absorption | L0 predicts absorption on real LLMs; Kendall's tau > 0.6 | 0.5 | Medium |
| Synthetic scale-up | Run on SynthSAEBench-16k (16,384 features) with same 6 variants | Effect sizes replicate at scale; confirms pilot-to-full generalization | 1.0 | Medium |
| Active-latent feature recovery | Measure MCC on active latents only (excluding dead) for TopK/MultiScale | If active-only MCC >> full-dictionary MCC: dead latents mask learning | 0.1 (CPU) | Medium |

### Falsifiability Criteria

1. **Dead latent artifact**: If active-only TopK absorption > 0.15 (vs. 0.056 full-dictionary), the TopK signal is partially artifactual. If > 0.20, it is largely artifactual.
2. **L0-matched baseline**: If L0=50 Baseline achieves absorption < 0.10, sparsity mediation is confirmed. If > 0.15, TopK has architectural benefits beyond sparsity.
3. **Dose-response**: If absorption is NOT monotonically increasing in k (e.g., U-shaped), the sparsity-absorption relationship is more complex than hypothesized.
4. **Real-LLM validation**: If Kendall's tau < 0.3, synthetic findings do not transfer; the synthetic-to-real gap is large.

---

## Honest Caveats

### Caveat 1: TopK's 78% Reduction May Be Partially Artifactual

**Counter-argument**: TopK reduces the effective dictionary size from 2048 to ~376 active latents. With fewer latents, there are simply fewer opportunities for parent-child co-activation. The absorption reduction could reflect dictionary shrinkage, not better hierarchical learning.

**Alternative explanation**: TopK achieves low absorption by "cheating" --- it kills most latents, and the surviving ones happen to avoid parent-child conflicts by chance.

**What would convince me**: Active-latent-only absorption < 0.10. If active latents alone show comparable absorption to Baseline, the effect is artifactual. If active-only absorption remains low, TopK genuinely learns better hierarchical structure.

### Caveat 2: Synthetic Data May Not Reflect Real-LLM Feature Structure

**Counter-argument**: SynthSAEBench features are hand-designed hierarchies (128 trees, depth 3, branching factor 4). Real LLM features may have different statistical properties (e.g., power-law activation, overlapping hierarchies, non-tree structures).

**Alternative explanation**: The L0-absorption correlation is an artifact of the synthetic data's clean hierarchical structure. On real LLMs, absorption may be driven by factors other than sparsity.

**What would convince me**: Real-LLM validation showing Kendall's tau > 0.6 between L0 and absorption across pretrained SAEs. Without this, the synthetic findings are suggestive but not definitive.

### Caveat 3: Small Sample Size (5 Replicates)

**Counter-argument**: With only 5 replicates per variant, the confidence intervals are wide. The TopK standard deviation (0.021) is 38% of the mean (0.056), indicating substantial replicate-to-replicate variance.

**Alternative explanation**: The large Cohen's d may be inflated by small sample size. With more replicates, the effect size could shrink.

**What would convince me**: Running 10+ additional replicates. If Cohen's d remains > 3.0, the effect is robust. If it drops below 2.0, the effect is moderate but still meaningful.

### Caveat 4: Orthogonality Null Result May Be Lambda-Dependent

**Counter-argument**: We tested orthogonality with lambda=0.001. A stronger penalty (lambda=0.01 or 0.1) might produce absorption reduction.

**Alternative explanation**: The orthogonality penalty was too weak to affect encoder behavior. A stronger penalty could change the story.

**What would convince me**: Testing lambda sweep {0.0001, 0.001, 0.01, 0.1}. If even lambda=0.1 shows d < 0.5, the null result is robust. If lambda=0.1 shows d > 1.0, the null result is conditional on penalty strength.

### Caveat 5: Full Matryoshka Antagonism May Be Training Instability

**Counter-argument**: Full Matryoshka combines three components (TopK + MultiScale + hierarchical loss), which may create optimization instability. The higher absorption could be a training artifact, not a genuine architectural interaction.

**Alternative explanation**: With better hyperparameter tuning or longer training, Full Matryoshka might match or exceed TopK/MultiScale.

**What would convince me**: Running Full Matryoshka with multiple learning rates and training durations. If absorption remains elevated across conditions, the antagonism is real.

---

## Bottom Line

There is a **strong, publishable story here**: TopK sparsity reduces absorption by 78% with Cohen's d = 4.93, while orthogonality penalties achieve only 2.7% (d = 0.13). The L0-absorption correlation (r = 0.865, p = 0.012) suggests absorption is primarily a sparsity-level phenomenon, not an architectural pathology. This reframes the field's optimization target from "which architecture?" to "what sparsity level?" --- a simpler, more actionable conclusion with direct practical implications for SAE practitioners. The critical follow-ups are: (1) active-latent-only analysis to rule out artifactual reduction, (2) L0-matched baseline to confirm sparsity mediation, and (3) real-LLM validation to test synthetic-to-real transfer. If all three hold, this is a NeurIPS/ICML-level contribution that challenges established claims in the SAE literature.
