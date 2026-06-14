# Experiment Critique -- Iteration 6

## Overall Assessment

The experimental execution is technically excellent (23/23 experiments, zero failures, 50/50 streak across iterations 4-6), but three critical design decisions undermine the paper's central claims and three high-value experiments remain unexecuted despite requiring only 1.5-2 additional GPU-hours. The experiments that WERE executed are methodologically sound: four-control suite across five domains, multi-L0 confound decomposition, k-NN CMI estimation with dimension sensitivity. The problem is not execution quality but (a) the hedging classification's tautological design, (b) the failure to control for probe quality in the CMI analysis, and (c) prioritization -- low-information experiments were executed while high-information experiments were deferred.

---

## Critical Issues

### 1. Hedging Classification Is Near-Tautological

The confound decomposition's headline finding -- 98.6% hedging at L0=22 -- is guaranteed by its operational definition.

**Design flaw**: A false negative at L0=22 is classified as "hedging" if the token ceases to be a false negative at ANY higher L0 value. At L0=176, only 10/1,195 tokens remain as false negatives (0.84%). Thus 99.2% of all tokens trivially "resolve" at L0=176 by any mechanism, capping the maximum hierarchy-driven fraction at ~1.5%.

**What is NOT checked**: Whether the specific parent latent (identified by cosine >= 0.025 to probe direction) fires at the higher L0. Alternative resolution mechanisms include: a different latent with incidental cosine alignment fires, the probe's behavior changes under different activation patterns, or a compensatory feature reconstructs the parent direction.

**Consequence**: The 98.6% figure is an upper bound that would hold even if 0% of the resolution involved actual hedging (information spreading resolved by the original parent latent firing). The paper's central narrative -- that hedging rather than competitive exclusion dominates -- rests on this tautological classification.

**Fix**: Implement strict classification checking whether the specific parent-associated latent activates at L0=176. Report both permissive and strict definitions. Computationally trivial using existing data.

### 2. Three Blocking Experiments Unexecuted

| Experiment | GPU Cost | Information Gain | Status |
|---|---|---|---|
| Activation patching on 9 core words | 0.5-1h | **Highest** -- only causal evidence for CE | Not done |
| Report threshold sensitivity grid | 0h | **High** -- diagnoses control failure mechanism | Data exists, unreported |
| CMI replication at L0=22 | 1h | **High** -- eliminates probe confound + pre-registers d' | Not done |

These three experiments are inexpensive relative to the 23 already completed and would determine whether the paper reaches main-conference quality. The activation patching experiment is the single most valuable: it resolves the two-interpretation ambiguity (metric miscalibrated vs. JumpReLU genuinely lacks absorption) that the paper currently cannot address.

### 3. Probe Quality Confound in CMI Analysis

The paper reports rho=-0.67 (p<0.001) between probe F1 and absorption rate, then uses these same absorption rates for the CMI analysis without controlling for the confound. At L0=82, only 10/25 letters pass F1>0.85. Letters hard to probe will have both inflated absorption and noisier CMI estimates.

**Fix**: Use L0=22 absorption rates (F1=1.0 for all letters) or compute partial correlation controlling for F1.

---

## Major Issues

### 4. Threshold Sensitivity Results Omitted

ablation_threshold_sensitivity.json (141KB, 5x4 grid, 577 words, 25 letters) was executed in 23.2 seconds but not reported. This experiment directly tests the paper's argument about metric miscalibration: if tighter thresholds (cosine >= 0.05 instead of 0.025) produce measured > shuffled, the metric is reclaimable. The methodology section promises this analysis. Not reporting it is a selective-reporting concern.

### 5. Unsupervised Pipeline Produced Complete Zero Signal

n_matching_pairs = 0 for all 25 letters. AUROC=0.47 and rho=-0.125 are computed on zero-signal data. The pipeline found nothing at all -- not a weak signal, but no signal. The paper should state this plainly rather than presenting statistics that imply some detection was attempted.

### 6. Cross-Architecture Comparison Fully Confounded

Section 5.3 devotes ~200 words and multiple tests (Hartigan's dip, bimodality coefficient) to comparing Gemma 2 2B (2B params, JumpReLU) with GPT-2 Small (117M params, L1-ReLU). No architecture-level conclusion is possible. The dip test statistics are meaningless for attribution. Compress to 2-3 sentences.

---

## Positive Findings (Preserve These)

1. **Zero experiment failures** across 50 consecutive tasks (iterations 4-6) -- infrastructure is a solved problem
2. **Four-control suite** (random probe, shuffled labels, dense ceiling, untrained SAE) across five domains is the most comprehensive in any absorption study
3. **L0 phase transition** (42.85% to 0.84%, cross-layer CV < 10%) is the most robust finding across all 6 iterations
4. **Multi-L0 confound decomposition design** is methodologically innovative even if the classification needs tightening
5. **Honest negative results** for H2, H4, H6, H7 -- exemplary scientific practice maintained for 6 consecutive iterations
6. **GPT-2 Small as reference anchor** provides a clean, reproducible data point on an open model
