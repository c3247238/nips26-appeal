# Skeptic Analysis: Component-Isolated SAE Absorption Study (Full Data)

## Executive Summary

With all 6 variants now complete (5 replicates each, n=35 total), the central confound identified in the partial analysis is **confirmed and amplified**: absorption reduction is almost perfectly predicted by sparsity level (L0), not by architectural components. The three variants achieving L0=50 (TopK, MultiScale, Full Matryoshka) all show comparable absorption (~0.055-0.066), while the three variants with high L0 (Baseline L0=964, Gating L0=966, Orthogonality L0=550) show high absorption (~0.245-0.261). The L0-absorption correlation across all 7 conditions is r = 0.865 (p = 0.012). This pattern strongly supports the sparsity-mediation hypothesis (H2) and undermines claims that any specific architecture independently reduces absorption.

**The most damaging finding**: MultiScale and TopK produce virtually identical absorption (0.055 vs 0.056, Cohen's d = 0.03) with identical L0 (50). If MultiScale's multi-scale decomposition were genuinely reducing absorption, we would expect it to outperform TopK at matched L0. It does not. This is strong evidence that the observed "architectural" effects are entirely sparsity-mediated.

---

## 1. Statistical Risk Inventory

### Risk 1: Perfect L0-Absorption Confound (FATAL FLAW for Architectural Claims)

**The numbers**:

| Variant | L0 | Absorption | Std |
|---------|-----|-----------|-----|
| Random Control | 1029 | 0.534 | 0.050 |
| Baseline ReLU | 964 | 0.252 | 0.046 |
| Gated | 966 | 0.261 | 0.050 |
| Orthogonality | 550 | 0.245 | 0.050 |
| TopK (k=50) | 50 | 0.056 | 0.021 |
| MultiScale | 50 | 0.055 | 0.024 |
| Full Matryoshka | 50 | 0.066 | 0.030 |

**Why it's unreliable**: The absorption rate tracks L0 with r = 0.865 (p = 0.012) across all 7 variant means. Every variant with L0=50 shows absorption ~0.055-0.066 regardless of architecture (TopK, MultiScale, or their combination in Full Matryoshka). Every variant with L0>900 shows absorption ~0.245-0.261 regardless of architecture (Baseline, Gating). Orthogonality, the only variant with intermediate L0 (~550), shows intermediate absorption (~0.245). This is exactly the pattern expected if absorption is a pure sparsity phenomenon.

**The smoking gun**: TopK vs MultiScale Cohen's d = 0.03 (negligible). These two architectures share nothing in common except L0=50. TopK uses explicit k-sparsity; MultiScale uses nested dictionaries with batch TopK. If architectural components mattered, these should differ. They do not.

**Precedent from evolution lessons**: The CMI dimension instability warning ("Sign reversal is qualitative failure, not sensitivity issue") applies here too. The L0-absorption relationship is not a subtle sensitivity -- it is a complete explanatory framework that renders architecture irrelevant.

### Risk 2: Massive Dead Latent Rates in Low-Absorption Variants (SERIOUS CONCERN)

**The numbers**: TopK has 1,672 dead latents (81.7%); MultiScale has 1,155 (56.4%); Full Matryoshka has 1,162 (56.7%). Baseline, Gating, and Random Control have 0 dead latents. Orthogonality has ~11 (0.5%).

**Why it's unreliable**: The "best" variants achieve low absorption by disabling most of the dictionary. A dead latent cannot absorb anything by definition. The absorption metric is being gamed by a pathological model state. The effective dictionary size for TopK is ~376 latents, not 2,048. For MultiScale and Matryoshka, the effective size is ~893 latents.

**Comparison**: Orthogonality achieves near-perfect reconstruction (MSE ~3e-5, explained variance ~0.994) with only 10 dead latents, yet its absorption is statistically indistinguishable from Baseline (d = 0.13, p = 0.845). This suggests that healthy, fully-utilized dictionaries naturally exhibit ~0.25 absorption on this synthetic data, and the only way to get substantially lower is to cripple the dictionary through extreme sparsity.

### Risk 3: The "r ~ -0.99" Claim is Inflated (SERIOUS CONCERN)

**The numbers**: The proposal and synthesis repeatedly claim "L0-absorption correlation r ~ -0.99." The actual Pearson correlation across all 7 variant means is r = 0.865 (p = 0.012). The sign is reversed in the claim (negative vs positive, because lower L0 correlates with lower absorption, but the raw correlation depends on coding).

**Why it's unreliable**: The r ~ -0.99 figure appears to have been computed on only 3 variant means (Baseline, Orthogonality, TopK) in the partial analysis. With all 7 variants included, the correlation drops to 0.865. While still significant, this is a material difference. More importantly, the correlation is driven by the three L0=50 variants clustering together and the three L0~960 variants clustering together -- there is almost no variance in absorption *within* each L0 band. This is not a continuous dose-response relationship; it is a threshold effect.

**Impact**: The inflated correlation claim exaggerates the strength of the L0-absorption relationship and may mislead readers into thinking the evidence is stronger than it is.

---

## 2. Alternative Explanations

### For the identical TopK/MultiScale absorption:

**Alternative A (sparsity sufficiency)**: Any mechanism that enforces L0=50 produces absorption ~0.055, regardless of architecture. The specific mechanism (TopK selection vs multi-scale decomposition) is irrelevant.

**Alternative B (dead latent artifact)**: Both TopK and MultiScale have >50% dead latents. The effective dictionary is much smaller than 2,048. With fewer active latents, the probability of parent-child co-activation drops, reducing absorption mechanically.

**Alternative C (metric ceiling)**: The absorption rate may have a floor at ~0.03-0.05 determined by the ground-truth hierarchy structure (992 parent-child pairs, tree depth 3, branching factor 4). Even with perfect architecture, some absorption is unavoidable due to the data structure itself.

### For the Full Matryoshka "antagonistic interaction":

**The numbers**: Component interaction analysis shows expected (additive) = -0.142, observed = 0.066, synergy = 0.208, interaction_type = "antagonistic."

**Alternative A (L0 saturation)**: Full Matryoshka combines TopK + MultiScale + hierarchical loss, but all three L0=50 variants show comparable absorption. There is no room for improvement beyond the L0=50 floor. The "antagonism" is not a genuine architectural interaction -- it is saturation at the sparsity limit.

**Alternative B (implementation overlap)**: Full Matryoshka and MultiScale may share the same underlying implementation (both use `matryoshka_batchtopk` architecture). The component interaction analysis treats them as independent when they may not be.

### For Orthogonality's null result:

**Alternative A (wrong target)**: The orthogonality penalty is applied to the decoder, but absorption is defined on encoder activations. Decoder orthogonality cannot affect which encoder latents co-fire. The null result is predicted by the metric definition, not informative about OrtSAE's claims.

**Alternative B (weak penalty)**: lambda=0.001 may be too weak to enforce meaningful orthogonality. The near-zero MSE (~3e-5) suggests the penalty is not competing effectively with reconstruction loss.

### For the overall pattern:

**Alternative A (metric measures sparsity, not architecture)**: The absorption rate metric is fundamentally a sparsity detector on synthetic data. All architectures that enforce low L0 show low absorption; all architectures with high L0 show high absorption. The architecture itself is irrelevant.

**Alternative B (synthetic data too structured)**: With 128 root trees, depth 3, branching factor 4, and 2M training samples, the hierarchical structure may be so regular that all trained variants converge to similar solutions modulo sparsity enforcement.

---

## 3. Proxy Metric Audit

| Claimed Measurement | What We Actually Measure | Gap | Severity |
|---|---|---|---|
| "Component-isolated causal effect on absorption" | Effect of changing sparsity level on a metric sensitive to sparsity | Architecture and sparsity are perfectly confounded; no L0-matched comparison exists | **Fatal** |
| "TopK reduces absorption by 78%" | Enforcing k=50 sparsity reduces absorption by 78% | Cannot distinguish "TopK architecture" from "k=50 sparsity" | **Fatal** |
| "MultiScale reduces absorption by 78%" | MultiScale (which also enforces L0=50) reduces absorption by 78% | MultiScale and TopK are identical at matched L0 (d=0.03); no independent MultiScale effect | **Fatal** |
| "Full Matryoshka shows antagonistic interaction" | Two L0=50 mechanisms combined produce L0=50 result | Interaction analysis is meaningless when both components saturate at the same sparsity limit | **Serious** |
| "Feature recovery MCC" | Hungarian matching correlation on overcomplete dictionary | MCC ~0.21-0.22 across ALL variants including Random; metric does not discriminate trained from random | **Serious** |
| "L0-absorption correlation r ~ -0.99" | L0-absorption correlation across 7 variant means | Actual r = 0.865 (p=0.012); r ~ -0.99 was from 3-variant subset and misreports sign | **Serious** |
| "Reconstruction quality" | MSE on synthetic data | Orthogonality achieves MSE ~3e-5 vs Baseline ~0.01; large differences exist but do not correlate with absorption | **Minor** |
| "Hedging score" | Fraction of latents mixing correlated features | Hedging ~0.23-0.24 across all variants; no trade-off observed | **Minor** |

**Key finding**: The primary metric (absorption rate) is measuring sparsity level, not architectural component effects. The secondary metrics (MCC, hedging) show no discriminative power across variants. The paper's central claim -- "first component-isolated causal analysis" -- is unsupported because components are not isolated from sparsity.

---

## 4. Severity Classification

### Fatal Flaws

1. **Sparsity-absorption confound is complete**: The observed absorption differences are perfectly explained by L0 differences. All L0=50 variants show absorption ~0.055-0.066 regardless of architecture. No L0-matched comparison has been performed. The causal claim that "TopK architecture reduces absorption" should read "enforcing L0=50 sparsity reduces absorption."

2. **MultiScale and TopK are indistinguishable at matched L0**: Cohen's d = 0.03 between TopK and MultiScale, with identical L0=50. If MultiScale's multi-scale decomposition had an independent effect, it should outperform TopK. It does not. This falsifies any claim that MultiScale independently reduces absorption.

3. **The r ~ -0.99 claim is factually incorrect**: The actual correlation across all 7 variants is r = 0.865. The r ~ -0.99 figure (with wrong sign) was computed on a 3-variant subset and has been propagated through the proposal, synthesis, and verdict without verification against the full data.

### Serious Concerns

4. **56-82% dead latents in low-absorption variants**: The "best" variants achieve low absorption through dictionary crippling. TopK has 81.7% dead latents; MultiScale and Matryoshka have ~56%. This is a degenerate solution.

5. **MCC metric failure**: MCC ~0.21-0.22 across ALL variants including Random means the feature recovery metric is not measuring feature recovery. Hungarian matching on overcomplete dictionaries produces chance-level correlations.

6. **Component interaction analysis is uninterpretable**: The "antagonistic interaction" for Full Matryoshka (expected=-0.142, observed=0.066) is an artifact of L0 saturation, not a genuine architectural interaction. Both TopK and MultiScale independently enforce L0=50; their combination cannot produce L0<50.

7. **Gating slightly increases absorption**: Gating shows absorption=0.261 vs Baseline=0.252 (d=-0.17, ns), with nearly identical L0 (~966). This suggests gating has no effect on absorption, consistent with the sparsity-mediation hypothesis.

### Minor Caveats

8. **Explained variance metric instability**: Negative explained variance despite good MSE is a known numerical issue with synthetic data. The experiment correctly uses MSE.

9. **ANOVA is significant but uninformative**: F=73.36, p=8e-16 reflects the large differences between L0 bands, not architectural differences within bands.

10. **Orthogonality's perfect reconstruction is unexplained**: MSE ~3e-5 is two orders of magnitude better than Baseline. This is a genuine architectural effect, but it affects reconstruction, not absorption.

---

## 5. Concrete Remediation

### For Fatal Flaw 1 (Sparsity Confound)

**Required experiment**: L0-matched comparison. Train Baseline with tuned L1 penalty to achieve L0=50, L0=550, and L0=964. Measure absorption at each L0 level.

**Expected outcome**: If absorption tracks L0 regardless of architecture, the sparsity confound is confirmed. If Baseline at L0=50 shows absorption comparable to TopK at L0=50, then absorption is purely a sparsity phenomenon.

**Alternative metric**: Compute absorption_rate / L0 (absorption per active latent) to normalize for sparsity level.

### For Fatal Flaw 2 (MultiScale-TopK Indistinguishability)

**Required experiment**: Test MultiScale WITHOUT its sparsity enforcement. If MultiScale has an independent architectural effect, it should reduce absorption even at high L0. Alternatively, test TopK with varying k (10, 25, 50, 100, 200, 500) to characterize the dose-response curve.

**Expected outcome**: If the dose-response curve is smooth and monotonic, absorption is a continuous function of sparsity. If there is a discontinuity at k=50, MultiScale may have an independent effect.

### For Fatal Flaw 3 (r ~ -0.99 Inflation)

**Required action**: Correct all instances of "r ~ -0.99" to "r = 0.865 (p = 0.012)" or report the correlation within L0 bands. Acknowledge that the correlation is driven by clustering at L0=50 and L0~960, not a continuous linear relationship.

### For Serious Concern 4 (Dead Latents)

**Required experiment**: Report absorption rate excluding dead latents. For each variant, recompute absorption using only non-dead latents as the effective dictionary.

**Expected outcome**: If absorption increases substantially when dead latents are excluded, the "improvement" was artifactual. If absorption stays low, the active latents genuinely avoid absorption.

### For Serious Concern 5 (MCC Failure)

**Required experiment**: Use a feature-recovery metric that works for overcomplete dictionaries. Options: (a) compute MCC at the feature level rather than latent level, (b) use cosine similarity between ground-truth and learned feature directions, (c) report the fraction of ground-truth features with a matching latent above a similarity threshold.

**Expected outcome**: Random control should score near zero; trained variants should score significantly above random.

### For Serious Concern 6 (Interaction Analysis)

**Required action**: Remove or heavily qualify the component interaction analysis. It is uninterpretable when both components enforce the same sparsity level. An interaction analysis requires components that vary independently.

---

## 6. Pre-Writing Checklist

Before any claims enter the paper, the following must be verified:

- [ ] L0-matched comparison performed (Baseline at L0=50, 550, 964)
- [ ] Dead-latent-adjusted absorption computed for all variants
- [ ] Feature recovery metric validated (Random < Trained)
- [ ] All "r ~ -0.99" claims corrected to r = 0.865
- [ ] Component interaction analysis removed or qualified
- [ ] All statistical tests use Holm-Bonferroni correction as pre-registered
- [ ] Dose-response curve (k sweep) completed

**Verdict**: The current results support only a **narrow claim**: "On synthetic hierarchical data, enforcing low L0 sparsity (L0=50) reduces absorption compared to high L0 (L0~960), but this effect is confounded with sparsity level and accompanied by 56-82% dead latents. MultiScale and TopK produce identical absorption at matched L0, suggesting no independent architectural effect." The broader claims about component-isolated causal analysis, component ranking, and architectural guidance are **not supported** by the current evidence.

**The most honest framing**: "Absorption is a sparsity phenomenon, not an architectural one. All mechanisms that enforce L0=50 produce comparable absorption (~0.055-0.066), regardless of whether they use explicit k-sparsity (TopK), multi-scale decomposition (MultiScale), or their combination (Full Matryoshka)."
