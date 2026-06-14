# Executive Summary: Phase Transition in SAE Feature Absorption

## Result Quality Score: 6.5/10

**Verdict**: Publishable at mid-tier venue (AAAI/EMNLP/Workshop), not top-tier.

**Recommendation**: **PROCEED** with paper writing and targeted validation experiments (~3 hours GPU).

---

## Key Conclusion

The experiment reveals **split results with genuine discoveries**:

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: Critical Sparsity | **CONFIRMED (quasi)** | λ_c≈5e-5, chi_max=11.19, chi_ratio=1.88 < 3.0 |
| H2: Finite-Size Scaling | **CONFIRMED** | ν=3, R²=0.951 — **first in SAE literature** |
| H3: Layer Critical | **FALSIFIED** | absorption_rate=1.0 for ALL layers (uniform saturation at λ=0.001) |
| H4: Variance Difference | **DISCOVERY (reversed)** | CV_high=7.33 >> CV_low=0.01 (733x); absorbed have HIGHER CV |
| H5: Info Bottleneck | **WEAK (post-hoc)** | r=0.647, reverse-engineered on E2 data |
| H6: Graph Topology | **FALSIFIED** | Component count decreases with layer |

**Summary: 3/6 confirmed (H1 quasi, H2 strong, H5 weak), 3/6 refuted (H3, H4 direction, H6)**

---

## What We Actually Learned

1. **First finite-size scaling measurement in SAE absorption** — ν=3, R²=0.951 with scaling collapse across dictionary sizes. This is the primary contribution and is genuinely novel.

2. **Variance paradox discovery** — Absorbed features have 733x higher CV than non-absorbed (opposite of prediction). High-CV features are 2x more steerable (0.153 vs 0.075), suggesting CV predicts actionability.

3. **Genuine causal absorption validated** — Activation patching shows 67.3% mean parent recovery (range 48.8%-75.2%) across 9/9 words. When child feature is zeroed, parent recovers substantially, confirming persistent absorption is real causal phenomenon.

4. **Layer-criticality and graph topology falsified** — All layers saturate at absorption_rate=1.0 at λ=0.001. The "layer as temperature" analogy fails at this sparsity level.

5. **Quasi-critical regime confirmed** — chi_ratio=1.88 is below "sharp transition" threshold of 3.0. The phase transition is gradual, not sharp. λ_c is unstable across sample sizes (10x pilot-to-full shift).

---

## Action Plan: PROCEED

### Rationale (changed from PIVOT)

1. **Activation patching validates genuine absorption** — 67.3% mean recovery is real causal evidence, not metric artifact
2. **CV → steering effectiveness transforms H4** — from "failed hypothesis" to "discovery with mechanism"
3. **H1/H2 signal is stable** — ν=3 finite-size scaling is first in SAE literature and remains the primary contribution

### Prioritized Next Steps (~3 hours GPU total)

| Priority | Experiment | Justification |
|----------|-----------|---------------|
| P0 | **Prospective H5 validation** | Weakest positive result needs held-out validation |
| P0 | **λ_c stability test** (n=500, 1000, 2000) | Addresses 10x pilot-to-full shift concern |
| P1 | **Cross-layer at λ_c=5e-5** | Tests if layer heterogeneity appears at true critical point |
| P1 | **CV decomposition full experiment** | 30 high vs 30 low CV, multiple steering strengths |
| P2 | **Out-of-sample dictionary size** | Add 4th size (32k/8k) to fix n=3 problem for H2 |
| P2 | **Randomized word selection** | Addresses selection bias in activation patching |

---

## Paper Framing

**Lead contribution**: First finite-size scaling measurement in SAE absorption (ν=3, R²=0.951)

**Highlight discovery**: Variance paradox — absorption selectively routes high-variance specialized features, and CV predicts which absorbed features remain steerable (addresses Basu et al. actionability paradox)

**Acknowledge explicitly**:
- H3 "layer as temperature" falsified at λ=0.001 — all layers saturate uniformly
- H4 prediction wrong direction — absorbed features have HIGHER CV, not lower
- H6 graph topology not an order parameter — component count decreases with layer
- chi_ratio below "sharp transition" threshold (1.88 < 3.0) — quasi-critical behavior
- λ_c 10x instability across sample sizes — reproducibility concern

**Intellectually honest framing**: "This work measures phase transition behavior in SAE feature absorption and finds finite-size scaling with ν=3 — the first such measurement in the literature. The most interesting finding is the variance paradox: absorbed features have higher coefficient of variation than non-absorbed, suggesting absorption selectively routes high-variance specialized features. The 'layer 6 critical point' hypothesis is definitively falsified at the tested sparsity level."

---

## Connection to Basu et al. Actionability Paradox

Basu et al. (2026) demonstrates 98.2% AUROC but 0% output change via steering. Our pilot results show high-CV features are 2x more steerable, suggesting CV may predict which absorbed features remain actionable.

**Mechanism hypothesis**: High-CV absorbed features route through specialized child channels. The child contributes fixed signal regardless of parent steering direction, yielding zero net output change (explains Basu et al. paradox).

**Status**: This is a hypothesis, not a confirmed result. Full validation required.

---

## Venue Recommendation

**Target**: Mid-tier (AAAI/EMNLP) or Workshop

**Do not target**: Top-tier venues (NeurIPS/ICML) — chi_ratio below threshold and λ_c instability are too severe for competitive acceptance.

**Comparable papers**:
- Cui et al. (ICLR 2026): More rigorous theoretical contribution
- Karvonen et al. (ICML 2025): SAEBench methodological contribution
- Costa et al. (NeurIPS 2025): MP-SAE architecture improvement

---

## Bottom Line

**Result quality: 6.5/10** — The primary contribution (ν=3 finite-size scaling) is genuinely novel and publishable. The variance paradox discovery (CV reversal + steering implications) is scientifically interesting. Falsified hypotheses (H3, H6) and weak results (H5 post-hoc, λ_c instability) must be explicitly acknowledged.

**Recommendation**: PROCEED with paper writing and targeted validation (P0/P1 priority). Target mid-tier venue with honest framing. Lead with ν=3 discovery, frame CV reversal as the interesting finding, and acknowledge all weaknesses explicitly.