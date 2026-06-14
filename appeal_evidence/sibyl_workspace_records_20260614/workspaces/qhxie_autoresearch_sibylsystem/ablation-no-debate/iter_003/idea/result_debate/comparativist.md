# Comparativist Analysis: Encoder-Driven Feature Absorption in SAEs

## Date: 2026-05-01
## Workspace: ablation-no-debate/current

---

## Executive Summary

**Contribution Margin: MARGINAL (<1%)**

This work proposes that feature absorption in SAEs is "encoder-driven" (rather than decoder-driven), with secondary claims about hierarchy strength non-monotonicity and safety-critical feature absorption. However, the experimental results show:

1. **H_Mech**: Partial confirmation with replication problems; stochastic setting reveals decoder contribution is non-zero
2. **H_Comp**: FALSIFIED - no monotonic relationship (R²=0.04)
3. **H_Pareto**: UNINTERPRETABLE - sensitivity metric broken (constant across all L0 levels)
4. **H_Safe**: FAILED on real SAEs (p=0.345 GPT2, p=1.0 Gemma)

**Novelty Verdict: WEAK** - The "encoder-driven" mechanism is partially undermined by stochastic results showing decoder saturation effects. The non-monotonic hierarchy finding is genuinely novel but requires validation. The safety claim failed.

**Venue Recommendation: Workshop tier (ACL Interpretability Workshop, NeurIPS Interpretability Workshop)** - The contribution margin is insufficient for a main-conference track paper. A workshop paper on the methodological contributions (factorial decomposition methodology, boundary conditions for sensitivity metrics) is achievable but requires fixing the broken H_Pareto metric first.

---

## 1. Baseline Landscape

### Comparison Table: Key Methods on Absorption Reduction

| Method | Absorption Reduction | Year | Notes |
|--------|----------------------|------|-------|
| **Matryoshka SAE** | 0.49 → 0.05 (90% reduction) | 2025 | Nested dictionaries; ~50% computational overhead |
| **HSAE (Feature Forest)** | Substantially outperforms baselines | 2026 | Joint parent-child training; less validated |
| **OrtSAE** | 65% absorption reduction | 2025 | Orthogonality penalty; strong at L0=70 |
| **AdaptiveK SAE** | Superior on SAEBench | 2025 | Dynamic sparsity; data-efficient |
| **JumpReLU SAE** | 0.0114 vs TopK 0.1402 | 2024 | Learnable thresholds |
| **This work (iter_003)** | Not directly measured | 2026 | Claims encoder-driven; partially confirmed |

### Key Published Numbers (from literature.md)

| Source | Metric | Value | Context |
|--------|--------|-------|---------|
| Chanin et al. (2024) | Absorption Rate | 0.49 (baseline) → 0.05 (Matryoshka) | Gemma 2B layer 12 |
| Korznikov et al. (2025) | OrtSAE reduction | 65% vs baseline | Standard SAEs |
| Hu et al. (2025) | Feature Sensitivity | Declines with SAE width | 7 variants up to 1M features |
| SAEBench (ICML 2025) | 8-metric benchmark | Proxy metrics do not predict performance | 200+ SAEs |

**Critical Gap**: This work does NOT report direct absorption rate reduction percentages comparable to Matryoshka (90%) or OrtSAE (65%). The H_Mech results are reported in arbitrary units (B=0.055 vs D=0.017) without calibration to established baselines.

---

## 2. Contribution Margin Analysis

### Delta Computation

| Comparison | Baseline | Ours | Delta | Classification |
|------------|----------|------|-------|----------------|
| Encoder vs Decoder (H_Mech) | N/A | B≈D confirmed | N/A | **Moderate** - encoder role validated BUT with caveats |
| Hierarchy strength (H_Comp) | Theoretical: monotonic | Observed: non-monotonic | R²=0.04 | **Marginal** - negative result, not improvement |
| Sensitivity-absorption frontier | N/A | Constant sensitivity | Unmeasurable | **Marginal** - metric broken |
| Safety feature absorption | N/A | p=0.345 (GPT2), p=1.0 (Gemma) | Not significant | **Marginal** - failed hypothesis |

### Classification Summary

- **Strong (>5%)**: NONE
- **Moderate (1-5%)**: H_Mech encoder-driven mechanism (PARTIAL)
- **Marginal (<1%)**: All hypotheses - either failed, broken, or uninterpretable

### Honest Assessment

Marginal gains on a well-studied benchmark (SAE absorption) do NOT constitute a publishable main-conference contribution without:
1. A working sensitivity metric to establish the Pareto frontier
2. Replication of H_Mech across more seeds to confirm encoder-driven story
3. A validated positive result (safety or otherwise) on real SAEs

The current trajectory produces a workshop paper at best, and only if the methodology is fixed.

---

## 3. Concurrent Work Scan

### Recent Papers (2025-2026)

| Paper | Date | Key Finding | Relationship to This Work |
|-------|------|-------------|---------------------------|
| Chanin et al. (2026) SynthSAEBench | Feb 2026 | Ground-truth benchmark; MP-SAEs exploit superposition | Establishes evaluation methodology |
| Luo et al. (2026) HSAE | Feb 2026 | Joint parent-child training; substantially outperforms | Superior absorption reduction |
| Korznikov et al. (2025) OrtSAE | Sep 2025 | 65% absorption reduction | Superior absorption reduction |
| Hu et al. (2025) Feature Sensitivity | Sep 2025 | Sensitivity declines with width | Methodology reference; this work's metric broken |

### Concurrent Overlap Assessment

**Risk: HIGH** - The field has moved quickly. By the time this work validates its findings:
- HSAE (Feb 2026) addresses absorption via hierarchical training with better results
- OrtSAE (Sep 2025) provides simpler orthogonal regularization with 65% reduction
- This work's claimed novelty ("encoder-driven mechanism") is undermined by:
  1. The finding that decoder contributes in saturation regime
  2. The broken sensitivity metric
  3. The failed safety hypothesis

---

## 4. Novelty Verdict

**ONE-SENTENCE NOVELTY**: "This work is the first to factorial-decompose encoder vs decoder contributions to SAE feature absorption, identifying the encoder's learned alignment with hierarchical structure as the primary driver."

**VERDICT: NOVELTY QUESTIONABLE**

### Why:

1. **Chanin et al. (2024)** already established that absorption correlates with hierarchy and sparsity - this work claims encoder alignment but does not measure alignment quality directly (only trained vs random status)

2. **The stochastic hierarchy experiment REVEALED decoder contribution** (Condition C wild variance: 17.3, 43.8 in some seeds) - the "encoder-only" story fails to replicate iter_001's deterministic result

3. **No quantitative improvement over baselines** - Matryoshka achieves 90% reduction; this work does not report a comparable absorption rate reduction

4. **The non-monotonic hierarchy strength finding is genuinely novel** but requires validation with finer grid and more seeds before claiming as contribution

---

## 5. Venue Recommendation

### Tier: Workshop (ACL Interpretability Workshop / NeurIPS Interpretability Workshop)

**Justification**:

| Factor | Assessment |
|--------|-------------|
| Contribution margin | <1% (marginal) - main conference too weak |
| Novelty | Questionable - encoder mechanism not fully validated |
| Methodology | 2/5 (poor) - broken metrics, failed hypotheses |
| Concurrency risk | HIGH - HSAE, OrtSAE already achieve better results |

**Comparable Papers at Workshop Tier**:
- Marks et al. (2025/2026) "A Unified Theory of Sparse Dictionary Learning" - theoretical contribution, workshop acceptable
- Song et al. (2025) "Feature Consistency in SAEs" - methodological contribution, workshop acceptable

**Comparable Papers at Main Conference (NOT achievable with current results)**:
- Bussmann et al. (2025) Matryoshka SAE - 90% absorption reduction, ICLR
- Luo et al. (2026) HSAE - substantially outperforms, arXiv preprint

---

## 6. Strengthening Plan

### Priority 1: Debug Sensitivity Metric (H_Pareto) - HIGH IMPACT

**Problem**: Sensitivity metric returns constant value (3.018) across all L0 levels, making Pareto frontier analysis impossible.

**Action**: 
1. Compare implementation against Hu et al. (2025) source code
2. Test on synthetic case where sensitivity SHOULD vary
3. Validate on Gemma Scope SAE before running full experiment

**Expected Impact**: Enables the core theoretical contribution (sensitivity-absorption trade-off). Without this, H_Pareto is unpublishable.

---

### Priority 2: Fix H_Safe Feature Selection - MEDIUM IMPACT

**Problem**: Safety features [1024, 2048, ...] are arbitrary indices with zero absorption readings.

**Action**:
1. Retrieve actual Neuronpedia annotations for Gemma Scope layer 12
2. Match safety vs non-safety by true activation frequency
3. Verify measurement works on non-zero features first

**Expected Impact**: If p<0.05 with proper feature selection, this becomes a 9/10 novelty safety finding. If p>0.3, properly document as negative result with larger sample (50 per group).

---

### Priority 3: Replicate H_Mech with More Seeds - MEDIUM IMPACT

**Problem**: Condition C shows extreme outliers (17.3, 43.8) in 2/5 seeds; Condition D suspiciously invariant at 0.0175.

**Action**:
1. Run 20 seeds across all 4 conditions
2. Test at varying parent activation levels (0.1, 0.5, 1.0, 2.0, 5.0)
3. Investigate whether Condition C extremes are bugs or real saturation effect

**Expected Impact**: Validates or invalidates the core "encoder-driven" contribution. If decoder saturation is real, the paper's main claim requires qualification.

---

## 7. Anti-Pattern Flags

| Anti-Pattern | Status in This Work |
|--------------|---------------------|
| Comparing against stale baselines | NO - literature.md is recent (2025-2026) |
| Claiming novelty without verifying concurrent work | **YES** - HSAE (Feb 2026) already addresses hierarchical absorption |
| Recommending top-tier for marginal improvement | NO - this analysis correctly recommends workshop |
| Ignoring recent SOTA | **YES** - HSAE, OrtSAE not compared against quantitatively |

---

## 8. Final Verdict

**Overall Assessment**: This work has a genuinely interesting negative result (non-monotonic hierarchy strength) and a partially validated mechanism (encoder-driven absorption), but:

1. **H_Pareto is broken** - the core theoretical contribution cannot be claimed
2. **H_Safe failed** - the highest-novelty safety claim is not supported  
3. **H_Mech replication problems** - the main contribution has caveats
4. **No quantitative improvement** over Matryoshka (90%) or OrtSAe (65%)

**Path to Publication**:
- Fix sensitivity metric → validate Pareto frontier
- Properly power H_Safe (50 per group) → either significant result or properly documented negative
- Replicate H_Mech with 20 seeds → confirm or qualify encoder-driven story
- Compare quantitatively against Matryoshka/OrtSAE baseline numbers

**Current State**: Not ready for submission. Workshop tier possible only after methodological fixes.

---

*Generated by sibyl-comparativist agent (sibyl-light tier)*
*Evidence contract: All claims backed by experiment outputs in /exp/results/*
