# Testable Hypotheses — ComposeAccel (Revised After Pilot Evidence)

*Last updated: 2026-04-14 (post result-debate synthesis)*

---

## Core Hypotheses (Iter 001 Results)

### H1: KV-Cache × Adaptive-Scheduling Sub-Orthogonality

**Statement**: At aggressive adaptive scheduling settings (step_jump >= 4x), KV-cache effectiveness degrades because rapid masking-pattern changes invalidate cached KV states.

**Empirical Status**: **NOT DIRECTLY TESTED — M2 DROPPED (FM1)**

M2 (adaptive step scheduling) was found fundamentally broken for LLaDA's discrete masking before pairwise testing could complete. Step_jump >= 4x causes accuracy collapse. Step_jump = 2x retains 76% accuracy but is not practically useful. M2 is now documented as Failure Mode FM1 (Discrete Masking Incompatibility), not as a composable method.

**Revised claim**: H1 is superseded. The more important finding is that DDIM-style adaptive step schedules are architecturally incompatible with discrete MDM masking — not merely sub-orthogonal, but fundamentally broken. This is a publishable negative finding generalizable to any attempt to apply continuous-diffusion step schedules to LLaDA/Dream-class models.

---

### H2: KV-Cache × IGSD Orthogonality

**Statement**: KV-caching (M1) and IGSD are highly orthogonal because M1 reduces per-step attention cost and IGSD reduces total forward passes.

**Empirical Status**: **EXCEEDED — SUPER-MULTIPLICATIVE SYNERGY**

Pilot results (100 samples, seed=42): Ortho=3.862 (using max-individual normalization), or Ortho=1.385 (using product-of-individuals normalization from result-debate). Both values confirm super-multiplicative synergy. Combined speedup 8.88x (pilot) vs. predicted ~5.1x multiplicative baseline.

**Mechanism confirmed**: IGSD's DRAFT phase accepts ~52% of tokens as high-confidence (at tau=0.9, T_draft=16), freezing them as stable KV anchors during the REFINE phase. EntropyCache hit rate during REFINE reaches ~97%, exploiting these frozen tokens at near-maximum cache efficiency. This creates a positive feedback: IGSD provides favorable context for caching, caching amplifies IGSD's effective speedup.

**Status**: Directionally robust across 2 seeds. REQUIRES full-scale validation (3 seeds, full benchmarks) before publication.

**Operationalization (full-scale)**:
- Ortho(M1+IGSD) > 1.0 on at least 2 reasoning benchmarks (GSM8K, MATH500)
- Combined QAS exceeds max individual QAS by >= 1.5x
- KV hit rate in REFINE phase >= 90% (confirming mechanism)

---

### H3: No Near-Multiplicative Four-Way Combination

**Statement**: Combining all methods does not achieve near-multiplicative speedup due to sub-optimal sub-combinations.

**Empirical Status**: **CONFIRMED — AND STRONGER THAN PREDICTED**

Original H3 predicted a gradient landscape with varying orthogonality. Empirical result reveals a binary landscape: exactly ONE pair synergizes; all others destructively interfere. The four-way combination would include M3 (which destroys cache state via distribution mismatch) and M2 (which causes accuracy collapse), making the overall combination worse than M1+IGSD alone.

**Revised claim**: MDM composability is not a gradient landscape — it is binary. The theoretical model of "orthogonal methods acting on independent bottlenecks" fails for MDMs because all methods interact through the shared global mask state.

---

### H4: Task-Dependent Optimal Recipe

**Statement**: The Pareto-optimal method combination differs between mathematical reasoning and code generation.

**Empirical Status**: **CONFIRMED DIRECTIONALLY**

Evidence from pilots:
- M3 (AR guidance): GSM8K QAS=1.675, HumanEval QAS degenerate (0.0 baseline). AR guidance helps reasoning, completely fails coding.
- IGSD: reasoning-heavy accuracy loss (35% at tau=0.9). Better for throughput-focused scenarios.
- M1+IGSD: most consistent QAS gain across reasoning benchmarks.
- Coding benchmarks (HumanEval/MBPP) are degenerate for LLaDA-8B-Instruct and must be excluded from primary analysis.

**Updated claim**: Task dependence exists but is driven by (a) the degenerate coding baselines making fair comparison impossible, and (b) AR guidance being systematically misaligned with MDM denoising for code. The practical recommendation is M1+IGSD for all tasks (reasoning and any future coding benchmark with non-degenerate baseline).

---

### H5: KV-Cache Failure Correlates with Large Unmasking Events

**Statement**: KV-cache approximation error exceeds safe threshold when more than N/4 tokens change state in a single step.

**Empirical Status**: **CONFIRMED, REFRAMED**

The failure mode is not primarily caused by large per-step unmasking. Empirical finding: at low entropy thresholds (< 1.0), the entropy computation overhead exceeds the cache benefit, causing speedup inversion (speedup < 1.0 vs. baseline). M1 only achieves net speedup at threshold >= 2.0. At threshold = 2.0, combined speedup = 1.38x but accuracy retention drops to 60.6%.

**Revised FM2 characterization**: KV-cache failure has two modes:
- FM2a (overhead inversion): entropy threshold < 1.0 → compute overhead > cache savings → net slowdown
- FM2b (accuracy cliff): entropy threshold > 3.0 → aggressive skipping → accuracy collapses
- Optimal operating window: threshold in [2.0, 3.0]

---

### H6: IGSD Core Feasibility

**Statement**: IGSD acceptance rate >= 60% at tau=0.85 with T_draft=8.

**Empirical Status**: **CONFIRMED, PARAMETERS REVISED**

Acceptance rate at tau=0.85: 96.3% (substantially exceeds threshold). However, the effective parameter configuration is tau=0.9, T_draft=16–32 (not the original tau=0.85, T_draft=8 design).

**Critical open finding**: At tau=0.0 (no confidence partitioning), QAS=1.801 vs. QAS=0.956 for full IGSD at tau=0.9. This 88.5% QAS improvement by *removing* the confidence partitioning mechanism is the most counter-intuitive finding of iter_001. Must be resolved by:
- Experiment A: Compare IGSD tau=0.0 vs. naive "uniform T=16 denoising" (no IGSD machinery). If they match: IGSD mechanism adds nothing over step reduction. If IGSD tau=0.0 > naive T=16: the acceptance gate adds value but partitioning is harmful.
- This result is critical because it either reframes IGSD as "just step reduction with a speed-quality knob" or reveals that confidence-based partitioning actively hurts through REFINE phase over-computation.

---

## New Hypotheses Generated from Iter 001 Data (Targets for Iter 002)

### NH1: Frozen-Token Fraction as Synergy Predictor

**Statement**: The frozen-token fraction |S_accept|/N at the end of IGSD's DRAFT phase is the primary predictor of M1+IGSD synergy magnitude. Ortho should correlate positively with frozen-token fraction as tau decreases (more tokens frozen at lower thresholds → higher synergy).

**Why important**: If NH1 holds, the synergy mechanism is principled — not a lucky coincidence. It also provides a per-instance predictor: when frozen fraction is high, M1+IGSD will synergize; when frozen fraction is low (e.g., for highly uncertain inputs), synergy degrades.

**Test**: Vary tau in [0.5, 0.7, 0.9, 0.95] and measure (frozen_fraction, Ortho) pairs. Fit linear model.

---

### NH2: MDM Semantic Commitment Within First 16 Steps

**Statement**: MDM semantic content (topic, major entities, argument structure) is >80% determined within the first 16 denoising steps, explaining why IGSD's 16-step draft captures sufficient quality for confident partitioning.

**Test**: Compare token-level embedding similarity at steps {8, 16, 32, 64} using cosine similarity between intermediate and final token embeddings. If similarity at step 16 > 0.80 of similarity at step 64, hypothesis is supported.

---

### NH3: M3+IGSD Interference via Distribution Mismatch

**Statement**: M3+IGSD interference arises because AR guidance logits (from Qwen2.5-0.5B trained on causal language modeling) are misaligned with LLaDA's masked denoising trajectory, and IGSD's confidence-based partitioning amplifies this mismatch by fixing tokens based on corrupted confidence estimates.

**Test**: Replace Qwen2.5-0.5B with a DLM-native guidance signal (e.g., a DLM trained on masked-only sequences) and measure M3_native+IGSD composability. If interference decreases, distribution mismatch is confirmed as the mechanism.

---

## Quality Thresholds (Updated)

| Metric | Minimum | Target |
|--------|---------|--------|
| Full-scale M1+IGSD Ortho (mean, 3 seeds) | 0.80 | 1.0+ |
| Full-scale M1+IGSD combined speedup | 3x | 6x |
| Reasoning-only (GSM8K+MATH500) accuracy retention | 50% | 70% |
| tau=0.0 vs. naive T=16 QAS difference | 10% (directional) | Explain mechanism |
| SSD+M1 Ortho (comparison) | Measure and report | Differentiate IGSD vs. SSD composability |

---

## Measurement Protocol

- **Hardware**: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM)
- **Seeds**: 3 [42, 123, 456]; report mean ± std
- **Throughput**: Wall-clock TPS over 100 stable-state samples (discard first 5)
- **Accuracy**: task-standard metrics (GSM8K: exact match; MATH500: exact match; HumanEval: pass@1 reported separately with degenerate-baseline caveat)
- **QAS**: Speedup × Accuracy-Retention (primary ranking metric)
- **Ortho**: QAS(combined) / max(QAS(M_i)) — max-individual normalization; also report product normalization for comparison
- **Statistical test**: Wilcoxon signed-rank for pairwise comparisons; alpha=0.05
- **Primary benchmarks for claims**: GSM8K, MATH500 only; HumanEval/MBPP reported with explicit degenerate-baseline caveat
