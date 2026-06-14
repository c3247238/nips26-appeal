# Comparativist Analysis: Positioning Against SOTA and Related Work

**Agent**: sibyl-comparativist (updated)
**Date**: 2026-04-14
**Workspace**: sae-absorption / iter_006
**Prior verdict score**: 4.5/10

---

## Executive Summary

After thorough analysis of the full experiment results against the current SOTA landscape, the honest assessment is: **this work's genuine contribution is narrower than originally framed but occupies a genuinely underserved niche**. The multi-L0 confound decomposition -- showing 98.6% of false negatives are hedging, not hierarchy-driven absorption at L0=22 -- is the single strongest finding and directly challenges the interpretation of Chanin et al.'s widely-cited absorption rates. The CMI diagnostic from rate-distortion theory is novel but empirically fragile (dimension-dependent, marginally significant). The cross-domain absorption characterization, while first-of-its-kind, is invalidated by universal control failure. The paper should be reframed as a "metric audit + L0 phase transition + rate-distortion diagnostic" rather than a "cross-domain absorption characterization."

---

## 1. Baseline Landscape: Top Existing Methods

### 1a. Published Absorption Rates (First-Letter Task)

| Method / Paper | Model | SAE Config | Absorption Rate | Metric | Year | Venue |
|---|---|---|---|---|---|---|
| Chanin et al. | GPT-2, Gemma 2 2B, Llama 3.2, Qwen2 | Gemma Scope, layers 0-17, widths 16k-65k | **15-35%** across all SAEs | Probe-based (cosine + magnitude gap) | 2024 | NeurIPS 2025 Oral |
| SAEBench (Karvonen et al.) | Gemma 2 2B | 200+ SAEs, 7 architectures | Varies by architecture; ReLU worst | Mean absorption fraction + full-absorption score | 2025 | ICML 2025 |
| ATM-SAE (Li et al.) | Gemma 2 2B L12 | Custom ATM | **0.0068** (vs TopK 0.1402, JumpReLU 0.0114) | SAEBench absorption score | 2025 | arXiv |
| Matryoshka SAE (Bussmann et al.) | Gemma 2 2B | Nested widths | Best on SAEBench absorption; positive scaling with width | SAEBench absorption score | 2025 | ICML 2025 |
| OrtSAE (Korznikov et al.) | Gemma 2 2B, Llama 3 8B | Custom OrtSAE | **65% reduction** vs BatchTopK | SAEBench absorption score | 2025 | arXiv |
| KronSAE | Gemma 2 2B | Kronecker factored | Reduced mean absorption fraction | Multiple | 2025 | arXiv |
| Masked Regularization (Narayanaswamy et al.) | Multiple architectures | Token masking | Reduced absorption + improved OOD robustness | SAEBench-derived | 2026 | arXiv |

### 1b. Our Results

| Experiment | Config | Absorption Rate | Probe F1 | Key Constraint |
|---|---|---|---|---|
| First-letter pilot (L0=82, 25 words/letter) | L12-16k | 13.4% [7.2%, 18.1%] | 0.565 | Low probe quality |
| First-letter improved (L0=82, 50+ words/letter) | L12-16k | ~16% aggregate | ~0.90 | Improved probes |
| **Confound decomposition L0=22** | L12-16k | **42.9%** [40.1%, 45.6%] | **1.0** | All probes perfect |
| Confound decomposition L0=41 | L12-16k | 37.5% [34.8%, 40.2%] | 1.0 | All probes perfect |
| Confound decomposition L0=82 | L12-16k | 14.4% [12.4%, 16.4%] | 1.0 | All probes perfect |
| Confound decomposition L0=176 | L12-16k | 0.8% [0.3%, 1.4%] | 1.0 | All probes perfect |
| Cross-domain: City-Continent | L12-16k | 6.5% [0%, 11.5%] | 0.795 | Controls fail |
| Cross-domain: City-Language | L12-16k | 6.6% [0%, 4.3%] | 0.745 | Controls fail |
| Cross-domain: City-Country | L12-16k | 0.0% | 0.602 | Controls fail |
| Cross-domain: Animal-Class | L12-16k | 1.4% [0%, 3.6%] | 0.696 | Controls fail |

---

## 2. Contribution Margin Assessment

### Contribution 1: Multi-L0 Confound Decomposition -- STRONG

**Delta vs SOTA**: This is the strongest finding and provides a genuinely new perspective on absorption.

At L0=22 with perfect probe quality (F1=1.0 for all 25 letters), 657 false negatives were detected. Classification:
- **98.6% hedging** (false negatives that also appear at adjacent L0 values -- the SAE spreads information across L0 regimes)
- **1.4% hierarchy-driven** (only 9 words are absorbed in the hierarchical sense)
- **0% reconstruction error**

The trend is perfectly monotonic (Spearman rho = 1.0): as L0 increases from 22 to 176, hedging drops from 98.6% to 10.0% and hierarchy-driven increases from 1.4% to 90.0%.

**Why this matters**: The Chanin et al. absorption metric (NeurIPS 2025 Oral) reports 15-35% absorption rates and interprets them as evidence that "a more specific feature has absorbed the parent's information under sparsity pressure." Our decomposition shows that at L0=22, **the metric is primarily detecting hedging (information spreading due to low L0), not hierarchy-driven competitive exclusion**. Only ~0.75% of all tested words (9/1195) exhibit genuine hierarchy-driven absorption at L0=22. This fundamentally changes how the community should interpret absorption measurements.

**Comparison to nearest work**:
- Chanin & Dulka (2025, "Feature Hedging"): Characterize hedging as a distinct phenomenon but do not decompose absorption metric output into hedging vs hierarchy components.
- Chanin & Garriga-Alonso (2025, "Sparse but Wrong"): Show incorrect L0 causes wrong features, but do not quantify the fraction of measured "absorption" attributable to L0. Our work provides this quantification.
- Neither paper has produced a multi-L0 decomposition with perfect probes.

**Contribution classification**: >5% = **strong contribution**. First quantitative decomposition of a widely-used metric's output into failure mode components.

### Contribution 2: Cross-Domain Absorption Characterization -- UNRELIABLE

**Delta vs SOTA**: First measurements of absorption outside the first-letter task.

| Domain | Rate | Shuffled Control | Random Control | Net Signal (rate - shuffled) | Credible? |
|---|---|---|---|---|---|
| First Letter | 13.4% | 59.5% | 9.2% | **-46.2%** | No |
| City-Country | 0.0% | 10.3% | 19.0% | **-10.3%** | No |
| City-Continent | 6.5% | 45.2% | 12.9% | **-38.7%** | No |
| City-Language | 6.6% | 18.0% | 20.8% | **-11.5%** | No |
| Animal-Class | 1.4% | 39.3% | 34.3% | **-37.9%** | No |

**Honest assessment**: In every domain tested, shuffled-label controls produce higher "absorption" rates than true labels. The net signal is negative everywhere. This means the measured absorption rates cannot be distinguished from metric noise. The cross-domain rates (0-6.6%) are first-of-their-kind measurements, but they are currently uninterpretable as genuine absorption signal.

**This is itself a finding**: No prior work has reported that the Chanin absorption metric fails control checks when applied to Gemma 2 2B JumpReLU SAEs. The control failure is informative about metric limitations and should be reported.

**Contribution classification**: <1% = **marginal as positive result**; moderate as **methodological negative result**.

### Contribution 3: Rate-Distortion CMI Diagnostic -- FRAGILE

**Delta vs SOTA**: No prior work applies conditional mutual information I(X; f_parent | f_child) from the successive refinement theorem to SAE absorption. Confirmed novel via systematic literature search.

**Key results**:
- Spearman rho = -0.383 (correct direction: absorbed letters have lower CMI)
- Mann-Whitney p = 0.045 (one-sided), Cohen's d = -0.924 (large effect)
- Predicted L0_crit = 24.7 vs empirical half-max = 22.4 (10.2% error)
- Geometric constant c degenerates for unit-normalized SAEs (CV = 2.16%)

**Critical caveat**: The negative correlation **only holds at subspace dimension d'=10**. At d'=20, 30, 50, the relationship disappears or reverses (rho goes to +0.048, +0.299, +0.197). This dimension sensitivity is a serious concern -- it may reflect a genuine signal in the most absorption-relevant subspace, or it may be an artifact of low-dimensional projection.

**Comparison**:
- Unified SDL Theory (arXiv:2512.05534): More complete theoretical framework explaining absorption via optimization landscape (WHY). Our work asks WHEN absorption is avoidable (complementary, not competing).
- MDL-SAEs (Ayonrinde et al., 2024): Information-theoretic framing of SAEs as compression, but does not address absorption specifically.
- No prior application of successive refinement theorem to SAEs found.

**Contribution classification**: 1-5% = **moderate contribution**. Novel theoretical framework but fragile empirical support.

### Contribution 4: Bifurcation Analysis -- CONFOUNDED

**Finding**: JumpReLU SAEs show dramatic L0-dependent phase transition (1.1% at L0=176 to 36.2% at L0=22 to 64.3% at 65k width) while L1 SAEs show uniformly high absorption (61-67%). KS test D=0.607 (p~0).

**Confound**: JumpReLU results are from Gemma 2 2B (2304-dim); L1 results are from GPT-2 Small (768-dim). The high L1 absorption may reflect model capacity differences rather than architecture effects.

**Contribution classification**: 1-5% = **moderate descriptive finding** with cross-model confound.

### Contribution 5: Unsupervised Detection -- FAILED

ITAC does not separate candidate from random pairs (median 1.35 vs 1.14, Mann-Whitney not significant). Most letters have zero matching pairs. All component correlations with gold-standard absorption are negative or near-zero. AUROC values below chance (0.39-0.47).

**Contribution classification**: **Pre-registered negative result**. Honest reporting is appropriate.

---

## 3. Concurrent Work Scan (Last 6 Months)

### Direct Competitors

| Paper | Date | Overlap | Threat Level | Assessment |
|---|---|---|---|---|
| **Chanin & Garriga-Alonso "Sparse but Wrong"** | 2025 | L0 causes wrong features -> our hedging finding is a corollary | **HIGH** | Our multi-L0 decomposition extends their work quantitatively but they established the theoretical foundation |
| **ATM-SAE** (Li et al.) | Oct 2025 | Absorption score 0.0068 -- near-total mitigation | **MEDIUM** | If absorption is "solved" by ATM, diagnostic work loses urgency; but understanding WHY remains valuable |
| **Unified SDL Theory** | Dec 2024 | Piecewise biconvex framework explains absorption | **MEDIUM** | More mature theoretical framework than our CMI approach; complementary (WHY vs WHEN) |
| **Masked Regularization** | Apr 2026 | Reduces absorption via token masking | **LOW** | Mitigation vs characterization -- complementary |
| **Domain-Specific SAEs ("Resurrecting the Salmon")** | Aug 2025 | Domain-specific training reduces absorption | **MEDIUM** | Overlaps with our critique of metric generalizability |
| **HSAE "Atoms to Trees"** | Feb 2026 | Hierarchical SAE with parent-child relationships | **LOW** | Different approach (architecture vs diagnostic) |
| **Feature Sensitivity** (Tian et al.) | Sep 2025 | Absorption as low sensitivity | **LOW** | Complementary metric framework |

### Key Threat Assessment

The most significant threat is **"Sparse but Wrong" (Chanin & Garriga-Alonso, 2025)**. Their central finding -- most SAEs have too-low L0, causing feature hedging/mixing -- directly explains our headline result (98.6% hedging at L0=22). **Our contribution narrows to**: (1) first explicit quantification of the hedging fraction within the absorption metric's output, (2) demonstration that this holds even when probe quality is perfect (F1=1.0), and (3) the monotonic L0-hedging profile across 4 sparsity levels.

No concurrent paper was found that:
- Performs cross-domain absorption measurement with controls
- Applies successive refinement / CMI to SAE absorption
- Decomposes absorption metric output into hedging vs hierarchy components

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does**:

This work provides the first quantitative decomposition of the Chanin absorption metric into hedging vs. hierarchy-driven components, revealing that at typical operating L0 (L0=22), 98.6% of what the metric detects is hedging, not the hierarchy-driven competitive exclusion it was designed to measure.

This single finding, if it holds up, changes how the community should interpret every published absorption rate number. It is a legitimate novelty that directly extends the most-cited work in this subfield (Chanin et al., NeurIPS 2025 Oral).

---

## 5. Venue Recommendation

### Scoring

| Factor | Score | Justification |
|---|---|---|
| Novelty | 6/10 | Confound decomposition is genuinely novel; CMI is novel but fragile; cross-domain is novel but unreliable |
| Significance | 5/10 | Challenges interpretation of a widely-used metric but proposes no solution |
| Rigor | 4/10 | Universal control failure on cross-domain; dimension-dependent CMI; classification methodology for hedging vs hierarchy needs external validation |
| Clarity | 6/10 | Well-documented experiments with clear pilot summaries |
| Reproducibility | 7/10 | Uses open Gemma Scope SAEs; builds on sae-spelling infrastructure |

### Current State: Workshop paper

The results in their current form are suitable for a workshop paper at a NeurIPS/ICML mechanistic interpretability workshop. The confound decomposition finding is interesting enough for a workshop audience, and the honest reporting of control failures and negative results (unsupervised detection) demonstrates research maturity.

### After Blocking Experiments: Mid-tier conference

If the three blocking experiments from the verdict pass:
1. **Control investigation**: Net positive signal in >=2 domains after recalibration
2. **CMI replication at L0=22**: Pre-registered d'=10, rho < -0.3, p < 0.05
3. **Activation patching**: Parent recovery in >=7/9 core absorbed words

Then a three-pillar paper (metric audit + L0 phase transition + rate-distortion diagnostic) could target AAAI, EMNLP, or BlackboxNLP.

### Top-Tier Venue (NeurIPS/ICML/ICLR): Unlikely

**Comparable NeurIPS/ICML main papers**:
- Chanin et al. (NeurIPS 2025 Oral): Discovered absorption, toy model proof, validated on hundreds of SAEs
- SAEBench (ICML 2025): 200+ SAEs, 8 metrics, standardized benchmark
- Matryoshka SAE (ICML 2025): Architectural innovation with SAEBench improvement on 5/8 metrics

Our work is narrower in scope than all of these. The contribution margin is one strong empirical finding (confound decomposition) plus one fragile theoretical contribution (CMI). This is below the threshold for top-tier venue, where papers typically combine strong theory with broad empirical validation.

---

## 6. Strengthening Plan (3 Highest-Priority Actions)

### 1. Resolve the Control Anomaly (CRITICAL -- blocks all cross-domain claims)

The shuffled-label control exceeding measured absorption in all 5 domains is the single biggest obstacle. Specific actions:
- Run a **null-domain benchmark**: create a random hierarchy with no true parent-child structure; target absorption rate < 2%.
- Implement **per-parent shuffled controls** (not aggregate) to identify which parents drive inflation.
- Compare our implementation against the **SAEBench reference implementation** of the Chanin metric to rule out implementation divergence.
- If shuffled controls measure feature density rather than absorption, **reframe with transparent discussion** of what the raw rate vs control-adjusted rate means.

### 2. Pre-Register CMI at d'=10 with L0=22 Data (HIGH -- unlocks rate-distortion claims)

Use the confound decomposition data (F1=1.0 at L0=22) and **pre-register d'=10** as the primary analysis dimension. Provide a theoretical justification for d'=10 (e.g., effective rank of parent-child decoder subspace). Report d'=20/30/50 as exploratory. Extend to at least one cross-domain hierarchy (city-continent).

### 3. Add Matryoshka SAE Comparison on First-Letter (MEDIUM -- maximally strengthens positioning)

Gemma Scope 2 includes Matryoshka SAEs on Gemma 2 2B. Measure absorption rates at matched L0 on the same first-letter task. If Matryoshka SAEs show different hedging/hierarchy ratios in the confound decomposition, this connects our diagnostic to the leading mitigation approach and substantially strengthens the paper's practical relevance.

---

## 7. Detailed Comparison with Closest Competitors

### vs. Chanin et al. (2024, NeurIPS 2025 Oral)

| Dimension | Chanin et al. | This Work | Winner |
|---|---|---|---|
| Domain coverage | First-letter only | 5 domains (but controls fail on all) | This work (scope) / Chanin (reliability) |
| Model coverage | Gemma 2 2B, Llama 3.2, Qwen2 | Gemma 2 2B + GPT-2 (bifurcation) | Chanin |
| SAE coverage | 200+ SAEs, all architectures | ~34 Gemma Scope + 3 L1 | Chanin |
| Metric creation | Created the absorption metric | Uses + audits the metric | Chanin (original) / **This work (audit)** |
| Theory | Toy model proof (sufficient condition) | Rate-distortion CMI (necessity condition) | Different questions |
| Control rigor | No shuffled controls reported | Shuffled + random (discovers anomaly) | **This work** |
| Confound decomposition | None | Multi-L0 with F1=1.0 probes | **This work** |

**Positioning**: Extends Chanin et al. in three orthogonal directions (cross-domain scope, information-theoretic theory, confound decomposition). The control anomaly discovery is itself a contribution.

### vs. "Sparse but Wrong" (Chanin & Garriga-Alonso, 2025)

| Dimension | Sparse but Wrong | This Work |
|---|---|---|
| Core finding | Incorrect L0 -> wrong features (hedging) | 98.6% of absorption metric output is hedging at L0=22 |
| Methodology | Toy model + empirical L0 sweep | Multi-L0 decomposition with F1=1.0 probes |
| Metric connection | Does not directly analyze absorption metric | Decomposes absorption metric output |
| Practical implication | Choose correct L0 | Reinterpret all published absorption rates |

**Positioning**: Our work is a natural empirical extension of "Sparse but Wrong." They establish the theoretical foundation; we quantify the impact on the community's primary absorption metric. **Risk**: reviewers may see our finding as an obvious corollary of their work.

### vs. Matryoshka SAE (Bussmann et al., ICML 2025)

| Dimension | Matryoshka SAE | This Work |
|---|---|---|
| Goal | Mitigate absorption | Diagnose and characterize |
| Key question | "How to reduce absorption?" | "When is absorption metric reliable? When is absorption necessary?" |
| SAEBench results | Best on 5/8 metrics | Not evaluated on SAEBench |

**Positioning**: Complementary. If our CMI diagnostic can predict which features Matryoshka successfully de-absorbs, the combination would be powerful. Without this connection, the two papers live in separate lanes.

### vs. ATM-SAE (Li et al., 2025)

ATM achieves absorption score 0.0068 (20x better than TopK, 1.7x better than JumpReLU). If ATM is viewed as "solving" absorption, our characterization work loses urgency. However, understanding WHY absorption occurs and WHEN it matters remains valuable even with solutions available -- analogous to how understanding disease mechanisms matters even when treatments exist.

---

## 8. Summary Verdict

| Contribution | Novelty | Evidence Strength | Delta vs SOTA | Venue Impact |
|---|---|---|---|---|
| Multi-L0 confound decomposition | Moderate (extends Chanin/Garriga-Alonso) | **Strong** (F1=1.0, 4 L0, rho=1.0) | **>5% (strong)** | Anchor for paper |
| Cross-domain absorption | **High** (first measurement) | **Weak** (all controls fail) | **<1% (marginal)** until controls resolved | Supplementary/negative |
| CMI rate-distortion diagnostic | **Very high** (no prior work) | **Moderate** (d'=10 only, p=0.059) | **1-5% (moderate)** | Secondary pillar |
| Bifurcation analysis | High (first LCA-SAE) | Moderate (cross-model confound) | **1-5% (moderate)** | Supporting result |
| Phase transition scale match | High | Moderate (partially circular) | **1-5% (moderate)** | Supporting result |
| Geometric constant degeneration | Novel but expected | Strong (clean negative) | **<1%** | One paragraph |
| Unsupervised detection | Moderate | Failed | **<1%** | Negative result |

### Bottom Line

The paper should be reframed around the confound decomposition as the primary empirical contribution ("What does the absorption metric actually measure?"), with the CMI diagnostic as the primary theoretical contribution ("When is absorption information-theoretically necessary?"), and cross-domain characterization as a methodological negative result ("The absorption metric does not transfer to knowledge hierarchies without recalibration"). This framing is honest, novel, and occupies a genuine gap in the literature. The blocking experiments (control investigation, CMI replication, activation patching) determine whether the paper targets a workshop or a mid-tier conference.

**Recommended venue**: Workshop (current state) -> Mid-tier conference (after blocking experiments pass) -> Top-tier unlikely without broader validation.

---

## Sources

- [A is for Absorption (Chanin et al., NeurIPS 2025 Oral)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., ICML 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAEs (Bussmann et al., ICML 2025)](https://arxiv.org/abs/2503.17547)
- [ATM-SAE (Li et al., 2025)](https://arxiv.org/abs/2510.08855)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [Masked Regularization (Narayanaswamy et al., 2026)](https://arxiv.org/abs/2604.06495)
- [Sparse but Wrong (Chanin & Garriga-Alonso, 2025)](https://arxiv.org/abs/2508.16560)
- [Feature Hedging (Chanin & Dulka, 2025)](https://arxiv.org/abs/2505.11756)
- [Unified SDL Theory (arXiv:2512.05534)](https://arxiv.org/abs/2512.05534)
- [MDL-SAEs (Ayonrinde et al., 2024)](https://arxiv.org/abs/2410.11179)
- [HSAE "Atoms to Trees" (Luo et al., 2026)](https://arxiv.org/abs/2602.11881)
- [SynthSAEBench (2026)](https://arxiv.org/abs/2602.14687)
- [Domain-Specific SAEs "Resurrecting the Salmon" (2025)](https://arxiv.org/abs/2508.09363)
- [Feature Sensitivity (Tian et al., 2025)](https://arxiv.org/abs/2509.23717)
- [DeepMind Safety SAE Negative Results (2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Gemma Scope (Lieberum et al., 2024)](https://arxiv.org/abs/2408.05147)
- [On the Limits of SAEs (Cui et al., 2025)](https://arxiv.org/abs/2506.15963)
- [KronSAE (2025)](https://arxiv.org/abs/2505.22255)
- [SAEBench at Neuronpedia](https://www.neuronpedia.org/sae-bench/info)
- [NeurIPS 2025 Poster for Chanin et al.](https://neurips.cc/virtual/2025/oral/130303)
