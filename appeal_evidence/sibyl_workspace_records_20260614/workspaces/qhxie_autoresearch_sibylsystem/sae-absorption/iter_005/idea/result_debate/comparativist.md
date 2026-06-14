# Comparative Analysis: Positioning Results Against SOTA and Related Work

**Agent**: Comparativist
**Date**: 2026-04-14
**Iteration**: 5
**Paper**: "Beyond the Spelling Task: Resolving Confounds, Extending Domains, and Mapping the Scaling Surface of Feature Absorption in Sparse Autoencoders"

---

## 1. Baseline Landscape

### Top Methods Addressing Feature Absorption (as of April 2026)

| Method | Absorption Metric | Model / Setting | Key Numbers | Year |
|--------|------------------|-----------------|-------------|------|
| **Chanin et al. (original)** | First-letter absorption rate | Gemma 2B, Llama 3.2 1B, Qwen2 0.5B | 15-35% absorption across widths; present in every tested SAE | NeurIPS 2025 |
| **Matryoshka SAE** (Bussmann et al.) | SAEBench absorption score | Gemma 2B, 65k width | Best absorption + RAVEL + sparse probing; positive scaling behavior; dramatic improvement on first-letter task | ICML 2025 |
| **OrtSAE** (Korznikov et al.) | Mean absorption fraction, full-absorption score | Gemma 2B | 65% absorption reduction; 9% more distinct features; 6% better SCR | 2025 |
| **ATM** (Li et al.) | Mean absorption score | Gemma 2B | 0.0068 vs TopK 0.1402, JumpReLU 0.0114 (20x reduction) | ICLR 2025 Workshop |
| **KronSAE** | Mean absorption fraction, full-absorption score | Multiple LLMs | Reduced absorption via Kronecker factorization; lower compute than TopK | 2025 |
| **Masked Regularization** (Narayanaswamy et al.) | Absorption + OOD robustness | Multiple architectures | Reduces absorption via token masking; April 2026 preprint | April 2026 |
| **SAEBench** (Karvonen et al.) | 8-metric suite including absorption | 200+ SAEs, Gemma 2B | Absorption inversely scales with width for all architectures except Matryoshka | 2025 |

### Benchmarks Used in the Field

| Benchmark | What It Measures | Coverage |
|-----------|-----------------|----------|
| SAEBench absorption task | First-letter absorption rate on Gemma 2B | 200+ SAEs, 8 architectures |
| SAEBench sparse probing | Concept detection via k-sparse probes | Same suite |
| SAEBench RAVEL/TPP | Feature disentanglement on entity-attribute data | Same suite |
| SAEBench SCR | Spurious correlation removal | Same suite |
| SynthSAEBench | Ground-truth feature recovery on synthetic data | Controlled hierarchies |
| Feature Sensitivity (Tian et al.) | Activation reliability on similar inputs | Gemma 2B, various widths |

---

## 2. Contribution Margin Analysis

### Contribution 1: Confound Resolution (H1)

**Our result**: Partial correlation between absorption and sparse probing r = -0.746 (p = 1.16e-09) AFTER controlling for L0. Suppression effect: controlling for L0 strengthened the association. Rosenbaum Gamma = 2.65. Mediation analysis: 3/4 quality metrics show significant indirect effect via absorption.

**What exists**: No prior work has performed mediation analysis, Rosenbaum sensitivity analysis, or formal L0 confound control on the absorption-quality relationship.
- Chanin et al. (2024) report absorption correlations but do not control for L0 or width jointly.
- SAEBench (2025) ranks architectures on absorption but does not test causal structure.
- Feature Hedging paper (Chanin & Dulka, 2025) identifies L0 as a confound for hedging but does not apply to absorption-quality.
- "Sparse but Wrong" (Chanin & Garriga-Alonso, 2025) shows incorrect L0 causes wrong features but does not perform mediation analysis.

**Contribution margin**: **STRONG (>5% equivalent)**. This is the first application of epidemiological causal methods (mediation analysis, propensity matching, Rosenbaum bounds, Bradford Hill criteria) to ANY SAE evaluation question. The finding that L0 control strengthens rather than weakens the absorption-quality link is a genuinely surprising result that overturns the most natural skeptical hypothesis ("absorption is just a proxy for L0"). No concurrent work addresses this.

**Caveat**: Sample size is n=48 SAEs from a single model (Gemma 2B). No individual width stratum achieves significance alone. The unlearning metric shows no association. These limitations are real and must be reported transparently, but do not undermine the core finding.

### Contribution 2: Cross-Domain Absorption (H2)

**Our result**: Dominance-based absorption detected at 51-85% across knowledge domains (country, continent, language) on GPT-2 Small with 3552 cities. HOWEVER, shuffled controls show 100% absorption rate. Cosine-calibrated metric shows 0%. The dominance-based metric does not discriminate real from shuffled hierarchies.

**What exists**: No prior work has measured absorption on knowledge hierarchies.
- Chanin et al. (2024) explicitly note the single-task limitation (first-letter spelling only). At least 5 other papers (SAEBench, SynthSAEBench, OrtSAE, KronSAE, Masked Reg.) repeat this call-out.
- SAE-RAVEL (Chaudhary & Geiger, 2024) measured disentanglement on RAVEL entity-attribute data, NOT absorption.
- SynthSAEBench (2026) provides controlled synthetic hierarchies but does not measure absorption on real-world knowledge hierarchies.

**Contribution margin**: **MODERATE (1-5% equivalent) as a methodological finding; MARGINAL as a positive result**. The fact that the Chanin absorption metric does not transfer to knowledge hierarchies on GPT-2 Small is itself informative. But the inability to distinguish real from shuffled hierarchies means this is primarily a negative/diagnostic finding about metric limitations, not a positive demonstration of cross-domain absorption. The 98% dead features in the GPT-2 SAE severely limit what can be concluded.

**Critical gap**: The study uses GPT-2 Small (124M params) rather than Gemma 2B (2B params), the model all prior absorption work targets. This model choice was forced by access constraints (no Gemma 2B HF token), but it fundamentally weakens the cross-domain contribution. GPT-2 Small has limited factual knowledge representation, and its SAEs have 98% dead features. Running on Gemma 2B would have been a far stronger test.

### Contribution 3: Absorption Scaling Surface (H3)

**Our result**: GAM interaction term p = 3.11e-15 on 420 SAEs from SAEBench. Interaction GAM R^2 = 0.693 vs additive R^2 = 0.620 vs linear R^2 = 0.488. Phase boundary detected at log2(L0) in [2.7, 3.8]. Absorption increases dramatically at 1M width with low L0.

**What exists**:
- Chanin et al. (2024) Figures 9b/9c show absorption vs. L0 at different widths, but no formal regression, no GAM, no interaction testing.
- SAEBench (2025) reports that absorption inversely scales with width for non-Matryoshka architectures, but does not fit a joint width-L0 surface.
- Feature Sensitivity paper (Tian et al., 2025) finds sensitivity decreases with width; 65k SAEs have 0.92-0.94, 1M SAEs have 0.85-0.87. This is a related but distinct metric.
- SAE scaling with feature manifolds (2025) develops theoretical analysis but does not produce an empirical phase surface.

**Contribution margin**: **STRONG (>5% equivalent)**. No prior work has constructed a formal 2D absorption surface with interaction term testing. The p = 3.11e-15 interaction is statistically overwhelming with N = 420. The practical implication (absorption depends on the JOINT structure of width and L0, not either independently) is directly actionable for SAE hyperparameter selection. The phase boundary at L0 ~ 6.5-14 is a novel finding.

### Contribution 4: Taxonomy Correction (H5)

**Our result**: Corrected rate drops from 92.3% to 19.2% using proper non-letter-context baselines. However, Chanin metric finds 73.1% absorption. The original 92.3% Type II rate was indeed an artifact of the n_comparison_tokens=0 fallback, as hypothesized.

**Contribution margin**: **MODERATE (1-5%)**. The correction from 92.3% to 19.2% for the Type II magnitude-based classification is a valuable calibration. However, the Chanin metric (which directly measures false negatives) still finds 73.1% absorption, suggesting the phenomenon is genuine even if the magnitude metric was inflated. This is a useful methodological finding but not headline material.

---

## 3. Concurrent Work Scan (Last 6 Months: October 2025 - April 2026)

| Paper | Date | Overlap | Threat Level |
|-------|------|---------|-------------|
| **Masked Regularization** (Narayanaswamy et al., arXiv:2604.06495) | April 2026 | Reduces absorption via masking; complementary to our characterization work | LOW: mitigation method, not characterization |
| **SynthSAEBench** (arXiv:2602.14687) | Feb 2026 | Controlled synthetic absorption study; does not do real-world cross-domain | LOW: different methodology (synthetic vs. real) |
| **HSAE** (arXiv:2602.11881) | Feb 2026 | Hierarchical SAE with parent-child structure; related to absorption | LOW: architectural solution, not characterization |
| **Sanity Checks for SAEs** (Korznikov et al., arXiv:2602.14111) | Feb 2026 | Shows SAEs barely beat random baselines; contextually relevant | MODERATE: strengthens the case for understanding absorption, does not directly compete |
| **Adversarial Robustness of SAE Concepts** (Li et al., EACL 2026) | 2026 | SAE concept representations are not robust to perturbations | LOW: orthogonal concern |
| **SAE for Sequential Recommendations** (July 2025) | 2025 | Domain-specific SAE application | NONE: different domain |

**No concurrent paper addresses the same three questions simultaneously (confound resolution + cross-domain + scaling surface).** The closest concurrent work is SynthSAEBench, which studies absorption in synthetic settings but does not perform causal analysis on real SAE quality metrics or map the scaling surface.

---

## 4. Novelty Verdict

**What is the ONE thing this work does that no prior work does?**

> This paper is the first to establish that feature absorption has an INDEPENDENT causal association with downstream SAE quality after controlling for L0, using epidemiological methods (mediation analysis, Rosenbaum bounds) never before applied to SAE evaluation. The suppression effect (L0 control strengthens the absorption-quality link) is a genuinely novel finding that overturns the default skeptical hypothesis.

Secondary novelties:
- First formal 2D absorption scaling surface with interaction term testing (p = 3.11e-15, N = 420)
- First attempt to measure absorption on knowledge hierarchies (primarily a negative/diagnostic finding)
- First application of Bradford Hill causal criteria to SAE evaluation

---

## 5. Venue Recommendation

### Assessment

| Factor | Rating | Justification |
|--------|--------|---------------|
| Methodological novelty | High | Epidemiological methods in SAE evaluation is genuinely new |
| Scaling surface result | Strong | p = 3.11e-15, N = 420, directly actionable |
| Confound resolution | Strong | Suppression effect, 3/4 metrics, Bradford Hill 3 strong |
| Cross-domain result | Weak | GPT-2 Small limitation, metric fails shuffled control |
| Statistical rigor | High | Mediation analysis, bootstrap CIs, sensitivity analysis |
| Practical impact | Moderate | Informs SAE hyperparameter selection but does not propose a fix |

### Venue Recommendation: **Mid-to-Top Tier Workshop or Main Conference (conditional)**

**With current results (GPT-2 Small for Phase 2)**: Workshop at NeurIPS/ICML or mid-tier venue (AAAI, EMNLP). The cross-domain contribution is too weak on GPT-2 Small to carry a top-tier submission. The confound resolution and scaling surface are strong, but the paper's three-contribution narrative is undermined by the failed Phase 2.

**If Phase 2 is re-run on Gemma 2B** (replacing GPT-2 Small with the model all SAEBench absorption data uses): **Top-tier workshop at NeurIPS/ICML or borderline main conference (ICLR)**. Comparable papers at these venues:
- Chanin et al. "A is for Absorption" was accepted at NeurIPS 2025 with empirical characterization and a simple toy model.
- Matryoshka SAE (Bussmann et al.) was accepted at ICML 2025 with architectural contribution + SAEBench evaluation.
- SAEBench (Karvonen et al.) was accepted as a benchmark paper.

Our paper's strength is in DEPTH OF ANALYSIS (epidemiological methods, scaling surface, taxonomy correction) rather than ARCHITECTURAL NOVELTY. This positions it as a strong characterization/analysis paper. The field values such papers (Chanin et al. was accepted at NeurIPS), but the bar is high.

### Specific Tier Assessment

- **NeurIPS/ICML main conference**: Possible IF Phase 2 on Gemma 2B shows genuine cross-domain absorption (or a clean negative result). The confound resolution + scaling surface form a solid two-contribution paper even without cross-domain. Confidence: 35%.
- **ICLR main conference**: Similar assessment. The epidemiological methods angle may appeal to the more methodologically focused ICLR reviewers. Confidence: 30%.
- **NeurIPS/ICML workshop (Mechinterp or SAE-specific)**: Strong submission with current results. The confound resolution alone would be of high interest to the community. Confidence: 70%.
- **AAAI/EMNLP main**: Suitable venue with current results. Confidence: 50%.
- **Insufficient for submission**: No. The confound resolution and scaling surface are independently publishable.

---

## 6. Strengthening Plan

### Priority 1 (CRITICAL): Re-run Phase 2 on Gemma 2B

The single most impactful improvement is replacing GPT-2 Small with Gemma 2B for cross-domain absorption measurement. Reasons:
- All prior absorption work uses Gemma 2B (Chanin et al., SAEBench, OrtSAE, ATM)
- GPT-2 Small has limited factual knowledge and 98% dead SAE features
- The failed shuffled control may be a GPT-2 artifact rather than a fundamental metric limitation
- Gemma Scope provides SAEs at multiple widths, enabling direct comparison with the 420-SAE scaling surface

**Estimated cost**: 2-4 GPU-hours. **Estimated impact**: Could elevate the paper from workshop to main conference.

### Priority 2 (HIGH): Add Architecture-Specific Analysis to Scaling Surface

The 420-SAE dataset contains 360 "standard" and 54 JumpReLU SAEs. Testing whether the interaction term holds within architecture subgroups would strengthen the scaling surface finding against the confound that architecture type drives the interaction. Also:
- Add Matryoshka SAEs (from SAEBench or Gemma Scope 2) to show whether they break the phase boundary
- This is a zero-GPU analysis on existing data

### Priority 3 (HIGH): Add Explicit Comparison to ATM and OrtSAE

Include the ATM absorption score (0.0068) and OrtSAE absorption reduction (65%) as baselines in the scaling surface analysis. Position the phase boundary finding as telling practitioners "where to expect absorption problems" while ATM/OrtSAE/Matryoshka tell them "how to mitigate once detected." This framing makes the contributions complementary rather than competitive.

### Priority 4 (MODERATE): Multi-Model Validation of Confound Resolution

The confound resolution uses only Gemma 2B SAEs (n=48). Adding GPT-2 Small SAEs from SAEBench (if absorption scores are available) would provide cross-model validation. Even n=20 additional SAEs from a different model family would strengthen the causal claim substantially.

---

## 7. Honest Assessment of Weaknesses

### Weaknesses That Reviewers Will Target

1. **GPT-2 Small for Phase 2**: This is the paper's Achilles heel. Every reviewer familiar with the SAE absorption literature will note that all prior work uses Gemma 2B and that GPT-2 Small is both too small and uses SAEs with 98% dead features. The shuffled control failure may be entirely attributable to this model choice.

2. **N=48 for confound resolution**: While the statistical tests are rigorous, 48 SAEs from a single model is a small sample. Individual width strata do not achieve significance. A reviewer focused on statistical power will rightly question the generalizability.

3. **No mitigation proposed**: The paper characterizes absorption but does not propose a fix. Matryoshka SAE, OrtSAE, ATM, and Masked Regularization all propose solutions. A pure characterization paper needs especially strong empirical results to justify the lack of architectural contribution.

4. **Taxonomy correction ambiguity**: The final results summary reports "92.3% validated, correction minimal" but the detailed JSON shows the corrected rate is actually 19.2% (not 92.3%). The Chanin metric finds 73.1%. This inconsistency needs to be resolved transparently.

### Strengths That Differentiate

1. **Methodological import**: Epidemiological causal methods are genuinely new to SAE evaluation and applicable far beyond absorption.
2. **Suppression effect**: L0 control strengthening (not weakening) the absorption-quality link is counterintuitive and highly informative.
3. **Scale of scaling surface**: N=420 SAEs with p=3.11e-15 is among the largest systematic analyses in the SAE literature.
4. **Honest negative results**: The failed cross-domain metric is reported transparently rather than hidden.

---

## 8. Summary Comparison Table

| Contribution | Our Result | Best Existing | Delta | Classification |
|-------------|-----------|---------------|-------|----------------|
| Absorption-quality causal chain | r=-0.746 after L0 control (n=48) | Chanin et al. r=-0.595 without L0 control (n=54) | +0.151 (strengthened by confound control) | STRONG: first confound-controlled analysis |
| Cross-domain absorption | Metric fails on GPT-2 Small (shuffled=100%) | None (no prior attempt) | N/A | MARGINAL: informative negative but on wrong model |
| Scaling surface | GAM interaction p=3.11e-15 (N=420) | Chanin Figs 9b/c (visual only, no formal test) | First formal surface | STRONG: first formal 2D surface with interaction test |
| Taxonomy correction | 19.2% corrected (vs 92.3% original) | 92.3% original (Chanin et al., our iter_1) | -73.1pp for Type II; Chanin metric gives 73.1% | MODERATE: important calibration |
| Epidemiological methods | Mediation + Rosenbaum + Bradford Hill | None in SAE literature | First application | STRONG: methodological novelty |
| Best absorption mitigation | N/A (characterization only) | ATM: 0.0068, OrtSAE: 65% reduction, Matryoshka: positive scaling | N/A | NOT COMPETING: different contribution type |

---

## Sources

- [A is for Absorption (Chanin et al., NeurIPS 2025)](https://arxiv.org/abs/2409.14507)
- [Matryoshka Sparse Autoencoders (Bussmann et al., ICML 2025)](https://www.alignmentforum.org/posts/zbebxYCqsryPALh8C/matryoshka-sparse-autoencoders)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [ATM: Adaptive Temporal Masking (Li et al., ICLR 2025 Workshop)](https://arxiv.org/abs/2510.08855)
- [Feature Hedging (Chanin & Dulka, 2025)](https://arxiv.org/abs/2505.11756)
- [Sparse but Wrong (Chanin & Garriga-Alonso, 2025)](https://arxiv.org/abs/2508.16560)
- [SynthSAEBench (2026)](https://arxiv.org/abs/2602.14687)
- [SAE Survey (Shu et al., EMNLP 2025)](https://arxiv.org/abs/2503.05613)
- [DeepMind SAE Negative Results (2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Feature Sensitivity (Tian et al., 2025)](https://arxiv.org/abs/2509.23717)
- [Understanding SAE Scaling with Feature Manifolds (2025)](https://arxiv.org/abs/2509.02565)
- [Transcoders Beat SAEs (Paulo et al., 2025)](https://arxiv.org/abs/2501.18823)
- [KronSAE (2025)](https://arxiv.org/abs/2505.22255)
- [Neuronpedia SAEBench Interactive](https://www.neuronpedia.org/sae-bench/info)
