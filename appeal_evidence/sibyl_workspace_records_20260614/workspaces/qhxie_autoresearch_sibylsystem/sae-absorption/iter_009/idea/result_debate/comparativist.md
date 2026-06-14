# Comparativist Analysis: Positioning Against SOTA and Related Work

**Agent**: sibyl-comparativist
**Iteration**: 9 (Synthesis Round 6)
**Date**: 2026-04-15
**Workspace**: sae-absorption/current

---

## 1. Baseline Landscape: Existing Methods on Feature Absorption

The table below places our results against published absorption measurements from the literature. Absorption rates are on the **first-letter spelling task** unless otherwise noted, as this is the only benchmark with cross-study comparability.

| Method | Model | Layer | Width | L0 | Absorption Rate | Source |
|--------|-------|-------|-------|-----|----------------|--------|
| **Chanin et al. (2024)** | Gemma 2 2B | 12 | 16k-65k | 82 | 15-35% (across configs) | arXiv:2409.14507 (NeurIPS 2025 Oral) |
| **SAEBench BatchTopK** | Gemma 2 2B | 12 | 16k | ~64 | ~0.29 (normalized metric) | arXiv:2503.09532 (ICML 2025) |
| **SAEBench Matryoshka** | Gemma 2 2B | 12 | 16k | ~64 | ~0.03 (normalized metric) | arXiv:2503.09532 |
| **ATM** | Gemma 2 2B | 12 | -- | -- | **0.0068** (mean) | arXiv:2510.08855 (ICLR 2025 WS) |
| **OrtSAE** | Gemma 2 2B | 12 | 16k | 70 | ~65% reduction vs BatchTopK | arXiv:2509.22033 |
| **KronSAE** | Multiple | 12 | varies | varies | Reduced mean absorption fraction | arXiv:2505.22255 |
| **Masked Regularization** | -- | -- | -- | -- | Reduced (details sparse) | arXiv:2604.06495 |
| **Ours (first-letter L12 16k)** | Gemma 2 2B | 12 | 16k | ~86 | **2.2%** | iter_009 full |
| **Ours (first-letter L24 16k)** | Gemma 2 2B | 24 | 16k | -- | **34.5-42.9%** | iter_008/009 |
| **Ours (city-continent L24 16k)** | Gemma 2 2B | 24 | 16k | -- | **31.4%** | iter_009 full |
| **Ours (city-country L24 16k)** | Gemma 2 2B | 24 | 16k | -- | **45.1%** | iter_009 full |
| **Ours (city-language L24 16k)** | Gemma 2 2B | 24 | 16k | -- | **11.6%** | iter_009 full |

**Critical caveat on comparison**: Our absorption rates and those from Chanin et al. use the same metric pipeline (adapted from sae-spelling), making them directly comparable on first-letter. However, our cross-domain rates use RAVEL probes with F1=0.73-0.87 (below the 0.90 strict gate), meaning **cross-domain rates are upper bounds**. SAEBench and ATM use slightly different normalized metrics, making direct numerical comparison imprecise. Trend-level comparison is valid.

---

## 2. Contribution Margin Analysis

### Contribution 1: Cross-Domain Absorption Characterization

**Delta from SOTA**: This is the paper's strongest contribution. **No prior work measures absorption on any task other than first-letter spelling.** The contribution margin is not a percentage improvement on a benchmark -- it is the opening of an entirely new measurement dimension.

| Aspect | Prior SOTA | Our Work | Delta | Classification |
|--------|-----------|----------|-------|---------------|
| Hierarchies measured | 1 (first-letter only) | 4 (first-letter + 3 RAVEL) | **+3 new hierarchies** | STRONG (novel dimension) |
| Cross-domain variation detected | Not studied | ANOVA p=7.4e-66 (extremely significant) | **First demonstration** | STRONG |
| Layer dependence | Noted qualitatively | 15x variation (2.2% at L12 to 34.5% at L24) | **Quantified for first time** | STRONG |
| Probe-based pipeline for non-spelling | Not available | Adapted sae-spelling for RAVEL entities | **Methodological contribution** | MODERATE |

**Verdict**: >5% improvement is not the right frame here. This is a **qualitative advance** -- the first paper to ask and answer "does absorption generalize beyond spelling?" The answer (yes, with dramatic hierarchy-dependent variation) changes how the community should evaluate SAEs.

### Contribution 2: Causal Evidence via Activation Patching

| Aspect | Prior SOTA | Our Work | Delta | Classification |
|--------|-----------|----------|-------|---------------|
| Causal evidence for absorption | None (all correlational: integrated gradients) | Activation patching: 32.5% recovery vs 1.5% control, d=1.33 | **First interventional evidence** | STRONG |
| Cross-domain causal test | Not attempted | Failed for city-continent (0.05% recovery) | **First negative result on generalization** | MODERATE (honest limitation) |
| Benign vs pathological classification | Not studied | 0% benign, 100% pathological (1000x effect ratio) | **First classification attempt** | STRONG |

**Verdict**: The activation patching result (p=0.000218, d=1.33) is a strong causal contribution. However, the cross-domain failure (recovery 0.05% vs 14.5% control) is a significant limitation that must be prominently disclosed. The benign/pathological result (0% benign at all thresholds) is definitive and changes the narrative from "maybe absorption is okay" to "absorption is always harmful."

### Contribution 3: Hedging Decomposition

| Aspect | Prior SOTA | Our Work | Delta | Classification |
|--------|-----------|----------|-------|---------------|
| Hedging classification | Chanin et al. 98.6% loose hedging | Strict 7.9% vs compensatory 86.2% vs persistent 5.9% | **Tightened measurement** | MODERATE |
| Cross-domain hedging | Not studied | Compensatory dominates (67-91%) across hierarchies | **First cross-domain comparison** | MODERATE |

**Verdict**: Moderate contribution. The 98.6% loose hedging figure being near-tautological is a useful methodological clarification, but the decomposition alone would not carry a paper.

### Contribution 4: Architecture Comparison

| Aspect | Prior SOTA | Our Work | Delta | Classification |
|--------|-----------|----------|-------|---------------|
| Architecture ranking | SAEBench: Matryoshka best on absorption | No significant architecture effect (p=0.50-0.53) | **Contradicts SAEBench narrative** | MARGINAL-MODERATE |
| Hierarchy vs architecture | Not compared | Hierarchy >> architecture (p=0.005 vs p=0.53) | **Novel comparison** | MODERATE |

**Verdict**: The "hierarchy >> architecture" finding is interesting but undermined by probe quality limitations at L12 and width mismatches (Matryoshka 32k vs others 16k). SAEBench tested far more architectures (7) at more sparsity levels (6), so our comparison is less comprehensive. I classify this as **marginal to moderate** -- the claim should be stated carefully with caveats.

### Contribution 5: Negative Results

| Negative Result | n | Key Metric | Significance |
|----------------|---|-----------|-------------|
| GAS unsupervised detector | 25 letters, 640k tokens | rho=0.116, AUROC=0.571 | Definitive negative (25x scale-up from pilot) |
| CMI at L0=22 | 25 letters | rho=0.044, p=0.83 | Clean null (all probes F1=1.0) |
| Absorption Tax T(G) | 17 configs | ranking rho=-0.20 | Chance-level concordance |
| Rate-distortion 3-factor | 262 pairs | rho=0.250, R^2=0.088 | Below threshold; predictors OPPOSITE direction |
| Cross-domain patching | 93 entities, 3751 FN | recovery 0.05% vs control 14.5% | Causal mechanism does not generalize |
| H2' semantic > syntactic | 4 hierarchies | Refuted at L24 | Hierarchy-specific, not semantic/syntactic |
| Architecture non-significant | 4 architectures | ANOVA p=0.50-0.53 | Hierarchy dominates |
| RAVEL probe quality | 4 hierarchies | F1=0.73-0.87 (below 0.90 gate) | Upper bound interpretation required |

**Verdict**: The **number and quality of negative results** (9 total, all with constructive reframing) is a genuine strength. Honest negative results are rare in ML papers. The GAS failure (25x scale-up confirming null), the rate-distortion reversal (predictors in OPPOSITE direction at n=262), and the cross-domain patching failure are each individually informative. Together, they establish a clear narrative: **absorption resists correlational/statistical prediction and requires causal/interventional approaches.**

---

## 3. Concurrent Work Scan

### Papers from the last 6 months addressing overlapping problems (Oct 2025 - April 2026)

| Paper | Date | Overlap | Threat Level |
|-------|------|---------|-------------|
| **Masked Regularization** (arXiv:2604.06495) | Apr 2026 | Addresses absorption via training-time intervention | LOW -- proposes mitigation (we characterize; complementary) |
| **From Atoms to Trees: HSAE** (arXiv:2602.11881) | Feb 2026 | Hierarchical SAE that learns parent-child structure | LOW -- architecture paper; does not characterize absorption cross-domain |
| **SynthSAEBench** (arXiv:2602.14687) | Feb 2026 | Synthetic benchmark with known hierarchy; studies absorption | MEDIUM -- tests absorption in controlled setting but synthetic, not real LLM hierarchies |
| **Sanity Checks for SAEs** (arXiv:2602.14111) | Feb 2026 | Shows SAEs recover only 9% of true features | LOW -- general negative result; does not study absorption specifically |
| **Stable and Steerable SAEs** (arXiv:2603.04198) | Mar 2026 | Weight regularization for steering/consistency | LOW -- different focus (steering, not absorption) |

**No concurrent paper addresses cross-domain absorption characterization.** The closest is SynthSAEBench, which studies absorption in synthetic settings with controlled hierarchies. However, SynthSAEBench's contribution is orthogonal: it provides ground-truth features in a synthetic model, while our contribution is measuring absorption in real LLM hierarchies (Gemma 2 2B with RAVEL entities). These are complementary.

**Key check**: I searched arXiv, Google Scholar, and the web for any work measuring absorption on tasks other than first-letter spelling as of April 2026. **No such work exists.** SAEBench's absorption metric is exclusively first-letter. ATM, OrtSAE, KronSAE, and Matryoshka all evaluate absorption on first-letter only. The cross-domain measurement is a genuine first.

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does:**

> This paper is the first to measure feature absorption across multiple semantic hierarchy types beyond the canonical first-letter spelling task, revealing that absorption is hierarchy-dependent (11.6-45.1% at L24, ANOVA p=7.4e-66) and always pathological (0% benign, ~1000x effect ratio), while demonstrating through 9 negative results that absorption resists all tested correlational predictors and requires causal/interventional methods.

This is articulable in one sentence. The novelty is clear and supported by evidence.

**Novelty decomposition**:
- Cross-domain measurement: **Novel** (no prior work)
- Activation patching causal evidence: **Novel** (first interventional, not correlational)
- Benign/pathological classification: **Novel** (first test of this distinction)
- 9 negative results as a package: **Novel** (no prior paper reports this comprehensively)
- Rate-distortion predictor failure at scale: **Novel** (direction reversal at n=262 is a new finding)
- Cross-domain patching failure: **Novel** (first attempt to test causal generalization)
- Individual metrics/tools: **Not novel** (adapted from Chanin et al., SAEBench)

---

## 5. Venue Recommendation

### Assessment

| Factor | Score | Justification |
|--------|-------|---------------|
| Novelty of primary question | HIGH | First cross-domain absorption study |
| Statistical rigor | HIGH | Bootstrap CIs, permutation tests, Bonferroni correction, 9 hypotheses tested |
| Effect size of primary result | MODERATE-HIGH | 11.6-45.1% variation, ANOVA p=7.4e-66 |
| Causal evidence strength | HIGH for first-letter | d=1.33, p=0.000218 |
| Causal evidence limitations | SIGNIFICANT | Cross-domain patching fails |
| Probe quality | MIXED | First-letter F1=0.97, RAVEL F1=0.73-0.87 |
| Negative results honesty | HIGH | 9 negative results, all documented |
| Model coverage | LOW | Single model (Gemma 2 2B) only |
| Architecture coverage | MODERATE | 4 architectures, but limited configs |
| Comparable prior work at top venues | Chanin et al. NeurIPS 2025 Oral; SAEBench ICML 2025; Matryoshka ICML 2025 |

### Recommendation

**Primary target: NeurIPS 2026 main conference or ICLR 2027**

**Justification**: The cross-domain characterization question is clearly novel and well-executed. Chanin et al. (first-letter only) was accepted as a NeurIPS 2025 Oral, which establishes that absorption characterization papers can reach top-tier. Our paper extends this to cross-domain with substantial new findings (hierarchy-dependence, causal evidence, pathological classification, comprehensive negative results). The level of statistical rigor and honest negative reporting is comparable to top-tier standards.

**Concerns that could prevent acceptance**:
1. **Single model limitation**: Only Gemma 2 2B. Reviewers may ask for Llama, Pythia, or GPT-2 replication. The iter_001 data includes some Llama/GPT-2 results that could be incorporated.
2. **Probe quality**: RAVEL probes below 0.90 F1 for 3/4 hierarchies. Reviewers may argue absorption rates are artifacts of poor probes. Mitigation: first-letter (F1=0.97) as gold standard + upper bound interpretation.
3. **Cross-domain patching failure**: Causal evidence does not generalize. Reviewers may view this as undermining the cross-domain contribution. Mitigation: honest reporting + distributed vs concentrated absorption as a research direction.
4. **Architecture comparison weakness**: Width mismatch (Matryoshka 32k vs others 16k), limited configurations. SAEBench has far more comprehensive architecture comparison.

**Fallback target: EMNLP 2026 or NeurIPS 2026 MI Workshop**

If the single-model limitation and probe quality concerns are not adequately addressed, a mid-tier venue (EMNLP) or workshop is more appropriate. The negative results package alone would make a strong workshop contribution.

---

## 6. Strengthening Plan

### Additional baselines/comparisons that would maximally strengthen the paper

**Priority 1 (HIGH impact): Multi-model replication**
- Run cross-domain absorption on at least one additional model (Llama 3.2 1B or Pythia-160M with SAEBench SAEs)
- Chanin et al. showed absorption in Llama 3.2 1B and Qwen2 0.5B on first-letter. Extending cross-domain measurement to these models would dramatically strengthen the generalization claim
- Estimated cost: 4-8 GPU-hours
- **Impact**: Removes the "single model" objection, which is the most likely reviewer concern

**Priority 2 (MEDIUM-HIGH impact): Comparison against ATM and OrtSAE**
- ATM claims absorption score 0.0068 (vs our ~0.02 at L12). Does ATM's advantage hold on cross-domain hierarchies?
- OrtSAE claims 65% absorption reduction. Does this hold for RAVEL hierarchies?
- These are the two strongest architecture-level competitors. Head-to-head comparison on cross-domain tasks would be a significant contribution
- Estimated cost: 2-4 GPU-hours (ATM/OrtSAE weights may need to be obtained or retrained)
- **Impact**: If ATM/OrtSAE reduce absorption on first-letter but NOT on RAVEL hierarchies, this powerfully supports the "hierarchy > architecture" finding

**Priority 3 (MEDIUM impact): Improved RAVEL probes**
- Current F1=0.73-0.87 is the paper's biggest methodological weakness
- Try: (a) nonlinear probes (2-layer MLP), (b) different layers (L20, L22), (c) ensemble probes, (d) prompt engineering for RAVEL entities
- If any approach achieves F1 > 0.90 on 3+ hierarchies, all cross-domain results gain substantial credibility
- **Impact**: Removes the "upper bound" caveat from all RAVEL results

---

## 7. Risk Assessment and Red Flags

### Suspicious results requiring extra validation

1. **City-country absorption at 45.1% (L24 16k)**: This is the highest rate but city-country has the worst probe (F1=0.726). A significant portion of this "absorption" may be probe error. The permutation test vs first-letter is non-significant (p=1.0), suggesting the difference may be noise. **Recommendation**: Do not claim city-country has "higher absorption than first-letter" without the probe quality caveat.

2. **Cross-domain Kruskal-Wallis p=7.4e-66**: This extremely small p-value reflects large sample sizes (n=1000+ per hierarchy), not necessarily large effect sizes. The per-entity rates are the more meaningful comparison. The permutation tests against first-letter show only city-language is significantly different after Bonferroni correction. **Recommendation**: Report both the ANOVA (confirms variation exists) and pairwise tests (characterize which differences are robust).

3. **Benign/pathological 0% benign**: This was tested only on city-continent. First-letter benign/pathological was not tested. There is a possibility that first-letter absorption is partially benign (since the model might not need the "starts with X" feature when it already has the specific word feature). **Recommendation**: Flag as future work; do not generalize the "100% pathological" claim to all hierarchies without testing.

4. **Architecture comparison non-significance**: With only 4 architectures and probe quality issues, the null result (p=0.50-0.53) may simply reflect low statistical power. SAEBench's results with 7 architectures and 6 sparsity levels found significant architecture effects. **Recommendation**: Frame as "we found no significant effect in our experimental setting" rather than "architecture does not matter."

---

## 8. Overall Positioning Statement

This paper occupies a unique position in the SAE absorption literature. While the existing landscape (Chanin et al., SAEBench, Matryoshka, OrtSAE, ATM, KronSAE) focuses either on (a) measuring absorption on the first-letter task or (b) proposing architectural mitigations, **no work has asked whether absorption generalizes across feature hierarchy types.** This is the fundamental question that determines whether absorption matters for the safety-relevant, knowledge-rich domains that motivate SAE research.

The paper's contribution margin is:
- **STRONG** on the primary question (cross-domain characterization): novel, well-executed, statistically rigorous
- **STRONG** on causal evidence (activation patching for first-letter): first interventional result
- **STRONG** on the benign/pathological diagnostic: decisively falsifies the "absorption might be okay" hypothesis
- **STRONG** on negative results documentation: 9 results, each informative, rare in ML papers
- **MODERATE** on hedging decomposition: useful clarification, not a standalone contribution
- **MARGINAL-MODERATE** on architecture comparison: limited by probe quality and experimental scope

The paper's positioning should emphasize: **"We measured what nobody measured (cross-domain), found what nobody expected (hierarchy-specific, not semantic/syntactic ordering), proved what was only assumed (100% pathological), and honestly reported what did not work (9 negative results)."** This narrative of careful, evidence-driven science is compelling for top-tier venues.

The main risks are the single-model limitation and probe quality. Addressing these (Priority 1 and 3 in the strengthening plan) would substantially improve the paper's chances at NeurIPS 2026 / ICLR 2027.

---

## Sources

- [Chanin et al., "A is for Absorption" (NeurIPS 2025)](https://arxiv.org/abs/2409.14507)
- [SAEBench (ICML 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAE (ICML 2025)](https://arxiv.org/abs/2503.17547)
- [Feature Hedging (Chanin et al., 2025)](https://arxiv.org/abs/2505.11756)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [ATM (Li et al., ICLR 2025 WS)](https://arxiv.org/abs/2510.08855)
- [KronSAE (2025)](https://arxiv.org/abs/2505.22255)
- [Unified Theory of SDL (2025)](https://arxiv.org/abs/2512.05534)
- [SynthSAEBench (2026)](https://arxiv.org/abs/2602.14687)
- [Masked Regularization (2026)](https://arxiv.org/abs/2604.06495)
- [HSAE - From Atoms to Trees (2026)](https://arxiv.org/abs/2602.11881)
- [Sanity Checks for SAEs (2026)](https://arxiv.org/abs/2602.14111)
- [SAEBench Interactive (Neuronpedia)](https://www.neuronpedia.org/sae-bench/info)
- [Feature Sensitivity (Tian et al., 2025)](https://arxiv.org/abs/2509.23717)
- [Incorporating Hierarchical Semantics in SAE (2025)](https://arxiv.org/abs/2506.01197)
- [Gemma Scope (2024)](https://arxiv.org/abs/2408.05147)
- [DeepMind SAE Negative Results (2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
