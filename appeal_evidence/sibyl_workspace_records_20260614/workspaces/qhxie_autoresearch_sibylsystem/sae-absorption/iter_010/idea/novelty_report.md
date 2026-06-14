# Novelty Report -- Iteration 10

**Date**: 2026-04-15
**Candidate**: `cand_crossdomain_causal_tax` (front-runner)
**Title**: The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains

---

## Executive Summary

The front-runner candidate retains **high novelty (score 7/10)** for its primary contribution: the first systematic cross-domain characterization of feature absorption rates beyond the first-letter spelling task. No published work as of April 2026 compares absorption rates across different hierarchy types (syntactic vs. semantic), and no work uses RAVEL entity-attribute hierarchies (city-country, city-continent, city-language) to measure absorption. However, novelty has **slightly narrowed since earlier iterations** due to accumulating related work: SAEBench now includes both an absorption metric and a RAVEL disentanglement metric (though measured separately), and several recent papers (HSAE, H-SAE, OrtSAE, ATM) use the first-letter absorption benchmark in their evaluations. The universal causal mechanism via activation patching remains strongly novel (8/10). The hedging decomposition has partial overlap with Chanin et al.'s new "Feature Hedging" paper (2505.11756) but the strict/loose/compensatory three-way classification is novel. Negative results (GAS, CMI, Absorption Tax) are valuable but not individually novel contributions.

---

## Per-Candidate Novelty Analysis

### Candidate: `cand_crossdomain_causal_tax` (Front-Runner)

#### Contribution 1: Cross-Domain Absorption Characterization

**Core claim**: Absorption rates vary significantly across hierarchy types, with the first-letter spelling task showing atypically LOW absorption compared to semantic hierarchies at L24.

**Prior art search results**:

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025 Oral)**
   - Overlap: Defines feature absorption, proposes absorption rate metric, validates on hundreds of SAEs across Gemma 2 2B, Qwen2, Llama 3.2
   - Severity: **related_work** -- This is the foundational work that the proposal extends. It exclusively uses the first-letter spelling task. The paper explicitly acknowledges that the first-letter task provides a convenient ground-truth setting but does not claim generalization to other domains. The limitation is well-known in the community.
   - Differentiation: The proposal measures absorption on 3 additional RAVEL hierarchy types and directly compares rates, finding that first-letter is NOT worst-case. This is the core novelty.

2. **SAEBench (Karvonen et al., 2025, arXiv:2503.09532, ICML 2025)**
   - Overlap: Includes BOTH a feature absorption metric AND a RAVEL disentanglement metric in the same benchmark suite
   - Severity: **partial_overlap** -- SAEBench measures absorption using the first-letter task exclusively. RAVEL is used as a separate "feature disentanglement" metric, not as an absorption measurement. The two metrics are never combined to compare absorption rates across hierarchy types. However, all the data and infrastructure exist in SAEBench to do so.
   - Differentiation: The proposal's key insight is to repurpose RAVEL's entity-attribute hierarchies as absorption measurement targets, enabling cross-domain comparison. SAEBench treats absorption and RAVEL as independent metrics evaluating different SAE capabilities. Nobody has connected these two evaluations.
   - Risk: A SAEBench follow-up paper could potentially extend the absorption metric to RAVEL hierarchies. The proposal should cite SAEBench and acknowledge that the infrastructure exists.

3. **H-SAE (Muchane et al., 2025, arXiv:2506.01197)**
   - Overlap: Proposes hierarchical SAE architecture; evaluates on first-letter absorption benchmark from SAEBench; discusses absorption in context of semantic hierarchies
   - Severity: **related_work** -- Evaluates absorption only on the first-letter task. The paper is an architecture contribution, not a characterization study. It does not compare absorption rates across different hierarchy types.
   - Differentiation: The proposal is a characterization study across hierarchies; H-SAE is an architectural mitigation.

4. **HSAE / "From Atoms to Trees" (2026, arXiv:2602.11881)**
   - Overlap: Hierarchical SAE with tree structure; evaluates on SAEBench absorption metric; shows HSAE reduces absorption; tests across layers and models (Gemma 2 2B, Qwen3 4B)
   - Severity: **related_work** -- Uses SAEBench's first-letter absorption metric. Cross-model generalizability is tested but NOT cross-domain absorption measurement. The paper focuses on whether HSAE's hierarchy mitigates absorption, not whether absorption varies by hierarchy type.
   - Differentiation: The proposal asks "does absorption severity depend on the type of hierarchy?" not "does this architecture reduce absorption?"

5. **Matryoshka SAE (Bussmann et al., 2025, arXiv:2503.17547, ICML 2025)**
   - Overlap: Reduces absorption via nested hierarchical training; evaluated on SAEBench absorption metric
   - Severity: **related_work** -- Architecture contribution evaluated on first-letter absorption only.

6. **OrtSAE (Korznikov et al., 2025, arXiv:2509.22033)**
   - Overlap: Reduces absorption ~70% via orthogonality; evaluated on SAEBench absorption
   - Severity: **related_work** -- Architecture contribution. Absorption evaluated on first-letter task only.

7. **ATM (Li et al., 2025, arXiv:2510.08855)**
   - Overlap: Achieves lowest absorption scores (0.0068); uses first-letter classification benchmark
   - Severity: **related_work** -- Training procedure contribution. First-letter task only.

8. **Feature Hedging (Chanin et al., 2025, arXiv:2505.11756)**
   - Overlap: Identifies "feature hedging" as complementary failure mode to absorption; studies interaction between hedging and absorption; discusses strict vs. loose hierarchy
   - Severity: **partial_overlap** -- The strict/loose hierarchy distinction is conceptually related to the proposal's strict/loose hedging classification. However, Chanin's "strict vs. loose" refers to whether parent-child feature relationships are deterministic or merely correlated, NOT to the classification of hedging instances into strict/compensatory/persistent categories.
   - Differentiation: The proposal's three-way hedging decomposition (strict/compensatory/persistent) with the finding that the widely-cited hedging classification is near-tautological (0-22.6% strict vs. 92.6% loose) is novel. The proposal redefines what hedging means operationally.

9. **SynthSAEBench (2026, arXiv:2602.14687)**
   - Overlap: Provides synthetic benchmarks with configurable hierarchical dependencies to study absorption; includes controlled hierarchy structure
   - Severity: **related_work** -- Synthetic setting with artificial hierarchies, not real LLM knowledge hierarchies. Does not compare absorption rates across different real-world hierarchy types.

10. **"Sparse but Wrong" (Chanin & Garriga-Alonso, 2025, arXiv:2508.16560)**
    - Overlap: Shows incorrect L0 causes feature mixing/hedging, related to absorption dynamics
    - Severity: **related_work** -- Focuses on L0 selection, not cross-domain absorption characterization.

11. **"Looking for Feature Absorption Automatically" (LessWrong, 2025)**
    - Overlap: Proposes automatic detection of absorption via encoder-decoder discrepancy and causal effect similarity; reports that the method does not work well in practice
    - Severity: **related_work** -- Related to the proposal's GAS negative result. Both attempt unsupervised absorption detection and report failure. However, the methods are different (encoder-decoder discrepancy vs. decoder-activation geometric score).

12. **Masked Regularization for SAE Robustness (2026, arXiv:2604.06495)**
    - Overlap: Proposes masking-based regularization to reduce absorption; very recent (April 2026)
    - Severity: **related_work** -- Architecture/training mitigation, not characterization. Uses standard absorption metric.

**Assessment**: No published work measures absorption rates across different hierarchy types. The entire literature evaluates absorption exclusively on the first-letter spelling task. The proposal's finding that first-letter is atypically LOW (not worst-case) would reframe the field's understanding. The closest risk is that SAEBench already contains both ingredients (absorption metric + RAVEL data) and someone could combine them, but nobody has.

**Novelty score: 7/10** (down from 9/10 in earlier iterations due to the density of related work and the fact that SAEBench provides the infrastructure to do this comparison)

#### Contribution 2: Universal Causal Mechanism via Activation Patching

**Core claim**: Zeroing child features recovers parent probe predictions across ALL hierarchy types (d=0.75-1.50), confirming a universal competitive exclusion mechanism.

**Prior art search results**:

1. **Chanin et al. (2024)** -- Uses ablation studies to verify causal effects of absorbing latents, but only on the first-letter task. Their causal validation is part of the absorption metric definition.
   - Severity: **partial_overlap** -- They use ablation (zero-out) as a validation tool within the first-letter domain. The proposal extends this to multiple hierarchy types and frames it as "competitive exclusion."
   - Differentiation: (a) Cross-domain application of the patching methodology, (b) the "competitive exclusion" framing as a universal mechanism, (c) effect size comparison across domains (d=1.33 vs 1.50 vs 0.75).

2. **"Causal Interpretation of SAE Features in Vision" (CaFE, 2025)**
   - Overlap: Uses causal methods (effective receptive fields) to validate SAE feature interpretations in vision transformers
   - Severity: **related_work** -- Different domain (vision, not language); different method (ERF-based, not activation patching); not about absorption.

3. **General activation patching literature** (e.g., Meng et al. ROME, Conmy et al. ACDC)
   - Overlap: Activation patching is a well-established technique
   - Severity: **related_work** -- The technique is standard; the application to absorption across hierarchy types is novel.

**Assessment**: The causal activation patching methodology is standard, but applying it specifically to confirm the universality of feature absorption across hierarchy types is novel. The "competitive exclusion" framing has no precedent in the SAE literature. The cross-domain effect size comparison is genuinely new.

**Novelty score: 8/10**

#### Contribution 3: Tightened Hedging Classification

**Core claim**: The widely-cited hedging rate (~92.6% loose) is near-tautological. Strict classification yields only 0-22.6%.

**Prior art search results**:

1. **Feature Hedging (Chanin et al., 2505.11756)**
   - Overlap: Directly studies feature hedging as a failure mode; provides theoretical and empirical analysis
   - Severity: **partial_overlap** -- Chanin et al. define hedging as merging correlated features in narrow SAEs. The proposal's "hedging classification" refers to whether non-absorbed tokens are hedged (using the Chanin et al. 2024 definition from "A is for Absorption"), not the newer 2025 hedging concept. The terminological overlap is confusing but the underlying measurements are different.
   - Differentiation: The proposal decomposes the original absorption-paper hedging metric into strict/compensatory/persistent categories. This three-way decomposition and the finding that the original metric is near-tautological is novel.

2. **SAEBench** -- Does not decompose hedging.

**Assessment**: The three-way hedging decomposition is a methodological contribution that has no direct precedent. However, it is a refinement of the original Chanin et al. metric, not a fundamentally new concept.

**Novelty score: 7/10**

#### Contribution 4: Decoder Direction Magnitude Analysis

**Core claim**: Child decoders carry large-magnitude parent information (3.98 nats vs. 0.12 control).

**Prior art search results**:

1. **OrtSAE** -- Measures MeanCosSim between decoder vectors; penalizes high cosine similarity to reduce absorption. The proposal's decoder direction magnitude analysis is conceptually related.
   - Severity: **partial_overlap** -- OrtSAE measures cosine similarity between decoder vectors as an orthogonality metric. The proposal measures the logit impact of the parent direction encoded in child decoders. Different metrics, related concept.

2. **Chanin et al. (2024)** -- Discusses encoder-decoder discrepancy as characteristic of absorption.
   - Severity: **related_work** -- Qualitative observation, not quantified in the same way.

**Assessment**: The specific measurement (logit change from ablating parent direction in child decoder) is novel, but the acknowledged circularity limits its impact.

**Novelty score: 6/10** (circularity acknowledged; reframed from original 7/10)

#### Contribution 5: Negative Results (GAS, CMI, Absorption Tax)

**Core claim**: Multiple unsupervised detection approaches failed.

**Prior art search results**:

1. **"Looking for Feature Absorption Automatically" (LessWrong)** -- Also reports failure of automatic detection (encoder-decoder discrepancy method)
   - Severity: **partial_overlap** -- Different method, same conclusion (unsupervised detection is hard).

2. **Measuring SAE Feature Sensitivity (arXiv:2509.23717)** -- Frames absorption as a special case of poor feature sensitivity
   - Severity: **related_work** -- Different framing of related problem.

**Assessment**: Negative results are valuable for the field but individually not high-novelty contributions. The convergence with the LessWrong post's negative finding strengthens the case that unsupervised detection is fundamentally hard.

**Novelty score: 6/10** (valuable but not surprising given multiple convergent negative findings)

#### Contribution 6: Probe Degradation Ablation (H10, NEW)

**Core claim**: Cross-domain absorption variation is genuine hierarchy effect, not probe quality artifact.

**Prior art search results**: No prior work systematically tests whether probe quality confounds absorption rate measurements.

**Assessment**: This is a novel methodological contribution. If it confirms the hierarchy effect, it substantially strengthens Contribution 1. If it reveals a probe artifact, it is an important methodological caution for the entire absorption measurement literature.

**Novelty score: 8/10** (genuinely new experimental design addressing an unexamined confound)

---

## Backup Candidates

### `cand_controlled_dictionary` -- Controlled Dictionary Experiment
**Novelty: 8/10** -- No prior work holds the dictionary constant while varying the encoder. MP-SAE (Matching Pursuit SAE) is related but does not perform this controlled comparison.

### `cand_ecological_phase_transition` -- Feature Absorption as Competitive Exclusion
**Novelty: 8/10** -- Zero prior work on competitive exclusion or phase transitions in SAE absorption. The Lotka-Volterra framing is entirely new. However, "A Unified Theory of Sparse Dictionary Learning" (arXiv:2512.05534) provides a different theoretical framework that partially addresses the same question.

### `cand_absorption_aware_correction` -- Absorption-Aware Post-Hoc Feature Correction
**Novelty: 7/10** -- All existing mitigations are training-time. Inference-time correction is novel. Risk: the "Masked Regularization" paper (arXiv:2604.06495) explores related territory of post-hoc intervention.

### `cand_within_hierarchy_variation` -- Within-Hierarchy Variation
**Novelty: 6/10** -- Per-class variation is a natural extension of cross-domain analysis. No prior work examines this specifically, but the analysis is relatively straightforward given existing data.

---

## Key Prior Work Citations

| Paper | Year | Relevance | Severity |
|-------|------|-----------|----------|
| Chanin et al., "A is for Absorption" (arXiv:2409.14507) | 2024 | Foundational; first-letter task only | **related_work** |
| SAEBench (arXiv:2503.09532) | 2025 | Contains absorption + RAVEL metrics separately | **partial_overlap** |
| Feature Hedging (arXiv:2505.11756) | 2025 | Hedging analysis; strict/loose hierarchy concept | **partial_overlap** |
| Matryoshka SAE (arXiv:2503.17547) | 2025 | Architecture mitigation; first-letter eval only | **related_work** |
| OrtSAE (arXiv:2509.22033) | 2025 | Orthogonality-based mitigation; cosine similarity | **related_work** |
| ATM (arXiv:2510.08855) | 2025 | Best absorption scores; first-letter only | **related_work** |
| H-SAE (arXiv:2506.01197) | 2025 | Hierarchical architecture; first-letter eval | **related_work** |
| HSAE / Atoms to Trees (arXiv:2602.11881) | 2026 | Tree-based SAE; SAEBench absorption eval | **related_work** |
| SynthSAEBench (arXiv:2602.14687) | 2026 | Synthetic hierarchical benchmarks | **related_work** |
| Sparse but Wrong (arXiv:2508.16560) | 2025 | L0 effects on feature quality | **related_work** |
| Masked Regularization (arXiv:2604.06495) | 2026 | Training-time absorption mitigation | **related_work** |
| Toy Models of Feature Absorption (LessWrong) | 2024 | Co-occurrence causes absorption | **related_work** |
| Looking for Absorption Automatically (LessWrong) | 2025 | Failed unsupervised detection | **partial_overlap** |
| Broken Latents (LessWrong) | 2024 | Tied SAE analysis; narrow SAE absorption | **related_work** |
| Unified Theory SDL (arXiv:2512.05534) | 2024 | Theoretical framework; feature anchoring | **related_work** |

---

## Recommendations

### Front-runner (`cand_crossdomain_causal_tax`): **PROCEED** with caveats

1. **Overall novelty remains sufficient for NeurIPS/ICLR.** No published work compares absorption rates across hierarchy types. The finding that first-letter is atypical would be impactful.

2. **SAEBench partial overlap is the biggest risk.** The paper must clearly articulate WHY connecting the existing absorption metric with RAVEL hierarchies is non-trivial and what the findings reveal. Frame as: "SAEBench measures absorption on first-letter and disentanglement on RAVEL independently. We show that using RAVEL hierarchies AS absorption measurement targets reveals that absorption severity is hierarchy-dependent -- a finding invisible to the current benchmark design."

3. **Probe degradation ablation (H10) is the most important remaining experiment.** If it confirms the hierarchy effect, novelty for Contribution 1 is solidified. If it reveals a probe artifact, the paper becomes a methodological caution paper (still publishable but different framing).

4. **Must cite and differentiate from all partial overlaps.** Especially:
   - SAEBench (both metrics exist but are never combined)
   - Feature Hedging paper (different "strict/loose" concept)
   - "Looking for Absorption Automatically" (convergent negative finding for unsupervised detection)

5. **Reframe headline claims conservatively.** The iter-9 review's recommendation to scope claims remains critical. "4x descriptive range" is fine; "first cross-domain characterization" is supportable; "universal mechanism" needs the probe degradation ablation to be bulletproof.

6. **Execute promptly.** The field is active. The probability of a competing cross-domain characterization paper increases with time, especially given that SAEBench makes this analysis straightforward for any motivated researcher.

---

## Overall Novelty Assessment

**Overall novelty: HIGH** -- All candidates score >= 7/10 on at least one contribution. The front-runner has no exact-match collision. The partial overlaps (SAEBench, Feature Hedging) are well-differentiated. The primary risk is not prior art but rather the possibility that someone else connects the existing SAEBench ingredients before this paper is submitted.
