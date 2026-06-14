# Novelty Report -- Iteration 11

**Date**: 2026-04-15
**Candidate**: `cand_crossdomain_causal_tax` (front-runner)
**Title**: The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains

---

## Executive Summary

The front-runner candidate retains **high novelty (score 7/10)** for its primary contribution: the first systematic cross-domain characterization of feature absorption rates beyond the first-letter spelling task. No published work as of April 15, 2026 compares absorption rates across different hierarchy types. The iter_011 restructuring -- promoting probe degradation methodology to Contribution #1 and universal causal mechanism to Contribution #2 -- strengthens the novelty profile, because probe quality as a confound in absorption measurement has **zero prior art** (novelty 8/10), and cross-domain activation patching for competitive exclusion is genuinely new (novelty 8/10).

Since the iter_010 novelty report (also dated April 15), the search landscape shows **no new competing cross-domain absorption paper**. Two new papers merit attention as related work but do not undermine novelty:

1. **"Stop Probing, Start Coding" (arXiv:2603.28744, March 2026)**: Shows SAE dictionary learning (not encoder amortization) is the bottleneck for compositional generalization. Relevant to the backup candidate `cand_controlled_dictionary` but does not overlap with the front-runner's cross-domain characterization.

2. **"Sparse Auto-Encoders and Holism about Large Language Models" (arXiv:2603.26207, March 2026)**: A philosophy-of-AI paper discussing whether SAE features genuinely decompose meaning. Cites absorption as evidence that SAE decomposition is unreliable. Related context but no empirical overlap.

The iter_010 novelty report identified 11 collisions; this iteration confirms all remain current with no new exact-match or partial-overlap additions. The field remains active on architectural mitigations (all using first-letter evaluation), not on cross-domain characterization.

---

## Per-Candidate Novelty Analysis

### Candidate: `cand_crossdomain_causal_tax` (Front-Runner)

#### Contribution 1 (Restructured as PRIMARY): Probe Degradation Methodology -- Novelty 8/10

**Core claim**: Probe quality is a major and previously unquantified confound in absorption measurement (R^2=0.777, Spearman rho=-1.0, p=0.009).

**Prior art search results**:

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025 Oral)**
   - Uses linear probes as the ground truth for absorption measurement but never systematically tests whether probe quality itself confounds the measurement.
   - Severity: **related_work**
   - Differentiation: The proposal shows that degrading probe F1 directly increases measured absorption rates with perfect monotonicity. This confound analysis has no precedent.

2. **SAEBench (Karvonen et al., 2025, arXiv:2503.09532, ICML 2025)**
   - Uses probe-based metrics (TPP, Sparse Probing) alongside absorption. Mentions probe quality in passing but does not test probe quality as a confound for absorption.
   - Severity: **related_work**
   - Differentiation: SAEBench assumes probe quality is stable. The proposal tests and refutes this assumption.

3. **"Are Sparse Autoencoders Useful?" (Heap et al., 2025, arXiv:2502.16681)**
   - Shows SAE probes underperform logistic regression baselines. Raises concerns about probe methodology broadly.
   - Severity: **related_work**
   - Differentiation: General critique of SAE probing, not specific to absorption measurement confounds.

4. **"Stop Probing, Start Coding" (Pacela et al., March 2026, arXiv:2603.28744)**
   - Shows linear probes and SAEs fail at compositional generalization. Isolates dictionary learning vs. encoder as failure source.
   - Severity: **related_work**
   - Differentiation: Studies compositional generalization failure, not absorption measurement confounds. Different phenomenon.

5. **"Measuring Sparse Autoencoder Feature Sensitivity" (Tian, 2025, arXiv:2509.23717)**
   - Frames absorption as a special case of poor feature sensitivity. Proposes scalable evaluation without probes.
   - Severity: **related_work**
   - Differentiation: Alternative evaluation methodology, not probe quality confound analysis.

**Assessment**: No published work systematically tests whether probe quality confounds absorption rate measurements. The three-way confound decomposition (probe-explained vs. residual vs. genuine hierarchy effect) is entirely novel. The perfect monotonicity finding (rho=-1.0) directly challenges every prior absorption measurement that did not control for probe quality.

**Novelty score: 8/10** -- Genuinely new experimental design addressing an unexamined confound. Strengthened by promotion to Contribution #1 in the restructured paper.

#### Contribution 2 (SECONDARY): Universal Causal Mechanism via Activation Patching -- Novelty 8/10

**Core claim**: Zeroing child features recovers parent probe predictions across ALL hierarchy types (d=0.75-1.50, all p<0.001), confirming a universal competitive exclusion mechanism.

**Prior art search results**:

1. **Chanin et al. (2024)** -- Uses ablation (zero-out) as part of the absorption metric definition, but only on the first-letter task. Their causal validation is within-domain, not cross-domain.
   - Severity: **partial_overlap**
   - Differentiation: (a) Cross-domain application (first-letter + city-continent + city-language), (b) "competitive exclusion" framing as a universal mechanism, (c) effect size comparison across domains (d=1.33 vs 1.50 vs 0.75).

2. **"Causal Interpretation of SAE Features in Vision" (CaFE, 2025, arXiv:2509.00749)**
   - Uses causal methods (ERF-based) for vision SAE interpretation. Different domain, different method.
   - Severity: **related_work**

3. **General activation patching literature** (Meng et al. ROME, Conmy et al. ACDC, Heimersheim 2024)
   - Activation patching is a well-established technique. The application to cross-domain absorption is novel.
   - Severity: **related_work**

4. **"Adversarial Activation Patching" (arXiv:2507.09406, 2025)**
   - Uses activation patching for detecting deception in safety-aligned transformers. Different application.
   - Severity: **related_work**

5. **BLOCK-EM (arXiv:2602.00767, 2026)**
   - Uses activation patching to localize emergent misalignment. Notes "competitive exclusion-like dynamics" in passing.
   - Severity: **related_work**
   - Differentiation: Brief mention of exclusion dynamics, no systematic study. The proposal provides the first quantitative evidence.

**Assessment**: The causal activation patching technique is standard, but applying it to confirm the universality of competitive exclusion across hierarchy types is novel. The "competitive exclusion" framing borrowed from ecology has no precedent in the SAE literature. Cross-domain effect size comparison (d=0.75 to d=1.50) is genuinely new.

**Novelty score: 8/10**

#### Contribution 3 (TERTIARY): Cross-Domain Absorption Characterization -- Novelty 7/10

**Core claim**: Absorption rates vary significantly across hierarchy types, with first-letter showing atypically moderate rates. Quality-gated range 2.7x (31.4/11.6).

**Prior art search results**:

1. **Chanin et al. (2024)** -- First-letter task exclusively. Foundation paper.
   - Severity: **related_work**

2. **SAEBench (2025, arXiv:2503.09532)**
   - Contains BOTH an absorption metric (first-letter) AND a RAVEL disentanglement metric (city-country/continent/language) in the same benchmark. The two metrics are never combined.
   - Severity: **partial_overlap**
   - Differentiation: SAEBench treats absorption and RAVEL as independent evaluations. Nobody has used RAVEL hierarchies as absorption measurement targets.

3. **Feature Hedging (Chanin et al., 2025, arXiv:2505.11756)**
   - Studies feature hedging as complementary failure mode. Discusses strict vs. loose hierarchy concept.
   - Severity: **partial_overlap**
   - Differentiation: Chanin's strict/loose refers to deterministic vs. correlated parent-child relationships. The proposal's three-way decomposition (strict/compensatory/persistent) is a different concept. The near-tautological finding (0-22.6% strict vs. 92.6% loose) is novel.

4. **Matryoshka SAE (Bussmann et al., 2025, arXiv:2503.17547, ICML 2025)** -- Architecture mitigation; first-letter eval only. Severity: **related_work**

5. **OrtSAE (Korznikov et al., 2025, arXiv:2509.22033)** -- Orthogonality-based mitigation; first-letter eval. Severity: **related_work**

6. **ATM (Li et al., 2025, arXiv:2510.08855)** -- Best absorption scores (0.0068); first-letter only. Severity: **related_work**

7. **H-SAE (Muchane et al., 2025, arXiv:2506.01197)** -- Hierarchical architecture; first-letter eval. Severity: **related_work**

8. **HSAE / Atoms to Trees (2026, arXiv:2602.11881)** -- Tree-based SAE; SAEBench absorption eval (first-letter); cross-model but not cross-domain. Severity: **related_work**

9. **SynthSAEBench (2026, arXiv:2602.14687)** -- Synthetic hierarchical benchmarks with configurable hierarchy structure. Severity: **related_work**. Differentiation: Artificial hierarchies, not real LLM knowledge hierarchies.

10. **KronSAE (2025, arXiv:2505.22255)** -- Kronecker factorization; reduced absorption; first-letter eval. Severity: **related_work**

11. **"Sparse but Wrong" (Chanin & Garriga-Alonso, 2025, arXiv:2508.16560)** -- L0 selection effects. Severity: **related_work**

12. **Masked Regularization (2026, arXiv:2604.06495)** -- Training-time absorption mitigation. Very recent (April 2026). Severity: **related_work**

13. **"Looking for Feature Absorption Automatically" (LessWrong, 2025)** -- Proposes encoder-decoder discrepancy for automatic detection; reports method doesn't work.
    - Severity: **partial_overlap**
    - Differentiation: Different method (encoder-decoder discrepancy vs. GAS/CMI). Convergent negative conclusion strengthens the proposal's negative results.

14. **"Resurrecting the Salmon: Domain-Specific SAEs" (2025, arXiv:2508.09363)** -- Notes that feature absorption is exacerbated in broad-domain SAEs and that domain-specific training mitigates it. Criticizes SAEBench first-letter evaluation as "not useful for evaluating domain-specific feature absorption."
    - Severity: **related_work**
    - Differentiation: Criticizes the same first-letter monoculture but proposes domain-specific training as solution, not cross-domain measurement.

15. **InverseScope (Luo et al., 2025, arXiv:2506.07406)** -- Benchmarks against SAEs on RAVEL dataset for attribute recovery. Shows InverseScope outperforms SAEs across all layers.
    - Severity: **related_work**
    - Differentiation: Uses RAVEL for disentanglement evaluation (not absorption measurement). Different task (activation inversion vs. absorption characterization).

16. **"A Unified Theory of Sparse Dictionary Learning" (Tang et al., 2025-2026, arXiv:2512.05534)** -- Theoretical framework for absorption via piecewise biconvex optimization. Proposes feature anchoring as solution.
    - Severity: **related_work**
    - Differentiation: Theoretical, not empirical cross-domain characterization.

17. **"Compute Optimal Inference and Provable Amortisation Gap" (O'Neill et al., 2024, arXiv:2411.13117)** -- Proves amortization gap in SAEs; decouples encoder and decoder.
    - Severity: **related_work**
    - Differentiation: Studies inference quality, not absorption measurement confounds.

**Assessment**: No published work measures absorption rates across different hierarchy types. The entire absorption evaluation literature uses the first-letter spelling task exclusively. Every architectural mitigation paper (Matryoshka, OrtSAE, ATM, HSAE, KronSAE, Masked Regularization) evaluates on first-letter only. The "Resurrecting the Salmon" paper explicitly criticizes this monoculture but proposes domain-specific training, not cross-domain measurement. SAEBench remains the closest risk -- it contains both ingredients -- but nobody has combined them.

**Novelty score: 7/10** (unchanged from iter_010; SAEBench partial overlap remains the main risk factor)

#### Negative Results (GAS, CMI, Absorption Tax) -- Novelty 6/10

Three unsupervised detection approaches failed:
- GAS: rho=0.12
- CMI: rho=0.044 (p=0.83)
- Absorption Tax quantitative: rho=0.08

The "Looking for Feature Absorption Automatically" LessWrong post reports convergent failure of a different unsupervised method (encoder-decoder discrepancy). This convergence strengthens the negative finding that unsupervised absorption detection is fundamentally difficult.

**Novelty score: 6/10** -- Valuable for the field but not individually novel; convergent with prior negative findings.

#### Tightened Hedging Classification -- Novelty 7/10

Three-way decomposition (strict/compensatory/persistent) with the finding that the widely-cited hedging classification is near-tautological (0-22.6% strict vs. 92.6% loose). The Feature Hedging paper (arXiv:2505.11756) uses different "strict/loose" concepts (deterministic vs. correlated parent-child relationships), creating terminological overlap but no methodological overlap.

**Novelty score: 7/10**

---

### Backup Candidates

#### `cand_controlled_dictionary` -- Controlled Dictionary Experiment -- Novelty 8/10

Hold decoder dictionary constant, vary encoder (feedforward vs. OMP vs. 2-pass). No prior work isolates encoder vs. dictionary contribution to absorption.

**New relevant work**:
- **"Stop Probing, Start Coding" (arXiv:2603.28744, March 2026)**: Identifies dictionary learning (not amortization) as the binding constraint for SAE compositional generalization. Uses FISTA on frozen SAE dictionaries.
  - Severity: **partial_overlap** (elevated from related_work in iter_010)
  - Differentiation: Studies compositional generalization failure, not absorption. Does not hold dictionary constant for absorption measurement. However, the finding that dictionary quality > encoder quality is related to the controlled dictionary hypothesis. Must cite and differentiate.

- **"Compute Optimal Inference and Provable Amortisation Gap" (arXiv:2411.13117)**: Decouples encoder and decoder for inference quality. Related decoupling methodology.
  - Severity: **related_work**

**Updated novelty score: 8/10** (unchanged; "Stop Probing" is related but different phenomenon)

#### `cand_ecological_phase_transition` -- Feature Absorption as Competitive Exclusion -- Novelty 8/10

Lotka-Volterra competitive exclusion formalization. Zero prior work on ecological competition theory applied to SAE absorption. "A Unified Theory of Sparse Dictionary Learning" (arXiv:2512.05534) uses different formalism.

**Novelty score: 8/10** (unchanged)

#### `cand_absorption_aware_correction` -- Post-Hoc Feature Correction -- Novelty 7/10

Inference-time correction of absorbed features. All existing mitigations are training-time.

**Novelty score: 7/10** (unchanged)

#### `cand_within_hierarchy_variation` -- Within-Hierarchy Variation -- Novelty 6/10

Per-class absorption variation within a single hierarchy. Natural extension; no specific prior work.

**Novelty score: 6/10** (unchanged)

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
| KronSAE (arXiv:2505.22255) | 2025 | Kronecker factorization; first-letter eval | **related_work** |
| HSAE / Atoms to Trees (arXiv:2602.11881) | 2026 | Tree-based SAE; SAEBench absorption eval | **related_work** |
| SynthSAEBench (arXiv:2602.14687) | 2026 | Synthetic hierarchical benchmarks | **related_work** |
| Sparse but Wrong (arXiv:2508.16560) | 2025 | L0 effects on feature quality | **related_work** |
| Masked Regularization (arXiv:2604.06495) | 2026 | Training-time absorption mitigation | **related_work** |
| Resurrecting the Salmon (arXiv:2508.09363) | 2025 | Domain-specific SAEs; criticizes first-letter eval | **related_work** |
| InverseScope (arXiv:2506.07406) | 2025 | RAVEL evaluation outperforming SAEs | **related_work** |
| Unified Theory SDL (arXiv:2512.05534) | 2025 | Theoretical framework; feature anchoring | **related_work** |
| Amortisation Gap (arXiv:2411.13117) | 2024 | Encoder-decoder decoupling | **related_work** |
| Stop Probing, Start Coding (arXiv:2603.28744) | 2026 | Dictionary learning as SAE bottleneck | **related_work** |
| SAE Holism (arXiv:2603.26207) | 2026 | Philosophy of SAE decomposition | **related_work** |
| Looking for Absorption Automatically (LessWrong) | 2025 | Failed unsupervised detection | **partial_overlap** |
| Toy Models of Feature Absorption (LessWrong) | 2024 | Co-occurrence causes absorption | **related_work** |
| Broken Latents (LessWrong) | 2024 | Tied SAE analysis; narrow SAE absorption | **related_work** |
| Are SAEs Useful? (arXiv:2502.16681) | 2025 | SAE probes underperform baselines | **related_work** |
| SAE Feature Sensitivity (arXiv:2509.23717) | 2025 | Absorption as poor feature sensitivity | **related_work** |
| Feature Consistency Position (arXiv:2505.20254) | 2025 | SAE reproducibility concerns | **related_work** |

---

## Changes from iter_010 Novelty Report

1. **No new exact-match collisions found.** The primary novelty claim (first cross-domain absorption characterization) remains supported as of April 15, 2026.

2. **Two new related_work papers added**: "Stop Probing, Start Coding" (arXiv:2603.28744) and "Sparse Auto-Encoders and Holism" (arXiv:2603.26207). Neither overlaps with the front-runner's core contributions.

3. **Contribution restructuring strengthens novelty profile.** Promoting probe degradation methodology (8/10) to Contribution #1 is well-justified -- no prior work exists on probe quality as an absorption confound. The cross-domain characterization (7/10), while still the central empirical question, is now properly contextualized by the probe quality confound analysis.

4. **SAEBench partial overlap remains the dominant risk.** SAEBench already contains both absorption and RAVEL metrics. Someone motivated could combine them. The proposal should cite this explicitly and execute promptly.

5. **"Resurrecting the Salmon" now additionally relevant.** This paper explicitly criticizes the first-letter monoculture, calling it "not useful for evaluating domain-specific feature absorption." This validates the proposal's central motivation but also suggests the community is aware of the gap. The paper proposes domain-specific training as the solution, not cross-domain measurement.

6. **Backup candidate `cand_controlled_dictionary` now has a closer neighbor.** "Stop Probing, Start Coding" identifies dictionary learning as the SAE bottleneck. While the research question is different (compositional generalization vs. absorption), the decoupling methodology is related. If the backup candidate is activated, it must cite and differentiate.

---

## Recommendations

### Front-runner (`cand_crossdomain_causal_tax`): **PROCEED** with high confidence

1. **Overall novelty remains sufficient for NeurIPS 2026 / ICLR 2027.** No competing cross-domain absorption paper exists. The restructured contributions (probe degradation methodology as #1, universal causal mechanism as #2) present two 8/10 novelty contributions plus a 7/10 tertiary contribution.

2. **SAEBench is the primary competitive risk.** The paper must explicitly state: "SAEBench measures absorption on first-letter and disentanglement on RAVEL independently. We show that using RAVEL hierarchies AS absorption measurement targets reveals hierarchy-dependent absorption severity -- a finding invisible to the current benchmark design."

3. **"Resurrecting the Salmon" validates the motivation.** Cite this paper as additional evidence that the first-letter monoculture is a recognized limitation, but note that their solution (domain-specific training) is orthogonal to the proposal's approach (cross-domain measurement).

4. **Execute Phase 0 (data integrity) immediately.** The iter_010 review identified blocking data issues. These must be resolved before submission. The novelty window remains open but time pressure is real.

5. **Time urgency is moderate.** No competing paper is imminent, but the gap between "ingredients exist in SAEBench" and "someone combines them" shrinks over time. The field has 6+ architectural mitigation papers, all using first-letter eval. The observation that this evaluation is inadequate is increasingly obvious.

---

## Overall Novelty Assessment

**Overall novelty: HIGH** -- All active candidates score >= 7/10 on at least one contribution. The front-runner has no exact-match collision. Partial overlaps (SAEBench, Feature Hedging, LessWrong negative results) are well-differentiated. The restructured contribution ordering (probe degradation 8/10, causal mechanism 8/10, cross-domain characterization 7/10) is stronger than the prior framing.
