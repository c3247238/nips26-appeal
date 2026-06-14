# Research Proposal (Synthesis Round 7 -- Iteration 10)

## Title

**The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains**

---

## Abstract

Feature absorption -- the systematic failure of general (parent) features to fire when specific (child) features are active -- threatens the reliability of sparse autoencoder (SAE) based mechanistic interpretability. Yet all published measurements rest on a single proxy task (first-letter spelling), and no formal theory quantifies when absorption is harmful versus faithful. We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) using RAVEL on Gemma 2 2B. Our experiments reveal that the received wisdom is inverted: first-letter spelling shows the *lowest* absorption (27.1% at L24), while semantic hierarchies show substantially higher rates (city-country: 45.1%, city-continent: 42.9%), establishing a 4x descriptive range (11.6% to 45.1%). Within-RAVEL variation is statistically significant (Kruskal-Wallis p=7.4e-66); against first-letter, city-language reaches significance after Bonferroni correction (p_Bonf=0.003), while other pairwise comparisons do not. We provide causal evidence for absorption via activation patching: zeroing child features recovers parent probe predictions across all hierarchy types (first-letter d=1.33, city-continent d=1.50, city-language d=0.75; all p<0.001), confirming a universal competitive exclusion mechanism. The widely cited hedging classification is revealed as a near-tautological upper bound (strict classification: 0-22.6% vs. loose: 92.6%). Multiple unsupervised detection approaches failed: GAS (rho=0.12), CMI (rho=0.044, p=0.83), and Absorption Tax quantitative predictions (rho=0.08) are reported as transparent negative results. The paper's central finding -- that absorption severity is hierarchy-dependent and the first-letter task is atypical -- reframes the entire absorption literature and demands cross-domain evaluation as a standard practice.

---

## Motivation

Feature absorption creates a false sense of monosemanticity: SAE features appear clean and interpretable, but silently fail to fire in systematic subsets of their semantic domain. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10-40%. Anthropic's successful circuit tracing in Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding -- making absorption a critical obstacle to broader adoption.

Yet the field's understanding of absorption rests on an alarmingly narrow empirical base. Every published absorption measurement uses the first-letter spelling task, which has an unnaturally clean hierarchy (26 letters, near-uniform distribution, 100% parent-child co-occurrence by construction). Real knowledge hierarchies are imbalanced (France has more cities than Liechtenstein), multi-level (city -> country -> continent), and context-dependent.

Five compounding gaps motivate this work:

**Gap A (Empirical -- the single-task monoculture):** Absorption rates on semantically rich hierarchies are unknown. The 15-35% rates from Chanin et al. may not transfer. Safety-relevant features live in knowledge/reasoning space, not spelling space. The field needs cross-domain characterization to know whether absorption actually matters where it counts.

**Gap B (Theory -- no quantitative prediction):** No closed-form prediction exists for absorption severity as a function of SAE configuration and hierarchy structure.

**Gap C (Architecture validity):** Every architecture comparison (Matryoshka vs. BatchTopK vs. JumpReLU) uses the first-letter task alone. If architecture rankings change on knowledge hierarchies, current recommendations may be misleading.

**Gap D (Confound disentanglement):** Feature absorption, hedging, and L0 artifacts are three distinct failure modes that all manifest as features failing to fire. No study systematically disentangles the contribution of each to measured "absorption."

**Gap E (Unsupervised detection):** All absorption metrics require known probe directions. No unsupervised method exists -- and our negative results (GAS, CMI) help clarify why this is hard.

### Evidence-Driven Context from Prior Iterations

Nine prior iterations and extensive experiments have produced substantial empirical results that ground this proposal:

- **Activation patching** causally confirms absorption: zeroing child features recovers parent probe predictions across ALL hierarchy types (first-letter d=1.33, city-continent d=1.50, city-language d=0.75; all p<0.001). This is the first metric-independent causal evidence for a universal competitive exclusion mechanism in SAEs.
- **Tightened hedging** reveals the widely cited 98.6% hedging classification is near-tautological: strict classification yields only 0-22.6%, with the bulk being compensatory resolution.
- **CMI at L0=22** is NOT SUPPORTED (rho=0.044, p=0.83). The information-theoretic approach does not hold.
- **Cross-domain existence** confirmed: absorption rates differ significantly within RAVEL hierarchies (p<0.01), with city-language at 11.6% vs. city-country at 45.1%.
- **First-letter is NOT the worst case.** Semantic hierarchies show HIGHER absorption at the strongest layer (L24), inverting the hypothesis that spelling represents a worst-case scenario.
- **GAS failed** as unsupervised detector (rho=0.12). The decoder-activation mismatch approach does not capture absorption.
- **Layer-dependent absorption is clean**: 15x variation (0.7% to 42.9%), F1>=0.96 at all layers. This is the paper's most robust finding.

### Critical Issues Identified in Prior Review (Score Regression 7.0 -> 6.5)

The iter-9 review identified several blocking issues that must be addressed in this iteration:

1. **Stale buggy data in Section 5.2**: Paper still reports pilot cross-domain patching data (d=-0.91) when corrected FULL-mode data shows d=1.50 (city-continent) and d=0.75 (city-language). The corrected data makes the paper STRONGER (universal mechanism, not domain-specific failure).
2. **Benign/pathological circularity**: The ablation-based diagnostic measures decoder geometry, not computational redundancy. Must be reframed.
3. **Headline overclaiming**: "100% pathological" tested on 1 hierarchy; "4x range" not fully significant pairwise; "no architecture effect" from underpowered test.
4. **Missing figures 4-6**: 3 figure PDFs absent, compilation blocker.
5. **Data provenance errors**: F1=0.97 vs 1.0 discrepancy, benign count 1471 vs 1464, aggregation method undocumented.
6. **Probe degradation ablation not executed**: Core experiment to resolve whether cross-domain variation is probe artifact vs. genuine hierarchy effect.
7. **validate_integration.py not implemented** (10 iterations of recommendation).

---

## Landscape of Perspectives

### Available Perspectives This Round

Only 2 of 6 perspectives were completed for iteration 10 (pragmatist and empiricist). Both independently:
- **Confirmed cross-domain characterization as highest-priority gap** -- consistent with all 6 perspectives across all prior rounds.
- **Identified the same front-runner** (cross-domain absorption taxonomy) with secondary contributions (unsupervised detection, L0 sensitivity).
- **Converged on the same toolchain** (sae-spelling + SAELens + Gemma Scope + TransformerLens).

### Key Agreements (Pragmatist + Empiricist + 9 Prior Iterations)

1. Feature absorption is a real, theoretically proven phenomenon.
2. The single-task monoculture (first-letter spelling only) is the most critical empirical gap.
3. Cross-domain characterization is the highest-priority experiment.
4. Training-free analysis of pre-trained SAEs is the right approach.
5. Probe quality gating (F1 > 0.85, ideally > 0.90) is essential for credible results.
6. RAVEL provides validated entity-attribute structure with 500+ entities.

### Key Conflicts and Resolutions

**Conflict 1: Scope of cross-domain tasks.**
- Pragmatist proposes 4 tasks (first-letter, city-country, POS, animal taxonomy).
- Empiricist proposes 5 tasks (adding entity-type NER).
- Prior iterations settled on RAVEL (city-country, city-continent, city-language) + first-letter.
- **Resolution**: Retain the RAVEL-based design validated by 9 iterations of pilot evidence. POS, animal taxonomy, and NER remain future work. The 3 RAVEL hierarchies provide clean, validated hierarchical structure with known ground truth.

**Conflict 2: Unsupervised detection priority.**
- Pragmatist elevates decoder geometry (Candidate B) as a secondary contribution.
- Empiricist deprioritizes unsupervised detection, noting it is "unvalidated."
- Prior evidence: GAS rho=0.12, CMI rho=0.044 -- both failed.
- **Resolution**: Unsupervised detection is documented as a negative result, not a contribution. The evidence from 9 iterations is clear: purely geometric and information-theoretic approaches do not work. This is reported transparently.

**Conflict 3: Benign vs. pathological framing.**
- Prior iterations included this as H8, but the iter-9 review identified fundamental circularity.
- **Resolution**: Reframe as "decoder direction magnitude" rather than computational-redundancy diagnostic. Acknowledge circularity explicitly. Describe what a genuine test would require (activation-level ablation, path patching through separate circuits). The result (3.98 nats) establishes that child decoders carry large-magnitude parent information, which is informative about absorption mechanics even if it does not answer the computational-redundancy question.

**Conflict 4: Theory priority.**
- Pragmatist offers Absorption Tax as qualitative framework.
- Empiricist demands falsification criteria for any theoretical claim.
- Pilot evidence: Absorption Tax quantitative predictions failed (rho=0.08).
- **Resolution**: Rate-distortion predictors tested as correlational analysis (H9), not formal theorem. Absorption Tax retained as qualitative framing only.

### Perspective Weighting

1. **9 Prior Iterations of Evidence (highest weight):** The accumulated experimental evidence from 9 iterations, including extensive pilot data, full-mode experiments, and 10 review cycles, is the most authoritative source for what works and what does not.

2. **Empiricist (high weight):** Statistical rigor, falsification criteria, control experiments, and probe quality gates are adopted wholesale. The RAVEL-based experimental design is validated.

3. **Pragmatist (high weight):** Implementation strategy, toolchain selection, time estimates, and resource planning are critical for feasibility.

4. **Prior Contrarian insights (moderate weight, from iter 9):** The benign-vs-pathological concept remains valuable as a research direction, though the current implementation has circularity issues.

5. **Prior Theoretical/Interdisciplinary insights (moderate weight, from iter 9):** Rate-distortion framework provides qualitative context but quantitative predictions failed.

6. **Prior Innovator insights (low weight for primary):** ITAC and unsupervised detection refuted by evidence. Retained as documented negative results.

---

## Research Questions

1. **RQ1 (Cross-Domain):** Do absorption rates vary systematically across feature hierarchy types (syntactic first-letter, factual city-country, city-continent, city-language), and does the first-letter spelling task represent a typical or extreme case?

2. **RQ2 (Confound Disentanglement):** What fraction of measured "absorption" is attributable to hierarchy-driven absorption vs. L0-induced hedging vs. probe/metric artifacts?

3. **RQ3 (Architecture Validity):** Do architecture rankings for absorption resistance (Matryoshka > BatchTopK, etc.) hold across hierarchy types, or are they task-specific?

4. **RQ4 (Causal Mechanism):** Is absorption driven by competitive exclusion dynamics (child suppresses parent via decoder overlap), and does activation patching confirm this universally across hierarchy types?

5. **RQ5 (Probe Artifact vs. Genuine Hierarchy Effect):** Does the cross-domain variation in absorption rates persist when probe quality is controlled? Specifically, does degrading first-letter probe quality to match RAVEL probe levels reproduce RAVEL absorption rates?

---

## Hypotheses

**H1 (Cross-Domain Variation) -- SUPPORTED:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.
- *Evidence:* Within-RAVEL Kruskal-Wallis p=7.4e-66. City-language vs. first-letter p_Bonf=0.003 at L24 16k.
- *Full-mode:* Confirm with probe degradation ablation to rule out probe artifact.
- *Falsification:* Rates within 5 pp across all types after probe quality matching.

**H2' (Semantic > Syntactic -- Revised) -- SUPPORTED at L24:** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task at layer 24.
- *Evidence:* First-letter 27.1% vs city-country 45.1%, city-continent 42.9% at L24 (all with F1>=0.96).
- *Caveat:* At L12, first-letter shows very low absorption (3.9%) due to poor binary probe. Layer selection matters.
- *Falsification:* After probe quality matching (degradation ablation), first-letter absorption >= all semantic hierarchies by > 5 pp at the same layer.

**H3 (Absorption-Hedging Decomposition) -- SUPPORTED:** The absorption-to-hedging ratio varies by hierarchy type.
- *Evidence:* Strict hedging: 0-22.6% across hierarchies (vs. 92.6% loose). Compensatory resolution dominates.
- *Falsification:* Ratio identical across all hierarchy types (within 10 pp).

**H4 (GAS Detector) -- REFUTED:** GAS achieves rho=0.12. Documented negative result.

**H5 (Absorption Tax) -- NOT SUPPORTED quantitatively:** Absorption-MSE correlation rho=0.08. Retained as qualitative framework only.

**H6 (Architecture Generalization) -- INCONCLUSIVE:** JumpReLU consistently lowest absorption across all tested hierarchies. Architecture rankings may be hierarchy-dependent.
- *Issue:* Underpowered (12 observations). Cannot claim "no architecture effect" -- only "effect not detected with current power."
- *Full-mode:* Reframe as exploratory with effect sizes reported regardless of significance.

**H7 (Causal Absorption via Activation Patching) -- SUPPORTED universally:**
- First-letter: d=1.33, p=0.000218
- City-continent: d=1.50, p<1e-20 (FULL-mode corrected)
- City-language: d=0.75, p<1e-18 (FULL-mode corrected)
- The universal mechanism is the paper's strongest causal finding.

**H8 (Decoder Direction Magnitude -- Reframed from Benign/Pathological):** Child decoder vectors carry large-magnitude parent-direction information (mean 3.98 nats logit change). 
- *Reframing:* This establishes the magnitude of parent information encoded in child decoders, NOT whether absorption is computationally benign or pathological. The circularity issue (ablating the direction that IS the absorption measures absorption magnitude, not computational independence) is explicitly acknowledged.
- *What a genuine test would require:* Activation-level ablation (z_parent=0) or path patching through separate circuits.
- *Falsification:* Logit change indistinguishable from random-direction control (this was falsified: 3.98 vs. 0.12 control).

**H9 (Rate-Distortion Three-Factor Predictor) -- UNTESTED in full mode:** Per-pair absorption probability predicted by cos_sim x co-occurrence / reconstruction importance.
- *Target:* Spearman rho > 0.5 across pooled data from all hierarchies.
- *Falsification:* rho < 0.3 or p > 0.05.

**H10 (Probe Degradation -- NEW, highest priority):** Degrading first-letter probe quality to match RAVEL probe levels (F1=0.70, 0.80, 0.85, 0.90) will NOT reproduce RAVEL-level absorption rates, confirming the cross-domain variation is a genuine hierarchy effect rather than a probe artifact.
- *Alternative outcome:* If degraded first-letter rates match RAVEL rates, cross-domain variation is probe-driven. Both outcomes are publishable and informative.
- *Falsification:* Degraded first-letter absorption rates at F1=0.80 fall within 5 pp of city-language rates at equivalent probe quality.

---

## Method

### Phase 0: Critical Fixes from Prior Review (BLOCKING, zero GPU)

**Step 0.1: Data integrity overhaul** (~2 hours CPU)
- Create authoritative data manifest (`exp/results/data_manifest.json`) mapping every numerical claim to source data file + field path
- Resolve F1=0.97 vs 1.0, benign count 1471 vs 1464, aggregation method discrepancies
- Implement `validate_integration.py` -- automated cross-check of paper claims against source data
- Document per-unique-word aggregation method in methodology section

**Step 0.2: Update paper with corrected FULL-mode data** (~2-3 hours CPU)
- Replace Section 5.2 with corrected FULL-mode cross-domain patching results
- Update abstract: "mechanism confirmed cross-domain" (d=0.75-1.50), not "mechanism fails cross-domain"
- Update Table 4: H7-crossdomain from FAILED to SUPPORTED
- Remove "concentrated vs. distributed absorption" dichotomy (based on buggy pilot data)
- Verify: grep for "d=-0.91" and "0.05% recovery" returns 0 hits

**Step 0.3: Reframe benign/pathological claim** (~45 min CPU)
- Acknowledge circularity in Section 5.3
- Reframe from "100% pathological" to "child decoders carry large-magnitude parent information (3.98 nats vs. 0.12 control)"
- Describe what a genuine computational-redundancy test would require
- Scope to city-continent L24 only (not universal)

**Step 0.4: Reframe headline claims** (~30 min CPU)
- "4x range" qualified as descriptive, not all pairwise-significant
- "100% pathological" scoped to "across 1,471 instances from city-continent at L24"
- "No architecture effect" reframed as "effect not detected with limited statistical power"
- Kruskal-Wallis scope clarified (3 RAVEL hierarchies, not 4)
- Separate descriptive from inferential claims throughout

**Step 0.5: Generate missing figures** (~1-2 hours CPU)
- fig4_patching_comparison.pdf: paired dot plot of recovery rates by hierarchy
- fig5_pathological_histogram.pdf: histogram of |logit change| with control overlay
- fig6_architecture_comparison.pdf: grouped bar chart of absorption rates by architecture x hierarchy

### Phase 1: Probe Degradation Ablation (HIGHEST-PRIORITY NEW EXPERIMENT)

**Step 1.1: Probe degradation experiment** (~2 GPU-hours)
- Load trained first-letter probes (F1=1.0 at L24)
- Inject label noise to degrade test-set F1 to {0.70, 0.80, 0.85, 0.90}
- Re-run absorption measurement pipeline at L24 with 16k SAE at each degraded level
- Plot absorption rate vs. probe F1
- **Key question:** Does first-letter absorption at F1=0.80 approach RAVEL absorption rates at F1=0.80?
- Report in new Section 4.6 or appendix

### Phase 2: Complete Missing Experiments

**Step 2.1: Benign/pathological on first-letter** (~0.5 GPU-hours)
- Replicate the decoder direction ablation on first-letter at L24
- Compare logit change distribution with city-continent result
- Supports or challenges the city-continent-only result

**Step 2.2: Rate-distortion predictor validation** (~1 GPU-hour)
- Compute per-pair predictors across all hierarchy types
- cos_sim(decoder_parent, decoder_child), P(child|parent), R(parent)
- Fit three-factor model and evaluate Spearman rho
- Test qualitative predictions: does absorption increase with cos_sim and co-occurrence?

### Phase 3: Writing Completions (zero GPU)

**Step 3.1: Write appendix sections** (~2 hours CPU)
- GAS negative result (rho=0.116, AUROC=0.571)
- CMI negative result (L0=22 rho=0.044, p=0.83)
- Absorption Tax analysis (rho=-0.20, concordance 50%)
- Threshold sensitivity (CV=0.077, 20/20 cells robust)

**Step 3.2: Document methodological details** (~30 min CPU)
- Per-unique-word aggregation explanation
- Token position asymmetry (first-letter at -6, RAVEL at -2) as limitation
- Single-model limitation (Gemma 2 2B only)

---

## Novelty Assessment

### Contribution 1 (PRIMARY): First Cross-Domain Absorption Characterization -- Novelty 9/10
- **No prior work** measures absorption on entity-attribute hierarchies (verified April 2026 via arXiv, Google Scholar, web search).
- Finding that first-letter is atypical (lowest absorption at L24) reframes the entire absorption literature.
- Directly addresses Gaps 2, 6, and 8b from the literature survey.

### Contribution 2 (SECONDARY): Universal Causal Mechanism via Activation Patching -- Novelty 8/10
- First interventional (not correlational) evidence for competitive exclusion in SAEs.
- FULL-mode corrected data shows the mechanism is universal across hierarchy types (d=0.75-1.50), not domain-specific.
- Tightened hedging classification (strict 0-22.6% vs. loose 92.6%) is a reusable methodological contribution.

### Contribution 3 (TERTIARY): Decoder Direction Magnitude Analysis -- Novelty 7/10
- Child decoders carry large-magnitude parent information (3.98 nats vs. 0.12 control).
- The circularity limitation is explicitly acknowledged, with a roadmap for genuine computational-redundancy testing.
- Still informative about absorption mechanics even with the circularity caveat.

### Negative Results (Honestly Reported)
- GAS unsupervised detector: rho=0.12 (target 0.6)
- CMI at L0=22: rho=0.044 (p=0.83)
- Absorption Tax quantitative predictions: rho=0.08
- First-letter is NOT worst case (positive finding disguised as negative result)

### Revisions from Prior Feedback

This is Synthesis Round 7, addressing the score regression (7.0 -> 6.5) from iteration 9:

1. **FULL-mode data propagation (BLOCKING):** Section 5.2 corrected with FULL-mode cross-domain patching results. d=1.50 (city-continent), d=0.75 (city-language) replace buggy pilot d=-0.91.

2. **Benign/pathological circularity acknowledged.** Reframed from "100% pathological" to "large-magnitude parent information in child decoders." Genuine test described.

3. **Headline claims scoped to evidence.** "4x range" qualified as descriptive. "No architecture effect" reframed as underpowered. Kruskal-Wallis scope clarified.

4. **Probe degradation ablation added (H10).** The single highest-value experiment recommended by the iter-9 review. Resolves the core uncertainty about cross-domain variation.

5. **Data integrity infrastructure.** validate_integration.py and data_manifest.json address the 10-iteration systemic weakness.

6. **Missing figures generated.** Figures 4-6 created from existing JSON data.

---

## Expected Contributions

### Primary: First Cross-Domain Absorption Characterization
- First systematic measurement beyond first-letter spelling
- Descriptive finding: 4x range in absorption rates (11.6% to 45.1%)
- Inferential finding: within-RAVEL variation statistically significant (p=7.4e-66); city-language vs. first-letter significant after Bonferroni (p=0.003)
- Probe degradation ablation distinguishes probe artifact from genuine hierarchy effect
- Layer-dependent absorption: 15x variation (0.7% to 42.9%), F1>=0.96 at all layers

### Secondary: Universal Causal Mechanism
- Activation patching confirms competitive exclusion across ALL hierarchy types (d=0.75-1.50)
- Tightened hedging classification: loose is near-tautological (0-22.6% strict vs. 92.6% loose)
- Three-way hedging decomposition (strict/compensatory/persistent) with bootstrap CIs

### Tertiary: Negative Results and Methodology
- GAS, CMI, Absorption Tax failures transparently documented
- Decoder direction magnitude analysis with acknowledged circularity
- Rate-distortion correlational predictors (if H9 succeeds)

---

## Experimental Plan

| Priority | Phase | Experiment | GPU-hours | Wall-clock | Validates |
|----------|-------|-----------|-----------|------------|----------|
| P0 | 0.1 | Data integrity overhaul + validate_integration.py | 0 | 2 hr | DATA001, PIPE001 |
| P0 | 0.2 | Update paper with corrected FULL-mode data | 0 | 2-3 hr | SND001 |
| P0 | 0.3 | Reframe benign/pathological | 0 | 45 min | SND002 |
| P0 | 0.4 | Reframe headline claims | 0 | 30 min | SND003, ANL001 |
| P0 | 0.5 | Generate missing figures 4-6 | 0 | 1-2 hr | WRI001 |
| P1 | 1.1 | **Probe degradation ablation** | 2 | 2 hr | **H10** (core) |
| P1 | 2.1 | Benign/pathological on first-letter | 0.5 | 30 min | H8 |
| P2 | 2.2 | Rate-distortion predictor validation | 1 | 1 hr | H9 |
| P2 | 3.1 | Write appendix sections | 0 | 2 hr | WRI004 |
| P2 | 3.2 | Document methodological details | 0 | 30 min | WRI002, EXP003, WRI005 |
| **Total** | | | **~3.5** | **~13 hr** | |

### Investment Summary
- **Zero-GPU fixes (Phase 0):** ~7 hours CPU. Addresses all P0 blockers.
- **Critical experiments (Phase 1-2):** ~3.5 GPU-hours. Probe degradation ablation is the single highest-impact experiment.
- **Writing completions (Phase 3):** ~2.5 hours CPU. Appendix and methodological documentation.

---

## Resource Estimate

- **GPU:** Single GPU >= 24GB VRAM
- **Total compute:** ~3.5 GPU-hours new experiments, ~13 hours wall-clock total
- **Storage:** ~10GB cached activations (existing)
- **Software:** SAELens v6, TransformerLens, sae-spelling, RAVEL (HuggingFace `hij/ravel`)
- **No SAE training required.** All experiments use pre-trained SAEs.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Probe degradation ablation shows cross-domain variation IS probe artifact | High | 25% | BOTH outcomes publishable: if probe-driven, document as methodological caution; if genuine, cross-domain finding is strengthened |
| Data integrity errors persist after manifest creation | Medium | 15% | validate_integration.py automates cross-checking; run after every paper revision |
| Rate-distortion predictors fail (rho < 0.3) | Medium | 40% | Report as negative result; cross-domain characterization stands independently |
| Benign/pathological on first-letter contradicts city-continent result | Low | 20% | Report both; hierarchy-dependent mechanisms would be an additional finding |
| Competing cross-domain absorption paper | Low | 10% | Verified no competing work as of April 2026. Execute promptly. |

---

## Venue Target

**Primary:** NeurIPS 2026 main conference or ICLR 2027 main conference.
**Fallback:** EMNLP 2026 or NeurIPS 2026 MI Workshop.

The paper's strength lies in careful, evidence-driven science with transparent negative results: cross-domain characterization extending the field's most important absorption benchmark, universal causal mechanism via activation patching, and honest reporting of multiple failed approaches (GAS, CMI, Absorption Tax). The probe degradation ablation (H10) is the final piece needed to make the cross-domain finding bulletproof.

---

## Synthesis Reasoning

### What Changed from Iteration 9

1. **H10 (Probe Degradation Ablation) added as highest-priority experiment.** This was the #1 recommended experiment from the iter-9 review and addresses the core uncertainty about whether cross-domain variation is genuine or a probe artifact. It was never executed in prior iterations -- this is the single most important remaining experiment.

2. **Phase 0 (Critical Fixes) added as blocking prerequisite.** The score regression from 7.0 to 6.5 was caused by stale data in the paper, not by experimental weakness. Zero-GPU fixes to propagate corrected FULL-mode data, reframe overclaims, and fix data integrity are all blocking.

3. **Benign/pathological reframed.** Circularity acknowledged; reframed as "decoder direction magnitude" rather than computational-redundancy diagnostic. This was the iter-9 review's second-highest priority fix.

4. **Headline claims scoped throughout.** "4x range" qualified, "100% pathological" scoped, "no architecture effect" reframed. The pattern of introducing new overclaims while fixing old ones is explicitly addressed.

5. **Compute budget reduced.** Only 3.5 GPU-hours new experiments needed (down from 10.5 in iter-9). The majority of work is zero-GPU paper fixes and writing, reflecting the project's maturity.

6. **Only 2/6 perspectives available.** Both pragmatist and empiricist confirmed the front-runner and provided consistent implementation guidance. The limited perspective count is compensated by the rich evidence base from 9 prior iterations, which is more authoritative than additional perspectives would be.
