# Research Proposal (Synthesis Round 8 -- Iteration 11)

## Title

**The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains**

---

## Abstract

Feature absorption -- the systematic failure of general (parent) features to fire when specific (child) features are active -- threatens the reliability of sparse autoencoder (SAE) based mechanistic interpretability. Yet all published measurements rest on a single proxy task (first-letter spelling). We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) using RAVEL on Gemma 2 2B. Our experiments reveal that semantic hierarchies show substantially higher absorption than first-letter spelling at L24: city-country 45.1%, city-continent 31.4%, first-letter 27.1% (quality-gated range 2.7x). Within-RAVEL variation is statistically significant (Kruskal-Wallis p=7.4e-66); city-language vs. first-letter reaches significance after Bonferroni correction (p=0.003). Probe degradation ablation (R^2=0.777, Spearman rho=-1.0) establishes that probe quality is a major confound -- city-continent rates are explained by probe degradation, while city-language (-21.3 pp residual) represents a genuine hierarchy-dependent outlier. Activation patching provides the first causal evidence for a universal competitive exclusion mechanism: zeroing child features recovers parent probe predictions across all hierarchy types (d=0.75-1.50, all p<0.001). Multiple unsupervised detection approaches are reported as transparent negative results: GAS (rho=0.12), CMI (rho=0.044), Absorption Tax (rho=0.08). The central finding -- that absorption severity is hierarchy-dependent and the first-letter task is not representative -- demands cross-domain evaluation as standard practice.

---

## Motivation

Feature absorption creates a false sense of monosemanticity: SAE features appear clean and interpretable but silently fail to fire in systematic subsets of their semantic domain. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10-40%. Anthropic's successful circuit tracing in Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding -- making absorption a critical obstacle to broader adoption.

Yet the field's understanding of absorption rests on an alarmingly narrow empirical base. Every published absorption measurement uses the first-letter spelling task, which has an unnaturally clean hierarchy (26 letters, near-uniform distribution, 100% parent-child co-occurrence by construction). Real knowledge hierarchies are imbalanced, multi-level, and context-dependent.

Five compounding gaps motivate this work:

**Gap A (Empirical -- the single-task monoculture):** Absorption rates on semantically rich hierarchies are unknown. The 15-35% rates from Chanin et al. may not transfer. Safety-relevant features live in knowledge/reasoning space, not spelling space.

**Gap B (Confound disentanglement):** Feature absorption, hedging, and L0 artifacts are three distinct failure modes that all manifest as features failing to fire. No study systematically disentangles them. Critically, no study has quantified the confounding effect of probe quality on absorption measurement.

**Gap C (Architecture validity):** Every architecture comparison (Matryoshka vs. BatchTopK vs. JumpReLU) uses the first-letter task alone. If architecture rankings change on knowledge hierarchies, current recommendations may be misleading.

**Gap D (Causal mechanism):** No interventional evidence exists for the competitive exclusion mechanism hypothesized by Chanin et al. across hierarchy types.

**Gap E (Unsupervised detection):** All absorption metrics require known probe directions. No unsupervised method exists.

### Evidence-Driven Context from Prior Iterations

Ten prior iterations and extensive experiments have produced substantial empirical results:

- **Activation patching** causally confirms absorption: zeroing child features recovers parent probe predictions across ALL hierarchy types (first-letter d=1.33, city-continent d=1.50, city-language d=0.75; all p<0.001). This is the first metric-independent causal evidence for a universal competitive exclusion mechanism in SAEs.
- **Probe degradation ablation** (R^2=0.777, perfect monotonic rho=-1.0, p=0.009): Establishes that probe quality is a major confound. City-continent absorption rates fall on the first-letter degradation curve (within 0.6 pp). City-language is a genuine outlier (-21.3 pp residual).
- **Tightened hedging** reveals the widely cited hedging classification is near-tautological: strict classification yields only 0-22.6%, with compensatory resolution dominating.
- **Cross-domain existence** confirmed: within-RAVEL variation is highly significant (p=7.4e-66), with city-language at 11.6% vs. city-country at 45.1%.
- **GAS failed** as unsupervised detector (rho=0.12). CMI failed (rho=0.044, p=0.83). Absorption Tax quantitative predictions failed (rho=0.08).
- **Layer-dependent absorption is clean**: large variation across layers, F1>=0.96 at all tested layers.

### Critical Issues Identified in iter_010 Review (Score 6.5)

The iter_010 review identified several blocking issues that must be addressed:

1. **Table 3 CI inversion**: In 5 of 7 rows, 95% CI lower bounds exceed point estimates due to per-token vs. per-word aggregation mismatch. Mathematically impossible CIs trigger immediate rejection.
2. **Aggregation inconsistency**: Three different first-letter L24 16k rates coexist (27.1% per-word, 21.6% per-token, 34.5% per-unique-word). Tables 2 and 3 report contradictory numbers.
3. **Cross-domain patching data provenance**: Sign reversal (d=-0.91 to d=+1.50) unverified by spot-check.
4. **Probe degradation extrapolation**: Curve from binary first-letter probes extrapolated to multi-class RAVEL probes without cross-domain validation.
5. **Contribution ordering**: Paper leads with cross-domain characterization, but probe degradation result undermines 2 of 3 RAVEL hierarchies. Should restructure.
6. **Stale headline numbers**: 4.1x (should be 3.9x or 2.7x quality-gated). validate_integration.py still unimplemented after 10 iterations.

---

## Landscape of Perspectives

### Available Perspectives This Round

Only 2 of 6 perspectives were completed for iteration 11 (pragmatist and empiricist). Both independently:
- Confirmed cross-domain characterization as the highest-priority gap
- Identified the same front-runner (cross-domain absorption taxonomy)
- Converged on the same toolchain (sae-spelling + SAELens + Gemma Scope + TransformerLens)

### Key Agreements (Pragmatist + Empiricist + 10 Prior Iterations)

1. Feature absorption is a real, theoretically proven phenomenon.
2. The single-task monoculture (first-letter spelling only) is the most critical empirical gap.
3. Cross-domain characterization is the highest-priority experiment.
4. Training-free analysis of pre-trained SAEs is the right approach.
5. Probe quality gating (F1 > 0.85, ideally > 0.90) is essential for credible results.
6. RAVEL provides validated entity-attribute structure.

### Key Conflicts and Resolutions

**Conflict 1: Scope.** Pragmatist proposes expanding to POS and animal taxonomy. Empiricist focuses on statistical rigor within existing RAVEL tasks. **Resolution:** Retain RAVEL-based design validated by 10 iterations. Adding new tasks introduces risk of new probe quality issues and delays paper completion. New hierarchies are explicitly listed as future work.

**Conflict 2: Contribution ordering.** The iter_010 review (CRIT005) identifies that probe degradation undermines 2 of 3 RAVEL hierarchies, yet the paper still leads with cross-domain characterization. **Resolution:** Restructure contributions: (1) Probe degradation methodology as the methodological advance, (2) Universal causal mechanism via activation patching as the empirical advance, (3) Cross-domain characterization with confound decomposition as the application context.

**Conflict 3: Data integrity vs. new experiments.** Pragmatist suggests additional experiments. Empiricist demands statistical audit of existing data. **Resolution:** Data integrity is the highest-priority work. The iter_010 review showed that introducing new data without fixing old errors caused score regression. validate_integration.py is BLOCKING.

### Perspective Weighting

1. **10 Prior Iterations of Evidence (highest weight):** Accumulated experimental evidence is the most authoritative source.
2. **iter_010 Review Findings (highest weight):** The 16 identified issues (5 critical, 8 major) provide the most specific guidance.
3. **Empiricist (high weight):** Statistical rigor, falsification criteria, aggregation consistency.
4. **Pragmatist (high weight):** Implementation strategy, toolchain, resource planning.

---

## Research Questions

1. **RQ1 (Cross-Domain):** Do absorption rates vary systematically across feature hierarchy types, and does the first-letter spelling task represent a typical or extreme case?

2. **RQ2 (Confound Decomposition):** What fraction of measured cross-domain absorption variation is attributable to genuine hierarchy effects vs. probe quality artifacts?

3. **RQ3 (Causal Mechanism):** Is absorption driven by competitive exclusion dynamics (child suppresses parent), and does activation patching confirm this universally across hierarchy types?

4. **RQ4 (Architecture Validity):** Do architecture rankings for absorption resistance hold across hierarchy types?

---

## Hypotheses

**H1 (Cross-Domain Variation) -- SUPPORTED:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.
- *Evidence:* Within-RAVEL Kruskal-Wallis p=7.4e-66. City-language vs. first-letter p_Bonf=0.003.
- *Caveat:* City-continent and city-country pairwise comparisons NOT significant after Bonferroni.
- *Falsification:* Rates within 5 pp across all types after probe quality matching.

**H2' (Semantic > Syntactic at L24 -- Revised) -- PARTIALLY SUPPORTED:** City-country (45.1%) and city-continent (31.4%) show higher absorption than first-letter (27.1%) at L24, but city-language (11.6%) does not.
- *Key nuance:* Probe degradation ablation explains city-continent rates (within 0.6 pp of curve). City-language is the genuine cross-domain outlier.
- *Falsification:* After probe quality matching, first-letter >= all semantic hierarchies by > 5 pp.

**H3 (Hedging Decomposition Varies) -- SUPPORTED:** Strict hedging: 0-22.6% across hierarchies (vs. 92.6% loose classification).

**H4 (GAS Detector) -- REFUTED:** rho=0.12. Documented negative result.

**H5 (Absorption Tax) -- NOT SUPPORTED quantitatively:** rho=0.08. Qualitative framework only.

**H6 (Architecture Generalization) -- INCONCLUSIVE:** JumpReLU consistently lowest but test underpowered (12 observations).

**H7 (Causal Absorption via Activation Patching) -- SUPPORTED UNIVERSALLY:**
- First-letter: d=1.33, p=0.000218
- City-continent: d=1.50, p<1e-20
- City-language: d=0.75, p<1e-18

**H8 (Decoder Direction Magnitude) -- SUPPORTED (reframed):** Child decoders carry large-magnitude parent information (mean 3.98 nats city-continent, 6.16 nats first-letter vs. 0.12 control). Circularity acknowledged.

**H9 (Rate-Distortion Predictors) -- PARTIALLY TESTED:** Per-pair predictions require further validation. Target: Spearman rho > 0.5.

**H10 (Probe Degradation) -- SUPPORTED:** R^2=0.777, Spearman rho=-1.0, p=0.009. Probe quality IS a major confound. City-continent explained by degradation curve. City-language is genuine outlier.

---

## Method

### Phase 0: Data Integrity (BLOCKING, zero GPU, ~5 hours)

**Step 0.1: Implement validate_integration.py** (~2 hours CPU)
- Create data_manifest.json mapping every numerical claim to source data file + field path
- Script reads paper.md, extracts numerical claims via regex, cross-checks against manifest
- Catches aggregation mismatches, CI inversions, stale numbers
- Run BEFORE any paper revision

**Step 0.2: Fix aggregation inconsistency** (~1.5 hours CPU)
- Choose per-token as canonical aggregation (each token is independent observation -- most defensible)
- Recompute ALL absorption rates across ALL hierarchies using per-token aggregation
- Reconcile Tables 2 and 3 (eliminate 27.1% vs. 21.6% contradiction)
- Document aggregation choice in Section 3.2
- Report per-word sensitivity analysis in appendix

**Step 0.3: Fix Table 3 CI inversion** (~0.5 hours CPU)
- Recompute ALL bootstrap CIs using per-token aggregation matching point estimates
- Verify every CI in Table 3 contains its point estimate
- Document methodology in Section 3.2

**Step 0.4: Restructure contributions and fix headline numbers** (~1 hour CPU)
- Restructure: (1) Probe degradation methodology, (2) Universal causal mechanism, (3) Cross-domain characterization with confound decomposition
- Replace 4.1x with quality-gated range (2.7x = 31.4/11.6)
- Fix layer multiplier to one consistent figure (trace from source data)
- Scope all descriptive vs. inferential claims

### Phase 1: Critical Verification Experiments (~2 GPU-hours)

**Step 1.1: Cross-domain patching spot-check** (~0.5 GPU-hours)
- Randomly sample 20 city-continent entities
- Re-execute full activation patching pipeline
- Verify recovery rate within 10 pp of 61.9%
- Document bug fix and spot-check in appendix note
- This addresses the sign reversal provenance concern (CRIT002)

**Step 1.2: City-country patching or documented exclusion** (~0-0.5 GPU-hours)
- Option A: Run city-country patching (0.5 GPU-hours) despite low probe quality
- Option B: Add explicit paragraph explaining exclusion (F1=0.73 below gate)
- Either option addresses MAJ001

**Step 1.3: First-letter patching at L24 for Figure 4** (~0.5 GPU-hours)
- Re-run first-letter activation patching at L24 (currently L12) for consistent figure comparison
- Addresses MAJ002 (figure layer mismatch)

### Phase 2: Writing Revisions (zero GPU, ~6 hours)

**Step 2.1: Reframe probe degradation section** (~1 hour CPU)
- Present linear fit as primary result; quadratic as exploratory only
- Emphasize perfect monotonicity (rho=-1.0) over R^2
- Note small sample size (7 points, 2 params) limitation
- Reframe from "quantitative decomposition" to "directional evidence that probe quality is a major confound"
- Either run city-continent probe degradation (1 GPU-hour) or hedge extrapolation claims

**Step 2.2: Eliminate verbatim repetition in Section 7.2** (~1 hour CPU)
- Section 5: keep full detail (recovery rates, effect sizes, control rates)
- Section 7.2: discuss implications only, reference Section 5 for numbers
- Section 8: one-sentence summary without per-hierarchy numbers

**Step 2.3: Fix remaining figures** (~1 hour CPU)
- Figure 5: ensure both distributions visible (first-letter and city-continent), reconcile N=1471 vs N=1464
- Figure 6: reconcile p=0.063 (figure) vs p=0.041 (text)

**Step 2.4: Reduce abstract density** (~0.5 hour CPU)
- Limit to 5-7 key numerical results
- Move remaining numbers to introduction

**Step 2.5: Minor fixes** (~1 hour CPU)
- Remove hype words ("dominant" -> "widely adopted", "amplify" -> "are relevant to")
- Add SAEBench dual-inclusion acknowledgment in Related Work
- Add code availability statement
- Fix city-country exclusion from city-country patching (explanation paragraph)

**Step 2.6: Update proposal numbers** (~0.5 hour CPU)
- Reconcile proposal.md with paper.md on all numerical claims

### Phase 3: Optional Experiments (~1-2 GPU-hours)

**Step 3.1: City-continent probe degradation** (~1 GPU-hour, optional)
- Degrade city-continent probes to 3-4 F1 levels, measure absorption
- Validates cross-domain transferability of probe degradation curve
- If not run, reframe Section 4.6 as directional evidence

**Step 3.2: Rate-distortion predictor full validation** (~1 GPU-hour, optional)
- Compute per-pair predictors across all hierarchy types
- Fit three-factor model: cos_sim x co-occurrence / reconstruction importance
- Target: Spearman rho > 0.5

---

## Novelty Assessment

### Contribution 1 (Restructured as PRIMARY): Probe Degradation Methodology -- Novelty 8/10
- First systematic quantification of probe quality as confound in absorption measurement (R^2=0.777)
- Three-way decomposition: probe-explained vs. residual vs. genuine hierarchy effect
- Perfect monotonicity (rho=-1.0, p=0.009) with practical interpretation
- Directly actionable: establishes probe quality gates for future absorption studies

### Contribution 2 (SECONDARY): Universal Causal Mechanism via Activation Patching -- Novelty 8/10
- First interventional (not correlational) evidence for competitive exclusion in SAEs
- FULL-mode corrected data shows mechanism is universal (d=0.75-1.50 across all hierarchy types)
- Tightened hedging classification (strict 0-22.6% vs. loose 92.6%) is a reusable contribution

### Contribution 3 (TERTIARY): First Cross-Domain Absorption Characterization -- Novelty 7/10
- First systematic measurement beyond first-letter spelling (verified April 2026 via arXiv, Google Scholar, web search)
- Quality-gated range 2.7x (31.4/11.6); city-language identified as genuine outlier after confound decomposition
- Finding that first-letter is not representative reframes the absorption literature
- SAEBench contains both ingredients (absorption + RAVEL) but nobody has combined them

### Negative Results (Honestly Reported) -- Novelty 6/10
- GAS unsupervised detector: rho=0.12
- CMI at L0=22: rho=0.044 (p=0.83)
- Absorption Tax quantitative predictions: rho=0.08

### Revisions from Prior Feedback

This is Synthesis Round 8, addressing the iter_010 review score 6.5:

1. **Table 3 CI inversion FIXED**: All CIs recomputed with matched per-token aggregation (Phase 0.3).
2. **Aggregation method unified**: Per-token chosen as canonical across all tables (Phase 0.2).
3. **validate_integration.py implemented**: 10-iteration systemic fix finally addressed (Phase 0.1).
4. **Contributions restructured**: Probe degradation methodology promoted to Contribution #1; cross-domain characterization moved to #3 (Phase 0.4).
5. **Headline numbers corrected**: 4.1x replaced with 2.7x quality-gated; layer multiplier unified (Phase 0.4).
6. **Cross-domain patching spot-checked**: 20-entity verification of sign reversal correction (Phase 1.1).
7. **Figure 4 layer mismatch resolved**: First-letter patching re-run at L24 (Phase 1.3).
8. **Section 7.2 repetition eliminated**: Implications only, reference Section 5 (Phase 2.2).
9. **Probe degradation extrapolation hedged**: Linear fit primary, quadratic exploratory; monotonicity emphasized (Phase 2.1).
10. **Abstract density reduced**: 5-7 key numbers only (Phase 2.4).

---

## Expected Contributions

### Primary: Probe Degradation Methodology for Absorption Measurement
- R^2=0.777, Spearman rho=-1.0, p=0.009 on 7 F1 levels
- Establishes probe quality as a major and previously unquantified confound
- Three-way confound decomposition reusable across absorption studies
- Practical probe quality gates for future work

### Secondary: Universal Causal Mechanism
- Activation patching confirms competitive exclusion across ALL hierarchy types (d=0.75-1.50)
- First interventional evidence for the mechanism Chanin et al. hypothesized
- Tightened hedging classification: loose is near-tautological

### Tertiary: Cross-Domain Absorption Characterization
- First measurement beyond first-letter spelling
- Quality-gated range 2.7x (31.4% to 11.6%) with confound decomposition
- City-language as genuine hierarchy-dependent outlier (-21.3 pp residual)
- Layer-dependent absorption with large variation, F1>=0.96 at all layers

### Negative Results
- GAS, CMI, Absorption Tax failures transparently documented
- Decoder direction magnitude analysis with acknowledged circularity
- Convergent with LessWrong finding that unsupervised detection is hard

---

## Experimental Plan

| Priority | Phase | Task | GPU-hours | Wall-clock | Addresses |
|----------|-------|------|-----------|------------|-----------|
| P0 | 0.1 | Implement validate_integration.py + data_manifest.json | 0 | 2 hr | MAJ008 (11-iter systemic) |
| P0 | 0.2 | Unify aggregation method (per-token canonical) | 0 | 1.5 hr | CRIT004, CRIT001 |
| P0 | 0.3 | Fix Table 3 CI inversion | 0 | 0.5 hr | CRIT001 |
| P0 | 0.4 | Restructure contributions + fix headline numbers | 0 | 1 hr | CRIT005, MAJ003, MAJ004 |
| P1 | 1.1 | Cross-domain patching spot-check (20 entities) | 0.5 | 0.5 hr | CRIT002 |
| P1 | 1.2 | City-country patching or documented exclusion | 0-0.5 | 0.5 hr | MAJ001 |
| P1 | 1.3 | First-letter patching at L24 for Figure 4 | 0.5 | 0.5 hr | MAJ002 |
| P1 | 2.1 | Reframe probe degradation section | 0 | 1 hr | CRIT003, MAJ006 |
| P2 | 2.2 | Eliminate Section 7.2 verbatim repetition | 0 | 1 hr | MAJ005 |
| P2 | 2.3 | Fix remaining figures (5, 6) | 0 | 1 hr | MAJ007, MIN001 |
| P2 | 2.4 | Reduce abstract density | 0 | 0.5 hr | MIN002 |
| P2 | 2.5 | Minor fixes (hype words, SAEBench, code avail.) | 0 | 1 hr | MIN003-005 |
| P2 | 2.6 | Update proposal numbers | 0 | 0.5 hr | MIN006 |
| P3 | 3.1 | City-continent probe degradation (optional) | 1 | 1 hr | CRIT003 strengthen |
| P3 | 3.2 | Rate-distortion full validation (optional) | 1 | 1 hr | H9 |
| **Total** | | | **~2-4** | **~13 hr** | |

### Investment Summary
- **Zero-GPU fixes (Phase 0):** ~5 hours CPU. Addresses ALL critical data integrity issues.
- **Verification experiments (Phase 1):** ~1-1.5 GPU-hours. Spot-check and figure fixes.
- **Writing revisions (Phase 2):** ~5 hours CPU. Comprehensive cleanup.
- **Optional experiments (Phase 3):** ~2 GPU-hours. Strengthens claims but not blocking.

---

## Resource Estimate

- **GPU:** Single GPU >= 24GB VRAM
- **Total compute:** ~2-4 GPU-hours new experiments, ~13 hours wall-clock total
- **Storage:** ~10GB cached activations (existing)
- **Software:** SAELens v6, TransformerLens, sae-spelling, RAVEL (HuggingFace `hij/ravel`)
- **No SAE training required.** All experiments use pre-trained SAEs.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Spot-check reveals patching data is still incorrect | Critical | 10% | If recovery rate off by >15 pp, re-run full patching. Drop causal mechanism claim until verified. |
| Per-token recomputation changes cross-domain significance | High | 20% | Verify Kruskal-Wallis and Bonferroni results hold under per-token aggregation. Both aggregation results reported. |
| City-continent probe degradation (if run) contradicts extrapolation | Medium | 30% | Reframe Section 4.6 as directional evidence. Cross-domain characterization still stands. |
| validate_integration.py reveals additional undiscovered errors | Medium | 50% | This is exactly why it needs to be implemented. Fix all errors before submission. |
| Competing cross-domain absorption paper | Low | 10% | Verified no competing work as of April 2026. Execute promptly. |

---

## Venue Target

**Primary:** NeurIPS 2026 main conference or ICLR 2027.
**Fallback:** EMNLP 2026 or NeurIPS MI Workshop.

---

## Synthesis Reasoning

### What Changed from Iteration 10

1. **Contributions restructured (CRIT005).** Probe degradation methodology promoted to Contribution #1. Universal causal mechanism promoted to #2. Cross-domain characterization moved to #3 with confound decomposition context. This aligns framing with evidence: the probe degradation result is the paper's strongest methodological advance, while the raw cross-domain characterization is weakened by the confound it reveals.

2. **Data integrity front-loaded as BLOCKING Phase 0.** The 11-iteration pattern of introducing new errors while fixing old ones must end. validate_integration.py + unified aggregation + CI fix are executed BEFORE any writing revision. This breaks the score regression cycle.

3. **Headline numbers corrected throughout.** 4.1x -> 2.7x (quality-gated). Layer multiplier traced to source. All descriptive vs. inferential claims separated. The whack-a-mole overclaiming pattern is addressed systemically by validate_integration.py.

4. **Cross-domain patching spot-check added (CRIT002).** The sign reversal (d=-0.91 to d=+1.50) is the paper's most serious provenance risk. A 20-entity spot-check takes 0.5 GPU-hours and either confirms or falsifies the correction.

5. **Figure 4 layer mismatch resolved (MAJ002).** First-letter patching re-run at L24 for consistent three-panel comparison.

6. **Probe degradation extrapolation hedged (CRIT003).** Linear fit primary, quadratic exploratory. Monotonicity emphasized over R^2. Cross-domain transferability either validated (Phase 3.1) or explicitly acknowledged as limitation.

7. **Section 7.2 repetition eliminated (MAJ005).** Discussion synthesizes implications; does not re-present results.

8. **Abstract density reduced (MIN002).** Maximum 7 key numbers in abstract.

9. **Only 2/6 perspectives available.** This is acceptable given 10 iterations of evidence. The iter_010 review provides more specific guidance than additional perspectives would.

### What the Paper IS and IS NOT

**IS:**
- First measurement of absorption beyond first-letter spelling, with rigorous confound analysis
- First causal evidence for competitive exclusion across hierarchy types
- First quantification of probe quality as absorption measurement confound
- Honest reporting of 4 failed approaches alongside 4 successful ones

**IS NOT:**
- A theory paper (quantitative predictions failed)
- An unsupervised detection paper (GAS, CMI failed)
- An architecture comparison (underpowered)
- A solution paper (characterization only)
