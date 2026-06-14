# Supervisor Review -- Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders

**Score: 6.5 / 10** | Verdict: CONTINUE | NeurIPS calibration: Borderline Reject (significant issues requiring resolution)

**Score trajectory**: 5.5 (x3) -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> 6.5 -> 7.0 -> **6.5** (regression due to stale data in paper)

---

## Executive Summary

The paper makes a genuinely novel empirical contribution: the first cross-domain absorption characterization beyond first-letter spelling. The within-RAVEL variation (Kruskal-Wallis p=7.4e-66), activation patching causal evidence (first-letter d=1.33, p=0.000218), and decisive pathological absorption result (0% benign, 1,471 instances, 1000x effect ratio) are individually strong findings. The comprehensive documentation of 9 negative results is exemplary.

However, the score **decreases from the prior 7.0** because the paper text uses buggy pilot data for its second anchoring finding (cross-domain patching) when corrected FULL-mode data exists and fundamentally changes the story. The "concentrated vs. distributed absorption" mechanistic dichotomy -- a centerpiece of Sections 5.2, 7.2, and the abstract -- is based on incorrect data. Two additional soundness issues (benign/pathological circularity, unquantified probe quality confound) compound the problem. Until these are resolved, the paper cannot be scored above 7.

---

## Dimension Scores

### 1. Novelty and Significance: 8/10

**Strengths:**
- First cross-domain absorption characterization. Verified: no competing work exists as of April 2026. Chanin et al. (NeurIPS 2025 Oral) established first-letter as the only benchmark; this paper extends to three RAVEL entity-attribute hierarchies on Gemma 2 2B.
- First interventional (activation patching) evidence for competitive exclusion in SAEs. Prior work (Chanin et al.) used only integrated gradients (correlational).
- Decisive benign/pathological result: 0% benign across 1,471 instances, settling the "computational redundancy" debate empirically.
- Comprehensive negative results: 5 correlational approaches (GAS, CMI, Absorption Tax, rate-distortion, competition coefficients) all fail, motivating a paradigm shift from correlational to causal methods.
- Architecture non-significance is a valuable "negative-as-positive" finding for the community.

**Concerns:**
- Cross-model generalization untested (single model, Gemma 2 2B).
- The probe quality confound limits the precision of cross-domain rate comparisons (the qualitative finding of variation survives; the quantitative rates are upper bounds).
- The corrected FULL data (patching works cross-domain, d=0.75-1.50) actually *strengthens* the novelty -- the causal mechanism is universal, not domain-specific -- but this narrative is missing from the paper.

### 2. Technical Soundness: 5/10

**Critical issues:**

**(A) Stale/buggy data in Section 5.2.** The paper's second anchoring finding -- that activation patching fails for semantic hierarchies (city-continent recovery 0.05%, d=-0.91) -- is based on pilot data with a confirmed bug. The corrected FULL-mode results show city-continent recovery = 61.9% (d=1.50, p<1e-20) and city-language recovery = 34.2% (d=0.75, p<1e-18). The bug zeroed features with highest positive cosine to the probe (features supporting the correct class) instead of features with the most negative contribution (absorbers). The experiment_analysis.md and the corrected data file both document this fix.

The "concentrated vs. distributed absorption" mechanistic divide -- discussed in Sections 5.2, 7.2, the abstract, and Table 4 -- does not exist. This is the paper's most serious soundness issue: a central finding is based on data the team itself has identified as incorrect.

**(B) Benign/pathological diagnostic circularity.** The diagnostic ablates the parent direction from the child feature's decoder vector and measures logit change. By definition, absorption means the child's decoder carries parent-relevant information. Removing parent information from a vector known to contain it will always produce large logit changes. The diagnostic measures absorption magnitude, not computational independence. The claim "absorption is 100% pathological" overstates what the experiment demonstrates.

A genuine test of computational redundancy would hold the child constant and test whether the parent feature has independent causal effects through separate downstream circuits (e.g., path patching through the SAE, or ablating the parent feature's activation z_parent directly rather than its direction from the child decoder).

**(C) Unquantified probe quality confound.** RAVEL probes achieve F1=0.73-0.87, and the paper correctly notes absorption rates are upper bounds. But the bias magnitude is never estimated. The probe-only FN baseline (control 3, Section 3.3) is never reported numerically. A probe degradation ablation -- injecting label noise into first-letter probes to degrade them to F1={0.70, 0.80, 0.85, 0.90} and re-measuring absorption -- would directly quantify whether cross-domain variation is a probe artifact. This is the single most important missing experiment.

**Major issues:**

- After Bonferroni correction, only city-language differs significantly from first-letter (p_Bonf=0.003 at 16k, 0.043 at 65k). City-continent (p_Bonf=1.0) and city-country (p_Bonf=1.0) do not. The "4x range" is descriptive, not inferentially supported for most hierarchy pairs.
- The Kruskal-Wallis test (p=7.4e-66, N=3,545) excludes first-letter, testing only within-RAVEL variation. The paper implies a 4-hierarchy comparison.
- Data provenance: consolidation reports 0.345 for first-letter L24_16k (per-instance); paper uses 0.2707 (per-unique-word). Paper says F1=0.97; data says 1.0. Consolidation says benign n=1,464; paper says 1,471. No authoritative manifest exists.

### 3. Experimental Rigor: 6/10

**Strengths:**
- Statistical reporting is thorough: bootstrap CIs (10k resamples), Bonferroni correction, Wilcoxon tests, Cohen's d, effect sizes throughout.
- Threshold sensitivity analysis (Appendix A, CV=0.077) demonstrates structural robustness.
- Three controls (random direction, shuffled hierarchy, probe-only) are well-designed.
- Layer-dependent absorption analysis (L6/L12/L18/L24) is confound-free for first-letter (F1=1.0 at all layers) and is the cleanest finding in the paper.
- 9 negative results transparently documented with exact metrics.

**Weaknesses:**
- Probe degradation ablation not performed. This is the highest-priority missing experiment.
- Benign/pathological tested only on city-continent. First-letter (F1=1.0, concentrated mechanism) is untested.
- Architecture comparison underpowered: 12 data points at L24, 3 architectures x 4 hierarchies, with width confounds.
- Three figures missing (fig4, fig5, fig6), blocking visual evaluation of three main contributions.
- Token position asymmetry (first-letter at -6, RAVEL at -2) is an uncontrolled confound.

### 4. Reproducibility: 7/10

**Strengths:**
- Seed fixed at 42 for all stochastic operations.
- SAE configurations fully specified (Gemma Scope JumpReLU, SAEBench BatchTopK/Matryoshka).
- Detailed methodology (Section 3) with explicit quality gates.
- Code release promised.

**Weaknesses:**
- Per-unique-word vs per-instance aggregation undocumented.
- Token position rationale incomplete (why -6 vs -2).
- Probe training protocol differs between first-letter (sae-spelling binary one-vs-all) and RAVEL (scikit-learn multi-class logistic regression).
- Full-mode vs pilot data sources unclear -- a replicator would not know which run generated the paper's numbers.
- SAEBench checkpoint versions not fully specified.

---

## Cross-Validation: Paper Claims vs. Raw Data

| Paper Claim | Source Data | Match? |
|---|---|---|
| First-letter L24_16k absorption = 27.1% | absorption_firstletter.json: 0.27065 | YES |
| City-continent L24_16k = 31.4% | phase1_absorption_crossdomain.json: 0.31429 | YES |
| City-language L24_16k = 11.6% | phase1_absorption_crossdomain.json: 0.11556 | YES |
| City-country L24_16k = 45.1% | phase1_absorption_crossdomain.json: 0.45096 | YES |
| KW p=7.4e-66 | phase1_absorption_crossdomain.json: 7.369e-66 | YES |
| First-letter patching d=1.33 | consolidation: d=1.33 | YES |
| **City-continent patching d=-0.91** | **Paper: -0.91 (buggy pilot)** vs **FULL data: d=1.50** | **NO -- CRITICAL** |
| Benign 0%, |logit|=3.98 | benign_pathological.json: 0% benign, mean=3.979 | YES (minor: paper says 3.98, data says 3.979) |
| First-letter F1=0.97 weighted | absorption_firstletter.json: f1_weighted=1.0 | **NO -- discrepancy** |
| Benign instances = 1,471 | benign_pathological.json: 1,471 | YES |
| City-continent probe F1=0.87 | phase1_absorption_crossdomain.json: 0.871 | YES |
| Architecture KW p=0.50 (L24) | architecture_comparison (from consolidation) | YES |
| Hedging: first-letter strict=0% | hedging_crossdomain.json L24_16k first-letter: strict=0, comp=291 | YES |
| Hedging: city-language strict=22.6% | hedging_crossdomain.json L24_16k: strict=28/124=22.58% | YES |

---

## Priority Actions for Next Iteration

### BLOCKING (must fix before re-scoring)

1. **[ZERO GPU, 2-3 hours] Replace Section 5.2 with corrected FULL-mode cross-domain patching results.** The paper currently presents buggy pilot data as a major finding. The corrected results (city-continent d=1.50, city-language d=0.75) fundamentally change the narrative from "mechanism does not generalize" to "mechanism IS universal." Update abstract, Section 1, Table 4, Sections 5.2, 7.1, 7.2. This is the single highest-impact fix.

2. **[2 GPU-hours] Probe degradation ablation.** Inject label noise into first-letter probes to degrade to F1={0.70, 0.80, 0.85, 0.90}, re-measure absorption. This resolves the core uncertainty about whether cross-domain variation is a probe artifact.

3. **[ZERO GPU, 1 hour] Reframe benign/pathological claim.** Acknowledge the circularity (ablating parent direction from child decoder measures absorption magnitude, not computational independence). Frame as "absorption carries large-magnitude parent information in the decoder that contributes to model output" rather than "absorption is always pathological." Add brief discussion of what a genuine computational-redundancy test would require.

### HIGH PRIORITY

4. **[0.5 GPU-hours] Replicate benign/pathological on first-letter** where probe quality is perfect and mechanism is concentrated. Either confirms universality or reveals a distinction.

5. **[ZERO GPU, 1 hour] Generate missing figures 4-6.** Compilation blocker.

6. **[ZERO GPU, 30 min] Fix data provenance.** Resolve F1=0.97 vs 1.0 discrepancy. Document per-unique-word aggregation. Update consolidation summary.

7. **[ZERO GPU, 30 min] Reframe headline claims.** (a) "4x range" should note that only city-language vs first-letter survives Bonferroni; (b) "100% pathological" should scope to city-continent; (c) "architecture non-significance" should acknowledge limited power.

### MEDIUM PRIORITY

8. Acknowledge token position asymmetry as confound.
9. Report probe-only FN baseline numerically for each hierarchy.
10. Write appendix sections (GAS, CMI, Absorption Tax, threshold sensitivity) that are referenced but do not exist.

---

## Score Justification

The score of 6.5 (Borderline Reject) reflects:

- **Novelty (8)**: The research questions are important and timely. The cross-domain characterization is genuinely the first of its kind.
- **Soundness (5)**: Two critical issues (stale data in a central finding; benign/pathological circularity) and multiple major issues (unquantified probe confound; data provenance; Kruskal-Wallis scope) undermine the paper's key claims as currently written.
- **Experiments (6)**: Good controls and statistical practice, but missing the single most important ablation (probe degradation), incomplete coverage of the pathological diagnostic, and three missing figures.
- **Reproducibility (7)**: Generally well-documented with fixed seeds and explicit configurations, but aggregation choices and data sources need clarification.

**Path to 8.0 (Accept):**
The underlying data actually supports a stronger paper than what is currently written. Incorporating the corrected cross-domain patching results would strengthen the causal contribution from "first-letter only" to "universal across hierarchy types" -- a more compelling finding. The probe degradation ablation would resolve the core confound. These two changes, plus honest reframing of the benign/pathological claim, would bring the score to 7.5-8.0 with approximately 3 GPU-hours of additional work.

The irony is that the corrected data makes the paper *better*, not worse. The "mechanism is universal" narrative is stronger than "mechanism is domain-specific." The team has already done the hard experimental work; the paper just needs to reflect the actual results.
