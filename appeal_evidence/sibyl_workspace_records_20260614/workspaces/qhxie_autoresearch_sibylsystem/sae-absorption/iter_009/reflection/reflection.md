# Iteration 10 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 10 (current workspace = iter_009 data, review cycle 10)
**Date**: 2026-04-15
**Supervisor Score**: 6.5 / 10
**Dimension Scores**: Novelty 8, Soundness 5, Experiments 6, Reproducibility 7
**Writing Critic Score**: 7 / 10
**Verdict**: CONTINUE
**Quality Trajectory**: Regression (7.0 -> 6.5, -0.5)

---

## 1. Iteration Summary

This iteration produced a comprehensive full-mode experiment campaign (13 completed tasks, 0 failures) and a fully rewritten paper with corrected cross-domain data. The supervisor score **regressed from 7.0 to 6.5** -- the first score decrease since iteration 5. The regression is not due to loss of capabilities but to the discovery that the paper's second anchoring finding (cross-domain activation patching in Section 5.2) is based on **buggy pilot data** when corrected FULL-mode data exists and fundamentally changes the narrative. The corrected data actually strengthens the paper (from "mechanism fails cross-domain" to "mechanism is universal"), but the paper text has not been updated.

**Key achievements this iteration:**
- Full-mode cross-domain activation patching: city-continent d=1.50 (p<1e-20, recovery=61.9%), city-language d=0.75 (p<1e-18, recovery=34.2%) -- corrected methodology
- Benign/pathological diagnostic: 0% benign across 1,471 instances, mean logit change 3.98 nats (1,000x control)
- Cross-domain absorption variation confirmed: Kruskal-Wallis p=7.4e-66, N=3,545
- 9 definitive negative results transparently documented
- Architecture non-significance: p=0.50-0.75

**Critical failure this iteration:**
- Paper Section 5.2 presents buggy pilot cross-domain patching data (d=-0.91, 0.05% recovery) when the corrected FULL-mode data shows d=1.50, 61.9% recovery. The "concentrated vs. distributed absorption" mechanistic dichotomy -- discussed in abstract, Section 5.2, Section 7.2, and Table 4 -- is based on incorrect data.

---

## 2. Issue Analysis by Category

### SOUNDNESS (2 critical, 1 major)

**[CRITICAL] SND001 -- Stale buggy data in Section 5.2 (paper's second anchoring finding)**
Section 5.2 reports city-continent activation patching recovery at 0.05% (d=-0.91), concluding "the mechanism does not generalize to semantic hierarchies." The corrected FULL-mode data shows city-continent recovery=61.9% (d=1.50, p<1e-20) and city-language recovery=34.2% (d=0.75, p<1e-18). The bug in the pilot zeroed features supporting the correct class rather than absorbing features. The paper's entire "concentrated vs. distributed absorption" narrative is based on incorrect data. The experiment_analysis.md identifies this as "the single most important update for the paper."

**[CRITICAL] SND002 -- Benign/pathological diagnostic circularity**
The diagnostic ablates the parent direction from the child feature's decoder vector. By definition, absorption means the child's decoder carries parent-relevant information. Ablating parent information from a vector known to contain it will always produce large logit changes. The test measures absorption magnitude, not computational independence. The "100% pathological" claim overstates what the experiment demonstrates. A genuine test would ablate the parent feature's activation directly (z_parent=0) or use path patching.

**[MAJOR] SND003 -- Headline claims overstate evidence**
Three headline claims need tempering: (1) "absorption is 100% pathological" -- tested only on city-continent at L24; (2) "4x range" -- only city-language differs significantly from first-letter after Bonferroni; (3) "architecture has no significant effect" -- from a 12-observation Kruskal-Wallis test (absence of evidence, not evidence of absence).

### EXPERIMENT (4 major, 2 minor)

**[MAJOR] EXP001 -- Probe degradation ablation not performed**
This is the single highest-priority missing experiment, identified by both supervisor and critic across multiple iterations. Injecting label noise into first-letter probes to degrade to F1={0.70, 0.80, 0.85, 0.90} and re-measuring absorption would directly quantify whether cross-domain variation is a probe artifact. Estimated 2 GPU-hours.

**[MAJOR] EXP002 -- Benign/pathological tested on only one hierarchy**
The "100% pathological" headline claim is tested exclusively on city-continent (F1=0.87) at L24 with 16k SAE. First-letter (F1=1.0, concentrated mechanism) is untested. Replication on first-letter (~0.5 GPU-hours) would either confirm universality or reveal an important distinction.

**[MAJOR] EXP003 -- Three figure PDFs missing**
fig4_patching_comparison.pdf, fig5_pathological_histogram.pdf, fig6_architecture_comparison.pdf correspond to three of five main contributions. This is a compilation blocker.

**[MAJOR] EXP004 -- Data provenance confusion**
At least three conflicting data sources: consolidation_summary.json (pilot/iter_008), full-mode data files (iter_009), and prior iteration results. Numbers conflict: first-letter weighted F1 is 0.97 (paper) vs 1.0 (data file); benign instances are 1,471 (paper) vs 1,464 (consolidation); first-letter L24_16k absorption is 27.1% (per-unique-word) vs 34.5% (per-instance, consolidation). No authoritative data manifest exists.

**[MINOR] EXP005 -- Token position asymmetry**
First-letter probes use position -6 (sae-spelling convention), RAVEL probes use position -2. Uncontrolled confound.

**[MINOR] EXP006 -- Architecture comparison underpowered**
Only 12 data points at L24, 3 architectures x 4 hierarchies, with width confounds (Matryoshka 32k vs JumpReLU 16k/65k).

### WRITING (3 major, 3 minor)

**[MAJOR] WRI001 -- Section 5.2 narrative entirely wrong**
The "concentrated vs. distributed absorption" dichotomy must be replaced. The corrected data shows universal mechanism (d=0.75-1.50), not domain-specific failure. Requires rewriting abstract, Section 1 anchoring findings, Table 4 verdicts, Section 5.2, and Sections 7.1-7.2.

**[MAJOR] WRI002 -- Table 3 format inconsistency**
Table 3 is inline markdown while all other tables are PDF figures. Will break LaTeX compilation.

**[MAJOR] WRI003 -- Appendix sections referenced but nonexistent**
GAS, CMI, Absorption Tax, and threshold sensitivity appendix sections are all cited in the paper but do not exist.

**[MINOR] WRI004 -- Aggregation method undocumented**
Paper uses per-unique-word absorption rates without explaining this choice. The 28% discrepancy with per-instance rates is never mentioned.

**[MINOR] WRI005 -- Kruskal-Wallis scope unclear**
Paper implies 4-hierarchy comparison but the test covers only 3 RAVEL hierarchies, excluding first-letter.

**[MINOR] WRI006 -- "Near-tautological" tone**
May provoke defensive reactions from prior authors. Consider softer phrasing.

### ANALYSIS (2 major)

**[MAJOR] ANL001 -- Unquantified probe quality confound**
RAVEL probes achieve F1=0.73-0.87. Absorption rates are acknowledged as upper bounds but the bias magnitude is never estimated. The probe-only FN baseline (control 3) is never reported numerically. City-country (F1=0.73, rate=45.1%) is most vulnerable to probe-induced inflation.

**[MAJOR] ANL002 -- Kruskal-Wallis statistical interpretation**
After Bonferroni correction of pairwise tests against first-letter, only city-language reaches significance. City-continent and city-country do NOT differ significantly from first-letter. The "4x range" is descriptive only, not inferentially supported for most hierarchy pairs.

### PIPELINE (1 major)

**[MAJOR] PIPE001 -- validate_integration.py still not implemented**
This is the **10th iteration** recommending this script. The 12.3% hallucinated number from iteration 9 perfectly demonstrates the need. The paper's first-letter weighted F1 discrepancy (0.97 vs 1.0) was caught by the critic, not by automation. Every iteration discovers new data integrity errors of the same structural type.

### EFFICIENCY (analysis below)

---

## 3. Resource Efficiency Assessment

### GPU Utilization Analysis

From gpu_progress.json:
- **13 tasks completed, 0 failed** -- perfect reliability (consecutive 5 iterations zero failures)
- **Total planned GPU time**: ~370 minutes (6.2 hours)
- **Total actual GPU time**: ~196 minutes (3.3 hours) -- 47% faster than planned
- **Largest time overrun**: phase1_probe_training (planned 45 min, actual 109.5 min -- 2.4x overestimate correction needed)
- **Largest time savings**: phase1_hedging_crossdomain (planned 30 min, actual ~0 min -- CPU-only analysis mistakenly categorized as GPU task)
- **GPU idle time**: Minimal -- tasks scheduled sequentially on single GPU with tight turnaround

### Bottleneck Analysis

1. **Writing update bottleneck**: The corrected FULL-mode cross-domain patching data was available but not incorporated into the paper. This is a zero-GPU, zero-cost fix that would have prevented the score regression. The pipeline did not gate paper revision on data freshness.

2. **Missing figure generation**: Three figures have been missing for at least 2 iterations. This is a zero-GPU task that blocks compilation. The pipeline has no enforcement mechanism for figure generation.

3. **Probe degradation ablation**: Identified as highest-priority experiment for 2+ iterations but not yet executed. Estimated 2 GPU-hours would resolve the core uncertainty about cross-domain variation.

### Scheduling Improvement Suggestions

- Run validate_integration.py (or equivalent cross-check) after every paper revision -- zero GPU cost
- Generate all referenced figures immediately after experiment completion, before writing begins
- Gate writing agent's access to data through a single authoritative manifest to prevent stale-data usage
- The probe training task took 109.5 minutes (2.4x planned); future plans should budget 120 minutes for probe training

---

## 4. Quality Trend Assessment

**Score trajectory**: 5.5 (x3) -> 6.5 -> 6.0 -> 6.5 (x3) -> 7.0 -> **6.5** (regression)

The trajectory shows two stagnation loops both broken by experimental execution:
- Iter 0-3 stagnation (5.5) broken by pivot + experiments in iter 4 (+1.0)
- Iter 5-8 stagnation (6.0-6.5) broken by critical experiments in iter 9 (+0.5)

The current regression (7.0 -> 6.5) is not a capabilities regression but a data-freshness failure: the paper text uses buggy pilot data when corrected FULL data exists. This is a fixable editorial issue, not a scientific weakness. The underlying experimental data actually supports a score of 7.5-8.0 if properly incorporated.

**Pattern**: Score improvements correlate perfectly with experiment execution. Score stagnation/regression correlates with writing-only iterations or stale-data-in-paper errors.

---

## 5. Root Cause Analysis

### Why did the score regress?

The score decreased from 7.0 to 6.5 because the supervisor identified that Section 5.2 (a central finding) is based on buggy pilot data. The corrected FULL-mode data was generated in this iteration (phase2_activation_patching_crossdomain, actual time: 34.1 min) but was not incorporated into the paper text. The root cause is a pipeline gap: the writing agent accessed the consolidation summary (which still reports the buggy pilot H7-crossdomain as "FAILED") rather than the authoritative FULL-mode data file.

### Why is validate_integration.py still missing after 10 iterations?

This is the project's most persistent systemic failure. Root causes:
1. The task is always deprioritized relative to experiments and writing
2. No enforcement mechanism exists -- it is purely advisory
3. Each iteration's reflection recommends it, but the recommendation competes with experiment-priority guidance
4. The system lacks a "technical debt" category with mandatory execution

### Why are figures still missing?

Figure generation has no dependency enforcement in the pipeline. The writing agent references figures without checking their existence. The LaTeX compilation check (if any) is not run before review.

---

## 6. Fix Tracking (vs. prev_action_plan.json from iter 7)

### FIXED
- **EXP001 (activation patching)**: EXECUTED. First-letter d=1.33 (p=0.000218), 25 words, 32.5% recovery. Was recurring for 2 iterations.
- **EXP002 (tightened hedging)**: EXECUTED. Strict hedging 0%-22.6% vs prior 98.6% loose. Hedging decomposition: strict/compensatory/persistent.
- **EXP003 (CMI replication at L0=22)**: EXECUTED. rho=0.044, p=0.83 -- definitive null. CMI approach confirmed as failure.
- **EXP004 (threshold sensitivity reporting)**: EXECUTED. CV=0.077, reported in consolidation.
- **SND001 (hedging tautological)**: FIXED. Strict hedging definition now implemented and reported.
- **SND003 (two-interpretation ambiguity)**: RESOLVED by activation patching results (confirms genuine competitive exclusion at d=1.33 for first-letter).
- **ANL001 (probe quality confound in CMI)**: MOOT. CMI approach abandoned after L0=22 null result.
- **PIPE001 (zero experiments executed)**: FIXED. 13 tasks completed this iteration.
- **WRI001 (title misleading)**: FIXED. Title restructured around actual findings.
- **WRI002 (cross-domain novelty contradiction)**: FIXED. Cross-domain results now genuine (absorption rates are statistically significant within RAVEL).

### RECURRING
- **PIPE002 (validate_integration.py)**: 10th iteration recommending. Still not implemented.
- **DATA001 (data provenance confusion)**: Still present. New manifestation: first-letter F1 = 0.97 vs 1.0, benign count 1,471 vs 1,464.
- **WRI005 (abstract length/structure)**: Abstract still needs restructuring around corrected findings.
- **Missing figures**: 3 PDFs still missing (was 2 last iteration; now 3 with the new architecture comparison figure).

### NEW
- **SND-NEW-001**: Buggy pilot data in Section 5.2 (concentrated vs. distributed dichotomy based on incorrect d=-0.91)
- **SND-NEW-002**: Benign/pathological diagnostic circularity
- **EXP-NEW-001**: Probe degradation ablation not performed (highest-priority missing experiment)
- **EXP-NEW-002**: Benign/pathological tested on only one hierarchy
- **ANL-NEW-001**: Kruskal-Wallis scope confusion (3 vs 4 hierarchies)
- **ANL-NEW-002**: Bonferroni correction reveals only city-language significantly differs from first-letter

---

## 7. Pattern Recognition

### Cross-Stage Recurring Issues

1. **Data freshness/provenance (10 iterations)**: Every single iteration has had data integrity or provenance issues. The system generates correct experimental data but fails to propagate it cleanly to the paper. Root cause: no authoritative data manifest, multiple data sources (consolidation, full-mode files, prior iterations), and no automated cross-check.

2. **Headline overclaiming (6 iterations)**: Iter 4-5: causal overclaiming. Iter 6-7: CMI "predicts." Iter 8-9: 100% pathological extrapolation. Iter 10: concentrated vs. distributed based on buggy data. The system systematically produces stronger claims than the evidence warrants. The bias toward publishable headlines overrides evidence-proportional framing.

3. **Missing figures (3 iterations)**: Figures referenced in paper but not generated. No dependency enforcement between experiment completion and figure generation.

### Systemic Strengths

1. **Experimental execution quality**: 13/13 tasks completed, 0 failures. Consecutive 5 iterations with zero experiment failures. Infrastructure is fully reliable.

2. **Statistical rigor**: Bootstrap CIs, Bonferroni correction, Cohen's d, Wilcoxon tests consistently applied. This is the project's most consistent strength.

3. **Negative result honesty**: 9 negative results transparently documented with exact metrics. This has been consistently strong for 10 iterations and is the paper's greatest asset.

4. **Strategic pivot capability**: The project successfully pivoted three times (EDA -> LV -> metric audit -> cross-domain characterization) based on evidence. Each pivot improved the research direction.

---

## 8. Success Pattern Extraction

1. **Experiment-first strategy works**: Score stagnation at 5.5 (x3) was broken by experiments in iter 4. Score stagnation at 6.5 (x3) was broken by experiments in iter 9. The pattern is definitive: experiments drive score improvement; writing-only iterations do not.

2. **Full-mode execution eliminates pilot artifacts**: The FULL-mode cross-domain patching (d=1.50) reversed the pilot finding (d=-0.91). The multi-L0 confound decomposition (1.4%) reversed the single-L0 finding (96.9%). Running experiments at full scale is not optional.

3. **Honest negative results build credibility**: Every reviewer across 10 iterations has praised the negative result reporting. This is the project's most consistent positive differentiator.

4. **Corrected data often strengthens the paper**: The corrected cross-domain patching (mechanism is universal, not domain-specific) is a stronger finding than the buggy pilot result. Data integrity fixes are not just corrections -- they are opportunities for better science.

5. **Layer-dependent absorption is confound-free**: 15x variation with F1>=0.96 probes at all layers. This has survived every review cycle without challenge. The cleanest finding should be the lead.

---

## 9. System Self-Check Response

No self_check_diagnostics.json found. No system self-check response required.
