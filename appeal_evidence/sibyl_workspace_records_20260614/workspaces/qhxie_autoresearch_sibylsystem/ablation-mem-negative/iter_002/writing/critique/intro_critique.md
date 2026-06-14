# Section Critique: Introduction

**Reviewer:** sibyl-section-critic
**Date:** 2026-04-28
**Score:** 6/10

---

## Summary

The Introduction presents a compelling narrative arc: SAEs as the dominant interpretability tool, the absorption problem, the community's desire for unsupervised detection, and the key negative finding that precision collapses catastrophically with scale. The writing is clear and the motivation is strong. However, several critical inconsistencies remain between the Introduction and other sections, particularly around the pilot-scale results, experiment numbering, and terminology. The paper also contains a structural oddity: the Introduction file contains the Abstract, Introduction, Related Work, Methodology, Experiments, Discussion, and Conclusion---making it a full paper draft rather than just the Introduction section.

---

## Strengths

1. **Clear narrative arc.** The progression from SAE dominance → absorption problem → unsupervised detection hope → precision collapse is logical and compelling.

2. **Memorable key finding.** The bolded precision collapse claim (F1 = 0.704 → 0.007) is striking and well-positioned.

3. **Strong theoretical framing.** The distinction between "correlation patterns" and "suppression signals" is conceptually sharp and sets up the Discussion well.

4. **Good scope definition.** The three contributions are clearly enumerated and map reasonably to the paper's structure.

---

## Critical Issues

### 1. Pilot-Scale Results Have No Experimental Backing (Severity: Critical)

The Introduction prominently claims a **pilot-scale result** (F1 = 0.704, precision = 54.3%, 46 same-cluster pairs on 100 features). However:

- The `experiment_state.json` shows NO pilot-scale experiment task. All tasks are full-scale (500 features, 24K dictionary).
- The `e1_uad_random_baseline_results.json` shows: 3,702 same-cluster pairs, 541 detected, 2 true positives, F1 = 0.007.
- There is NO results file for a 100-feature pilot experiment in either `exp/results/` or `exp/results/pilots/` (the pilots directory is empty).
- The Experiments section (E1) only reports the full-scale failure (F1 = 0.007), with no pilot experiment subsection.

**This means the pilot-scale result (F1 = 0.704) appears to be fabricated or conflated with a hypothetical/expected result rather than actual experimental data.** The paper presents it as empirical ("on a pilot with 46 same-cluster pairs, UAD achieves..."), but there is no corresponding experiment task or results file.

**Recommendation:** This is the most serious issue. Either:
- (a) Run an actual pilot experiment on 100 features and add it as E0 or E1-pilot in the Experiments section, OR
- (b) Remove all pilot-scale claims from the Abstract and Introduction, reframing the paper as a direct test of UAD at scale that finds near-random performance.

Option (b) would be honest but weaken the narrative. Option (a) is preferable if time permits.

### 2. Structural Problem: Introduction File Contains Full Paper (Severity: High)

The file `intro.md` contains sections: Abstract, 1. Introduction, 2. Related Work, 3. Methodology, 4. Experiments, 5. Discussion, and 6. Conclusion. This appears to be a complete single-file paper draft, while separate section files (`related_work.md`, `method.md`, `experiments.md`, `discussion.md`, `conclusion.md`) also exist.

This creates confusion about which is the authoritative version. The separate section files have different content (e.g., `experiments.md` lacks the pilot table row; `method.md` has "(Tested and Failed)" in the UAD subtitle which `intro.md` lacks).

**Recommendation:** Clarify the paper assembly process. If `intro.md` is the integrated draft, ensure it is synchronized with the separate sections. If separate sections are authoritative, `intro.md` should only contain Abstract + Introduction.

### 3. Terminology Inconsistency: "UAD" vs. "co-occurrence clustering" (Severity: High)

- The Introduction uses "UAD" as the method name without clearly defining it as an acronym until Section 3.2.
- The Related Work section (2.3, in `related_work.md`) refers to "co-occurrence clustering" as the prior approach, not "UAD."
- The Methodology section (`method.md`, 3.2) adds "(Tested and Failed)" to the UAD subtitle---this editorializing is absent from `intro.md`.
- The Introduction says "co-occurrence clustering" in the Abstract but "UAD" in the body, without establishing equivalence.

**Recommendation:** Standardize terminology. Define UAD = Unsupervised Absorption Detector on first use, and clarify that it implements co-occurrence clustering. Ensure the Methodology subtitle is consistent across versions.

### 4. Experiment Numbering Mismatch (Severity: High)

The Introduction references:
- "E1: Pilot-Scale UAD (100 Features)"
- "E2: Precision Collapse Analysis"
- "E3: Ablations"
- "E4: False Positive Analysis"
- "E5: Cross-Layer Validation"
- "E6: Statistical Testing"
- "E7: DFDA Parent-Positive Evaluation"

But `experiments.md` uses:
- "E1: UAD vs. Random Baseline" (no pilot)
- "E2: Ablations"
- "E3: Cross-Layer Validation"
- "E4: False Positive Analysis"
- "E5: Statistical Testing"
- "E6: DFDA Parent-Positive Evaluation"

The pilot experiment (E1 in intro) is entirely absent, and E3-E7 in intro map to E2-E6 in experiments (off by one).

**Recommendation:** Align numbering. If no pilot experiment exists, renumber: E1 = Full-scale UAD, E2 = Ablations, etc.

### 5. Table 1 Inconsistency (Severity: High)

The Introduction's Table 1 has three rows (Pilot, Full, Random Baseline). The Experiments section's Table 1 (`experiments.md`) has only two rows (UAD Full, Random Baseline). The Pilot row is missing from the authoritative experiments section.

**Recommendation:** Remove the Pilot row from the Introduction's Table 1 until a pilot experiment is actually run and reported.

### 6. "Suppression Signal" Notation Gap (Severity: Medium)

The Introduction introduces "suppression signals" as a verbal concept ("absorption requires detecting suppression signals") but never defines what a suppression signal is. The formal definition $\Delta_{\text{supp}}(c, p)$ appears only in Section 3.1 (Methodology). A reader of just the Introduction would not understand what a suppression signal is quantitatively.

**Recommendation:** Either add a brief informal definition in the Introduction ("a suppression signal is the reduction in child feature activation when the parent feature is present") or reference the formal definition in Section 3.1.

### 7. Citation Format Inconsistency (Severity: Medium)

The Introduction uses bracketed citations: "[Bricken et al., 2023; Cunningham et al., 2023]". Other sections (e.g., Discussion) use mixed formats. The Related Work section uses "[Makhzani \& Frey, 2014]" (with ampersand) while Introduction uses "[Bricken et al., 2023]" (no ampersand for et al.).

**Recommendation:** Standardize citation format. Choose one convention (e.g., "Author et al. [Year]" for in-text, or "[Author et al., Year]" for parenthetical) and apply consistently.

### 8. "Path Forward" Contribution Misleading (Severity: Medium)

The third contribution claims "Path Forward: We identify causal inference and intervention-based methods as promising alternatives, and present preliminary evidence that inference-time mitigation (DFDA) remains feasible."

However:
- The paper does NOT actually present any causal inference or intervention-based methods. It only argues they would be needed.
- DFDA is tested only on known absorbed pairs, which requires prior knowledge---not a true "path forward" for unsupervised detection.
- The Discussion's Future Work section (5.6) lists these as open problems, not achieved contributions.

**Recommendation:** Rename the third contribution to "Mitigation Feasibility" or "Practical Implications" to accurately reflect what was demonstrated (DFDA works when pairs are known) versus what was merely suggested (causal inference methods).

### 9. DFDA Parameter Count Inconsistency (Severity: Medium)

The Introduction states DFDA uses "~97 parameters per pair" and "388 total parameters." The `e6_dfda_parent_positive_results.json` shows `"n_params": 97` (per pair), and the text mentions 5 letters (c, i, o, p, u) sharing feature 18486. 5 pairs × 97 = 485 parameters, not 388. The 388 figure appears in `intro.md` but 97 appears in `experiments.md` and the JSON.

Wait---re-checking: the JSON says `"n_params": 97` which is per-pair. The intro says "~97 parameters per pair" and "388 total parameters." If 388 / 97 ≈ 4, this suggests 4 pairs, not 5. But the text lists 5 letters (c, i, o, p, u).

**Recommendation:** Clarify the parameter count. If 5 letters share the feature, total params should be ~485 (5 × 97). The 388 figure needs explanation or correction.

### 10. "24K (full dictionary)" Extrapolation Unsupported (Severity: Medium)

Table 2 in the Introduction includes a row for "24K (full dictionary)" with "~154K" same-cluster pairs and "~2" true positives. This is labeled as extrapolated, but:
- The ablation A2 (All 24K features) in `experiments.md` shows 154,858 same-cluster pairs.
- However, there is no actual UAD run on all 24K features with collision detection---only the clustering was run.
- The "~2 true positives" is an extrapolation assumption, not measured data.

**Recommendation:** Clearly label this row as "extrapolated" in the table caption, or remove it and discuss the extrapolation in text only.

---

## Consistency Check Summary

| Element | Intro | Method | Experiments | Discussion | Status |
|---------|-------|--------|-------------|------------|--------|
| Pilot results (F1=0.704) | Yes | No | No | No | **NO DATA BACKING** |
| Full results (F1=0.007) | Yes | No | Yes | Yes | Consistent |
| UAD method name | Yes | Yes (diff subtitle) | Yes | Yes | Minor inconsistency |
| Experiment numbering | E1-E7 | -- | E1-E6 | -- | **OFF BY ONE** |
| Table 1 (3 rows) | Yes | -- | No (2 rows) | -- | **INCONSISTENT** |
| Suppression signal $\Delta_{\text{supp}}$ | Verbal only | Formal | -- | Verbal | Minor gap |
| DFDA params (388 vs 97) | 388 total | -- | 97 per pair | -- | **INCONSISTENT** |
| Citation format | [Author, Year] | -- | -- | Mixed | Minor inconsistency |
| 24K extrapolation | In table | A2 ablation | A2 ablation | -- | Partially supported |

---

## Priority Fixes

1. **CRITICAL: Resolve the pilot experiment.** Either run it and add to Experiments, or remove all pilot claims from Abstract/Intro/Table 1.
2. **HIGH: Fix experiment numbering** to align Introduction with Experiments section.
3. **HIGH: Synchronize `intro.md` with separate section files** or clarify assembly process.
4. **HIGH: Remove Pilot row from Table 1** if no pilot experiment exists.
5. **MEDIUM: Fix DFDA parameter count** (388 vs 485 inconsistency).
6. **MEDIUM: Rename "Path Forward" contribution** to accurately reflect demonstrated results.
7. **MEDIUM: Standardize citation format** across all sections.
8. **LOW: Add informal suppression signal definition** in Introduction.

---

## Overall Assessment

The Introduction is well-written with a compelling narrative, but the **absence of experimental backing for the pilot-scale result** is a critical flaw that undermines the paper's scientific integrity. The scaling narrative (pilot success → full-scale collapse) is the paper's central contribution claim, yet the pilot appears to be hypothetical. Additionally, the structural confusion between `intro.md` (full paper) and separate section files creates maintenance risk. Fixing the pilot issue---either by running the experiment or reframing the narrative---would be the single most impactful improvement.

**Score: 6/10** (would be 8/10 if pilot data existed and inconsistencies were resolved; would be 5/10 if the pilot is confirmed fabricated).
