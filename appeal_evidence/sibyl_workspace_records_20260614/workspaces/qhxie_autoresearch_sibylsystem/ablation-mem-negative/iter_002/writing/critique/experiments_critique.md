# Critique: Experiments Section

**Reviewer:** sibyl-section-critic
**Section:** 4. Experiments
**Score:** 5/10
**Verdict:** Major revision needed. The standalone section file is an abridged version that omits the pilot-scale results and scaling analysis present in the full paper, creating serious consistency issues. Additionally, there are data inconsistencies, missing statistical details, and structural problems.

---

## 1. Critical Issue: Section Mismatch Between Standalone and Full Paper

The standalone `experiments.md` (sections/experiments.md) is a **severely abridged version** compared to the Experiments section in `paper.md`. The standalone version:
- Omits E1 (Pilot-Scale UAD with 100 features, F1 = 0.704)
- Omits E2 (Precision Collapse Analysis table)
- Re-labels experiments: E2→Ablations, E3→Cross-Layer, E4→False Positive, E5→Statistical, E6→DFDA
- Has only 6 experiments vs. 7 in the full paper

**This is a critical inconsistency.** The standalone section presents a flat "UAD fails" narrative without the pilot-scale success that makes the scaling collapse meaningful. The pilot result (F1 = 0.704 at 100 features) is essential context---without it, the full-scale F1 = 0.007 is just a number, not a collapse.

**Fix:** The standalone `experiments.md` must be updated to match the full paper structure (E1-E7) or the paper must be reconciled with the standalone sections.

---

## 2. Data Inconsistency: Same-Cluster Pair Counts

The standalone section reports different numbers than the full paper for the same experiment:

| Metric | Standalone (experiments.md) | Full Paper (paper.md) |
|--------|----------------------------|----------------------|
| Full UAD same-cluster pairs | 3,702 | 3,702 |
| Full UAD detected pairs | 541 | 541 |
| Full UAD true positives | 2 | 2 |
| Ablations: Full UAD (Ward) pairs | **7,608** | **7,608** |
| Ablations: K-means pairs | **7,648** | **7,648** |
| Ablations: 24K features | **154,858** | **154,858** |

The ablation table in the standalone section uses "Same-Cluster Pairs" as the column header but the values match the full paper. However, the standalone section's Table 1 shows 3,702 same-cluster pairs for "Full (500 features)" while the ablation table shows 7,608 for "Full UAD (Ward linkage, top 500 features)" on 500 features. **These should be the same experiment but report different pair counts (3,702 vs 7,608).**

This inconsistency is present in both versions and needs explanation. Is 3,702 the number of pairs *after* the top-10% co-occurrence filter, while 7,608 is the raw same-cluster count before filtering? The text does not clarify this distinction.

**Fix:** Explicitly state that 3,702 = pairs passing the top-10% co-occurrence threshold, while 7,608 = all pairs within the same clusters. Or reconcile these numbers if they should be identical.

---

## 3. Missing Pilot-Scale Results in Standalone

The standalone section jumps directly to "UAD vs. Random Baseline" at full scale (F1 = 0.007) without the pilot-scale result (F1 = 0.704). This undermines the paper's central claim about "precision collapse with scale." A reader of only the standalone section would see:
- "UAD achieves F1 = 0.007" -- so what? Maybe it was always bad.

The pilot result establishes that UAD *can* work at small scale, making the collapse a meaningful finding rather than just a confirmation of poor performance.

**Fix:** Add E1 (Pilot-Scale UAD) to the standalone section, or at minimum reference the pilot result in the E1/UAD vs. Random section.

---

## 4. Terminology and Notation Inconsistencies

### 4.1 "Feature Collision" vs. "Absorption"
The Methodology section (method.md) defines "Feature Collision" as the formal metric but the Experiments section consistently uses "absorption" as the ground-truth label. The Methodology says:
> "Absorption. Following Chanin et al. [2024], absorption measures parent-feature suppression of child features under co-occurrence. This requires hierarchy labels and is our gold-standard metric where available."

But the experiments report "True Positives" as features "known to participate in absorption from Chanin et al.'s protocol." Are collision and absorption used interchangeably? The paper should be consistent.

**Fix:** Clarify whether the ground-truth labels are "collision" (same feature, multiple concepts) or "absorption" (parent suppresses child). These are related but distinct phenomena.

### 4.2 "UAD (Full)" vs. "Full UAD"
The standalone section uses "UAD (Full)" in Table 1 but "Full UAD" in the ablation table. Be consistent.

### 4.3 "pilot config" vs. "Full (500 features)"
The full paper's Table 2 labels 500 features as "(pilot config)" which is confusing---the pilot is 100 features. The standalone section correctly labels it "Full (500 features)."

**Fix:** Remove "(pilot config)" from the 500-feature row in the full paper.

---

## 5. Missing Statistical and Methodological Details

### 5.1 Random Baseline Method
The section states "random baseline achieves F1 = 0.0075 (± 0.005)" but does not explain:
- How many random trials were averaged?
- What does ± 0.005 represent (std dev? std error?)?
- The random baseline in the full paper lacks the ± notation entirely

**Fix:** Specify "mean ± standard deviation over N trials" or similar.

### 5.2 Cross-Layer Validation (E3/E4) Lacks Detail
The standalone section states: "Testing UAD on layers 0, 2, 4, 6, 8, 10 shows consistent failure: all layers produce F1 ≈ 0.007."

But:
- Was this at 100-feature or 500-feature scale?
- Is there a table? The full paper says pilot scale gives F1 ≈ 0.65--0.75 at all layers.
- The standalone omits this pilot-scale cross-layer result entirely.

**Fix:** Include a small table or at least clarify the scale at which cross-layer validation was performed.

### 5.3 False Positive Analysis Methodology
"99.6% of detected pairs are features that are semantically related" -- how was this categorization performed? Manual inspection? Automated semantic similarity? The number of categories?

**Fix:** Add a sentence describing the categorization methodology (e.g., "We manually inspected 100 random false positive pairs and categorized them by semantic relationship type").

---

## 6. DFDA Section (E6/E7) Issues

### 6.1 Parameter Count Inconsistency
The standalone section says "using 388 total parameters" while the Methodology says "~97 parameters per pair." With 5 letter pairs (c, i, o, p, u), 5 × 97 = 485, not 388. Or if the MLP has a different structure:
- Methodology: "2-layer, 16 hidden units" → input × 16 + 16 × 1 + biases
- For a single feature pair with input dim = 1 (z excluding one feature): 1×16 + 16 + 16×1 + 1 = 49 parameters
- For 5 pairs: 5 × 49 = 245, or if input dim is larger...

**Fix:** Reconcile the parameter count or explain the architecture more precisely.

### 6.2 "Parent-Positive Evaluation" Terminology
The section title "E6: DFDA Parent-Positive Evaluation" is unclear. What does "Parent-Positive" mean? Is this a standard term?

**Fix:** Use clearer terminology or define "parent-positive" on first use.

---

## 7. Table Formatting Issues

### 7.1 Table 1 Column Alignment
The "Same-Cluster Pairs" column header has a colon for right-alignment (`-------------------:`) but the values below are not consistently formatted. The "Random Baseline" row has "--" which should align right.

### 7.2 Missing Table Numbers
The standalone section does not label tables (no "Table 1", "Table 2"). The full paper does.

**Fix:** Add table captions/numbers to the standalone section.

---

## 8. Structural Recommendations

1. **Reorder for narrative flow:** The current order (Setup → UAD vs Random → Ablations → Cross-Layer → False Positive → Statistical → DFDA → Summary) is acceptable, but the standalone version would benefit from starting with the pilot result to establish the "before/after" narrative.

2. **Add a figure:** The scaling collapse (F1 from 0.704 to 0.007) would be much more impactful as a log-scale plot. Consider adding a figure showing precision vs. feature set size.

3. **Connect to Methodology:** The experiments section rarely references the formal definitions from Section 3. For example, the suppression signal Δ_supp is defined but never measured or discussed in the experiments.

---

## 9. Minor Issues

- **Line 9:** "precision 0.37%" -- should have backslash escape: `0.37\%` for LaTeX compatibility (already present, good)
- **Line 15:** "2.1" true positives for random baseline -- fractional true positives are acceptable for averaged baselines but should be explained
- **Line 40:** "$p$ = 0.87" -- this p-value is surprisingly high (not significant), which supports the claim but should be noted explicitly: "the high p-value indicates no significant difference from random"
- **Section 4.8 Summary:** "Two findings stand out" -- but three findings are listed in the full paper. The standalone should match.

---

## Summary of Required Changes

| Priority | Issue | Location |
|----------|-------|----------|
| **Critical** | Reconcile standalone section with full paper (missing E1, E2) | experiments.md |
| **Critical** | Explain 3,702 vs 7,608 pair count discrepancy | Table 1 + Ablations table |
| **High** | Clarify collision vs. absorption terminology | Throughout |
| **High** | Add false positive categorization methodology | Section 4.5 |
| **Medium** | Reconcile DFDA parameter count (388 vs ~97/pair) | Section 4.7 |
| **Medium** | Add random baseline trial details | Section 4.2 |
| **Medium** | Add table numbers/captions | All tables |
| **Low** | Consistent "UAD (Full)" vs "Full UAD" naming | Tables |
| **Low** | Explain "parent-positive" terminology | Section 4.7 title |

**Overall Assessment:** The experiments section contains solid empirical results but suffers from a critical version mismatch between the standalone file and the full paper, data inconsistencies that need explanation, and missing methodological details. With these fixes, the section would score 7-8/10.
