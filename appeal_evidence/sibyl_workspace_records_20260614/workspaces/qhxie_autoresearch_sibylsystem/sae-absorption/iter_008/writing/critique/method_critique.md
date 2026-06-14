# Section Critique: Method (Section 3)

**Score: 7/10**

---

## Summary

The Method section provides a thorough and largely well-organized description of the experimental setup: model, SAEs, feature hierarchies, probe training, absorption measurement, activation patching, and hedging decomposition. The pipeline is clearly adapted from Chanin et al. (2024) and extended to novel domains. Statistical rigor is generally strong with bootstrap CIs and multiple hypothesis tests. However, the section has notable weaknesses in internal consistency, omitted justifications for key design choices, a mismatch between what the Method promises and what the Results deliver, and some presentation issues that would concern a careful NeurIPS/ICML reviewer.

---

## Strengths

1. **Pipeline transparency.** The step-by-step description of false negative detection (Step 1), IG attribution (Step 2), and absorption detection (Step 3) in Section 3.4 is clear and reproducible. The formal notation for AR is well-defined and consistent with the notation table.

2. **Honest quality gate reporting.** Table 1 with probe F1 scores across all hierarchy-layer combinations, combined with the explicit acknowledgment that only first-letter probes pass the strict gate, is exemplary scientific transparency. This directly supports the probe quality confound discussion in Section 4.3.

3. **Statistical protocol.** Bootstrap CIs (10k resamples), paired permutation tests, Wilcoxon signed-rank, and Cohen's d are all appropriate choices. The threshold sensitivity grid (Section 3.4, 5x4 parameter sweep) adds robustness.

4. **Activation patching design.** The control procedure (15 magnitude-matched random latents) is well-designed, and the transparent acknowledgment that the word-discovery procedure biases toward finding absorption is commendable.

5. **Hedging decomposition.** The three-way decomposition (strict, compensatory, persistent) is a genuine methodological contribution that sharpens the prior binary hedging/not-hedging classification.

---

## Weaknesses

### Major Issues

**M1. Inconsistency in probe methodology between first-letter and RAVEL hierarchies.**
The section describes two different probing approaches for first-letter: the `sae_spelling` pipeline (position -2, ICL prompts, `LinearProbe`) and sklearn logistic regression (position -1). Section 3.3 states "We use the higher-quality sklearn probes for absorption measurement to maximize the denominator of false-negative detection," yet Section 3.2 states the first-letter hierarchy uses "the `sae_spelling` pipeline (Chanin et al., 2024) with in-context learning prompts." These two claims conflict. Table 2 in Section 4.1 then refers to "sae_spelling ICL pipeline (F1 >= 0.97)" for first-letter absorption measurement. The reader cannot determine which probe is actually used for the primary first-letter absorption measurements. This must be resolved unambiguously.

**Cross-reference conflict:** The Introduction states "first-letter probes achieve F1 = 0.97 at all layers," but Table 1 in Section 3.2 shows F1 = 0.69 at layer 6 and F1 = 0.31 at layer 12 for first-letter probes. The discrepancy arises because the Introduction refers to sae_spelling probes while Table 1 reports sklearn probes, but this is never made explicit in the text. A reviewer will flag this as a data contradiction.

**M2. The absorption rate definition is class-level, not token-level, but implications are underexplored.**
AR = fraction of K classes with at least one absorbed FN. This means a single absorbed token in a class counts the same as 100 absorbed tokens. For city-country with K=80 classes (many with very few examples), a single false positive in the absorption detection pipeline could inflate AR disproportionately. The section does not discuss the sensitivity of this metric to class count or to low-sample classes. The Experiments section (Section 4.2) later reports city-country AR=18.5%, but with 80 classes, that is ~15 classes with absorption -- some of which may contain just 1-2 tokens. The Method should discuss this or provide a token-level absorption rate alongside the class-level one.

**M3. Activation patching is described only for layer 12, but the Discussion (Section 8.4) draws causal conclusions about absorption in general.**
Section 3.5 specifies: "For each word t with detected absorption at layer 12 (Gemma Scope JumpReLU 16k)." The strongest absorption signal is at layer 24 (34.5%), yet patching is performed at layer 12 where absorption is 5.7%. No justification is provided for this choice. Was it because the sae_spelling probe at L12 was used before the sklearn probes at L24 were trained? The Discussion's limitation section (8.5) acknowledges this, but the Method should justify the design choice upfront rather than leaving the reader confused until page 8.

**M4. Missing justification for threshold defaults.**
The absorption detection thresholds (tau_cos=0.025, tau_gap=1.0) are stated as "defaults" but their provenance is not explained. Are these from Chanin et al. (2024)? Were they tuned on this data? The sensitivity grid partially addresses this, but the initial choice needs grounding. A reviewer will ask why 0.025 and not 0.05 or 0.01 for the cosine threshold.

### Minor Issues

**m1. Table 1 and Figure tab1_probe_quality redundancy.**
The section includes both Table 1 (inline) and "Figure tab1_probe_quality" referencing a PDF. These appear to contain the same information. One should be removed to save space.

**m2. RAVEL dataset sizes vary without explanation.**
Section 3.2 reports n=2039 (city-continent), n=1881 (city-country), n=1859 (city-language) from RAVEL. The differences come from "filtering entities with missing country labels," but filtering should affect city-country and city-language (which depend on country/language labels), not city-continent. Why does city-continent have more entities? Is the filtering applied differently per hierarchy? This needs one sentence of clarification.

**m3. Hedging decomposition uses a different SAE than the main experiments.**
Section 3.6 specifies L0_base=22 with L0_target=176 (8x multiplier). The Gemma Scope JumpReLU SAEs operate at L0=75-87. How is L0_base=22 achieved? Is this a different SAE (BatchTopK from SAEBench has L0=20)? Or is L0 being artificially modified? This is critical information that is completely missing.

**m4. The "200 contexts per word" for activation patching is stated but not justified.**
Why 200? Is this a power calculation result, a computational budget constraint, or a convention from prior work? Similarly, the choice of 15 control features and the 10% magnitude matching tolerance are not justified.

**m5. Cross-reference to Figure 2 is unclear.**
The section opens by referencing "Figure 2 illustrates the full pipeline" but the figure placeholder points to a markdown description file, not an actual generated figure. If this figure does not yet exist, the section reads as incomplete.

---

## Cross-Section Consistency Check

| Check | Status | Details |
|-------|--------|---------|
| Notation vs. notation.md | PASS | All symbols (x^(L), AR, FN, IG, tau_cos, tau_gap, d_i, w_y) consistent with notation table. |
| Terminology vs. glossary.md | PASS | Uses "latent" (not "neuron"), "feature absorption," "SAEBench" -- all per glossary. |
| Method vs. Introduction claims | PARTIAL FAIL | Intro claims "F1=0.97 at all layers" for first-letter but Table 1 shows F1=0.69/0.31 at L6/L12. Inconsistency due to mixing probe types (see M1). |
| Method vs. Experiments data | PARTIAL FAIL | Section 3.4 reports n=87 FN in threshold sensitivity, but Section 4.1 Table 2 shows n_FN values of 4-30 per SAE config. The n=87 appears to be from a specific (unspecified) config. Clarify which. |
| Method vs. Discussion limitations | PASS | Discussion Section 8.5 correctly identifies the L12-only patching limitation described here. |
| Method vs. Outline | MINOR GAP | Outline specifies "C=1.0" for logistic regression; Method Section 3.3 says "C in {0.01, 0.1, 1.0, 10.0} selected by 5-fold stratified cross-validation." This is an improvement over the outline but creates an inconsistency with the plan's fixed C=1.0. The Method's version is better. |
| AR definition vs. Experiments | PASS | AR formula matches usage in Section 4. |
| Hedging decomposition vs. Section 5 | PARTIAL MISMATCH | Section 3.6 describes the decomposition but references "n=304 first-letter false negatives" nowhere; that number appears in Section 5.2 of the outline but not in the Method. Section 5.2 uses these numbers. The Method should state the sample it applies to. |
| Activation patching stats vs. Section 5.1 | PASS | Statistical tests listed in 3.5 match those reported in 5.1. |

---

## Specific Recommendations

1. **Resolve the probe duality (M1).** Add a dedicated paragraph or table explicitly mapping: "For first-letter absorption measurement, we use [X] probe at [Y] position. For cross-domain absorption, we use [Z] probe. Table 1 reports sklearn probe quality for cross-hierarchy comparison; sae_spelling probe quality is reported separately in Appendix [N]." Ensure the Introduction's F1 claims match.

2. **Add a token-level absorption rate (M2).** Define AR_token = |{t in FN : t classified as absorbed}| / |{t : y_hat_raw = y}| and report it alongside the class-level AR. This is especially important for city-country (K=80, many low-count classes).

3. **Justify the L12 activation patching choice (M3).** One sentence suffices: "We performed activation patching at layer 12 because this was the layer with available probes at the time of experiment design; the subsequent discovery that L24 shows 15x higher absorption motivates future L24 patching studies."

4. **Cite the threshold provenance (M4).** If from Chanin et al., say so. If modified, explain why.

5. **Clarify the L0_base=22 in hedging (m3).** State which SAE has L0~22 (BatchTopK 16k from SAEBench) or explain the mechanism for operating at non-native L0.

6. **Trim redundant Table 1 / Figure tab1 (m1).** Keep the PDF figure for visual appeal; remove the inline markdown table, or vice versa.

---

## Score Justification

| Criterion | Score (1-10) | Weight |
|-----------|:---:|:---:|
| Technical correctness | 7 | 25% |
| Completeness of method description | 7 | 25% |
| Internal consistency (within section) | 6 | 15% |
| Cross-section consistency | 6 | 15% |
| Writing clarity and organization | 8 | 10% |
| Reproducibility | 8 | 10% |
| **Weighted total** | **7.0** | |

The section is solid and mostly reproducible, but the probe methodology confusion (M1), the unjustified layer choice for activation patching (M3), and the unexplained L0_base=22 in hedging (m3) are issues that a thorough reviewer would catch. Resolving these would bring the score to 8-9.
