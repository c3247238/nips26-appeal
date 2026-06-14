# Critique: Results (Experiments)

## Summary Assessment
The Results section presents numerical findings clearly and follows a logical structure from main results through hypothesis tests to controls and replication. However, it contains a **critical factual error** in the hierarchy-specificity claim (all architectures show NH > SH, not "except PAnneal"), misidentifies the architecture with the largest gap, and has several issues with figure references, data provenance, and cross-section consistency with the Method section.

## Score: 5/10
**Justification**: The section earns points for clear presentation of the three main results (Table 1, H1, H2, H3) and generally accurate statistical reporting. It loses significant ground due to the critical factual error in Section 4.3 (wrong exception architecture and wrong largest gap), unsupported claims about tau_fs correlation values, and inconsistencies with the Method section's hypothesis labels. To reach 7/10, fix the factual errors, align hypothesis labels with Method, and clarify data provenance. To reach 9/10, add effect sizes, address the missing Figure 1, and reconcile the section's framing with the proposal's broader scope.

---

## Critical Issues

### Issue 1: Factual Error — "Except PAnneal" Claim Is Wrong
- **Location**: Section 4.3, paragraph 2
- **Quote**: "Every trained architecture except PAnneal shows higher non-hierarchy than hierarchy absorption, with the Standard SAE exhibiting the largest gap (0.416 vs. 0.352)."
- **Problem**: This claim is factually incorrect on both counts. **All seven trained architectures** show higher non-hierarchy (NH) than semantic-hierarchy (SH) absorption, including PAnneal (NH = 0.131, SH = 0.064). Furthermore, the **Standard SAE does NOT exhibit the largest gap** — GatedSAE does (gap = 0.192 vs. Standard's 0.064). The gaps are: BatchTopK 0.039, GatedSAE 0.192, JumpReLU 0.118, Matryoshka 0.130, PAnneal 0.067, Standard 0.064, TopK 0.060.
- **Fix**: Replace with: "Every trained architecture shows higher non-hierarchy than hierarchy absorption. GatedSAE exhibits the largest gap (0.379 vs. 0.188), followed by Matryoshka (0.333 vs. 0.203) and JumpReLU (0.348 vs. 0.230)."

### Issue 2: H2 Hypothesis Label Mismatch with Method Section
- **Location**: Section 4.3 heading and text
- **Quote**: "H2 is rejected: the absorption metric is not specific to hierarchical structure."
- **Problem**: The Method section (3.6) labels the hierarchy-specificity hypothesis as **H2**, but the proposal labels H2 as "Utility Disconnect" (correlation with downstream metrics), which is not tested at all in this paper. The Method section's H1/H2/H3 labels are internally consistent but do not match the proposal's four-hypothesis framework. A reader cross-referencing the proposal will be confused.
- **Fix**: Add a footnote or parenthetical in the Results section acknowledging that the hypothesis numbering follows the Method section's adapted framework, which differs from the proposal's original four-hypothesis design due to scope narrowing.

### Issue 3: Unsupported tau_fs Correlation Values
- **Location**: Section 4.4
- **Quote**: "r = 0.468 at tau_fs = 0.01, 0.463 at 0.03, and 0.471 at 0.05"
- **Problem**: The tau_fs parameter controls feature splitting in the k-sparse probe, but the per-architecture mean absorption scores (which determine the correlation) should be **identical** across tau_fs values because tau_fs only affects how many latents are retained — it does not change the underlying SAE latents or the per-hierarchy absorption computation. The reported differences (0.468 vs. 0.463 vs. 0.471) are small but unexplained. If they come from different data sources or rounding, this should be stated.
- **Fix**: Clarify why the correlations differ across tau_fs. If the differences are due to rounding of the input data, state this. If tau_fs genuinely affects the per-architecture means, explain the mechanism.

---

## Major Issues

### Issue 4: Missing Figure 1 Reference
- **Location**: Entire section
- **Problem**: The outline's Figure & Table Plan includes Figure 1 ("Architecture Ranking Comparison") as a grouped bar chart of first-letter vs. semantic-hierarchy absorption. This figure is never referenced in the Results section. Table 1 serves a similar purpose, but the outline promised both. If Figure 1 was generated but not included, the section should reference it or explain its omission.
- **Fix**: Either add a reference to Figure 1 in Section 4.1 ("Three patterns emerge..." would be a natural place) or remove Figure 1 from the outline's figure plan and consolidate with Table 1.

### Issue 5: Inconsistent Random-SAE Semantic Score Between Sections
- **Location**: Section 4.1 (Table 1) vs. Section 4.5
- **Quote**: Table 1 shows Random semantic-hierarchy = 0.175, non-hierarchy = 0.233. Section 4.5 says "0.175" and "0.233" respectively. The Discussion section (5.3) says "0.352" for semantic-hierarchy.
- **Problem**: The Discussion section claims the Random SAE "produces identical semantic-hierarchy absorption scores (0.352) to the trained Standard SAE." But Table 1 and Section 4.5 both report Random semantic-hierarchy = 0.175, which is NOT identical to Standard's 0.352. This is a **cross-section inconsistency**: the Discussion section appears to be using outdated or incorrect data. The 0.352 value may come from iter_002 (where Random and Standard were indeed identical), but iter_003 corrected this.
- **Fix**: The Discussion section must be corrected to use the iter_003 values (Random = 0.175, Standard = 0.352). The Results section is internally consistent and correct.

### Issue 6: No Effect Size Reported for H2
- **Location**: Section 4.3
- **Quote**: "paired t-test: t = -4.748, p = 0.0032"
- **Problem**: The t-statistic and p-value are reported, but no effect size (e.g., Cohen's d) is given. For a paired t-test with n=7, t = -4.748 corresponds to Cohen's d = -1.79, which is a very large effect. Reporting only the p-value understates the practical significance of the finding.
- **Fix**: Add Cohen's d = -1.79 (or similar) to the H2 results. This strengthens the claim that the hierarchy-specificity failure is not just statistically significant but practically meaningful.

### Issue 7: GPT-2 Replication Is Underdeveloped
- **Location**: Section 4.6
- **Problem**: The GPT-2 replication is described in one short paragraph with only two architectures (Standard, TopK). No statistical test is performed — the section merely states the values are "an order of magnitude lower." The implications are vague ("model-dependent"). The outline promised this as Figure 5, but the figure is not referenced in the section.
- **Fix**: Add a statistical comparison (e.g., ratio test or simple t-test if more architectures were tested). Reference Figure 5 explicitly. Clarify whether the GPT-2 results support or challenge the main findings — currently the paragraph reads as an afterthought.

---

## Minor Issues

- **Section 4.1, paragraph 2**: "Bold indicates best (lowest) score per column" — but lower absorption is not necessarily "best" in this context. The paper's thesis is that absorption scores are suspect, so labeling low scores as "best" subtly undermines the critique. → Remove "best" label; just bold the minimum.

- **Section 4.1, paragraph 3**: "The Random SAE's semantic-hierarchy score (0.175) exceeds that of PAnneal (0.064)" — true, but the more striking comparison is that Random (0.175) falls within the trained range (0.064–0.359), not just that it exceeds one architecture. → Rephrase to emphasize the within-range finding.

- **Section 4.2**: "Including the Random SAE (n = 8) yields r = 0.497 (CI: [-0.206, 0.958])" — the CI is reported without explaining why it matters. The point is that including Random doesn't change the conclusion, but this should be stated explicitly. → Add: "Including Random does not alter the inconclusive assessment."

- **Section 4.3**: "H2 is rejected" — since H2 was one-tailed (expecting SH > NH), the observed direction (NH > SH) is not just a failure to support but an active reversal. → Use "H2 is rejected; the effect is in the opposite direction" for precision.

- **Section 4.4**: "The hierarchy-specificity rejection (t = -4.748, p = 0.003) holds identically across all thresholds" — "identically" is slightly overstated since the per-architecture means are the same but the correlation values differ. → Replace "identically" with "consistently."

- **Section 4.5**: "This pattern implies that the semantic-hierarchy adaptation... captures properties of the base-model residual stream or the probe setup" — "or" is vague. The paper should commit to one explanation or acknowledge both as untested alternatives. → Clarify: "captures properties of the base-model residual stream, the probe setup, or both."

- **Figure captions**: Figure 2 caption says "Pearson r = 0.463 with bootstrap 95% CI [-0.389, 0.981]" but does not specify n = 7. → Add "(n = 7 trained SAEs)" to the caption.

- **Table 1**: The Random SAE is listed last, separated from trained architectures. This is good. But the table does not indicate that Random is a control, not a trained architecture. → Add a footnote or dagger symbol to distinguish the control.

---

## Visual Element Assessment
- [x] Figures/tables match outline plan: Table 1, Figures 2-4 are present and match. Figure 1 is missing from the section text (though planned in outline). Figure 5 is not referenced.
- [x] All visuals referenced before appearance: Table 1 is introduced before data discussion. Figures 2-4 are referenced before their embed lines.
- [ ] Captions are self-explanatory: Figure 2 caption lacks sample size (n=7). Table 1 lacks a control-condition footnote.
- [ ] No text-heavy sections that need visual support: Section 4.6 (GPT-2) is very short and would benefit from Figure 5 being explicitly referenced.

---

## What Works Well

1. **Clear hierarchical structure**: The section moves naturally from main results (Table 1) through the three hypotheses (H1, H2, H3) to controls (Random SAE) and replication (GPT-2). This mirrors the Method section's hypothesis structure and makes the section easy to follow.

2. **Honest reporting of inconclusive results**: Section 4.2 does not overclaim the H1 result. The phrase "H1 is neither supported nor rejected" and the explicit acknowledgment that "the upper bound of 0.981 permits a strong relationship that our sample is simply too small to detect" demonstrate appropriate epistemic humility.

3. **Effective use of Table 1**: The table is well-structured with clear column headers and the Random-SAE anomaly is immediately visible. The three patterns identified in the text (ranking differences, Standard's inversion, Random's within-range score) are all directly supported by the table data.
