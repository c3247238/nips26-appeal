# Critique: Experiments

## Summary Assessment
The Experiments section is compact and technically honest, reporting negative results (ARS_v2 does not improve over encoder_norm alone) alongside the positives. The OMP falsification result is presented with pre-committed criteria and unambiguous data. However, the section as written deviates significantly from the outline's specification — the mandatory "Experimental Setup" subsection (5.1) is missing entirely, the ordering of subsections does not match the outline, figure references are absent throughout, and two data discrepancies exist between the text and the underlying JSON results.

## Score: 5/10
**Justification**: Solid data and honest reporting save this section from a lower score, but the structural omissions are critical for a NeurIPS submission. The missing setup subsection forces readers to hunt through the Methods section for basic experimental parameters. Every figure reference is absent. Two numeric inconsistencies (EncNorm means and CI bounds) undermine reproducibility trust. To reach 7/10, the section needs: (1) the setup subsection added, (2) all six figure references restored, (3) the two numeric errors corrected, (4) the ARS_v2 negative result surfaced (currently suppressed from this section entirely).

---

## Critical Issues

### Issue 1: Experimental Setup subsection entirely absent
- **Location**: Section 4 (Experiments) header
- **Quote**: Section begins directly with "## 4.1 Encoder Weight Norm Detection Performance" — no setup subsection
- **Problem**: The outline mandates a "5.1 Experimental Setup" subsection containing model details (GPT-2-small, SAELens releases), label sources (Chanin IG FeatureAbsorptionCalculator, n_pos=18 / n_neg=24,558), dataset (OpenWebText 10k tokens), and primary metrics (AUROC, AUPRC, DeLong). Without this subsection, a reader encountering the section standalone cannot reproduce or assess the experiments. This information is currently scattered across the Methods section (Section 3.4), which is correct per the outline — but that content is not repeated here where readers need it as context for the tables.
- **Fix**: Add a concise "## 4.1 Experimental Setup" subsection (3–5 sentences maximum) that states: GPT-2-small (117M) with two SAELens SAEs (gpt2-small-res-jb Standard/L1 d_SAE=24,576 at resid_pre; gpt2-small-resid-post-v5-32k TopK-32k d_SAE=32,768 at resid_post); gold labels via Chanin IG FeatureAbsorptionCalculator (n_pos=18, n_neg=24,558 Standard; n_pos=77, n_neg=32,691 TopK-32k); OpenWebText corpus (10k tokens for co-occurrence); primary metric AUROC with 95% bootstrap CI. This is a 6-line addition that makes the section self-contained.

### Issue 2: All six figures are unreferenced — figures do not exist in the section text
- **Location**: Entire section
- **Quote**: The words "Figure 4", "Figure 5", "Figure 6" do not appear anywhere in the section text
- **Problem**: The outline's Figure & Table Plan specifies Figure 4 (ROC curves, Section 5.2), Figure 5 (ablation bar chart, Section 5.3), and Figure 6 (OMP per-letter bar chart, Section 5.4). None are referenced. At a NeurIPS workshop, reviewers expect every visual to be cited before it appears; unreferenced figures are treated as supplementary noise or, worse, a sign the figures do not exist.
- **Fix**: In Section 4.1/4.2 (detection performance), add "(see Figure 4)" after the AUROC comparison sentence. In Section 4.2 (co-occurrence), add "(see Figure 5)" before or after the Precision@50 claim. In Section 4.3 (OMP), add "(see Figure 6)" after the absorption rate table. These are single-phrase insertions at the end of each relevant paragraph. Actual figure generation must accompany this.

### Issue 3: EncNorm mean for absorbed latents is inconsistent with source data
- **Location**: Section 4.1, last sentence of the first bold paragraph
- **Quote**: "Mean EncNorm for absorbed latents: $3.26$; for non-absorbed: $2.58$ (ratio $= 1.26$)."
- **Problem**: The A2 JSON (`task3_layer_monotonicity`, L6 entry) reports `mean_enc_norm_absorbed = 3.263` and `mean_enc_norm_non_absorbed = 2.576`, giving ratio 3.263/2.576 = 1.267. The overall SAE mean (all latents, not just non-absorbed) from `task1_spearman_full_latents` is 2.577. The section writes "non-absorbed: 2.58" which conflates non-absorbed mean with overall mean — these are not the same value. Additionally, the ratio is 1.267, not 1.26. This is a rounding and labeling error that affects reproducibility.
- **Fix**: Change to: "Mean EncNorm for absorbed latents: $3.26$; overall SAE mean: $2.58$ (ratio $= 1.267$)." Or use the actual non-absorbed mean directly: "for non-absorbed: $2.58$" is acceptable if verified against the non-absorbed-only statistics; the JSON summary shows non-absorbed mean at L6 is 2.576, so the value is correct but the label should be "non-absorbed" not "overall". The ratio should read $1.267$, not $1.26$.

---

## Major Issues

### Issue 4: AUROC confidence interval for EncNorm Standard/L1 does not match source data
- **Location**: Table 1, EncNorm Standard/L1 row
- **Quote**: `AUROC = 0.757, 95% CI [0.634, 0.864]`
- **Problem**: The A3 JSON reports `auroc_ci95 = [0.665, 0.849]` for Standard/L1 encoder_norm. The text writes `[0.634, 0.864]` — both bounds are wrong. The lower bound is too low (0.634 vs 0.665) and the upper bound is too high (0.864 vs 0.849). This appears to be a stale number from an earlier iteration.
- **Fix**: Replace `[0.634, 0.864]` with `[0.665, 0.849]` to match the A3 experiment output (`auroc_ci95: [0.6654916666666667, 0.8492361111111112]`).

### Issue 5: AUROC confidence interval for TopK-32k EncNorm does not match source data
- **Location**: Table 1, EncNorm TopK-32k row
- **Quote**: `AUROC = 0.837, 95% CI [0.746, 0.911]`
- **Problem**: The A3 JSON reports `auroc_ci95 = [0.807, 0.870]` for TopK-32k encoder_norm. The text writes `[0.746, 0.911]` — wider interval with incorrect bounds. The actual bootstrap CI is substantially narrower due to the larger n_pos=77.
- **Fix**: Replace `[0.746, 0.911]` with `[0.807, 0.870]` to match the A3 JSON output (`auroc_ci95: [0.8072704545454545, 0.8698618506493506]`).

### Issue 6: ARS_v2 negative result is entirely absent from the Experiments section
- **Location**: Section 4.2 (Co-occurrence Jaccard Signal) — should be present, is not
- **Quote**: Section 4.2 discusses O_jaccard AUROC=0.721 and Spearman independence, then immediately pivots to "two-signal audit strategy." The B2 experiment also produced ARS_v2 AUROC=0.586 (DeLong z=-2.455, p=0.993 vs encoder_norm) — a clear negative result about product combinations.
- **Problem**: The outline (Section 5.3) explicitly requires: "ARS_v2 (encoder_norm × A_cooccur) AUROC=0.586 — does NOT improve over encoder_norm alone; DeLong z=-2.455, p=0.993" and "Figure 5: Ablation bar chart." This is a key negative result that practitioners need to avoid a dead-end path. Its omission makes the results appear uniformly positive.
- **Fix**: Add one paragraph at the end of Section 4.2: "Combining encoder_norm with the directed co-occurrence asymmetry ($A_\text{cooccur}$) into a product score (ARS_v2) does not improve detection: AUROC $= 0.586$ (DeLong vs.\ encoder_norm: $z = -2.455$, $p = 0.993$). Product formulations dilute both signals; the two detectors should be applied independently."

### Issue 7: Precision@50 enrichment calculation is not reproducible from the data presented
- **Location**: Section 4.2, first paragraph
- **Quote**: "Precision@50 $= 0.100$ — the top 50 latents by $O_\text{Jaccard}$ contain $5$ of the 18 absorbed latents, a $13.9\times$ enrichment over random."
- **Problem**: The claim that P@50=0.10 contains 5 absorbed latents is correct (5/50=0.10). The enrichment calculation is: (5/50) / (18/24576) = 0.10 / 0.000733 = 136.5×, not 13.9×. Alternatively, enrichment over the random P@50 baseline = 18/24576 × 50 ≈ 0.037 expected absorbed in top 50 by chance, enrichment = 5/0.037 = 136×. The 13.9× figure appears to be a miscalculation — possibly (18/24576) × 13.9 ≈ 0.010, which is the base rate, not the enrichment.
- **Fix**: Verify the enrichment calculation. If using random baseline: enrichment = P@50_observed / P@50_random = 0.10 / (18/24576) ≈ 136×. If using a different baseline (e.g., 50-percentile random draw = 50 × 18/24576 ≈ 0.037), report it explicitly. Replace "13.9×" with the correct value or show the calculation.

### Issue 8: "Hook confound note" placement breaks the logical flow of Section 4.4
- **Location**: Section 4.4, final paragraph
- **Quote**: "*Hook confound note*: The Standard-24k and TopK-32k SAEs differ in both dictionary size and hook point..."
- **Problem**: The hook confound is already documented in Section 3.4 (Method) and Table 1 footnote. Repeating it here as an italicized note at the end of Section 4.4 treats it as an afterthought. Worse, it undermines the F1 claim — the section first says "Two-thirds of absorbed features have geometric counterparts in the wider dictionary" and then concedes the comparison "may reflect the different activation spaces rather than true feature recovery." This undercuts the result.
- **Fix**: Move the hook confound note to a "Limitations of this analysis" bullet within the paragraph that presents the finding, and reframe it: "Two-thirds of absorbed features (12/18, cos\_sim $> 0.80$) have direction-aligned counterparts in the TopK-32k SAE. This comparison carries a hook confound (resid\_pre vs.\ resid\_post), which we cannot resolve without a matched-hook wider SAE experiment (see Section 5.2 limitations). Treating this as a lower bound on geometric recovery, one-third (6/18) have no close match even cross-hook, suggesting genuine semantic coverage gaps." This acknowledges the limit without burying the result.

---

## Minor Issues

- **Section 4.1, DeLong citation**: "DeLong test: $p = 0.0012$" — report the z-statistic as well. Standard practice: "DeLong test: $z = X$, $p = 0.0012$." The A3 JSON should contain the z-statistic; add it.
- **Section 4.1, "mean $L_0 = 55.7$"**: This value appears in Method (Section 3.3) but not in Experiments. If mentioned here for the OMP cross-reference, it needs a forward reference or reminder. Currently no mention of mean L0 in experiments at all — needed for reproducibility context of the OMP result.
- **Table 2 (OMP)**: The absorption rates shown are 0.978/0.867/0.733 for a/e/s feedforward, but the C2 JSON reports 0.967, 1.000, 0.967 for feedforward (letters a, e, s). Table 2 values differ — the 0.867 for "e" does not match the JSON 1.000 for letter "e." Either Table 2 is reporting different letters or different tokens; clarify which. The mean "0.859" in the table also does not match the JSON mean_absorption_rate of 0.978.
- **Notation consistency**: The section uses "EncNorm" (capitalized, no subscript) in the text but the notation.md establishes `enc_norm(j)` as the preferred formula notation and "encoder weight norm" as the preferred full-form English name. Section 3 (Method) correctly uses "EncNorm" as an abbreviation after defining it. Ensure first-use in experiments adds the full form: "encoder weight norm (EncNorm)."
- **Section 4.3, "OMP ($K=53$)"**: No mention of how K=53 was selected (matched to feedforward mean L0=55.7). Without this, the matched-sparsity claim is opaque. One sentence suffices: "$K = 53$ was chosen to match the feedforward mean $L_0 = 55.7$ (rounded)."
- **Glossary deviation**: The glossary specifies "OMP oracle" as acceptable shorthand only after "Orthogonal Matching Pursuit" on first use. Table 2 header uses "OMP AbsRate" without a first-use expansion. Section 4.3 header says "Amortization Gap Experiment: H2 Falsified" without referencing OMP in the title — inconsistent with the outline's "Amortization Gap Oracle Experiment."

---

## Visual Element Assessment
- [ ] **Figures/tables match outline plan**: No. Figure 4 (ROC curves), Figure 5 (ablation bar chart), and Figure 6 (OMP per-letter results) are specified in the outline for this section but are not referenced anywhere in the text.
- [ ] **All visuals referenced before appearance**: Cannot assess — no figure references exist in the section.
- [ ] **Captions are self-explanatory**: Cannot assess — captions are not present in the markdown draft (presumably to be added at LaTeX stage), but no caption text is drafted.
- [ ] **No text-heavy sections that need visual support**: Section 4.3 (OMP) presents a table but no figure. The outline calls for Figure 6 (per-letter bar chart) to make the 0% reduction immediately visible. Without it, the result requires careful table reading. Section 4.2 similarly lacks Figure 5.

**Missing figures are the most significant visual gap.** The ROC curves in Figure 4 are the primary visual evidence for the detector comparison; their absence means the current draft cannot serve as a complete experiments section.

---

## What Works Well

1. **Pre-committed falsification criterion is stated explicitly and honored.** Section 4.3 states "The pre-committed falsification criterion was: OMP $\geq 80\%$ of feedforward absorption rate $\Rightarrow$ H2 falsified" and then confirms the outcome with "ratio $= 1.000$ across all three letters, $p_\text{Fisher} > 0.99$." This is methodologically exemplary — the outcome is unambiguous and the criterion was set before the experiment.

2. **Hook confound is disclosed proactively.** The cross-architecture disclaimer ("The AUROC cannot be directly compared to the Standard/L1 result due to the hook confound") is stated in Section 4.1 before readers might draw incorrect comparisons. This matches the Level of disclosure required at a rigorous venue.

3. **OMP technical validity control is included.** Section 4.3 explicitly verifies that OMP achieves lower reconstruction MSE than feedforward (0.219 vs 0.242, $R^2 = 0.87$ vs 0.84), ruling out the possibility that OMP simply fails. This preempts the obvious reviewer objection.
