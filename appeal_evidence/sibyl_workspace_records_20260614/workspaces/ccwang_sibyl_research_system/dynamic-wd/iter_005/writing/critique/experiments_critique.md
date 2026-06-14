# Critique: Experiments (Sections 4 and 5) — REVISION ROUND

## Summary Assessment

This is the REVISION round critique. Cross-referenced against review.md four flagged issues: (1) no_wd seed count error, (2) Figure 6 missing, (3) rho_low data underreported, (4) run count verification. Positive changes from Round 1: Section 5.3 (Ratio Regime Analysis) now exists; Section 5.7 diagnostic analysis now reports exact Spearman statistics (r_s = 0.03, p = 0.81, n = 28); Section 5.6 matched-rho seed 42 footnote is appropriately clarified. **None of the four review.md critical/major issues have been fixed.** The Table 3 no_wd footnote still claims "seed 456 missing" despite source data showing all 3 seeds present. Figure 6 is still absent and the body numbering jumps 5→7. The rho_low row still omits the cwd_hard data point. The "150+ runs" claim remains unsubstantiated. These constitute three factual errors (one provably false, two misleadingly incomplete) that will draw immediate reviewer scrutiny.

## Score: 5/10

**Justification**: Round 1 score was 6/10. The section fixed the ratio regime analysis gap (now Section 5.3 with a proper 4-row table), diagnostic Spearman statistics, and matched-rho seed 42 footnote. These improvements warrant acknowledging. However, the four issues explicitly flagged in review.md as Critical or Major remain unaddressed. Two are factually verifiable errors (wrong footnote, missing figure with broken numbering) and two are incomplete data reports (rho_low, run count). Together they constitute a data-integrity credibility problem that drops the score to 5/10. Fixing just the Table 3 footnote and rho_low data would restore 6/10 immediately; adding Figure 6 or renumbering would bring it to 7/10.

---

## REVISION FOCUS: Four Review.md Issues — Status

### Review Issue 1: Table 3 no_wd Seed Count Footnote — **NOT FIXED (Critical)**

**Current state (Section 5.2, line 304):**
> `$^\dagger$2 seeds only (seed 456 missing).`

This is provably false. Source data in `full/nobn/../vgg_16_bn/no_wd/seed_456/` (or equivalent path per review.md) shows all three seeds present: seed_42 = 92.01, seed_123 = 92.0, seed_456 = 92.08. The reported mean (92.03 ± 0.04) already matches the 3-seed computation (mean = 92.03, std = 0.044), meaning the numbers are correct but the footnote text asserting "seed 456 missing" is a factual error. A reviewer who inspects the appendix or asks to verify seed counts will find the footnote directly contradicted by the data. The fix is a single deletion: remove `$^\dagger$` from the no-WD row and remove the footnote line entirely, then update the table header from "no-WD uses 2 seeds" to "3 seeds." This takes 30 seconds and the numbers require no change.

**Why it matters at this stage:** A false data-provenance statement in the final-round revision is worse than in a first draft — it signals the authors did not verify their own footnotes before submission.

---

### Review Issue 2: Figure 6 Missing and Figure Numbering Gap — **NOT FIXED (Major)**

**Current state:** Section 5.5 (NoBN vs. BN Ablation) has no figure reference. The body-text figure sequence is Figure 4 (Section 5.2), Figure 5 (Section 5.4), Figure 7 (Section 5.7). Figure 6 does not appear anywhere in the body. The figure list in the paper's TODO comment still shows "[TODO: NoBN vs. BN Cohen's d bar chart]."

**Two independent problems:**

1. **Broken numbering:** Readers encountering "Figure 7" after "Figure 5" will immediately check for Figure 6. Its absence will be interpreted as either a typo or an incomplete manuscript. Either interpretation damages credibility.

2. **NoBN section lacks visual support:** Section 5.5 reports Cohen's d = 9.14 — a very large effect size — but presents no figure. Cohen's d = 9.14 between BN and NoBN on constant WD is arguably the most visually striking single-number result in the experiments. A bar chart showing this (with error bars) would take a paragraph's worth of explanation and compress it to immediate visual impact.

**Minimum fix (lower effort):** Renumber Figure 7 to Figure 6 and update the diagnostic panel reference throughout the paper. This eliminates the numbering gap without generating a new figure. Cost: 10 minutes.

**Preferred fix (higher impact):** Generate the NoBN vs. BN Cohen's d bar chart as Figure 6; shift the diagnostic panel to Figure 7. Cost: generate the figure from the data in Table 5.

---

### Review Issue 3: rho_low Data Underreported — **NOT FIXED (Major)**

**Current state (Section 5.3, ratio regime table):**

| $\rho$ regime | Source | $\rho$ value | Phi spread |
|---|---|---|---|
| $\rho$-low (AdamW) | Pilot sweep | $\sim$0.05 | constant only: 90.13 ± 0.07 |

Two problems persist:

**3a. cwd_hard data omitted.** Review.md confirms `rho_low/cwd_hard/seed_42 = 90.09` exists in source data. The paper says "constant only" but CWD hard data is available. Including it gives a partial Phi spread of 90.13 − 90.09 = 0.04% at rho_low, which is directionally consistent with Theorem 1 (even lower spread than rho_standard's 0.25%). This is a positive result that supports the paper's argument and should be reported, not omitted.

**3b. Numerical coincidence not explained.** The rho_low constant value (90.13 ± 0.07) is identical to rho_standard constant (90.13 ± 0.31, Table 1). This identical number in two different rows with different standard deviations is precisely the pattern that triggers copy-paste suspicion in a careful reviewer. No explanatory parenthetical appears in the current text. The fix is a single sentence: "The rho_low constant accuracy (90.13 ± 0.07) coincidentally matches the rho_standard constant (90.13 ± 0.31 from Table 1); the two experiments use different learning rates and are independent."

---

### Review Issue 4: "150+ Runs" Claim Unverified — **NOT FIXED (Major)**

**Current state:** "totaling 150+ runs" appears in the abstract. Section 5.1 Theorem 1 validation paragraph states 5 complete configurations. The claim "150+ runs" is never broken down anywhere in the paper.

**Arithmetic check:** Complete 200-epoch runs = ResNet-20 (7 methods × 3 seeds × 4 configs = 84) + VGG-16-BN (7 methods × 3 seeds × 1 config = 21) = 105 total. Reaching 150+ requires counting partial/pilot runs. Section 5.3's rho_low and rho_high pilots, Section 5.5's NoBN partial data, and Section 5.6's matched-rho partial data could add 40–50 more, but these are experiments with 1 seed or 5 epochs — not the "systematic evaluation" the abstract implies.

**Required fix:** Either (a) add a footnote: "105 completed 200-epoch runs plus ~45 pilot and partial runs totaling 150+" so readers understand what the count includes, or (b) revise the abstract claim to "105 completed runs across 5 configurations."

---

## Changes From Round 1 — What Was Fixed

The following Round 1 issues are now resolved and should not be re-raised:

1. **Section 5.3 (Ratio Regime Analysis) now exists.** The section was entirely absent in Round 1; it now presents a 4-row spread-vs-rho table with appropriate data-gap acknowledgment. The table structure is correct. ✓
2. **Diagnostic statistics now specific.** Section 5.7 now reports Spearman r_s = 0.03, p = 0.81, n = 28 for CSI vs. accuracy, and r_s = 0.05, p = 0.72 for AIS vs. accuracy. ✓
3. **Matched-rho seed 42 footnote is clarified.** Table 6 now correctly says "$^\dagger$Seed 42 constant ran for only 5 epochs (pilot configuration); we cannot determine from this data whether full training would converge" — no longer claims "training instability." ✓
4. **"5 complete configurations" count is internally consistent.** The paper no longer claims "7/7 predictions" inconsistently with "5 complete configurations." ✓
5. **Section 5.5 BN/NoBN qualifiers improved.** "within 1-std margin" added for the 0.10% gap. ✓

---

## Remaining Round 1 Issues (Still Present)

The following Round 1 issues were NOT fixed and are still present:

### Issue: Table Numbering Inverted vs. Outline

The outline specifies Table 2 = VGG-16-BN results, Table 3 = statistical tests. The paper has Table 2 = statistical tests (Section 5.1), Table 3 = VGG-16-BN (Section 5.2). This inversion is still present. It is a consistency issue rather than a factual error — the tables are correctly labeled within the text — but the outline should be updated or the tables renumbered. **Priority: Low** (does not affect reviewer evaluation of data quality).

### Issue: NoBN metric inconsistency (best vs. final test accuracy)

Section 5.5 Table 5 reports "ResNet-20-NoBN | 87.74 ± 0.20" for constant WD. Round 1 raised the concern that `best_test_acc` for NoBN seed_42 is 87.97 while `final_test_acc` is 87.74 — the reported value corresponds to `final_test_acc`, inconsistent with Table 1 (which uses `best_test_acc`). This issue remains unresolved. Verify the metric used across all NoBN seeds and ensure consistency with Table 1. **Priority: Moderate** (affects cross-table comparability).

---

## Critical Issues

- **Location**: Between Section 5.2 and 5.3; outline Section 5.3
- **Quote**: Outline specifies "Section 5.3: Ratio Regime Analysis — Figure 2: Method spread vs. log(rho) with three data points." This section does not exist in experiments.md.
- **Problem**: The ratio-regime analysis is the central empirical claim of the paper (the "differentiation" regime at elevated rho). The outline planned it as Section 5.3 with a dedicated figure showing the three-regime phase diagram. Instead, Section 5.3 in the actual text covers "SGD vs. AdamW Effect Size," which partially overlaps but does not present the three-regime framework (inhibition/transition/differentiation) or the rho_low data point. The Introduction (intro.md, line 37) explicitly promises: "As shown in Figure 1, method spread increases monotonically with the gradient-to-weight ratio ρ, with a phase transition from the 'inhibition' regime (ρ < 0.1, spread < 0.1%) through 'transition'..." but Figure 1 is a teaser in the intro; the experiments section should contain the supporting analysis. Without this section, the paper cannot validate its central theoretical claim experimentally.
- **Fix**: Add Section 5.3 "Ratio Regime Analysis" that explicitly presents: (a) rho_low constant result (90.13% ± 0.07 from pilots/rho_sweep/rho_low/summary.json), (b) rho_standard result (90.13% ± 0.31 from main table), (c) SGD as a low-rho proxy (0.91% spread at rho ≈ 0.005), (d) rho_high data gap clearly noted. Plot or tabulate the spread-vs-rho curve. Reference Figure 2 (the regime figure currently described in the outline as being in the Theory section — clarify which figure goes where). This is the most important missing element.

### Issue 2: Table numbering inverted relative to outline

- **Location**: Sections 5.1 and 5.2
- **Quote**: Section 5.1 introduces "Table 1" (main accuracy) and "Table 2" (statistical significance). Section 5.2 introduces "Table 3" (VGG-16-BN results).
- **Problem**: The outline's Figure & Table Plan specifies: Table 1 = Main Accuracy Results, Table 2 = VGG-16-BN Results, Table 3 = Statistical Tests. In the actual text, Table 2 and Table 3 are swapped — statistical tests appear before VGG results. This creates a cross-reference failure: any reader (or LaTeX label) following the outline's numbering will find the wrong table. More importantly, statistical tests are logically secondary to the VGG results (they both support the null result), and the current ordering has the statistical analysis interrupting the presentation of experimental results.
- **Fix**: Renumber so VGG-16-BN results are Table 2 (immediately following the main accuracy table) and statistical tests are Table 3, or update the outline's Table Plan to reflect the actual ordering. The latter requires confirming that all cross-references in intro.md, method.md, discussion.md, and conclusion.md are consistent. A uniform decision is needed before the final draft.

### Issue 3: NoBN constant accuracy discrepancy — best vs. final test accuracy conflated

- **Location**: Section 5.4, Table 4
- **Quote**: "ResNet-20-NoBN | 87.74 ± 0.20 | 87.64 ± 0.17 | +0.10 | 0.50"
- **Problem**: The raw data file `full/nobn/cifar10/resnet20_nobn/constant/seed_42/summary.json` shows `best_test_acc: 87.97` and `final_test_acc: 87.74`. The value 87.74 in the table corresponds to `final_test_acc`, not `best_test_acc`. The main accuracy Table 1 for ResNet-20/CIFAR-10 uses best test accuracy (the section header says "best test accuracy"), making this table internally inconsistent. If the NoBN row uses final accuracy, it underreports NoBN performance by ~0.2% relative to the BN rows. This may not change the conclusion (constant still slightly outperforms CWD), but it introduces measurement inconsistency between tables.
- **Fix**: Confirm which metric (best vs. final) is used for all NoBN runs across all 3 seeds, and use the same metric as Table 1. If best_test_acc, recompute the NoBN constant mean across seeds 42, 123, 456 using best_test_acc values.

---

## Major Issues

### Issue 4: Missing Figure 6 — NoBN vs. BN Cohen's d visualization

- **Location**: Section 5.4 (NoBN vs. BN Ablation)
- **Quote**: Section 5.4 presents Table 4 with Cohen's d = 9.14 but contains no figure reference.
- **Problem**: The outline's Figure & Table Plan explicitly specifies "Figure 6: NoBN vs. BN Cohen's d (Section: Experiments) — bar chart showing Cohen's d for constant method between BN and NoBN." This figure is completely absent from the section. The section ends with a theoretical interpretation paragraph but no visual support. A Cohen's d of 9.14 is a striking result (very large effect size), and a figure would make this immediately clear to readers.
- **Fix**: Either generate and reference Figure 6, or explicitly note in the section that this figure is deferred to the appendix. If deferred, add a note to the figure comment block at the bottom of the file.

### Issue 5: Matched-rho SGD seed 42 description is imprecise

- **Location**: Section 4.3, Section 5.5
- **Quote (Section 4.3)**: "The ρ_high regime produced only 5-epoch pilot data (77.69% at epoch 5) before training instability terminated the full run."
- **Quote (Section 5.5)**: "Seed 42 constant ran only 5 epochs (training instability at high λ = 5 × 10^{-3}); excluded from mean."
- **Problem**: The raw data shows `epochs_completed: 5` for matched_rho_sgd/constant/seed_42, which is a pilot run configuration — it ran exactly 5 epochs as designed, not a "terminated" full run. The rho_high pilot also ran exactly 5 epochs by design. The section conflates a planned pilot (5 epochs) with a "training instability" termination event. A reviewer would ask: was seed_42 a 5-epoch pilot or did a 200-epoch run crash? The footnote in Table 5 ("Seed 42 constant ran only 5 epochs") is ambiguous — if the plan was 200 epochs and it ran only 5, "instability" is a reasonable inference, but the current text implies a full run failed rather than clarifying this was a pilot.
- **Fix**: Clarify explicitly: "The matched-rho SGD constant seed_42 experiment ran for only 5 epochs (pilot configuration). Its 76.12% accuracy reflects an early training checkpoint; we cannot determine from this data alone whether full training would converge. Seeds 123 and 456 completed 200 epochs stably (90.94% and 90.89%)." Remove the causal claim about "training instability" unless there is evidence the run was terminated mid-execution rather than completed as a 5-epoch pilot.

### Issue 6: "7/7 empirical predictions confirmed" claim is not supported in the experiments section

- **Location**: Section 5.1 (referenced from method.md line 59; absent from experiments.md)
- **Quote (method.md)**: "Across all 7 tested configurations—{AdamW, SGD} × {CIFAR-10, CIFAR-100} × {ResNet-20} plus VGG-16-BN/CIFAR-10—constant WD matches or outperforms CWD (all p > 0.05 after Bonferroni correction), confirming all 7 predictions from Theorem 1."
- **Problem**: The experiments section does not explicitly map these 7 configurations or confirm the prediction count. The count is: AdamW×CIFAR-10×ResNet-20 (1), AdamW×CIFAR-100×ResNet-20 (2), SGD×CIFAR-10×ResNet-20 (3), SGD×CIFAR-100×ResNet-20 (4), AdamW×CIFAR-10×VGG-16-BN (5) — that is 5 complete configurations. NoBN and matched-rho SGD are incomplete. It is unclear how the method section arrives at 7. The experiments section should explicitly enumerate the 7 predictions and their confirmation status in a table or list, both to support the method section's claim and to give readers a clear accounting.
- **Fix**: Add a brief "Theorem 1 Validation" paragraph in Section 5.1 or 5.6 that explicitly lists all 7 predictions and their outcomes. If only 5 are complete, the claim should be "5 of 5 complete configurations confirm the prediction; 2 configurations (NoBN, matched-rho SGD) are incomplete."

### Issue 7: CSI and AIS diagnostic analysis lacks any specific numbers

- **Location**: Section 5.6
- **Quote**: "No correlation (Spearman ρ ≈ 0) between CSI and accuracy across methods."
- **Problem**: "≈ 0" is not a number. The section should report the actual Spearman correlation coefficient, its confidence interval, and the sample size (number of (method, configuration) pairs). Similarly, "AIS values cluster in [0.18, 0.59]" is asserted but not backed by a table or inline citation to specific experiment logs. The diagnostic section is the weakest in terms of evidence specificity.
- **Fix**: Report the exact Spearman ρ value (e.g., "Spearman ρ = 0.03, p = 0.81, n = 28 method-configuration pairs") for both CSI and AIS correlations with accuracy. Add a parenthetical citing which specific experiment logs these values come from.

---

## Minor Issues

- **Section 5.2, line "Figure 4"**: The text says "As shown in Figure 4, the accuracy distributions of all 7 methods overlap substantially." The multi-architecture figure is referenced after the text describing VGG results. The outline's Figure & Table Plan places this as Figure 4. However, Section 5.3 references "Figure 5" and Section 5.6 references "Figure 7 (diagnostic panel)". The outline's diagnostic panel is Figure 7, which is consistent — but there is no reference to Figure 6 anywhere in the text. Verify the figure numbering sequence is gapless: Figure 4 (multi-arch), Figure 5 (SGD vs AdamW spread), Figure 6 (NoBN Cohen's d — missing), Figure 7 (diagnostic panel). A missing Figure 6 reference is a structural gap.
- **Section 4.1**: "Both architectures use batch normalization (BN)" — this sentence appears before mentioning ResNet-20-NoBN in the same paragraph. The phrasing is confusing; the NoBN variant should be introduced with an explicit "except ResNet-20-NoBN, which replaces BN with identity."
- **Section 5.2, "phi spread"**: "The VGG-16-BN Phi spread is 0.16%—smaller than ResNet-20's 0.25%." The Phi spread in Table 3 is listed as 0.16, but the text elsewhere (including the outline, which says 0.23%) gives a different number. The outline Section 5.2 says "VGG phi spread: ~0.23% (across available methods)." If no_wd is missing seed_456 and therefore uses a 2-seed average, the spread calculation may differ from the 3-seed spread. The source of the 0.16% vs 0.23% discrepancy should be clarified.
- **Section 5.4**: "2.4 percentage-point accuracy drop" for NoBN vs. BN — this number (90.13 - 87.74 = 2.39) uses final_test_acc for NoBN but best_test_acc for BN (see Issue 3). With consistent best_test_acc, the drop would be 90.13 - 87.97 = 2.16%. This minor inconsistency follows from Issue 3.
- **Section 5.5, Table 5**: "CWD | 90.81 | --- | --- | 90.81" lists only 1 seed for CWD (seed 42), but the text says "cwd_hard: seed_42=90.81% only." Checking the data: there is no `_DONE` file confirmed for matched_rho_sgd cwd_hard seeds 123 or 456. However, the section does not say whether seed_42 for CWD ran 5 or 200 epochs. If it ran 5 (like the constant seed_42), the 90.81% is also a pilot number and should be flagged with a dagger.
- **Section 4.2, BEM column**: The BEM for "No WD" is listed as 1.0 in the experiments table, but in notation.md and method.md it is defined as |lambda_eff^method - lambda_eff^constant| / lambda_eff^constant. For no_wd, lambda_eff = 0, so BEM = |0 - lambda_eff^constant| / lambda_eff^constant = 1.0. This is correct, but the range in notation.md says [0, 1] while the AIS section description says "AIS is an intrinsic network-dataset property." No inconsistency here, but the BEM value of 1.0 for no_wd is counterintuitive (1.0 = 100% deviation from constant budget); the section would benefit from a one-sentence note: "BEM = 1.0 for no_wd indicates complete removal of the WD budget."
- **Passive voice in Section 5.1**: "Three observations emerge from Table 1" — restructure as "Table 1 reveals three findings:" to be more direct.
- **Terminology**: The section uses "WD method" and "phi modulator" interchangeably. method.md uses "Phi modulator" consistently. The experiments section should pick one and stick to it.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan
  - Figures 4, 5, 7 are referenced correctly.
  - **Figure 6 (NoBN Cohen's d) is missing entirely from the text.**
  - Table numbering (Table 2 = stats, Table 3 = VGG) is inverted vs. the outline plan (Table 2 = VGG, Table 3 = stats).
- [x] All referenced visuals appear before appearance in the narrative (Figures 4, 5, 7 are referenced in the relevant subsections).
- [ ] Captions are self-explanatory — Figure 4's markdown inline caption in the text is not a LaTeX caption; it's a description embedded in the image tag. Confirm these will render as proper captions in LaTeX.
- [x] Diagnostic panel (Figure 7) explicitly states what each subplot shows in the inline description.
- [ ] Section 5.4 (NoBN) has no visual support despite being a complete 200-epoch result set.
- [x] Section 5.3 correctly defers to Figure 5 for the SGD vs. AdamW effect size comparison.

---

## What Works Well

1. **Transparency about data gaps**: Sections 5.4 and 5.5 explicitly caveat incomplete results with footnotes and clear language ("insufficient for statistical conclusions," "preliminary evidence"). This is exactly what a rigorous reviewer expects and it avoids overclaiming.

2. **Statistical rigor in Section 5.1**: Reporting both raw p-values and Bonferroni-corrected p-values alongside Cohen's d for all methods is thorough. The conclusion ("SWD shows the largest effect size (d = 0.88), driven entirely by its 0.25% accuracy deficit, which falls within inter-seed variance") is honest and well-calibrated.

3. **The BEM finding in Section 5.6**: "A 10× variation in effective WD budget produces negligible accuracy change" is a strong, concrete result stated with evidence. This is the section's best single sentence — it leads with the empirical finding and explains it without hedging.
