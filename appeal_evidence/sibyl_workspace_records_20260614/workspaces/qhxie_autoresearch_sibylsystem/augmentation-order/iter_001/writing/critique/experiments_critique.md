# Critique: Experiments (Results)

## Summary Assessment
The section presents pilot-scale results across all five hypotheses with honest reporting of falsifications and inconclusive findings. The hypothesis verdict summary table is a strong organizational device. However, the section suffers from a fundamental credibility problem: all ordering experiments use 10 epochs on 100-sample subsets with 1 seed, while baselines use 30 epochs on full datasets, making the entire results section read as a placeholder rather than a publishable contribution. The section also lacks most of the visual elements planned in the outline and has structural issues with hypothesis numbering.

## Score: 4/10
**Justification**: The intellectual honesty about negative results is commendable and the structure is logical, but no venue would accept results from 100-sample, 10-epoch, 1-seed experiments as evidence for or against any hypothesis. The missing figures (2, 3, 4, 5, 6), missing error bars, missing statistical tests (paired t-tests, ANOVA), and the asymmetric baseline comparison collectively undermine every claim. To reach a 7, the section needs full-scale results, all planned figures, proper statistical tests with p-values and confidence intervals, and resolution of the structural issues identified below.

## Critical Issues

### Issue 1: Pilot-scale results cannot support hypothesis verdicts
- **Location**: Throughout; explicitly acknowledged in line 3 but then contradicted by verdict language
- **Quote**: "H1 is confirmed at pilot scale." / "H5 is falsified."
- **Problem**: The section declares hypotheses "confirmed" or "falsified" based on 10 epochs, 100 samples, and 1 seed. With n=1 seed, no statistical test is possible. The proposal specifies 200 epochs, 5 seeds, paired t-tests with Bonferroni correction, and Cohen's d > 0.2 as the pre-registered analysis plan. None of these are executed. A 0.96% spread on ResNet-18/CIFAR-10 where absolute accuracy is ~10% (near random chance for 10 classes) is meaningless noise, not evidence.
- **Fix**: Either (a) run the full-scale experiments before declaring verdicts, or (b) reframe all verdicts as "directional signals pending full-scale confirmation" and remove the bold "confirmed"/"falsified" language. Add a prominent warning box at the section start stating that no statistical tests were performed due to n=1 seed.

### Issue 2: ResNet-18 CIFAR-10 ordering accuracies are at random chance
- **Location**: Table 1, lines 20-25
- **Quote**: "CJ->Flip->Crop ... 10.97" (CIFAR-10 ResNet-18 column)
- **Problem**: CIFAR-10 has 10 classes, so random chance is 10%. All six orderings achieve 10.01-10.97%, meaning the model has learned essentially nothing. A 0.96% "spread" at the random-chance floor is not evidence of ordering effects -- it is noise in a degenerate training regime (100 samples, 10 epochs). The discussion section (line 64) acknowledges this but the experiments section does not flag it.
- **Fix**: Add explicit text noting that ResNet-18/CIFAR-10 ordering results are at the random-chance floor and should not be interpreted as evidence. Exclude this block from the H1 confirmation count, or at minimum add an asterisk. The current "3/4 blocks exceed 0.5% spread" count is misleading because one of those three blocks is degenerate.

### Issue 3: Missing statistical tests
- **Location**: H1 (line 39), H2 (line 54)
- **Quote**: "H1 is confirmed at pilot scale. Three of four architecture-dataset blocks exhibit accuracy spreads exceeding 0.5%"
- **Problem**: The proposal pre-registered paired t-tests with Bonferroni correction, Cohen's d, and two-way ANOVA for the architecture interaction. The section reports zero p-values, zero confidence intervals, zero effect sizes for H1 and H2. The only statistical tests reported are for H3/H4 (Spearman correlations). With 1 seed, paired t-tests are impossible, but this should be stated explicitly rather than silently omitted.
- **Fix**: State explicitly: "Paired t-tests and ANOVA require 5 seeds and are deferred to full-scale experiments. The spreads reported here are single-seed point estimates without confidence intervals." For H3/H4, report sample sizes and degrees of freedom alongside p-values.

### Issue 4: Asymmetric baseline comparison acknowledged but table layout misleads
- **Location**: Table 1, lines 8-37
- **Quote**: "direct accuracy comparisons between the two sections of Table 1 are not valid"
- **Problem**: Despite the caveat on line 41, presenting orderings (10 epochs, 100 samples) and baselines (30 epochs, full dataset) in the same table invites exactly the comparison the authors disclaim. A reader scanning the table sees orderings at ~10% and baselines at ~92% and draws the obvious (but invalid) conclusion that orderings are catastrophically bad. This framing actively harms the paper.
- **Fix**: Split into two separate tables, or add a visual separator with different column headers making the training conditions unmissable (e.g., "Pilot Conditions" vs. "Reference Conditions"). Better yet, defer baselines to the full-scale experiment where all conditions are matched.

## Major Issues

### Issue 5: H2 hypothesis numbering confusion
- **Location**: Lines 43-54 (labeled "H2"), lines 57-60 (labeled "Reversibility-Sorted Ordering Performance" with no hypothesis number)
- **Quote**: Section header "H2: Architecture-Dependent Ordering Sensitivity" vs. outline which splits this into "H2: Architecture Differential" and "H2 (DPI): Reversibility-Sorted Ordering"
- **Problem**: The proposal defines H2 as architecture-dependent sensitivity, but the outline uses "H2 (DPI)" for the reversibility-sorted ordering. The experiments section drops the hypothesis label entirely for the reversibility analysis, creating a gap in hypothesis coverage. The reversibility-sorted ordering analysis is presented as a standalone subsection rather than as a hypothesis test.
- **Fix**: Either assign the reversibility-sorted analysis to H2 (matching the outline) or create a clear mapping. The hypothesis verdict table (line 140) labels it "H2: Reversibility-sorted wins" which contradicts the section header "H2: Architecture-Dependent Ordering Sensitivity." Pick one definition and be consistent.

### Issue 6: Missing figures -- 5 of 6 planned figures absent
- **Location**: Throughout
- **Problem**: The outline plans Figure 2 (ordering heatmap), Figure 3 (NC scatter), Figure 4 (MI bar chart), Figure 5 (magnitude line plot), and Figure 6 (architecture violin plots). The section references Figure~\ref{fig:magnitude} (line 97) but includes no actual figure. No heatmaps, no scatter plots, no bar charts. The section is entirely tables and text.
- **Fix**: Generate all planned figures. At minimum, the section needs: (a) a heatmap of pairwise ordering accuracy differences (Figure 2), (b) the magnitude inverted-U curve (Figure 5), and (c) the NC scatter plot (Figure 3). These are essential for a results section at any venue.

### Issue 7: Category-level ordering (Tier 2) uses different training conditions than Tier 1
- **Location**: Table 2, line 69
- **Quote**: "pilot: 10 epochs, 5k samples, 1 seed, ResNet-18"
- **Problem**: Tier 2 uses 5k samples while Tier 1 uses 100 samples, but both are labeled "pilot." The 9.01% spread is impressive but could be an artifact of the specific 5k subset chosen with 1 seed. No statistical test is possible. Also, Tier 2 was run only on CIFAR-10 with ResNet-18 -- the outline specifies 2 architectures x 2 datasets.
- **Fix**: Note the training condition mismatch between tiers explicitly. Acknowledge that a single architecture-dataset combination is insufficient to support the claim that "interleaved orderings consistently outperform block orderings" (line 88) -- "consistently" implies multiple settings, but only one was tested.

### Issue 8: H5 falsification based on only 3 data points
- **Location**: Lines 97-99
- **Quote**: "spread does not increase monotonically with magnitude ... M=5: 0.35%, M=9: 0.88%, M=14: 0.00%"
- **Problem**: Three magnitude levels with 1 seed each provide 3 data points. The inverted-U interpretation is one of many possible explanations (including noise). The M=14 convergence to identical accuracy (24.50%) with both orderings at exactly the same value is suspicious -- this could indicate a training collapse or a numerical coincidence with 1 seed.
- **Fix**: Flag the 3-point limitation. Note that the identical M=14 accuracy could indicate training collapse rather than a genuine "noise floor" effect. Report the individual ordering accuracies at each magnitude level (not just the spread) so the reader can assess whether the model is learning at M=14.

### Issue 9: Claim about "the largest signal in our study" without context
- **Location**: Line 86
- **Quote**: "The spread of 9.01 percentage points between the best (interleaved P->G) and worst (all-geometric-first) orderings is the largest signal in our study."
- **Problem**: This 9.01% spread comes from 5k samples, 10 epochs, 1 seed on a single architecture-dataset pair. Calling it "the largest signal in our study" implies comparable reliability to other results, when it actually has the least statistical support (single condition). The phrase "significant practical implications" (line 93) is premature given n=1.
- **Fix**: Qualify: "the largest point-estimate spread in our pilot, though from a single seed on one architecture-dataset combination."

## Minor Issues
- Line 3: "All results reported here are from pilot-scale experiments (10 epochs, 1 seed); full-scale results with 200 epochs and 5 seeds are forthcoming" -- this reads as a disclaimer for a workshop paper, not a submission. Either run the experiments or restructure the narrative.
- Line 39: "this exceeds the accuracy difference between several well-known augmentation methods" -- which methods? Provide specific numbers or remove.
- Line 54: "A formal two-way ANOVA test ... will be reported in the final version" -- do not promise future results in a submission.
- Line 60: "This suggests that the optimal ordering is determined by factors beyond simple reversibility" -- vague. What factors? At least enumerate candidates.
- Line 86: "Three patterns emerge" -- followed by an enumerated list, which is fine, but pattern 3 ("random-per-image ordering is competitive") conflates "competitive with block orderings" and "competitive overall." Random is 25.40% vs. interleaved P->G at 29.39%, a 4% gap.
- Line 93: "significant practical implications" -- banned pattern. Replace with specific implication (e.g., "interleaving adds 9 accuracy points over block ordering in this pilot").
- Lines 105-115: NC_2 values listed as bullet points would be clearer in a small table.
- Line 140: Hypothesis verdict table labels H2 as "Reversibility-sorted wins" but the section header calls H2 "Architecture-Dependent Ordering Sensitivity." This is confusing.
- The section title is "Results" but the file is named `experiments.md` and the outline section is "5. Results." This is fine but verify the final paper uses a consistent title.

## Visual Element Assessment
- [ ] Figures/tables match outline plan -- **FAIL**: 5 of 6 planned figures missing; only Tables 1, 2, and 3 are present
- [ ] All visuals referenced before appearance -- **PARTIAL**: Figure~\ref{fig:magnitude} is referenced but no figure is included
- [ ] Captions are self-explanatory -- **OK** for the tables that exist
- [ ] No text-heavy sections that need visual support -- **FAIL**: H5 magnitude results desperately need the planned line plot; H3 NC results need the scatter plot; architecture comparison needs violin plots

## What Works Well
1. **Honest negative results.** The section forthrightly reports that H3 is falsified and H4 is inconclusive, with specific numbers ($\rho_s = -0.20$, combined $\rho_s = -0.06$). This is rare and valuable -- many papers would bury or spin these results.
2. **Hypothesis verdict summary table.** Table 3 (lines 130-146) is an excellent organizational device that gives reviewers a one-glance summary with pre-registered thresholds alongside observed values. This should be prominent in the final paper.
3. **Mechanistic interpretation of H4 sign flip.** The paragraph on lines 119-126 offering two competing explanations (estimator noise vs. regularization effect on harder tasks) is a thoughtful analysis that goes beyond simple reporting.
