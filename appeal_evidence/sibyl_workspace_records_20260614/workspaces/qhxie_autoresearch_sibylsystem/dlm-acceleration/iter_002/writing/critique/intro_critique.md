# Critique: Introduction

## Summary Assessment
The introduction is well-structured and makes a compelling case for the composition gap in DLM acceleration. It leads with concrete numbers (LLaDA speed deficit, the 0.86x destructive interference example) and provides an honest AR comparison upfront. However, there are numerical inconsistencies between the intro and the experiments/method sections, the contribution list is overlong and mixes findings with scope caveats, and the M3 speedup numbers cited in the intro do not match those in the experiments section.

## Score: 7/10
**Justification**: Strong problem framing, honest positioning, and concrete numbers. To reach 8+, the section needs to fix numerical inconsistencies, tighten the contribution list from 6 items to 4, eliminate the scope statement's defensive tone that reads like a rebuttal, and address the mismatch between the opening's "34 TPS" baseline and the method section's "33.8 TPS."

## Critical Issues

### Issue 1: Baseline TPS inconsistency across sections
- **Location**: Paragraph 1, line 3
- **Quote**: "generates text at 34 tokens per second (TPS)"
- **Problem**: The method section (4.5) reports the GSM8K baseline as "33.8 TPS" and the iter_001 summary reports "31.0 +/- 4.0 tok/s." The rounded "34 TPS" is plausible as a round of 33.8 but does not match the 31.0 TPS from the baseline experiment. The AR comparison paragraph also uses "34 TPS" as the denominator for the QAS=3.08 calculation. If the real baseline is 33.8, the claimed QAS=3.08 should be 71/33.8 * 0.96 = 2.02... unless QAS is computed differently. Actually, QAS = S * AccRet = (71/33.8) * (96/71.2) = 2.10 * 1.35 = 2.83, which also does not equal 3.08. The number 3.08 does not self-consistently derive from any baseline TPS value in the paper.
- **Fix**: Recompute QAS=3.08 and show the derivation explicitly. Use the exact baseline TPS from the experimental setup (33.8 TPS on GSM8K) consistently throughout the intro. If the AR baseline QAS is computed using a different TPS baseline (e.g., 34), state so. The mismatch suggests a calculation error that must be traced and corrected.

### Issue 2: M3 speedup number does not match experiments section
- **Location**: Paragraph 3 (The Composition Gap, "No failure characterization"), line 13
- **Quote**: "AR-guided unmasking ($w_g = 0.3$, 1.65x speedup, 102.5% accuracy retention)"
- **Problem**: The experiments section (Table 3) reports M3 at $w_g$=0.3 as 1.68x speedup with 103.9% accuracy retention on GSM8K. The outline (Section 5.1) says 1.65x. The intro uses 1.65x and 102.5%. These are three different numbers for the same configuration. The proposal (iter_001 data) reports 1.68x / 103.9%. The discrepancy undermines trust in all numbers.
- **Fix**: Align all M3 numbers to the full-scale experimental result: 1.68x speedup, 103.9% accuracy retention at $w_g$=0.3 on GSM8K. Update the intro, the outline, and any other references.

### Issue 3: M1 measured speedup inconsistency
- **Location**: Paragraph 3 (The Composition Gap), line 13
- **Quote**: "entropy-based KV caching ($\eta = 0.5$, 1.16x speedup, 94.5% accuracy retention)"
- **Problem**: These numbers match the experiments section for GSM8K. However, the composition example that follows -- "will yield 1.91x speedup or 0.86x slowdown" -- uses 1.16x * 1.65x = 1.91x as the multiplicative expectation. But if M3 is actually 1.68x (per experiments), the multiplicative expectation should be 1.16 * 1.68 = 1.95x, not 1.91x. The 0.86x result is stated as the actual outcome, but the experiments section reports M1+M3 as achieving a TPS of 50.3 vs. baseline 58.5, which is 0.86x of the *HF baseline* -- not of the LLaDA 64-step baseline (33.8 TPS). At 50.3 TPS vs. 33.8 TPS baseline, the actual speedup is 1.49x, not 0.86x. The "0.86x slowdown" appears to be the ratio relative to the HF baseline, not the denoising baseline.
- **Fix**: Clarify the reference point for the 0.86x number. If it means "0.86x of the HuggingFace forward-pass throughput" rather than "0.86x of the 64-step denoising baseline," this must be explicit. If 0.86x means the composition is slower than M3 alone (50.3 TPS vs. M3's ~56.8 TPS), state that clearly: "the composition is 11% slower than M3 alone." Recalculate the multiplicative expectation using consistent M3 speedup numbers.

## Major Issues

### Issue 4: Contribution list is too long and mixes findings with methodology
- **Location**: Lines 24-36, "Contributions" subsection
- **Quote**: Six numbered contributions spanning 12 lines
- **Problem**: Six contributions is excessive for an introduction. Items 1, 2, and 3 are genuine contributions. Item 4 (cross-model validation) is methodology, not a contribution. Item 5 (honest AR comparison) is a finding, and item 6 (practical recipes) is a deliverable. The list dilutes impact by mixing levels of abstraction. Top venues expect 3-4 crisp contributions.
- **Fix**: Consolidate to 4 contributions: (1) first systematic composition study with Ortho/QAS metrics, (2) composition taxonomy (M1+IGSD orthogonal, M3+IGSD task-dependent, M1+M3 interference), (3) IGSD as composability study vehicle, (4) honest AR comparison and practical recipes. Fold cross-model validation into contribution 2 ("confirmed transferable on Dream-7B") and fold recipes into contribution 4.

### Issue 5: Scope Statement reads as preemptive defense, not as scientific boundary-setting
- **Location**: Lines 37-41, "Scope Statement"
- **Quote**: "ComposeAccel is an analysis paper, not a methods paper. IGSD is a composability study vehicle --- simple by design so that composition effects are attributable to method interactions rather than implementation complexity."
- **Problem**: The opening sentence is defensive ("not a methods paper"). The M1 kernel-gap explanation ("15.2x framework overhead due to eager attention incompatibility on RTX PRO 6000 Blackwell GPUs") reads as an excuse. Reviewers will read this as acknowledging weakness rather than as boundary-setting. The HumanEval/MBPP explanations belong in the experimental setup, not the intro.
- **Fix**: Reframe as positive scope: "ComposeAccel focuses on composition behavior rather than individual method engineering. We use LLaDA-8B-Instruct as the primary model and GSM8K/MATH500 as benchmarks with sufficient signal strength; Section 4.5 details setup constraints." Move M1 kernel-gap, HumanEval/MBPP exclusions, and M2 NO_GO details to the method or discussion sections where they have full context.

### Issue 6: The IGSD contribution description front-loads implementation detail over insight
- **Location**: Contribution 3, lines 29
- **Quote**: "IGSD (Information-Geometric Step Distillation). A 50-line training-free step scheduler using inter-step logit KL divergence to partition tokens into draft and refine phases. At $T_{\text{draft}} = 32$, $\tau = 0.9$, IGSD achieves 1.71x speedup at 67.8% accuracy retention (QAS = 1.16 on GSM8K). $T_{\text{draft}} = 32$ is Pareto-optimal; the per-step KL divergence profile is monotonically decreasing (not inverted-U as hypothesized), explaining IGSD's low sensitivity to the threshold $\tau$."
- **Problem**: This contribution description is 4 sentences and contains detailed ablation findings (monotonic KL profile, tau insensitivity) that belong in the results section preview, not in the contribution bullet. A contribution should state what was done and why it matters, not present results. The "not inverted-U as hypothesized" reference to H6 is meaningless to a first-time reader who has not seen the hypotheses.
- **Fix**: Shorten to: "IGSD (Information-Geometric Step Distillation): a 50-line training-free step scheduler that partitions denoising into draft and refine phases using inter-step KL divergence. IGSD achieves 1.71x speedup (QAS = 1.16) and composes near-orthogonally with M1 (Ortho = 0.96)." Move the monotonic-KL finding and tau-insensitivity to the results roadmap at the end.

### Issue 7: Figure 1 is referenced but its content is not adequately previewed
- **Location**: Line 17 and lines 19 (figure caption)
- **Quote**: "Figure 1 shows the speed-quality landscape that motivates this work."
- **Problem**: The figure is described narratively ("Individual methods cluster in distinct regions... M1+M3 pair falls below both individual methods") but the reader has not yet been told what M1, M3, or IGSD stand for. These abbreviations are introduced in the Composition Gap section but only as parenthetical references. A reader encountering Figure 1 for the first time needs a one-line reminder of what each method does.
- **Fix**: Add a parenthetical when first introducing Figure 1: "Figure 1 shows the speed-quality landscape: M1 (KV caching), IGSD (step distillation), and M3 (AR-guided decoding) cluster in distinct regions when applied individually, and their pairwise compositions span the space unpredictably."

## Minor Issues
- **Line 3**: "In Q1 2026 alone, over 20 training-free acceleration methods" -- This is a strong empirical claim. Add a footnote or citation listing the 20+ methods or reference a survey. Currently only 10 are bracketed.
- **Line 3**: The bracketed citation lists "[Fast-dLLM, EntropyCache, dKV-Cache, Elastic-Cache]" etc. without proper citation format. These should be `\cite{}` references in the final LaTeX, but even in the draft, using author-year or arXiv IDs would help reviewers verify.
- **Line 11**: "TORS (arXiv:2603.00763)" -- Good that an arXiv ID is given. Apply the same treatment to "Kolbeinsson et al. (2024)" which lacks a specific reference.
- **Line 25**: "three-seed validation" -- the glossary and notation table use "seeds {42, 123, 456}" but the intro does not specify which seeds. Minor, but useful for reproducibility claims.
- **Line 27**: "Ortho = 0.96 on the combined metric" then later "Ortho = 0.96" for M3+IGSD on GSM8K -- using the same Ortho value for different pairs/benchmarks in the same paragraph is confusing. Add the benchmark qualifier each time.
- **Line 31**: "amplified composition effects (Dream M1+IGSD+M3$_{\text{off}}$ QAS = 2.18 on GSM8K vs. LLaDA QAS = 1.07)" -- The subscript "off" notation for M3 is not defined until the reader reaches the methods section. Define it on first use or use "without M3 guidance."
- **Line 39**: "M1 reports measured cache hit rate (56--99%)" -- This range is misleading in isolation. The 56% is at eta=0.5 and the 99% is at eta=2.0, but the experiments section (Table 3) reports 56.2% and 60.2%. The "99%" figure does not appear in the experiments. Check source.
- **Line 41**: "Section 3 surveys... Section 4 defines... Section 5 reports..." -- This roadmap paragraph is conventional but adds no information beyond what section titles convey. Consider cutting or integrating into the scope statement.

## Visual Element Assessment
- [x] Figure 1 (teaser scatter) matches the outline plan (Figure 1: Speed-Quality Landscape Teaser)
- [x] Figure 1 is referenced before it appears (line 17, figure on line 19)
- [ ] Caption is somewhat self-explanatory but overloaded with detail ("M1+M3 achieves 0.86x speedup (a net slowdown) despite 102.5% accuracy retention, falling below both M1 alone and M3 alone in QAS due to overhead stacking from dual model loading"). This reads as body text, not a caption. Shorten and move the explanatory detail to the text.
- [ ] No Table 1 (Related Work Speed Comparison) appears in the intro despite the outline listing it for "Introduction/Related Work". Either add it or note it belongs in Section 3.

## What Works Well
1. **Opening paragraph** (lines 1-3): Leading with "LLaDA-8B-Instruct achieves 71.2% accuracy on GSM8K but generates text at 34 tokens per second" is exactly the right structure -- concrete numbers first, context second. The 2.1x/13.9x speed deficit framing immediately establishes stakes.
2. **The 0.86x destructive interference example** (line 13): Stating "Our experiments show the answer is 0.86x --- destructive interference from overhead stacking" before the reader expects it is an effective hook. This transforms an abstract question ("do methods compose?") into a concrete, surprising empirical finding.
3. **Honest AR comparison** (contribution 5, line 33): "DLM acceleration does not close the speed gap with AR models; the value of this study lies in characterizing the composition design space, not in claiming speed parity." This level of intellectual honesty is rare and will earn reviewer respect. It preempts the most obvious criticism.
