# Critique: Discussion

## Summary Assessment
The discussion section is well-structured and technically substantive, providing mechanistic explanations for composition outcomes that go beyond restating results. The six subsections cover synergy/interference mechanisms, design principles, the AR gap, cross-model transferability, and limitations with commendable specificity. However, the section suffers from substantial overlap with the experiments and conclusion sections, contains several numerical inconsistencies with earlier sections, and the limitations subsection -- while thorough -- is long enough to dominate the section's impression, undermining the paper's contributions.

## Score: 6/10
**Justification**: The mechanistic analysis in 7.1 and the three design principles in 7.3 are the strongest parts and would earn a 7+ on their own. The score is held back by: (1) significant content duplication with the experiments section (multiple paragraphs essentially restate results from Section 5), (2) numerical inconsistencies that erode trust, (3) the AR gap discussion (7.4) reads as a rehash of Section 6.2 rather than offering new insight, and (4) the limitations section is extensive but repetitive with the conclusion's limitations paragraph. Reaching a 7 requires eliminating duplication, fixing number mismatches, and sharpening the "so what" of each subsection.

## Critical Issues

### Issue 1: Numerical inconsistency -- M3 TPS and speedup values conflict with experiments section
- **Location**: Section 7.1, paragraph "M1+M3 interference mechanism"
- **Quote**: "M3 standalone (96.5, which benefits from IGSD-like pipelining effects absent here)"
- **Problem**: The experiments section (Table 3) reports M3 speedup as 1.68x relative to a baseline of 33.8 TPS on GSM8K, which implies M3 TPS = 56.8. The discussion claims M3 standalone achieves 96.5 TPS. This 96.5 value appears nowhere in the experiments section or the outline. If this is a MATH500 TPS value (baseline 79.1 TPS * 1.68 = 132.9, still not 96.5), or a different measurement basis, it is unexplained and creates confusion.
- **Fix**: Verify the source of the 96.5 TPS figure. If it is from a different benchmark or batch configuration, state this explicitly. If it is an error, replace with the correct M3 TPS consistent with Table 3. The parenthetical claim about "IGSD-like pipelining effects" also needs either a citation to a specific experiment or removal, since M3 standalone does not use IGSD.

### Issue 2: Numerical inconsistency -- baseline TPS discrepancy
- **Location**: Section 7.1, M1+M3 interference paragraph
- **Quote**: "reducing TPS from the baseline 58.5 to 50.3"
- **Problem**: The experiments section reports baseline TPS as 33.8 on GSM8K and 79.1 on MATH500. The value 58.5 matches neither. If this is a different measurement (e.g., a different sequence length, or the M1-only TPS), it must be labeled. The reader who cross-references Section 5.1 will find a contradiction.
- **Fix**: Use the canonical baseline TPS values from the experiments section (33.8 for GSM8K, 79.1 for MATH500), or if 58.5 is a combined/weighted metric, define it explicitly on first use. The 50.3 TPS value must also be reconciled.

### Issue 3: M1 speedup value inconsistency across sections
- **Location**: Section 7.3, Principle 1
- **Quote**: "1.16x * 1.65x = 1.91x expected speedup"
- **Problem**: The introduction and experiments section report M3 speedup as 1.68x (Table 3), not 1.65x. The discussion uses 1.65x. This is a minor but trust-eroding discrepancy. For the "expected" multiplicative product: 1.16 * 1.68 = 1.95, not 1.91.
- **Fix**: Use the consistent M3 speedup value (1.68x) from Table 3 and recompute: 1.16 * 1.68 = 1.95x expected. Then state the measured 0.86x for contrast.

## Major Issues

### Issue 4: Section 7.4 (The AR Gap) is largely duplicated from Section 6.2
- **Location**: Section 7.4, entire subsection
- **Quote**: "Qwen2.5-7B at batch = 1 reaches 96% GSM8K accuracy at 71 TPS (QAS = 3.08 relative to the LLaDA baseline), while the best composed DLM acceleration achieves QAS = 1.07 at 1.71x speedup."
- **Problem**: Section 6.2 already presents the AR vs. DLM comparison with the same numbers (Table 7). The discussion's contribution should be interpretation and forward-looking analysis, not restating the comparison. The second paragraph offers a mechanistic explanation (64 forward passes with O(N^2) attention vs. N incremental steps), which is valuable, but the first paragraph is pure repetition. The third paragraph ("The honest comparison establishes that the value of DLM composition research lies in understanding the design space") is a claim also made in the introduction (contribution 5) and the conclusion.
- **Fix**: Cut the first paragraph entirely (the reader has already seen Table 7). Open with the mechanistic comparison (paragraph 2) and then move directly to the "what this means for the field" argument. Remove "honest comparison" framing that was already established in the intro.

### Issue 5: Section 7.5 (Cross-Model Transferability) repeats Section 6.1 without sufficient new analysis
- **Location**: Section 7.5, entire subsection
- **Quote**: "Dream-7B-Instruct validation confirms that composition patterns transfer across DLM architectures. The Max-Speed recipe ... achieves QAS = 2.18 on Dream-7B GSM8K versus QAS = 1.07 on LLaDA-8B."
- **Problem**: Section 6.1 already presents the Dream-7B results. The discussion adds one new insight ("Dream-7B's iterative refinement in later steps is less productive than LLaDA's") but mostly repackages Table 6 data. The "key transferable finding" at the end (M1+IGSD without M3 remains Pareto-optimal) was already stated in Section 5.3.
- **Fix**: Merge the one new insight (Dream-7B refinement productivity hypothesis) into Section 7.1's mechanism discussion as a brief cross-model corroboration. Delete Section 7.5 as a standalone subsection, or compress it to 2-3 sentences within Section 7.3's design principles.

### Issue 6: Limitations section is disproportionately long and partially duplicated with conclusion
- **Location**: Section 7.6, entire subsection
- **Quote**: Multiple paragraphs on model coverage, M1 projected speedup, benchmark limitations, seed variance, hardware specificity, M2 exclusion.
- **Problem**: The limitations subsection is 6 paragraphs spanning roughly 40% of the discussion's word count. The conclusion section repeats many of these limitations verbatim (model coverage, M1 projection, MATH500 weakness, HumanEval/MBPP exclusion). For a paper targeting ~1 page for the discussion (per outline), this subsection alone exceeds that budget. Reviewers may perceive this as the authors "listing weaknesses defensively" rather than discussing implications.
- **Fix**: Consolidate limitations into 3 concise paragraphs: (1) model/benchmark scope, (2) M1 measurement gap, (3) statistical power. Move M2 exclusion rationale to Section 4 (methods) where it is more natural. Remove the conclusion's separate limitations paragraph (it should refer to the discussion's limitations or briefly summarize).

### Issue 7: The "Reconciling with iter_001" paragraph may confuse readers unfamiliar with the iteration structure
- **Location**: Section 7.1, final paragraph
- **Quote**: "The pilot experiment in iter_001 reported M1+M3 Ortho = 1.34 on 100 GSM8K samples..."
- **Problem**: The term "iter_001" is internal project terminology. While the experiments section briefly mentions the discrepancy with "iter_001 pilot finding" in Section 5.2, a discussion section should provide the narrative context for why an earlier measurement was wrong without relying on internal iteration labels. A reviewer will wonder: is this from a prior paper? A preprint? An internal experiment?
- **Fix**: Rephrase as "our initial pilot study" or "preliminary experiments on a smaller sample (n=100, single seed)" without the iteration label. The key information (small-sample variance, inflated combined metric including degenerate benchmarks) should remain.

## Minor Issues

- **Section 7.1, paragraph 1**: "Ortho = 0.96 combined, 0.99 on GSM8K" -- the outline states combined Ortho = 0.96 and GSM8K Ortho = 0.99. The experiments section (Table 4) shows GSM8K Ortho = 0.99 and combined = 0.96 for M1+IGSD. The presentation order (combined first, then per-benchmark) is unintuitive; lead with GSM8K first as it is the stronger result.
- **Section 7.1**: "the undisclosed 0.5x penalty" -- this phrasing implies the prior metric was deliberately hidden. Prefer "the previously included 0.5x weighting factor" or "a 0.5x penalty in the original metric formula."
- **Section 7.2**: "Qwen2.5-0.5B achieves 96% accuracy on GSM8K independently (Table 7)" -- Table 7 in the outline/experiments is the AR vs. DLM comparison table, which lists Qwen2.5-7B, not Qwen2.5-0.5B. The 0.5B guide model's standalone accuracy is not in Table 7. Either add it to a table or cite the source.
- **Section 7.3, Principle 2**: The sentence "The 15.2x gap between theoretical and measured M1 speedup" is ambiguous. The 15.2x is the d2Cache framework overhead (d2Cache runs 15.2x slower than HF baseline), not a gap between theoretical and measured M1 speedup. The M1 projected-to-measured gap is 2.27x vs. 1.16x (approximately 2x, not 15.2x).
- **Section 7.3, Principle 2**: "published speedups (15--26x)" -- this range is not sourced. The experiments section mentions d2Cache achieving "4.39x internal speedup." Published KV caching papers report various numbers. Either cite the specific papers and their claimed speedups, or remove the range.
- **Section 7.3, Principle 3**: "M3 achieves AccRet > 100% on GSM8K at all guidance weights" -- Table 3 shows M3 at $w_g$=1.0 has AccRet = 84.9%, which is below 100%. The claim is only true for $w_g \in \{0.3, 0.5, 0.7\}$.
- **Section 7.6, M2 exclusion paragraph**: "simplified Saber implementation produced catastrophic accuracy collapse (AccRet < 50% at step jump > 3x)" -- this is the first time a specific AccRet threshold for M2 is given in the paper. If this data exists, it should appear in the experiments section (or appendix) first, then be referenced here.
- **Section 7.4**: "DLMs offer architectural advantages (parallel generation, bidirectional context) that may prove valuable in settings not captured by sequential exact-match benchmarks." -- This is a vague forward-looking claim with no evidence or citation. Either cite a paper demonstrating these advantages or remove.

## Visual Element Assessment
- [x] No figures or tables planned for the discussion section (outline confirms "None") -- appropriate for a discussion
- [x] Cross-references to earlier figures/tables (Table 7) are present
- [ ] Table 7 cross-reference in Section 7.2 appears incorrect (references Qwen2.5-0.5B accuracy, but Table 7 covers Qwen2.5-7B)
- [ ] No visual support for the three design principles -- a summary diagram showing the composition taxonomy (synergy/task-dependent/interference) would strengthen Section 7.3

## What Works Well

1. **Section 7.1's mechanistic analysis of M1+IGSD synergy** is the highlight of the discussion. The explanation of how frozen tokens create low-entropy positions that M1's cache exploits is specific, falsifiable, and grounded in measured quantities (CHR = 83.4%, alpha = 88.6%). This paragraph does what a discussion section should: explain *why* a result occurs, not just restate that it does.

2. **The three design principles in Section 7.3** distill the composition analysis into actionable guidance. Principle 1 (overhead stacking is subadditive) and Principle 3 (quality-preserving methods are best standalone) are clear, evidence-backed, and immediately useful to practitioners. These principles are the section's main contribution beyond the results.

3. **The limitations section is genuinely thorough.** While it is too long (see Major Issue 6), the content is honest: the M1 projection caveat, the MATH500 statistical power issue, the hardware specificity of d2Cache failure, and the M2 exclusion rationale are all important for reader trust. The paper does not hide its weaknesses.
