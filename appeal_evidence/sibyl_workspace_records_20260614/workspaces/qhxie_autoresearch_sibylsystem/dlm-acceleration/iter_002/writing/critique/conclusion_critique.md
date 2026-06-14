# Critique: Conclusion

## Summary Assessment
The conclusion is dense, evidence-rich, and avoids the filler patterns common in ML paper conclusions. It accurately echoes the paper's major findings with specific numbers and presents an honest self-assessment of the AR gap. However, it reads more like a compressed results summary than a conclusion -- it re-reports data without providing the higher-level synthesis, implications, or forward-looking framing that a strong conclusion should deliver. The section also deviates from the outline's intended structure in minor but noticeable ways.

## Score: 6/10
**Justification**: The technical content is accurate and well-supported by data (which is hard to get wrong given the extensive experiment sections). The main weaknesses are structural: the conclusion is nearly twice the ~0.5 page target in the outline, it is overly detailed (re-reporting configuration-level hyperparameters and per-benchmark Ortho values that belong in Results), and it lacks the synthesis-level framing (design principles, future directions, broader implications) that separates a 6-score section from an 8-score one. Reaching 8/10 requires cutting ~40% of the data recitation, strengthening the "so what" framing, and adding a forward-looking paragraph that connects to the three design principles from the Discussion.

## Critical Issues

### Issue 1: Conclusion reads as a results summary, not a conclusion
- **Location**: Entire section (paragraphs 1-5)
- **Quote**: "M1+IGSD composes near-orthogonally. Combined Ortho = 0.96, with 2.75x speedup at 58.9% accuracy retention on GSM8K. IGSD's draft-partition-refine pipeline creates frozen tokens ($\alpha = 0.886$) with near-zero entropy, which M1 exploits at 83% cache hit rate."
- **Problem**: This level of mechanistic detail (frozen token fractions, cache hit rates, per-benchmark Ortho values, specific hyperparameter configs) belongs in Section 5 and Section 7 -- and indeed it already appears there verbatim. A conclusion should synthesize the key takeaway ("M1+IGSD composes because they target non-overlapping bottlenecks") without re-explaining the mechanism. The reader who has reached page 9 already knows the details.
- **Fix**: Replace the three bullet points (paragraphs 2-4) with a single paragraph that states the composition taxonomy at the verdict level: (1) M1+IGSD is near-orthogonal, (2) M3+IGSD is task-dependent, (3) M1+M3 interferes destructively. Reference the Discussion (Section 7.1-7.2) for the mechanisms. Remove all specific Ortho values, CHR percentages, alpha values, and hyperparameter settings from the conclusion -- the reader can find these in Tables 4-5.

### Issue 2: The three-way composition paragraph repeats results without adding insight
- **Location**: Paragraph 5 (beginning "Three-way composition confirms...")
- **Quote**: "The Pareto-optimal operating point -- M1 ($\eta = 0.5$) + IGSD ($\tau = 0.85$, $T_{\text{draft}} = 32$) with M3 off -- achieves 1.71x speedup, QAS = 1.07, and Ortho = 1.02 (3-seed mean, QAS CV < 10%)."
- **Problem**: This is nearly identical to the corresponding entry in Table 5 and the Experiments section 5.3. The conclusion should state the high-level takeaway ("the pairwise composition structure fully determines three-way outcomes") rather than re-reporting the same numbers with the same precision.
- **Fix**: Condense to: "Three-way composition confirms the pairwise structure: the Pareto-optimal three-way recipe is simply M1+IGSD with M3 off (QAS = 1.07, Ortho = 1.02, stable across 3 seeds). Adding M3 guidance to any composition degrades Ortho to ~0.5."

## Major Issues

### Issue 3: No forward-looking paragraph or implications for the field
- **Location**: Missing content -- the conclusion ends with "Released artifacts" without any future directions
- **Quote**: N/A
- **Problem**: The outline specifies this section should be 0.5 pages and lists five bullet points. The current draft covers the first four but entirely omits any forward-looking statement. A conclusion at a top venue should connect findings back to the broader research agenda: what should the DLM community do differently based on these results? The Discussion (Section 7.3) lays out three design principles, but the conclusion does not reference them. Without this, the conclusion feels like it stops mid-thought.
- **Fix**: Add a short paragraph (3-4 sentences) after the limitations and before the released artifacts. Possibilities: (a) The composition taxonomy suggests DLM acceleration research should shift from developing individual methods to designing methods that are compositionally aware (e.g., reducing per-step overhead, as Principle 1 from Discussion 7.3 suggests). (b) Kernel-level KV cache integration is the highest-leverage missing piece. (c) Future work should extend the factorial design to MDLM, SEDD, and models above 10B parameters.

### Issue 4: Section is significantly over length
- **Location**: Entire section
- **Problem**: The outline allocates 0.5 pages for Section 8. The current draft, with its three detailed bullet points, two full paragraphs, plus Limitations and Released Artifacts paragraphs, will run to approximately 1-1.2 pages in a NeurIPS two-column layout. This steals space from other sections and signals poor editorial discipline to reviewers.
- **Fix**: Target 12-15 sentences total. The current draft has roughly 30 sentences. Cut by: (a) removing mechanism explanations from the bullet points, (b) merging the three-way and cross-model paragraphs into a single sentence each, (c) shortening the limitations to a single sentence referencing Section 7.6.

### Issue 5: Limitations paragraph duplicates Discussion Section 7.6 almost verbatim
- **Location**: Paragraph beginning "**Limitations.**"
- **Quote**: "All experiments use LLaDA-8B-Instruct and Dream-7B-Instruct; generalization to MDLM, SEDD, or models above 10B parameters is untested."
- **Problem**: This sentence and the subsequent ones about M1 projected speedup, MATH500 baseline, and HumanEval/MBPP exclusion are near-identical to Discussion 7.6. A conclusion should either (a) cross-reference ("see Section 7.6 for detailed limitations") or (b) provide a one-sentence highest-level limitation statement, not re-enumerate every limitation from the Discussion.
- **Fix**: Replace the entire Limitations paragraph with: "Key limitations include single-family model coverage (LLaDA and Dream only), projected rather than measured M1 speedup due to d2Cache integration failure, and limited statistical power on MATH500 (11.1% baseline); see Section 7.6 for details." This saves 4-5 sentences.

## Minor Issues

- **Paragraph 1, line 1**: "ComposeAccel provides the first controlled factorial study of training-free acceleration composition for diffusion language models" -- this claim appears verbatim in the Introduction (Section 1) contributions list. Vary the phrasing to avoid a copy-paste feel. Suggestion: "This work presented ComposeAccel, a controlled factorial study..." (past tense, no superlative).
- **Paragraph 6, "average transfer ratio = 1.86"**: This statistic does not appear in Section 6 (Cross-Model) or the Discussion. Either it should be introduced in Section 6 first, or removed from the conclusion. New data should not debut in a conclusion.
- **Paragraph 6, "4 of 5 recipes showing consistent synergy patterns"**: Vague. Which recipe is the exception? If this detail matters, state it; if not, remove.
- **Paragraph 7, "QAS = 20.5"**: The batch=8 AR comparison, while honest and useful, is somewhat tangential to a conclusion about *composition*. Consider moving this to the AR gap sentence in one clause rather than a full sentence.
- **Paragraph 8, "Released artifacts"**: Bold label "**Released artifacts.**" breaks the paragraph flow. Consider moving this to a dedicated "Reproducibility" sentence or integrating it naturally.
- **Terminology**: "M3+IGSD is task-dependent" -- the glossary defines "task-dependent" for the first time here and in Results, but the Conclusion bullet uses it without the parenthetical explanation from Section 5.2. A reader skimming just the conclusion might not understand what "task-dependent" means in this context. Add a one-clause clarification: "task-dependent (near-orthogonal on GSM8K, interference on MATH500)."
- **Paragraph 5**: "Activating M3 guidance ($w_g = 0.3$) drops Ortho to 0.49 due to per-step TPS overhead" -- "$w_g = 0.3$" level of detail is unnecessary for the conclusion. Simplify to "Activating M3 guidance drops Ortho to ~0.5."

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies "None" for Conclusion)
- [x] All visuals referenced before appearance (no visuals in this section)
- [x] Captions are self-explanatory (N/A)
- [x] No text-heavy sections that need visual support (the conclusion is appropriately text-only)

## What Works Well
1. **Honest AR comparison framing** (paragraph 7): "The value of this study is in mapping the composition design space, not in claiming DLM speed parity with AR inference." This is exactly the kind of mature, self-aware framing that reviewers respect. It preempts the obvious criticism and reframes the contribution constructively.
2. **Concrete released artifacts** (paragraph 8): Listing the IGSD implementation size (50 lines) and specific deliverables (acceleration recipes, composability benchmark suite) gives the conclusion a tangible, actionable ending. Few papers are this specific about what they release.
3. **Accurate data throughout**: Every number in the conclusion cross-checks against the Experiments and Discussion sections. No overclaiming, no rounding errors, no inconsistencies between the conclusion and the body of the paper.
