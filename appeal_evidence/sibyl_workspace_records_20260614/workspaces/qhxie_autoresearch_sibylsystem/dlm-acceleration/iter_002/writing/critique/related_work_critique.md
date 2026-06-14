# Critique: Related Work (Section 3)

## Summary Assessment
The related work section is well-organized into three subsections that mirror the paper's method taxonomy and concludes with an effective framing of the composability gap. The unique strength of this section is its systematic integration of the authors' own experimental findings alongside each literature category, grounding the survey in concrete evidence rather than vague comparisons. However, the section suffers from several claim-evidence misalignments, inconsistencies with other sections, and an incomplete treatment of the speculative denoising family that weakens its positioning function.

## Score: 6/10
**Justification**: The structure is solid and the composability gap framing in Section 3.3 is genuinely effective. Reaching a 7 requires fixing the numerical inconsistencies flagged below (particularly the M3 speedup and model size discrepancies), tightening the claim-evidence chain for the published speedup numbers in the KV-cache survey, and adding the missing Table 1 reference promised in the outline. The banned-pattern violations and missing cross-references to notation.md are fixable but currently hurt readability.

## Critical Issues

### Issue 1: M3 speedup inconsistent across sections
- **Location**: Section 3.2, paragraph on AR-Guided Unmasking
- **Quote**: "M3 at $w_g = 0.3$ achieves 1.65x speedup"
- **Problem**: The method section (4.4) reports M3 speedup as 1.68x ("The measured speedup is 1.68x on GSM8K"), and the experiments section (5.1, Table 3) also reports 1.68x across all $w_g$ values. The introduction uses 1.65x in the contributions list. Two different numbers circulate for the same measurement: 1.65x (related work, intro) and 1.68x (method, experiments). This is the kind of inconsistency reviewers seize on to question data integrity.
- **Fix**: Unify to a single number across all sections. Table 3 and the method section report 1.68x from the full-scale experiment; use that everywhere. Search-and-replace 1.65x with 1.68x in related work and introduction.

### Issue 2: Qwen2.5-0.5B model size inconsistent
- **Location**: Section 3.2, paragraph on AR-Guided Unmasking
- **Quote**: "using Qwen2.5-0.5B ($g_\phi$)"
- **Problem**: The related work section does not state the parameter count. The method section (4.4) says "Qwen2.5-0.5B, 0.49B parameters" while the Discussion (7.1) says "loading Qwen2.5-0.5B (0.95 GB VRAM)". The glossary says "0.5B parameter AR model." The related work needs to be consistent with the method section's 0.49B number when the parameter count is first introduced, or defer the parameter count entirely to Section 4.4. Currently, the related work mentions the model name without parameters but the method section introduces "0.49B" --- this is not itself an error, but it means the reader encounters the model in Section 3 without knowing its size, which weakens the overhead argument made in the same paragraph.
- **Fix**: Add "(0.49B parameters)" after "Qwen2.5-0.5B" on first mention in Section 3.2 to give the reader the context needed to understand the 12% overhead claim made two sentences later.

### Issue 3: M3 accuracy retention number inconsistent
- **Location**: Section 3.2, AR-Guided Unmasking paragraph
- **Quote**: "M3 at $w_g = 0.3$ achieves 1.65x speedup with 102.5% accuracy retention on GSM8K"
- **Problem**: The experiments section (5.1) reports M3 at $w_g=0.3$ as 103.9% accuracy retention, matching the method section (4.4): "at $w_g = 0.3$, GSM8K accuracy retention is 103.9%." The introduction also uses 102.5% in one place and 103.9% in another. The related work's 102.5% is orphaned --- it matches no other section.
- **Fix**: Replace 102.5% with 103.9% to match Section 4.4 and Table 3.

## Major Issues

### Issue 4: Table 1 promised in outline but absent
- **Location**: Section 3.2, first paragraph
- **Quote**: "Table 1 summarizes published speedup claims and evaluation protocols for representative methods."
- **Problem**: The section references Table 1 but no table is actually present in the section draft. The outline specifies "Table 1: Related Work Speed Comparison" with columns for Method, Axis, Model, Reported Speedup, Eval Protocol, and Composition Tested. The text currently dumps all the speedup numbers into dense paragraphs, which is far harder to parse than a structured table.
- **Fix**: Create Table 1 as specified in the outline. Move the per-method speedup numbers into the table and use the prose to synthesize observations (e.g., "Published speedups range from 2x to 99x, but no two methods share an evaluation protocol").

### Issue 5: Published speedup claims presented without critical context
- **Location**: Section 3.2, KV-Cache Approximation paragraph
- **Quote**: "Elastic-Cache [...] reaching 45.1x on long sequences" and "Window-Diffusion [...] reporting up to 99x speedup"
- **Problem**: These extreme speedup numbers (45x, 99x) are presented without noting the conditions that make them incomparable to the paper's own measurements. The 99x from Window-Diffusion sacrifices global bidirectional context (mentioned parenthetically), but the 45.1x from Elastic-Cache has no caveat at all. A reviewer will ask: "If Elastic-Cache achieves 45x, why does your M1 only achieve 1.16x?" The section partially addresses this for EntropyCache (paragraph 2 of KV-Cache Approximation), but the other methods' claims float without comparable scrutiny.
- **Fix**: Add a summary sentence after the enumeration: "These published numbers span 2x to 99x, but use different hardware, batch sizes, sequence lengths, and quality thresholds (Table 1). Our controlled protocol measures all methods on identical infrastructure (Section 4.5)." This is partly present in Section 3.3 but should be foreshadowed here.

### Issue 6: DyLLM and ES-dLLM cited without citation markers
- **Location**: Section 3.2, end of KV-Cache Approximation paragraph
- **Quote**: "DyLLM uses saliency-based token selection [...] ES-dLLM applies early-layer skipping"
- **Problem**: These two methods lack [CITE:...] markers, unlike every other method in the section. If these are real papers, they need citations. If they were added speculatively, they should be removed or properly cited.
- **Fix**: Add [CITE:dyllm] and [CITE:esdllm] (or the correct citation keys), or remove if no citable source exists.

### Issue 7: Logical gap between M1 paragraph and IGSD description
- **Location**: Section 3.2 to Section 3.2 (transition from KV-Cache to Adaptive Step Scheduling)
- **Problem**: The section describes four acceleration families, but IGSD is introduced under "Speculative Denoising" (Section 3.2, paragraph 4). However, IGSD is functionally a step-reduction method --- it reduces the number of steps for high-confidence tokens. The section's own taxonomy places it under speculative denoising, but the mechanism (draft fewer steps, then refine) is closer to adaptive step scheduling than to SSD-style verification. The intro positions IGSD as "a composability study vehicle" and the method section describes it as a "speculative step scheduler." The related work needs to explicitly acknowledge this boundary-straddling nature rather than simply slotting IGSD into one category.
- **Fix**: Add a bridging sentence at the end of the IGSD paragraph: "IGSD thus occupies the boundary between step scheduling and speculative decoding: it reduces computation by running fewer steps (like M2-family methods) but uses a confidence-based partition to selectively refine (like speculative methods). We classify it under speculative denoising because the draft-refine architecture is structurally analogous to draft-verify in SSD, though IGSD is approximate while SSD guarantees lossless output."

### Issue 8: Missing reference to FlashDLM's "FreeCache" component
- **Location**: Section 3.2, AR-Guided Unmasking paragraph
- **Quote**: "FlashDLM [CITE:flashdlm] combines FreeCache (a KV approximation method) with a lightweight AR supervisor"
- **Problem**: FlashDLM is listed under "AR-Guided Unmasking" but is explicitly described as combining KV caching (FreeCache) with AR guidance. This cross-axis composition within a single published method is highly relevant to the paper's thesis about composability, yet the section does not flag this. FlashDLM is arguably the closest prior work to ComposeAccel's composition study --- it composes two axes within a single method, achieving 12.14x. The section should acknowledge FlashDLM as partial prior art on composition and explain how ComposeAccel differs (systematic factorial study vs. a single pre-designed combination).
- **Fix**: Expand the FlashDLM description: "FlashDLM is the closest precedent to our composition approach: it combines FreeCache (KV approximation) with AR guidance in a single integrated system, achieving 12.14x combined speedup. However, FlashDLM evaluates only this one pre-designed combination; it does not measure whether FreeCache and AR guidance compose near-orthogonally, nor does it compare against alternative pairings. ComposeAccel provides this systematic analysis."

## Minor Issues

- **Section 3.1, line 5**: "Starting from a fully masked sequence $\mathbf{x}_T$" --- notation.md defines $T$ as the total denoising steps and $t$ as the index. The fully masked sequence should be $\mathbf{x}_0$ (the noise end) or $\mathbf{x}_T$ depending on convention. Check which direction the forward process runs (0 to T or T to 0) and ensure consistency with notation.md. The method section (4.3) uses "steps $0 \to T_{\text{draft}}$" suggesting 0 is the starting point, which would make the fully masked sequence $\mathbf{x}_0$, not $\mathbf{x}_T$.
- **Section 3.1, line 5**: "the transformer $f_\theta$ produces logits $p_t(\cdot \mid \mathbf{x}_t) \in \mathbb{R}^{N \times V}$" --- notation.md defines this as $p_t \in \mathbb{R}^{N \times V}$, consistent. Good.
- **Section 3.2**: "PRR [CITE:prr] trains a lightweight controller" --- PRR requires auxiliary training, so it is not purely training-free. The section should note this distinction since the paper's scope is training-free methods.
- **Section 3.2**: "Model scheduling replaces the full model with a smaller one at less-critical middle steps" --- no citation marker. Add [CITE:...] or remove.
- **Section 3.2**: The sentence "the only method that improves accuracy over baseline" (about M3) uses a superlative that could be challenged if any other DLM method achieves >100% AccRet. Soften to "the only method in our study that improves accuracy over baseline."
- **Section 3.3**: "the exact same gap we address" --- casual tone for a venue paper. Replace with "the analogous gap we address" or "precisely the gap we address."
- **Section 3.3**: The Ortho formula is introduced here: "$\text{Ortho}(A, B) = \text{QAS}(A+B) / \max(\text{QAS}(A), \text{QAS}(B))$." This duplicates the formal definition in Section 4.1. Either remove the formula here and forward-reference Section 4.1, or present it informally here and formally in Section 4.1. Currently, having the identical formula in both sections is redundant.
- **Section 3.2**: "IGSD differs from SSD in a structurally important way" --- the word "structurally" is vague here. The difference is concrete: SSD runs all T steps and prunes via batch verification; IGSD runs fewer steps. Say that directly.
- **Section 3.1**: Baseline TPS for LLaDA is stated as "34 TPS" but Section 4.5 reports "33.8 TPS." Use the precise number (33.8) or round consistently.

## Visual Element Assessment
- [ ] **Figures/tables match outline plan**: Table 1 is planned in the outline but missing from the section.
- [x] All visuals referenced before appearance: Table 1 is referenced in the text, but the table itself is absent.
- [ ] Captions are self-explanatory: N/A (no table/figure rendered).
- [x] No text-heavy sections that need visual support: The KV-Cache paragraph is extremely dense with 8 methods and their speedup numbers. Table 1 would solve this.

## What Works Well

1. **Section 3.3 (Composability Gap) is the section's strongest contribution.** The framing of TORS as addressing the same gap for vision diffusion, and Kolbeinsson et al. as addressing weight-space composability, creates a clear intellectual positioning. The final paragraph's three-point contribution summary (formal metric, controlled experiments, mechanistic analysis) is crisp and gives the reader a concrete preview of what the paper delivers beyond the survey.

2. **Integrating own results alongside prior work (Section 3.2).** The decision to present M1's measured 1.16x speedup alongside EntropyCache's published 15--26x, with a clear explanation of the implementation gap (full forward passes vs. kernel-level sparse attention), is honest and builds trust. The M2 NO_GO negative result is similarly valuable --- it demonstrates the paper's willingness to report failures, which strengthens the credibility of the positive results.

3. **The speculative denoising taxonomy (Section 3.2, paragraph 4).** The differentiation between DualDiffusion (draft + verify), SSD (self-speculative with tree verification), SSMD (non-causal/causal toggle), S2D2 (block-diffusion specific), and IGSD (step-reduction with confidence partition) is precise and informative. The explicit statement that SSD is lossless while IGSD is approximate is the kind of honest comparison reviewers value.
