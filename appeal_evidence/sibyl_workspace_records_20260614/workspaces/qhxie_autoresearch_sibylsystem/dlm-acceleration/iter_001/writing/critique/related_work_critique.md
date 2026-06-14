# Critique: Related Work

## Summary Assessment

The Related Work section is unusually strong for a paper of this type. It is specific, evidence-grounded, and honest about the negative result for M2. The positioning subsections at the end of each sub-section effectively delineate what this paper adds versus what exists. However, three issues require attention before submission: (1) a critical factual error on SSD's classification as training-based, (2) incomplete coverage of the outline's scope statement for Section 3.2 Background vs. Section 7 Related Work, and (3) several minor inconsistencies with notation.md and glossary.md.

## Score: 7/10
**Justification**: The section reads like it was written by someone who actually read the cited papers, not skimmed their abstracts. The method-by-method breakdowns in §5.1 are better than most related work sections. The main deduction is the SSD misclassification (critical — SSD is listed in the method section as training-free), one structural redundancy with the Methods section, and underdeveloped positioning in §5.5 relative to the paper's core novelty claim.

---

## Critical Issues

### Issue 1: SSD Misclassified as Training-Based in Related Work, Contradicts Method Section

- **Location**: Section 5.3, paragraph 4: "SSD ... achieves lossless 3.46× speedup via discrete token self-speculation but is training-based, not training-free."
- **Quote**: "IGSD operates training-free, requires no external model, and stays within the MDM regime, filling the gap at the cost of an approximate (not lossless) quality guarantee."
- **Problem**: The glossary.md lists SSD as a training-free method (External Methods table: "Achieves 3.46x lossless speedup"). The Method section (§4.2, IGSD's "Relation to prior work" paragraph) explicitly states SSD is training-free: "SSD achieves lossless 3.46× speedup via discrete token self-speculation but is training-based, not training-free." Wait — this is the same claim verbatim. Cross-checking against the outline (§7.3): "SSD (self-speculative, lossless)" — the outline classifies SSD under Speculative Decoding as a valid comparator. The Discussion section (§6.4, Limitation 6) says "SSD + M1 composability test" is the most important follow-up — implying SSD is a comparable training-free method. The proposal abstract (§motivation) and glossary both list SSD without any "training-based" qualifier. The IGSD positioning in Method §4.2 says SSD is training-based but that claim appears internally contradicted by every other document. This needs resolution: is SSD [CITE:gao2025ssd] training-free or not? If SSD is actually training-free (which the glossary, discussion, and outline imply), the "IGSD fills the gap" claim in Related Work is false — the gap was already filled by SSD before this paper. If SSD is training-based, then the Method section §4.2 (where the same "training-based" claim appears) is inconsistent with the paper's self-positioning as "first training-free self-speculative MDM method."
- **Fix**: Resolve definitively. Gao et al. 2025 (SSD) appears to be training-free based on paper description (uses existing model's own forward pass with top-k confidence selection, no auxiliary model). If training-free: (a) rewrite the "filling the gap" claim throughout to acknowledge SSD, (b) reposition IGSD as offering a different trade-off (lossy but faster draft, different architecture for KV-synergy). If training-based: cite specific evidence from the SSD paper (what training is required?) and add that citation in both §5.3 and Method §4.2. The current text is ambiguous in both places and the contradiction will be caught by reviewers.

### Issue 2: SSD Speedup Figures Disagree Between Sections

- **Location**: Section 5.3, paragraph on SSD: "Reported speedup is 2.11–3.46× on GSM8K, MATH500, HumanEval, and MBPP." Discussion §6.4: "SSD achieves 3.46× lossless speedup." Glossary: "Achieves 3.46x lossless speedup."
- **Quote**: "Reported speedup is 2.11–3.46×" (Related Work) vs. "SSD achieves 3.46× lossless speedup" (Discussion)
- **Problem**: A reviewer will check both sections. Reporting a range in one place and a scalar in another looks like inconsistency or cherry-picking. The 2.11× lower bound from Related Work does not appear in Discussion or Glossary.
- **Fix**: Use a consistent figure across all sections. If the range 2.11–3.46× is task-dependent, state that explicitly everywhere: "2.11–3.46× depending on task (peak 3.46× on GSM8K)." If 3.46× is the task-average, state that and drop the range from §5.3.

---

## Major Issues

### Issue 3: Related Work Section Title Numbering Does Not Match Outline

- **Location**: Section header in related_work.md reads "# 5. Related Work"
- **Problem**: The outline (outline.md) numbers Related Work as Section 7: "## 7. Related Work (0.75 pages)". The Introduction (intro.md) explicitly states "Section 5 positions this work against related acceleration methods." The method section header is "# 4. Methods" and experiments is "# 5. Experiments." If Related Work is Section 7 in the outline but is numbered Section 5 in the file, this will create cross-reference inconsistencies throughout the paper (intro mentions Section 5, but the actual content is in what is labeled Section 5 in the file but should be Section 7).
- **Fix**: Confirm the final section numbering with the editor. The Introduction's roadmap gives the definitive order: Methods (§2), Experiments (§3), Discussion (§4), Related Work (§5), Conclusion (§6). Update the section header to match the intro's roadmap. All cross-references in the paper should use the section number from the Introduction roadmap.

### Issue 4: Outline Coverage — §5.5 Is Underspecified Relative to Paper's Core Claim

- **Location**: Section 5.5, "Composability of Intervention Methods"
- **Problem**: The paper's primary novelty claim is "first systematic pairwise composability study of training-free MDM acceleration methods." §5.5 covers only Kolbeinsson et al. (2024), which studies weight-space interventions. No other composability-adjacent work is cited. The outline (§7) does not explicitly mention what should go in 7.5, but given that the paper's central contribution is composability measurement, this subsection should be the most thoroughly defended. A reviewer will ask: "Is there any inference-time composability literature for AR models or other generation paradigms that this work should acknowledge?" PEFT composition, LoRA + quantization composition, or speculative decoding with KV-cache in AR models are all relevant adjacent areas.
- **Fix**: Add 1–2 sentences on composability of inference optimizations in AR models (e.g., speculative decoding + PagedAttention), even if just to argue why MDM's mask-state coupling creates a fundamentally different composability challenge. The current contrast with Kolbeinsson et al. is valuable but feels isolated — one reference does not establish a research tradition.

### Issue 5: "DLM" Used in Body Text, Violating glossary.md's Banned Terms

- **Location**: Section 5.4, line 59: "D2F is the first open-source DLM to surpass AR throughput in practice"
- **Quote**: "D2F is the first open-source DLM to surpass AR throughput in practice"
- **Problem**: glossary.md explicitly bans "DLM" in the paper body, requiring "MDM" instead. Using "DLM" here (and potentially elsewhere) violates the glossary contract. The glossary note: "Do NOT use 'discrete diffusion model' or 'DLM' interchangeably in the paper body (use MDM consistently). 'DLM' acceptable only when quoting external literature."
- **Fix**: Replace "DLM" with "MDM" or the full "masked diffusion language model" if this is first use. Check the entire section for additional occurrences.

### Issue 6: Section 5.1's Coverage Exceeds the 0.75-Page Scope Allocated in Outline

- **Location**: Section 5.1 (KV-Cache Methods for MDMs)
- **Problem**: The outline allocates 0.75 pages total for Related Work (§7). Section 5.1 alone covers seven methods in detailed paragraphs, followed by a position statement. Sections 5.2–5.5 add similar density. The resulting section will be approximately 2–2.5 pages, far exceeding the outline's budget. For a top-venue submission, exceeding the page budget in related work means compressing contributions or experiments.
- **Fix**: Consolidate §5.1. The seven KV-cache methods can be grouped into 2–3 clusters based on approach (entropy-based: EntropyCache; drift-based: dKV-Cache, Elastic-Cache; structure-based: Window-Diffusion; system-level: Fast-dLLM, SlowFast, FlashDLM-FreeCache) and described collectively within each cluster. Save individual detail for the 2–3 methods most directly compared to M1. Currently EntropyCache gets the same depth as Window-Diffusion even though only EntropyCache is instantiated as M1.

---

## Minor Issues

- **§5.1, EntropyCache paragraph, notation inconsistency**: Uses $\eta$ for the entropy threshold inline, which matches notation.md. But the paragraph uses the subscript $i$ for position, consistent with notation.md. No issue here — notation is correct.
- **§5.2, first paragraph**: "creating a sequential dependency chain" — this introduces "sequential dependency chain" without the glossary term "mask-state coupling." Cross-reference to the defined term would improve clarity.
- **§5.3, SSMD paragraph**: "approximately 2× reduction in forward pass count" — this is the only claim in the section without a specific benchmark or model scale qualifier. SSMD is tested "at GPT-2 scale" per the text, which means results are not comparable to 8B-scale models. Add a brief caveat that this scale difference limits direct comparison.
- **§5.3, S2D2 paragraph**: "block-diffusion architectures (SDAR, LLaDA2.1-Mini with partial AR structure)" — "SDAR" is not expanded anywhere in the paper or glossary. Expand on first use, even briefly, or replace with "semi-autoregressive diffusion models."
- **§5.4, Block Diffusion paragraph**: "BD3-LMs are trained from scratch and achieve SOTA on language modeling benchmarks." The phrase "SOTA" without qualification is on the banned list per glossary.md ("state-of-the-art (as standalone claim)" → "compare with specific methods and numbers"). Replace with a specific claim, e.g., "BD3-LMs achieve lower perplexity than pure-diffusion baselines on [benchmark] by [margin]."
- **§5.4, scope statement, last sentence**: "The contribution of this paper is not competitive throughput but structural insight: understanding which training-free methods compose and why." This is a strong thesis restatement but is oddly placed in the Related Work section. It belongs in the Introduction's scope statement (which already contains a similar sentence). In Related Work, it reads as defensive. Replace with the specific positioning claim: "Unlike training-based methods, which modify the generation architecture to enable KV-cache reuse, this paper asks which existing training-free methods compose without architectural changes."
- **§5.5, last paragraph**: "The gap encompasses both the measurement framework (our orthogonality metric $\text{Ortho}(M_a + M_b)$) and the causal analysis (the frozen-token synergy mechanism documented in Section 4 and Figure 8)." The reference to "Section 4" for the frozen-token mechanism is internally inconsistent with the section numbering issue raised in Issue 3 — if Methods is Section 2 and Discussion is Section 4 in the final paper, the reference needs to update accordingly.

---

## Visual Element Assessment

- [x] No figures are planned for Related Work per the outline ("None" in the FIGURES comment block) — consistent with standard practice.
- [x] No visuals are referenced or missing from this section.
- [N/A] No captions to assess.
- [x] The Table 1 (literature speed comparison) is introduced in the Introduction and positions the same methods covered in §5.1; Related Work provides the detailed narrative complement. The cross-reference from Related Work back to Table 1 (if any) would improve reader navigation but is absent. A single sentence like "Table 1 in the Introduction summarizes the speedup fragmentation across these methods" would help.

---

## What Works Well

1. **Method-level specificity**: The EntropyCache paragraph (§5.1) correctly reproduces the entropy formula $H_i = -\sum_v p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)$ and explains the $O(V)$ constant-cost property — this is the level of engagement with prior work that distinguishes a strong submission from a superficial one.

2. **Negative result integration**: The "Negative finding — M2 in this work" paragraph in §5.2 is excellent. It positions the experimental result within the related work narrative, explains the root cause, and makes the publishability of the negative result explicit. This is exactly how a reviewable paper handles a NO_GO result.

3. **IGSD vs SSD positioning** (§5.3, final paragraph): The architectural contrast between IGSD's reduced-step drafting and SSD's full-step drafting with tree verification is technically precise and well-argued. The explanation of why IGSD enables KV synergy (token freezing in $S_{\text{accept}}$) while SSD does not is the clearest statement of IGSD's structural contribution anywhere in the paper. Assuming the SSD training-free status issue (Issue 1) is resolved, this paragraph will be the section's strongest contribution.
