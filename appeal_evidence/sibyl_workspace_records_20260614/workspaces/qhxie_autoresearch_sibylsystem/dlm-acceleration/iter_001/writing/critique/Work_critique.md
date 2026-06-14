# Critique: Related Work

## Summary Assessment

The Related Work section (labeled "# 5. Related Work" in the file, but Section 5 in the Introduction roadmap and Section 7 in the outline) is technically detailed and above-average for its depth of engagement with prior work. Each subsection contains concrete method descriptions with specific speedup figures and architectural distinctions. However, two blocking issues prevent submission as-is: (1) "IGSD" is used throughout the entire section when the glossary bans this term and mandates "CD-SSD" everywhere in the paper body; (2) the section is approximately 3.5–4 pages in practice against a 0.75-page outline budget, which will force page-budget violations at NeurIPS. Beyond these, several cross-section consistency issues were identified when comparing against intro.md, method.md, experiments.md, and discussion.md.

## Score: 6/10

**Justification**: The section would score 8/10 on technical accuracy and specificity. The deduction to 6 reflects: (a) the IGSD→CD-SSD rename is a blocking consistency error — the method section, experiments section, discussion section, glossary, and notation all use "CD-SSD" but related work uses "IGSD" throughout §5.3; (b) scope bloat that will require approximately 50% cuts; (c) a missed opportunity in §5.5 to anchor the composability gap claim within any inference-time composability literature beyond one weight-space reference. Reaching 8 requires fixing the rename and tightening §5.1; reaching 9 requires §5.5 to cite at least one AR inference-time composability analog.

---

## Critical Issues

### Issue 1: "IGSD" Used Throughout — Banned Term, Not Updated to CD-SSD

- **Location**: §5.3 (four instances), §5.1 position statement (one instance)
- **Quotes**:
  - §5.3 heading: "**Position of IGSD relative to SSD.**"
  - §5.3: "IGSD and SSD both eliminate the external draft model requirement."
  - §5.3: "IGSD is approximate (35.1% accuracy retention on GSM8K at $\tau=0.9$, $T_{\text{draft}}=16$); SSD is lossless."
  - §5.3: "IGSD's primary value in this paper is as a composability enabler for the M1+IGSD synergy"
  - §5.1 position statement: "the M1+IGSD composability analysis (Section 2 and Section 3) is the first systematic evaluation of KV-cache composition with speculative denoising."
- **Problem**: The glossary.md explicitly bans "IGSD": "Do NOT use 'IGSD' in the paper body. Use CD-SSD." The notation.md section on "CD-SSD Notation" states: "method renamed from IGSD to CD-SSD to avoid name collision with Info-Gain Sampler (Yang et al., arXiv:2602.18176)." The outline's pre-submission checklist (Major item): "IGSD renamed consistently to CD-SSD throughout all files." The method section consistently uses "CD-SSD" and "Coarse-Draft Self-Speculative Denoising"; the experiments section uses "IGSD" in table labels (a separate issue), but the narrative uses "IGSD" when the correct term is "CD-SSD." A reviewer who reads the method section first will encounter "CD-SSD" and then find "IGSD" in related work, creating confusion about whether these are the same method.
- **Fix**: Global search-and-replace "IGSD" → "CD-SSD" throughout `related_work.md`. Update the positioning paragraph heading "Position of IGSD relative to SSD." to "CD-SSD Relative to SSD." Also update the §5.1 cross-reference from "M1+IGSD" to "M1+CD-SSD." Verify $S_{\text{accept}}$, $\tau$, and $T_{\text{draft}}$ notation in §5.3 remains consistent with notation.md (it does, but the CD-SSD label should appear at their first introduction).

---

## Major Issues

### Issue 2: Section Scope Exceeds Outline Budget by ~5x

- **Location**: Entire section (§5.1–§5.5)
- **Problem**: The outline allocates 0.75 pages to Related Work. The current section contains: §5.1 (7 method paragraphs, each 4–6 sentences, plus a position statement), §5.2 (3 method paragraphs plus a negative-result integration), §5.3 (5 method paragraphs plus a positioning statement), §5.4 (3 method paragraphs plus scope statement), §5.5 (2 comparison paragraphs). Estimated word count: 1,400–1,500 words, or approximately 3.5–3.75 pages. For NeurIPS 8-page format, related work at 3.5+ pages leaves fewer than 2 pages for the 2.5-page methods section and 3-page experiments section combined.
- **Fix**: Consolidate §5.1 from seven individual paragraphs to three clusters described in 1–2 sentences each, reserving a full paragraph only for EntropyCache (the M1 instantiation). Example cluster grouping: (i) drift-aware methods (dKV-Cache, Elastic-Cache); (ii) structure-aware / system-level methods (Window-Diffusion, Fast-dLLM, SlowFast, FlashDLM-FreeCache). The entropy formula $H_i$ and the implementation gap note should stay in the EntropyCache paragraph. All other methods in §5.1 can be reduced to cited speedup figures with one-sentence characterization. This consolidation alone recovers approximately 1 page.

### Issue 3: Section Numbering Conflict with Introduction Roadmap

- **Location**: File header reads "# 5. Related Work"
- **Problem**: The Introduction's roadmap (intro.md, final paragraph) gives the canonical ordering: "Section 2 formalizes the composability framework...Section 3 provides experimental results...Section 4 analyzes...Section 5 positions this work against related acceleration methods. Section 6 concludes." This matches the file header "# 5." However, the method section file is labeled "# 4. Methods" and the experiments section is labeled "# 5. Experiments" — creating a direct conflict where two sections claim the same number (5). The outline (outline.md) places Related Work at "## 7.", creating a three-way inconsistency (file says §5; outline says §7; experiments file also claims §5). Internal cross-references — e.g., §5.5's "Section 4 and Figure 8" — will be wrong in the final compiled paper regardless of which numbering wins.
- **Fix**: The Introduction's roadmap is the authoritative document (readers encounter it first). The final numbering should be: §2=Methods, §3=Experiments, §4=Discussion, §5=Related Work, §6=Conclusion. All section files must be renumbered accordingly. For Related Work, the file header is already §5 (matching the Introduction), so it is correct; the experiments file header "# 5." is wrong and should be "# 3." The editor pass should reconcile all headers before LaTeX compilation.

### Issue 4: §5.5 Composability Gap Claim Rests on One Tangentially-Related Reference

- **Location**: §5.5, final two paragraphs
- **Quote**: "No prior work evaluates pairwise composability of training-free inference acceleration methods for MDMs, or provides a failure-mode atlas characterizing the conditions under which method combinations destructively interfere."
- **Problem**: The only prior composability work cited is Kolbeinsson et al. (2024), which operates on model weights (knowledge editing, compression, unlearning). While the contrast is valid, a single weight-space reference does not establish that inference-time composability is a well-defined research challenge. A reviewer might ask: "Why is composability of inference methods non-trivial? Why can't you just run them sequentially?" The argument needs at least one analog from AR inference to establish that (a) combining inference optimizations is a known engineering challenge, and (b) MDM's mask-state coupling creates a structurally different — and harder — problem.
- **Fix**: Before the gap claim, add 2 sentences: "In autoregressive models, combining speculative decoding with paged memory management (vLLM [CITE:vllm]) requires careful rollback accounting when draft tokens are rejected, to avoid stale cache entries. Combining quantization with speculative decoding introduces distribution mismatch between the draft and verifier acceptance distributions. MDM's globally coupled mask state introduces an analogous but harder composability challenge: interactions propagate through the full token canvas at every denoising step, rather than being confined to the current cache position." This positions the gap as an instance of a broader inference composability problem rather than an isolated MDM-specific claim.

---

## Minor Issues

- **§5.1, position statement uses "M1+IGSD"**: "the M1+IGSD composability analysis (Section 2 and Section 3) is the first systematic evaluation..." — Two problems: (a) "IGSD" should be "CD-SSD" (Critical Issue 1); (b) Section cross-references "Section 2 and Section 3" will be wrong pending Issue 3 resolution. Flag for editor pass.

- **§5.1, self-reference phrasing**: "All seven KV-cache methods are evaluated independently...None evaluates whether KV-cache methods compose with any other acceleration family." This is factually accurate but reads as self-promotional in Related Work. Replace with a direct transition: "Section [Methods] evaluates M1 in combination with CD-SSD, M3, and each other under a unified benchmark protocol."

- **§5.2, first paragraph "sequential dependency chain"**: "creating a sequential dependency chain" introduces this concept without linking to the paper's defined term "mask-state coupling" (glossary.md). First use should read "creating a sequential dependency chain — an instance of MDM's mask-state coupling (see §[Methods])."

- **§5.2, paragraph on "Model scheduling" [CITE:modelscheduling]**: "This is a single-method analysis focused on model allocation, not step count reduction or composability." This dismissal adds no information beyond what the prior two sentences already imply. Cut the sentence.

- **§5.2, PRR classification**: "PRR requires training an auxiliary controller, making it a training-based method by the definitions used in this paper." The definitions are in the Methods section, which per the Introduction's roadmap (§2) precedes Related Work (§5). This forward reference is therefore fine — but verify section ordering is respected in the final compiled paper before submission.

- **§5.3, SSMD paragraph**: "approximately 2× reduction in forward pass count" — no benchmark qualifier. Add: "This scale gap (GPT-2 scale ~117M–1.5B parameters vs. LLaDA-8B) precludes direct comparison with results in this work."

- **§5.3, S2D2 paragraph**: "block-diffusion architectures (SDAR, LLaDA2.1-Mini with partial AR structure)" — "SDAR" is unexpanded. Replace with "semi-autoregressive diffusion with AR reversal (SDAR)."

- **§5.3, SSD speedup figures inconsistency**: §5.3 reports "Reported speedup is 2.11–3.46× on GSM8K, MATH500, HumanEval, and MBPP." The discussion section (§6.2) references "KV hit rate during refine phase: 94.0%" and the glossary entry for SSD says "Achieves 3.46x lossless speedup" without a range. A reviewer comparing sections will notice the mismatch. Harmonize: use "2.11–3.46× (task-dependent; peak 3.46× on GSM8K)" everywhere SSD speedup is cited.

- **§5.4, "BD3-LMs...achieve SOTA"**: "SOTA" is on the glossary banned list ("state-of-the-art (as standalone claim)" → "compare with specific methods and numbers"). Replace with a specific claim, e.g., "BD3-LMs outperform standard masked-diffusion baselines on [benchmark] by [margin]," or simply "BD3-LMs achieve lower perplexity than pure-diffusion baselines on standard language modeling benchmarks."

- **§5.4, thesis restatement misplaced in Related Work**: "The contribution of this paper is not competitive throughput but structural insight: understanding which training-free methods compose and why." This belongs in the Introduction's scope statement (intro.md already contains an equivalent sentence). In Related Work, it reads as defensive hedging. Replace with: "This paper studies training-free methods applied to LLaDA-8B-Instruct at inference time; training-based approaches define the upper bound of achievable throughput but are not directly comparable under a composability framework."

- **§5.4, line 59 "DLM" banned term**: "D2F is the first open-source DLM to surpass AR throughput in practice" — glossary bans "DLM" in the paper body (acceptable only when quoting external literature). Replace "DLM" with "MDM" or "masked diffusion language model."

- **§5.5, cross-reference "Section 4 and Figure 8"**: These section/figure numbers will be stale pending Issue 3 numbering resolution. Flag for the editor pass; do not update until section numbering is finalized.

- **§5.1, Table 1 forward reference missing**: No sentence points readers toward Table 1 in the Introduction, which positions these same methods by speedup. Add at the top of §5.1: "Table 1 (Introduction) summarizes reported speedup figures for these methods; the following provides the architectural context necessary to interpret those figures."

---

## Cross-Section Consistency Check

These issues were identified by comparing related_work.md against intro.md, method.md, experiments.md, and discussion.md:

1. **IGSD in experiments.md**: The experiments section (Table 2 header, §5.1 IGSD paragraph) uses "IGSD" in several places where "CD-SSD" is required. This is a separate issue from related_work.md but confirms the rename is incomplete across multiple files.

2. **KV hit rate figures vary across sections**: experiments.md §5.2 states "Measured KV cache hit rate during the refine phase: ~96%"; method.md §4.2 (CD-SSD synergy paragraph) states "94.0% (from per-seed results at $\tau=0.9$, $T_\text{draft}=16$)"; related_work.md does not cite this figure directly. The hit rate should be cited consistently as "94.0%" (the figure from verified per-seed JSON) or "~94–96%" if the spread is genuine. The experiments section's "~96%" appears to be an approximation that may not match the cited data source.

3. **Method section uses "CD-SSD" consistently; intro.md uses "IGSD"**: The Introduction (intro.md) uses "IGSD" and "Information-Gain-Driven Self-Speculative Denoising" throughout — the entire intro needs the same rename treatment as related_work.md. This is a global problem; related_work.md is not the only file affected.

4. **§5.3 positioning paragraph claims IGSD fills "the gap"**: "IGSD fills the gap left by DualDiffusion and DFlash." The Introduction also makes this claim. However, SSD (Gao et al., arXiv:2510.04147) is a training-free self-speculative MDM method — if SSD is indeed training-free (which the glossary and method section confirm), then the "gap" was already filled before this paper. The Related Work section does not resolve this tension cleanly: it says "No training-free, no-auxiliary-model speculative approach for MDMs exists" in §5.3's IGSD positioning, immediately after describing SSD as training-free. The sentence in §5.3 "IGSD operates training-free, requires no external model, and stays within the MDM regime, filling the gap at the cost of an approximate (not lossless) quality guarantee" implies SSD does not fill this gap — but the prior paragraph calls SSD training-free. **Fix**: Reframe from "filling a gap" to "providing an alternative with different design trade-offs concurrent with SSD": "CD-SSD offers a coarse-step reduced-depth draft (T_draft=16) as an alternative to SSD's full-step hierarchical tree verification; the key differentiation is composability behavior, not priority claim."

---

## Visual Element Assessment

- [x] No figures are planned for Related Work per the inline comment block ("None") — consistent with standard practice.
- [x] No table is planned for this section; Table 1 in the Introduction serves as the literature comparison table and is referenced from the introduction, not from Related Work.
- [N/A] No captions to assess.
- [ ] A forward reference to Table 1 is absent from §5.1 (see Minor Issues above). One sentence would suffice.

---

## What Works Well

1. **EntropyCache paragraph specificity (§5.1)**: The paragraph correctly reproduces the entropy formula $H_i = -\sum_v p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)$, identifies the $O(V)$ constant-cost property, and cross-references the implementation gap (1.38× vs. published 15.2–26.4×) to §4.2 and §6.3. This level of engagement distinguishes a strong submission; this paragraph should be preserved verbatim after the scope consolidation.

2. **Negative-result integration in §5.2**: The "Negative finding — M2 in this work" paragraph positions the NO_GO result within the related literature on adaptive scheduling, explains why the structural mask-state incompatibility is not resolved by hyperparameter tuning, and stakes a generalizable negative claim ("practitioners who attempt to deploy Saber-style schedules on fully masked MDMs should expect similar degradation"). This is the correct way to handle a negative result in related work — it adds value to the community rather than burying the finding.

3. **CD-SSD vs. SSD architectural contrast (§5.3, final paragraph)**: Despite the naming error, the technical content of the IGSD/CD-SSD vs. SSD comparison is the strongest paragraph in the section. The distinction between reduced-step drafting (CD-SSD's $T_\text{draft}=16$ vs. SSD's full-step hierarchical tree) is precise, and the explanation of why the frozen-token partition ($\alpha \approx 0.52$) enables KV-cache synergy while SSD's architecture does not is the clearest statement of CD-SSD's structural contribution anywhere in the paper. After the rename fix, this paragraph should be preserved verbatim.
