# Writing Quality Review

## Summary
ComposeAccel presents the first systematic factorial study of training-free acceleration composition for diffusion language models, evaluating three method families (entropy-based KV caching, AR-guided unmasking, and information-geometric step distillation) individually and in pairwise/three-way combinations on LLaDA-8B-Instruct and Dream-7B-Instruct. The paper finds that M1+IGSD composes near-orthogonally (Ortho = 0.96), M1+M3 shows destructive interference (Ortho = 0.41--0.43), and the best composed DLM configuration (QAS = 1.07) does not close the gap with AR inference (QAS = 3.08). The argument structure is clear and the results are honestly reported, including negative results and an AR comparison that works against the paper's own methods.

## Detailed Assessment

### Structural Coherence: 8/10
The paper follows a logical progression: motivation (Introduction) -> prior work (Related Work) -> framework and methods (Methods) -> results (Experiments) -> cross-model validation (Section 5) -> discussion (Section 6) -> conclusion (Section 7). Transitions between sections are generally motivated.

Two structural issues reduce the score:

1. **Section numbering inconsistency.** The paper contains a parenthetical note at line 47: "*(Note: Section numbering in the original sections used '3' for Related Work; we renumber here for the integrated paper.)*" This editorial artifact must be removed before submission.

2. **Section 5 naming mismatch with content.** Section 5 is titled "Cross-Model and AR Comparison" but also contains batch sensitivity analysis (Section 5.3), which is neither cross-model nor AR-related. The batch sensitivity material feels orphaned --- it is only three sentences long and lacks the figure (Figure 8) planned in the outline. Either expand it into a proper subsection with the planned figure or relocate it to the appendix.

3. The outline planned 8 figures. The paper references 7, and Figure 8 (batch sensitivity) was dropped without explanation. The batch sensitivity section (5.3) consequently reads thin compared to adjacent sections.

### Notation & Terminology Consistency: 9/10
Cross-checking against `notation.md` and `glossary.md` reveals strong consistency overall. All core symbols ($\eta$, $\tau$, $T_{\text{draft}}$, $T_{\text{full}}$, $w_g$, $\mathcal{S}_{\text{accept}}$, $\mathcal{S}_{\text{reject}}$, $\alpha$, QAS, Ortho, CHR, AccRet) are used consistently throughout.

Minor deviations found:

1. **$\tau$ dual definition.** In `notation.md`, $\tau$ is defined as "KL-divergence confidence threshold." In the paper (Section 3.3, line 163), $\tau$ is described as a confidence threshold on maximum softmax probability, not KL divergence. The notation file should be updated to match the paper's actual usage, or vice versa. This is a documentation inconsistency, not a paper inconsistency --- the paper itself is internally consistent.

2. **$N_{\text{gen}}$** appears in the CHR formula (line 147) and in the $\alpha$ definition (line 167) but is never formally defined. From context it means the number of generation-token positions (excluding prompt). Adding a one-line definition would help.

3. **$r_{\text{accept}}$ vs. $\alpha$.** Table 3 uses $r_{\text{accept}}$ (accept rate) for IGSD, while the text in Section 3.3 uses $\alpha$ (frozen fraction). `notation.md` defines both but their relationship is unclear. Are they the same quantity measured at different points? The paper should clarify.

### Claim-Evidence Integrity: 8/10
Most claims are backed by specific numbers, and the paper is admirably transparent about limitations (projected vs. measured speedup, pilot-scale pairwise data, weak MATH500 baseline). Numbers in the paper are internally consistent after the standardization pass noted in `visual_audit.md`.

Issues found:

1. **Proposal numbers vs. paper numbers.** The proposal (`proposal.md`) reports M1+IGSD achieving 8.88x speedup at 52% accuracy retention (iter_001 data). The paper reports M1+IGSD at 2.75x speedup and 58.9% accuracy retention (iter_002 data). This is expected (different iterations), but the abstract still references "2--14x slower" for DLMs, which partially derives from the iter_001 context. The "14x" figure at line 1 corresponds to Qwen2.5-7B at batch=8 (471 / 33.8 = 13.9x), which is correct.

2. **Table 3 sample size inconsistency.** M1 results use N=1319 (full GSM8K), IGSD uses N=200, and M3 uses N=100. This is disclosed in the text (line 251: "Note: M3 results are from 100-sample evaluation") but not in Table 3 itself. A reader scanning the table would compare M1 QAS=0.98 (N=1319) against M3 QAS=1.69 (N=100) without realizing the different statistical power. The N column exists but is easy to miss. Consider adding a table footnote.

3. **Confidence gate ablation claim.** Section 4.4 (line 320) states "confidence partitioning provides zero measurable benefit at this operating point" and that "IGSD at $T_{\text{draft}}$=32 is functionally equivalent to naive step reduction." This is a strong claim that somewhat undercuts IGSD's stated mechanism. The paper handles this honestly, but the abstract and contributions still frame IGSD as using "inter-step KL divergence" as a design principle. The paper should more explicitly acknowledge in Section 1.2 (Contributions) that the confidence gate is a design vehicle rather than a performance driver.

4. **M3 accuracy retention > 100%.** The paper reports M3 at $w_g$=0.3 achieving 103.9% accuracy retention. This is a striking result that deserves a brief mechanistic explanation (which is provided in Section 3.4: "the guide model corrects some DLM errors"). The claim is adequately supported.

### Visual Communication: 6/10
This is the weakest dimension. Two of seven planned figures are missing, and one section lacks its planned visual entirely.

**Missing figures:**
- **Figure 2 (architecture diagram):** A `[TODO]` placeholder appears in the paper text at line 231. This is the single most important figure for reader comprehension --- it shows how the three methods integrate. A detailed specification (`fig2_architecture_desc.md`) exists but no PDF has been generated. **Critical gap.**
- **Figure 7 (KL divergence profile):** Referenced in text at line 324 with a `[TODO]` tag. Raw data exists (`igsd_kl_profiles_raw.json`). This figure is key evidence for the ablation analysis (refuting the inverted-U hypothesis). **Major gap.**
- **Figure 8 (batch sensitivity):** Planned in outline but never generated or referenced. Section 5.3 reads as a stub without it.

**Present figures (5 of 7):**
- Figures 1, 3, 4, 5, 6 are present as PDFs and properly referenced in text before they appear.
- Figure captions are descriptive and self-explanatory.
- The figure plan from the outline is mostly followed, with the two exceptions noted above.

**Text-heavy sections needing visual support:**
- Section 2.2 (Training-Free Acceleration Families) is dense text with a long table. Table 1 helps, but a taxonomy diagram showing the four families and their computational axes would aid comprehension.
- Section 6.1 ("Why M1+IGSD Composes but M1+M3 Does Not") is pure text explanation of mechanisms. The visual audit suggested a "composition taxonomy summary diagram" --- this would strengthen the discussion significantly.

### Writing Quality: 8/10
The writing is generally strong: direct, evidence-driven, and free of most banned patterns. Sentences are specific and quantified. The paper leads with results rather than preamble.

**Banned patterns found:**

1. Line 11: "Over 20 training-free acceleration methods have targeted this bottleneck across three computational axes" --- borderline. The sentence is factual and specific (citing examples immediately after), but "over 20" is the kind of vague quantification that could be made precise.

2. No instances of "In recent years...", "It is worth noting...", "Furthermore...", "groundbreaking", "game-changing", "revolutionary", or "novel" (except one use of "novel" in `proposal.md` which is not in the paper itself).

**Unclear or complex sentences:**

1. Line 89 (Section 2.2): "IGSD occupies the boundary between step scheduling and speculative decoding: it reduces computation by running fewer steps (like M2-family methods) but uses a confidence-based partition to selectively refine (like speculative methods)." This sentence is grammatically fine but cognitively heavy --- three comparisons in one sentence. Consider splitting.

2. Line 96--98 (Section 2.3): "Kolbeinsson et al. study composability of LLM interventions---knowledge editing, model compression, and unlearning applied to the same model weights. Their framework is conceptually related to ours, but operates on static weight-space modifications rather than dynamic inference-time algorithms. The DLM setting introduces a temporal composability challenge: the mask state evolves at every step, and two methods may interact differently at step 10 (when 85% of tokens are masked) versus step 60 (when 5% remain masked)." This is well-written and specific --- highlighting as a positive example.

3. Line 380--381 (Section 5.3): "M1+IGSD at batch=1: 1.64x speedup, 96 TPS. At batch=4: accuracy improves to 50% (from 45%) but TPS drops to 56. At batch=8: accuracy 52%, TPS=34." These numbers appear without context --- what is the baseline batch=4 and batch=8 TPS? The reader cannot interpret "TPS drops to 56" without knowing the batch=4 baseline.

**Residual editorial artifacts:**

1. Line 47: "*(Note: Section numbering in the original sections used '3' for Related Work; we renumber here for the integrated paper.)*" --- must be removed.
2. Line 231: "[TODO: Figure 2 -- generate from fig2_architecture_desc.md specification]" --- must be resolved.
3. Line 324: "[TODO: Figure 7 -- Per-step KL divergence profile...]" --- must be resolved.
4. Lines 233--237 and 328--337: HTML comment blocks with figure/table inventories (<!-- FIGURES ... -->) --- these are internal bookkeeping and should be removed before submission.
5. Lines 444--457: A "Figures and Tables" index section at the end --- this is unusual for ML venue papers and should be removed (or converted to a supplementary material index).

## Issues for the Editor

1. **Critical** -- **Missing Figure 2 (architecture diagram)**: Section 3.5 / line 229--231. The paper has a `[TODO]` placeholder where the architecture diagram should be. This is the key visual for understanding the method. **Fix**: Generate Figure 2 from the detailed specification in `fig2_architecture_desc.md`. Use TikZ, draw.io, or equivalent. Remove the `[TODO]` tag and the HTML comment block.

2. **Critical** -- **Missing Figure 7 (KL divergence profile)**: Section 4.4 / line 324. The ablation analysis describes the KL profile finding in text only; the figure is a `[TODO]` placeholder. Raw data exists in `igsd_kl_profiles_raw.json`. **Fix**: Write a generation script (matplotlib, line plot with shaded std band, horizontal reference lines at tau values), generate the PDF, and replace the `[TODO]` tag.

3. **Major** -- **Editorial artifacts remaining**: Lines 47, 233--237, 328--337, 444--457. Internal notes, HTML comments, and a trailing figure index survive in the manuscript. **Fix**: Remove the parenthetical note at line 47, delete all `<!-- FIGURES ... -->` comment blocks, and delete the "Figures and Tables" section at the end (lines 442--457).

4. **Major** -- **Table 3 sample size transparency**: Section 4.1 / Table 3. Methods in the same table are evaluated on different sample sizes (N=1319, 200, 100) but the table does not highlight this adequately. **Fix**: Add a table footnote explaining the different sample sizes and their implications for statistical comparison. Consider adding a visual separator or grouping rows by sample size.

5. **Minor** -- **$N_{\text{gen}}$ undefined**: First appears in the CHR formula (line 147). **Fix**: Add a one-line definition after the CHR formula: "$N_{\text{gen}}$ denotes the number of generation-token positions (excluding the prompt)."

## What Works Well

1. **Honest AR comparison (Section 5.2, Table 7).** The paper voluntarily reports that its best result (QAS = 1.07) falls 2.9x short of AR inference (QAS = 3.08). This level of intellectual honesty is rare and significantly strengthens the paper's credibility. The framing --- "the value of this study is in mapping the composition design space, not in claiming speed parity" --- is exactly right.

2. **Mechanistic explanations for composition outcomes (Section 6.1).** The paper does not just report that M1+IGSD composes while M1+M3 does not; it explains *why* with specific mechanisms (frozen tokens creating low-entropy KV entries, guide model overhead dominating marginal cache speedup). The CHR = 83.4% during composition vs. 56.2% standalone is a concrete, verifiable data point.

3. **Transparent negative results (Sections 4.4 and 6.5).** The confidence gate ablation showing zero benefit at $T_{\text{draft}}$=32, the M2 NO_GO verdict, and the reconciliation of the iter_001 M1+M3 pilot discrepancy are all reported forthrightly. This builds trust in the positive results.

SCORE: 7
