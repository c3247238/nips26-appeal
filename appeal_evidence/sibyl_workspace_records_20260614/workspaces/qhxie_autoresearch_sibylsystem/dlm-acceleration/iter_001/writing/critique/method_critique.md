# Critique: Method (v3 — full 7-dimension review, cross-section validated)

## Summary Assessment

The Method section is technically rigorous and unusually transparent: the M1 implementation gap (1.38× vs. published 15–26×) is disclosed proactively, the M2 NO_GO verdict is stated with exact failure numbers, and the CD-SSD frozen-token synergy mechanism is derived from first principles (H_i = 0 for deterministic positions). The composability framework in §4.1 is cleanly formalized and immediately operational. Three problems prevent a higher score: (1) a persistent IGSD/CD-SSD naming inconsistency — the glossary mandates "CD-SSD" throughout the paper body, but the Experiments and Discussion sections use "IGSD" extensively, creating terminological whiplash for any reader moving from §4 to §5; (2) the Figure 2 reference is a Markdown description file, not a rendered artifact, which will break PDF compilation; (3) a three-way numerical inconsistency in CD-SSD's GSM8K accuracy retention (63.7% in §4.2 and Table 2, 35.1% in the Introduction's scope section). These are fixable before submission, but the naming inconsistency requires a global replace across multiple files.

## Score: 7/10
**Justification**: Framework formalization (§4.1) and implementation transparency (M1 gap, M2 NO_GO) are genuinely strong. The score is capped by the naming inconsistency that runs through the downstream sections, the broken figure reference, and the three-way AccRet inconsistency that a reviewer will flag as a data integrity concern. Resolving those three issues would push this to 8/10.

---

## Critical Issues

### Issue 1: IGSD/CD-SSD naming inconsistency violates glossary mandate across sections

- **Location**: Methods §4.2 uses "CD-SSD" correctly, but Experiments §5.1–5.4 and Discussion §6 use "IGSD" throughout (Table 2 row headers, §5.1 body "IGSD ($\tau = 0.9$, $T_{\text{draft}} = 16$)...", §5.2 "M1 + IGSD," §5.3 Table 4 "IGSD" column, §5.4 "IGSD ablation," §6.2 and §6.3 "CD-SSD" in discussion but "IGSD" in intro roadmap)
- **Quote (violation example)**: Experiments §5.1: "**IGSD** ($\tau = 0.9$, $T_{\text{draft}} = 16$): IGSD achieves 4.57× speedup on GSM8K..."
- **Problem**: The glossary.md explicitly bans "IGSD (in paper body)" and mandates "CD-SSD." It also states: "Do NOT use 'IGSD' in the paper body." The Methods section correctly uses CD-SSD, but the Experiments section reverts to IGSD in every table row header and every text reference. This creates terminological inconsistency across an entire section. A reviewer who notices this will question whether the rename was actually completed, and may flag it as sloppy presentation. More practically: if the paper is compiled as-is, readers see CD-SSD in §4 and IGSD in §5, and must infer they are the same thing.
- **Fix**: Global replace in sections/experiments.md and sections/discussion.md: every instance of "IGSD" → "CD-SSD." This includes Table 2 row headers, Table 3 pair names ("M1 + IGSD" → "M1 + CD-SSD"), Table 4 failure mode entries, Table 5 configuration labels, and all body text. Also rename the figure description file from `fig2_igsd_architecture_desc.md` to `fig2_cd_ssd_architecture_desc.md` if it is retained, and update its internal title.

### Issue 2: Figure 2 reference is a Markdown description file, not a rendered artifact

- **Location**: §4.2, CD-SSD subsection (line 163)
- **Quote**: `![CD-SSD three-phase pipeline: Draft, Partition, and Refine](figures/fig2_igsd_architecture_desc.md)`
- **Problem**: The figure reference points to a `.md` description file. This will fail to compile in any LaTeX or PDF pipeline — `\includegraphics` cannot embed a Markdown file. Figure 2 is the sole figure in the Methods section, and the CD-SSD three-phase pipeline (draft → partition → refine) is described in substantial text that relies on the reader having a visual. The figure description file (`fig2_igsd_architecture_desc.md`) is well-specified with TikZ-ready content, but the artifact itself has not been rendered.
- **Fix**: Render Figure 2 as a PDF or PNG using the spec in `writing/figures/fig2_igsd_architecture_desc.md`. Update the reference to `figures/fig2_cd_ssd_architecture.pdf` (following the naming convention of Figures 3–8 which are `fig3_pareto_curves.pdf`, `fig4_ortho_bars.pdf`, etc.). If rendering is blocked, change the image reference to a `[Figure 2 placeholder — render from fig2_cd_ssd_architecture_desc.md]` comment so the broken reference does not silently fail.

### Issue 3: Three-way numerical inconsistency in CD-SSD GSM8K accuracy retention

- **Location**: §4.2 CD-SSD operating point (line 197), Introduction §1 scope section, Experiments Table 2
- **Quote (methods)**: "CD-SSD achieves 4.57$\times$ speedup on GSM8K with 63.7% accuracy retention"
- **Quote (intro §1, scope)**: "its 35.1% accuracy retention on GSM8K is not deployable independently"
- **Quote (Table 2)**: "IGSD $\tau=0.9, T_d=16$: GSM8K Acc. 45.3%, AccRet 0.637"
- **Problem**: Three different numbers appear for the same quantity (CD-SSD GSM8K accuracy retention at the operating point): 63.7% in §4.2, 35.1% in the Introduction's scope section, and 63.7% (= 45.3% / 71.2%) from Table 2. The Introduction and the Methods section disagree by 28.6 percentage points on a central performance claim. A reviewer will notice this immediately and raise a fatal objection about data integrity. The 35.1% value appears to come from an older experiment snapshot or from the pilot data; the full-scale 3-seed result (Table 2) gives 63.7%.
- **Fix**: Audit `exp/results/full_igsd/igsd_pareto_full.json` (or the equivalent) to confirm the authoritative AccRet at $\tau=0.9$, $T_{\text{draft}}=16$. If 63.7% is correct (as Table 2 implies), update the Introduction scope paragraph from "35.1% accuracy retention" to "63.7% accuracy retention (45.3% absolute on GSM8K vs. 71.2% baseline)." If 35.1% is correct, update both §4.2 and Table 2. Either way, all three locations must show the same number before submission.

---

## Major Issues

### Issue 4: M3 operating-point speedup stated as 1.68× in §4.2 but 1.33× in Table 2

- **Location**: §4.2, M3 operating point description (line 144)
- **Quote**: "**Operating point**: $w = 0.3$, which achieves 1.68$\times$ speedup on GSM8K with accuracy retention of 103.9%"
- **Problem**: Experiments Table 2 shows M3 at $w=0.3$ achieving 1.33× speedup (identical for w=0.3, 0.5, 0.7; 1.34× at w=1.0). The 1.68× figure does not appear in Table 2. The glossary "Key Resolved Findings" table explicitly flags this: "M3 speedup 1.68x vs. 1.33x — USE 1.33x (combined); 1.68x is GSM8K-specific only." The outline pre-submission checklist also flags it as "Major (required before venue submission)." This is a known inconsistency that was not fixed when writing this section.
- **Fix**: Update the operating-point description to: "achieves 1.68$\times$ speedup on GSM8K (1.33$\times$ combined across all benchmarks) with accuracy retention of 103.9%." Both numbers must appear together since they measure different things; the combined figure (1.33×) is the one used in Table 2 and pairwise comparisons.

### Issue 5: The tau=0.0 ablation preview in §4.2 presents QAS values from a different scale than the main results

- **Location**: §4.2, CD-SSD subsection, "tau=0.0 ablation preview" paragraph (lines 221–229)
- **Quote**: "yields QAS = 4.198, statistically indistinguishable from naive 16-step denoising (QAS = 4.458, $\Delta$ = $-$5.8%)"
- **Problem**: The main single-method results (§5.1, Table 2) report CD-SSD QAS = 1.194 at the operating point. The tau=0.0 preview uses QAS values of 4.198 and 4.458. These are from the 200-sample ablation subset (§5.4, Table 5), not from the full benchmark, and the QAS scale differs substantially. A reader encountering these numbers in the Methods section — before seeing §5.4's scale qualification — will be confused by QAS values more than 3× higher than anything in the main results. Additionally, the Discussion section's Table 7 (tau=0.0 resolution) shows yet another set of QAS values (4.198 for CD-SSD(tau=0.0), 4.458 for naive-T16, 2.950 for CD-SSD tau=0.9), which are consistent with §5.4 but inconsistent with §5.1's 1.194 — because they use different subsets and include HumanEval in QAS differently.
- **Fix**: Either (a) remove the QAS values from the methods section preview entirely (replace with a forward pointer: "Section 5.4 ablates $\tau$ and §6.3 resolves the paradox"), or (b) add a parenthetical immediately after the first QAS value: "QAS = 4.198 (measured on 200 GSM8K + HumanEval subset, §5.4; this scale differs from the full-benchmark QAS = 1.194 in Table 2 due to subset selection and benchmark weighting)." Option (a) improves flow and eliminates the numerical confusion.

### Issue 6: Ortho derivation in §4.2 produces 1.094 from the stated inputs, not 1.385

- **Location**: §4.2, CD-SSD "Synergy with M1" paragraph (lines 217–218)
- **Quote**: "This yields a combined speedup of 5.13$\times$ for M1+CD-SSD, exceeding the product of individual speedups ($1.38 \times 3.40 = 4.69\times$) and producing Ortho = 1.385."
- **Problem**: Using the formula Ortho = 5.13 / (1.38 × 3.40) = 5.13 / 4.69 = 1.094, not 1.385. The Ortho = 1.385 from Table 3 (pairwise experiments) uses individual speedups measured within the pairwise experimental setup (200 samples, 2 seeds, where CD-SSD's individual speedup is different from the full-scale 3.40×). The methods section's presentation gives the impression that the denominator is 1.38 × 3.40 = 4.69, which yields a different Ortho than reported. This arithmetic inconsistency will be caught by any reviewer who verifies the formula.
- **Fix**: Either (a) state explicitly that "Ortho = 1.385 is computed from the pairwise experiment's own individual speedup measurements, not from the full-scale single-method results; the denominator in the pairwise setting is 1.38 × 3.71 = 5.12, giving Ortho ≈ 1.385" (with the actual pairwise CD-SSD speedup from Table 3 raw data), or (b) quote the formula as "Ortho = 5.13 / (M1_pairwise × CD-SSD_pairwise)" and state the pairwise individual speedups in a parenthetical. The current text implicitly uses different denominators in the method explanation vs. the reported value.

### Issue 7: Section numbering mismatch with the Introduction's roadmap

- **Location**: Intro §1 closing roadmap paragraph vs. §4 section header
- **Quote (intro)**: "Section 2 formalizes the composability framework and presents IGSD and the three other method families. Section 3 provides experimental results..."
- **Problem**: The Introduction's roadmap labels this "Section 2" but the file header is "# 4. Methods." The Introduction was written with a different section numbering (likely a 6-section structure: Intro, Methods, Experiments, Discussion, Related Work, Conclusion) while the current outline has a 7-section structure. This is internally inconsistent and will confuse readers navigating by section number.
- **Fix**: Update the Introduction's roadmap paragraph to use the correct section numbers: "Section 4 formalizes the composability framework... Section 5 provides experimental results... Section 6 analyzes why composability is binary... Section 7 reviews related work... Section 8 concludes."

---

## Minor Issues

- **§4.1, Ortho formula — denominator interpretation**: The sentence "The denominator is the multiplicative ideal: the combined speedup that would be observed if the two methods eliminated independent, non-overlapping bottlenecks" — technically, multiplicative combination requires not just non-overlapping bottlenecks but also that each method's speedup is measured under the same conditions. Since the individual speedups in the denominator come from standalone measurements and the combined speedup from joint measurement, there is an implicit assumption of identical execution environments. Add one clause: "provided both measurements use the same hardware, batch size, and inference protocol."

- **§4.2, M1 entropy formula — missing position index**: The entropy formula at line 77 shows $H_i = -\sum_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)$ but notation.md defines $p_\theta$ as a function mapping to $\Delta(\mathcal{V})^N$ (a distribution over all positions). The formula should clarify it evaluates $p_\theta$ at position $i$: write $p_\theta(v \mid \tilde{x}_t)_i$ or $p_\theta(v \mid \tilde{x}_t)[i]$ to match the notation convention. This is how notation.md defines $H_i$ and how the cache refresh decision is actually made.

- **§4.2, M2 failure mode label**: The M2 subsection says it is "documented as failure mode FM1 (Section 5.3)" — this is correct (FM1 = step_starvation = M2). Good. However, the methods section calls the cache overhead mode "documented as failure mode FM2" (line 88), while the failure atlas in §5.3 labels it "F2" (without the "M" prefix). Standardize to "FM2" or "F2" across the entire paper; the methods section uses "FM2" while experiments uses "F2."

- **§4.2, M2 accuracy numbers discrepancy**: The methods section states "At $J = 2$, GSM8K accuracy retention falls to 54.4%; at $J = 4$ it collapses to 13.0%; at $J = 8$ it reaches 24.3% (catastrophic)." Experiments Table 2 shows M2 $J=2$ AccRet = 0.544 (54.4% — matches), $J=4$ AccRet = 0.130 (13.0% — matches), $J=8$ AccRet = 0.243. These are consistent. Good.

- **§4.3, baseline TPS explanation**: GSM8K baseline is 31.0 ± 4.0 tok/s vs. MATH500 at 79.2 ± 0.1 tok/s — a 2.56× throughput difference at baseline without any acceleration. The methods section provides no explanation for this large gap. A reviewer will ask. Add one sentence: "The substantially higher TPS on MATH500 relative to GSM8K reflects shorter generated output lengths: MATH500 problems typically produce brief numeric answers, while GSM8K problems require multi-step chain-of-thought responses."

- **§4.3, MBPP sample count**: The baseline table says MBPP has 257 samples. The outline and glossary also say 257. However, the standard MBPP test set has 374 or 500 samples depending on the split used. Verify that 257 is the correct count for the evaluation subset used and add a note: "MBPP test set (257 samples, standard evaluation split)."

- **§4.2, MATH500 AccRet 243.9% for M3**: The methods section mentions "The MATH500 retention figure (243.9%) is inflated by the low 11.1% baseline and should not be interpreted as a real 2.4× improvement." This is correct and important to flag. However, it should also note the absolute accuracy to give context: "The MATH500 retention figure (243.9%, absolute accuracy 0.27 at w=0.3 vs. baseline 0.111) is inflated by the near-floor baseline and should not be interpreted as a real 2.4× reasoning improvement." Without the absolute accuracy, readers may not realize how tiny the actual improvement is.

- **§4.2, CD-SSD KV hit rate inconsistency across sections**: Section 4.2 ("Synergy with M1") states "The measured KV-cache hit rate during the refine phase reaches 94.0%." Section 5.2 states "Measured KV cache hit rate during the refine phase: ~96% (vs. ~60% during the draft phase)." The value is 94.0% in the Methods but ~96% in the Experiments. Additionally, section 6.2 of Discussion says "Measured KV hit rate during refine phase: 94.0%." All three references should cite the same number. The ~96% in Section 5.2 appears to be an approximation that was rounded up; the authoritative value from the igsd_p2 per-seed JSON is 94.0%. Standardize to 94.0% (or the exact figure from the raw JSON) across all three references.

---

## Visual Element Assessment

- [x] Figure 2 is referenced before the CD-SSD pipeline is fully described (line 163, "Figure 2 illustrates the three-phase pipeline")
- [ ] **Figure 2 is NOT a rendered artifact** — the reference points to `figures/fig2_igsd_architecture_desc.md`, a Markdown file. This will produce a broken image in any compiled output and must be generated before submission.
- [x] The description file (`fig2_igsd_architecture_desc.md`) provides a detailed, TikZ-ready spec with color coding, annotations, and dimensions — sufficient to generate the figure
- [ ] The figure description file is named "igsd" not "cd_ssd" — a naming inconsistency that should be resolved when the file is rendered
- [x] No other figures are embedded in the Methods section; §4.3 uses inline tables only (consistent with the outline's Figure & Table Plan)
- [ ] The synergy discussion (§4.2, lines 209–219) describes a KV hit rate jump from 60% to 94% that is the paper's key mechanistic evidence — add a forward reference: "(see Figure 8, §6.2 for empirical KV hit rate measurements)" to help readers find the visual confirmation

---

## What Works Well

1. **M1 implementation gap disclosure (§4.2, lines 92–100)**: The explanation that "our M1 implementation computes entropy and applies the cache-refresh logic but still executes full transformer forward passes" is exemplary transparency. Most papers would quietly omit a 10–19× gap between claimed and implemented speedup. The argument that the dimensionless Ortho metric remains valid despite the absolute speedup gap is logically correct and preempts the most predictable reviewer objection.

2. **M2 NO_GO verdict structure**: The subsection states the verdict in the header ("**Verdict: NO\_GO**"), provides exact numbers at each tested parameter ($J=2$: 54.4%, $J=4$: 13.0%, $J=8$: 24.3%), explains the root cause (sequential cumulative conditioning → mask inconsistency cascade), distinguishes this from a hyperparameter failure, and includes the appropriate caveat about Saber's backtracking. This is the correct way to report a structural negative result.

3. **CD-SSD frozen-token KV synergy derivation (§4.2, "Synergy with M1," lines 209–219)**: The argument chain is tight: frozen tokens → deterministic distribution → H_i = 0 → guaranteed cache hit (since 0 < η = 2.0) → 94% CHR during refine → super-multiplicative speedup. Each step follows from the previous. The mechanism is testable, falsifiable (by measuring CHR during refine), and produces a quantitative prediction (Ortho > 1.0) that the experiments confirm. This is the paper's strongest scientific argument and it is presented clearly in the Methods section rather than being buried in the Discussion.
