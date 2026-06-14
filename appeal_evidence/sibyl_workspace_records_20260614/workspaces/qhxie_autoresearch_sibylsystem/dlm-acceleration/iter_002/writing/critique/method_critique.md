# Critique: Method (Section 4)

## Summary Assessment
The Method section is well-structured and provides clear formal definitions of the composability framework (QAS, Ortho), three acceleration methods (M1, IGSD, M3), and experimental setup. The writing is direct, evidence-rich, and avoids most banned patterns. However, the section has a significant structural tension: IGSD receives disproportionate depth relative to M1 and M3, creating an identity crisis between "analysis paper" and "methods paper." Several numerical claims lack cross-verification against the experiments section, and the M1 subsection's dual-speedup reporting strategy, while honest, is confusingly organized.

## Score: 7/10
**Justification**: Strong metric definitions, honest reporting of engineering failures (d2Cache), and clear algorithmic exposition. Reaching 8/10 requires resolving the M1 speedup inconsistencies between this section and the experiments section, tightening the IGSD description to match the "analysis paper" framing, and adding the missing Figure 2 reference before the architecture is described.

## Critical Issues

### Issue 1: M1 speedup numbers inconsistent between method and experiments sections
- **Location**: Section 4.2, paragraph on measured speedup
- **Quote**: "At eta = 0.5: 1.16x speedup, 56% CHR on GSM8K (full-scale, 1319 samples). At eta = 2.0: 1.50x speedup, 60% CHR."
- **Problem**: The method section reports M1 eta=2.0 CHR as 60% and speedup as 1.50x. The experiments section (Table 3) reports M1 eta=2.0 CHR as 60.2% and speedup as 1.50x with AccRet=55.5%. These are consistent. However, the method section states M1 eta=0.5 has "56% CHR" while the experiments section reports CHR=56.2%. More critically, the method section claims "projected speedup" at eta=0.5 of 2.27x with "CHR = 93.3%" -- but the experiments section reports measured CHR at eta=0.5 as 56.2%, not 93.3%. Where does the 93.3% figure come from? The discussion section (7.1) also cites "CHR = 93.3% on GSM8K" for standalone M1 at eta=0.5. If this is the projected/ideal CHR vs. measured CHR, the distinction needs explicit labeling in both locations.
- **Fix**: Add a clear sentence explaining that measured CHR (56.2%) reflects the Python-level implementation where only positions that *would have* reused cache are tracked, while the projected CHR (93.3%) is computed differently (e.g., from entropy distribution statistics). Define the two CHR measurement methodologies before reporting either number.

### Issue 2: M3 TPS baseline discrepancy across sections
- **Location**: Section 4.4, final paragraph
- **Quote**: "The measured speedup is 1.68x on GSM8K (52.0 TPS vs. 31.0 TPS baseline)"
- **Problem**: The experimental setup in Section 4.5 states the GSM8K baseline is 33.8 TPS. The M3 subsection claims the baseline is 31.0 TPS. The discussion section (7.1) refers to "baseline 58.5" TPS. The intro section references 34 TPS. These are four different baseline TPS numbers: 31.0, 33.8, 34, and 58.5. If 31.0 is a different measurement context (e.g., pre-warmup or different hardware state), it must be explained. Readers will immediately flag this inconsistency.
- **Fix**: Standardize all baseline TPS to the Section 4.5 reference value (33.8 TPS on GSM8K). If M3's 1.68x speedup is relative to 31.0 TPS, recalculate against the canonical 33.8 TPS baseline and note the discrepancy explicitly.

## Major Issues

### Issue 3: Figure 2 referenced before it is shown -- but placement is ambiguous
- **Location**: Section 4, opening paragraph (line 3) and line 5
- **Quote**: "Figure 2 illustrates the full composed pipeline." followed immediately by the figure markdown
- **Problem**: The figure appears at the top of the section, which is correct for forward-referencing. However, the figure markdown `![ComposeAccel architecture...](figures/fig2_architecture.pdf)` is placed before any method content is described, so the reader encounters an architecture diagram with "IGSD draft-partition-refine pipeline with M1 and M3 integration points" before learning what IGSD, M1, or M3 are. The outline specifies Figure 2's purpose as "Show the IGSD draft-partition-refine pipeline and how M1 KV caching and M3 AR guidance integrate" -- this purpose is better served after Section 4.1 or after Section 4.3 (IGSD description).
- **Fix**: Move the figure placement to after Section 4.3 (IGSD) so the reader has context for all three methods before seeing the architecture diagram. Keep the forward reference in the opening paragraph.

### Issue 4: IGSD confidence partitioning uses max softmax probability, not KL divergence
- **Location**: Section 4.3, confidence partitioning paragraph
- **Quote**: "Token i is frozen if its logit confidence (measured as the maximum softmax probability) exceeds the threshold tau"
- **Problem**: The method name is "Information-Geometric Step Distillation" and the KL divergence signal is described prominently. However, the actual partitioning criterion uses max softmax probability, not KL divergence. The tau threshold in the notation table (notation.md) is defined as "KL-divergence confidence threshold," but in practice it is a softmax-probability threshold. The KL divergence is only used as a post-hoc analysis signal (Section 5.4), not as the operational partitioning criterion. This creates a mismatch between the method name/framing and the actual mechanism.
- **Fix**: Acknowledge this explicitly in the text: "Despite the method's information-geometric framing (which motivated the design), the operational partitioning criterion is the maximum softmax probability at T_draft, not the KL divergence directly. Section 5.4 validates that the KL profile provides theoretical justification for the threshold insensitivity." Also update notation.md to define tau as "confidence threshold (maximum softmax probability)" rather than "KL-divergence confidence threshold."

### Issue 5: Missing MATH500 data throughout method descriptions
- **Location**: Sections 4.2, 4.3, 4.4
- **Problem**: The combined benchmark metric is defined as 0.7 * GSM8K + 0.3 * MATH500 in Section 4.1, but all method descriptions in 4.2--4.4 report only GSM8K numbers. M1 reports "56% CHR on GSM8K"; IGSD reports "frozen fraction measured at 0.886 +/- 0.133 across 100 GSM8K samples"; M3 reports "GSM8K accuracy retention is 103.9%". No MATH500 numbers appear in any method subsection. The outline notes MATH500 baseline is 11.1%, but method-level MATH500 behavior is never mentioned.
- **Fix**: Add at least one MATH500 data point per method to justify the combined metric weighting. If MATH500 data is deferred to Section 5, state this explicitly: "MATH500 results for all methods are reported in Section 5.1."

### Issue 6: Table 2 caption format inconsistent with paper conventions
- **Location**: After the metric definitions table (line 33)
- **Quote**: "**Table 2.** Metric definitions and interpretation thresholds."
- **Problem**: The caption is placed *below* the table as bold markdown text, but it lacks the standard LaTeX-style `\caption{}` or consistent formatting with other tables. More importantly, this is the only table in the method section with an explicit "Table N" label; the outline assigns Table 2 to this exact location, which is correct, but the placement after the table conflicts with the convention of having captions above tables in ML papers.
- **Fix**: Move the caption above the table or ensure the final LaTeX conversion handles this consistently. This is a minor formatting issue but will confuse the LaTeX writer.

## Minor Issues
- **Line 3**: "Sections 4.2--4.4" -- ensure the final LaTeX uses an en-dash, not double hyphens. Current markdown is fine but verify in compilation.
- **Line 11, QAS formula**: The text says "No penalty factor is applied" -- this is a response to an implicit criticism from iter_001 that used a 0.5x penalty. The iter_001-specific context should be removed from the method section; it belongs in supplementary material or an appendix. A reader of the paper does not know about iter_001.
- **Line 23**: "HumanEval results are reported in the appendix but excluded from combined metrics due to a degenerate 2.4% baseline" -- the word "degenerate" is imprecise in a mathematical context (where it has a specific meaning). Use "uninformative" (as the intro section does) or "near-zero."
- **Line 43, M1 entropy threshold**: "eta in {0.5, 1.0, 2.0}" -- the notation table uses $\eta$ consistently, but the method text sometimes writes "eta" in code-like font and sometimes in math mode. Ensure consistent $\eta$ throughout.
- **Line 50, d2Cache overhead**: "15.2x overhead" -- this number appears in the method section, experiments section, discussion section, and intro section. While repetition across sections is expected, the method section should note it once cleanly and reference it rather than restating the full d2Cache failure narrative (which takes a full paragraph). The same story appears in 4.2, 5.1, 7.3, and the intro scope statement.
- **Line 77, KL formula**: The set $\mathcal{M}_t$ is defined as "set of masked positions at step t" in both the method section and notation.md. However, Section 4.3 also uses $\mathcal{S}_{\text{accept}}$ and $\mathcal{S}_{\text{reject}}$ with $\{1, \ldots, N\}$ as the universe. Is $\mathcal{M}_t$ a subset of $\{1, \ldots, N\}$? Clarify whether $N$ here is total sequence length or generation-token count only, since CHR uses $N_{\text{gen}}$.
- **Line 95**: "The computational overhead of the confidence gate (a single argmax over the vocabulary dimension) is O(N x V)" -- argmax is O(V) per position, so O(N x V) total. This is correct but might confuse readers who think argmax is O(1). Clarify: "O(V) per position, O(N x V) over all positions."
- **Line 107, M3**: "GSM8K accuracy retention is 103.9% (the guide model corrects some DLM errors)" -- this parenthetical claim (guide corrects errors) lacks evidence here. It is a mechanistic hypothesis. Either remove the parenthetical or add "suggesting that" to qualify it.
- **Section 4.5, baseline**: "GSM8K: 71.2% +/- 1.5% accuracy, 33.8 TPS (generation-only, post-warmup)" -- the +/- 1.5% is cross-seed variance (seeds 42, 123, 456). State this explicitly: "71.2% +/- 1.5% accuracy (mean +/- std across 3 seeds)."
- **Section 4.5, M2 exclusion**: The M2 NO_GO paragraph is placed at the end of experimental setup, which makes it feel like an afterthought. The outline places M2 exclusion in the scope statement (Introduction). Since it is also here, it is mentioned in two places -- acceptable for emphasis, but the method section version could be shortened to a single sentence with a cross-reference.

## Visual Element Assessment
- [x] Figure 2 (architecture diagram) is planned and matches the outline
- [ ] Figure 2 is referenced before it appears -- but it appears *before* any method content, reducing its pedagogical value (see Major Issue 3)
- [ ] Table 2 caption is placed below the table, not above (see Major Issue 6)
- [x] No text-heavy sections that urgently need visual support beyond what is planned
- [ ] Algorithm 1 (IGSD pseudocode) is rendered as a code block, not as an algorithm environment. The final LaTeX should use `\begin{algorithm}` with `\caption{IGSD}`. This is a formatting note for the LaTeX writer.

## What Works Well
1. **QAS and Ortho metric definitions (Section 4.1)** are crisp, self-contained, and immediately interpretable. The summary table with formula, interpretation thresholds, and the combined benchmark weighting rationale (GSM8K signal strength vs. MATH500 weakness) is exactly what a reviewer wants to see. The threshold breakdown (>1.0 synergy, 0.8--1.0 near-orthogonal, <0.8 interference) gives the reader a ready-made vocabulary for the results section.

2. **Honest d2Cache reporting (Section 4.2)** distinguishes this paper from acceleration papers that report only projected speedups. The dual-reporting strategy (measured vs. projected) with explicit labeling is intellectually honest and preempts the most likely reviewer objection. The sentence "The gap between projected and measured speedup is itself a finding" turns a limitation into a contribution.

3. **M1+IGSD interaction explanation (Section 4.3, final paragraph)** concretely explains why the two methods compose well -- frozen tokens create low-entropy positions that M1 exploits, with a specific CHR number (94.3% at eta=0.5 during refine phase vs. 56% standalone). This mechanistic explanation is the strongest paragraph in the section and directly supports the results in Section 5.2.
