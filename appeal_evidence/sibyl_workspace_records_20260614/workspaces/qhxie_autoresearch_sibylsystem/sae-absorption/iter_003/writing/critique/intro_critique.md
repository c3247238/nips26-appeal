# Critique: Introduction

## Summary Assessment

The Introduction is unusually strong for a mechanistic interpretability paper. It opens with the precise technical problem, states the competing theories and their practical implications with clarity, previews the decisive result with exact numbers, and closes with a clean enumerated contribution list. The hypothesis-testing framing ("controlled experiment that resolves this debate") is well-motivated and honest about what the experiment can and cannot establish. The main weaknesses are: one terminological inconsistency with notation.md, an absorption rate discrepancy between the intro's preview and the actual experiments table, a missing prevalence motivation that the outline promises, and the planned Figure 1 teaser is not referenced anywhere in the section.

## Score: 8/10

**Justification**: The intro earns its score on clarity, evidence density, and honest framing — rare virtues in MI writing. It would reach 9/10 by (1) reconciling the absorption rate numbers with the experiments section, (2) referencing Figure 1, (3) adding one sentence on prevalence (the outline explicitly required this), and (4) fixing the notation inconsistency for the encoder weight norm. It cannot reach 10 until the contribution list's AUROC range is reconciled with the experiments table.

---

## Critical Issues

### Issue 1: Absorption Rate Numbers in Preview Do Not Match the Experiments Table

- **Location**: Paragraph 4 (result preview), sentence "OMP achieves a $0\%$ absorption reduction relative to feedforward encoding across all tested features (absorption rate ratio $= 1.000$)"
- **Quote**: "Our result is decisive: OMP achieves a $0\%$ absorption reduction relative to feedforward encoding across all tested features (absorption rate ratio $= 1.000$)"
- **Problem**: The experiments section (Table 2) reports per-letter feedforward absorption rates of AR(a)=0.978, AR(e)=0.867, AR(s)=0.733, with a mean of 0.859. The introduction says "absorption rate ratio = 1.000" — which is technically the ratio OMP/FF per letter — but the phrase "across all tested features" implies universality whereas the experiment is conducted on exactly 3 letters. More critically, Table 2 reports the mean FF AbsRate as 0.859, yet the intro-paragraph does not give the absolute level at all. This creates a disconnect: a reader who takes "0% reduction" at face value might expect OMP scored 0% absorption (i.e., fixed it entirely), not that both conditions score ~0.86–0.98. This is a precision issue, not a fabrication, but precision is required for a falsification claim.
- **Fix**: Change the preview sentence to read: "OMP achieves a $0\%$ reduction in absorption rate relative to feedforward encoding on all three tested letters (mean AR$_\text{OMP}$ = mean AR$_\text{FF}$ = 0.859; absorption rate ratio $= 1.000$), falsifying the amortization gap hypothesis."

### Issue 2: Missing Prevalence Motivation Required by Outline

- **Location**: Paragraph 1, operational definition of absorption
- **Quote**: "A key failure mode of this approach is **feature absorption**: a child feature $c$ absorbs the activation budget of a semantically related parent feature $p$, causing $p$ to fail to fire on inputs where it should activate"
- **Problem**: The outline (Section 2 key content, first bullet) explicitly requires: "Feature absorption: operational definition... **prevalence** (15–35% first-letter task on Gemma Scope 16k/65k)". The Introduction never states a prevalence figure. A reviewer who does not already know the absorption literature will not understand how consequential this failure mode is. The motivation for DeepMind's safety deprioritization (mentioned in the proposal) is also absent. Without prevalence, the urgency is implicit rather than demonstrated.
- **Fix**: Append to the first sentence of paragraph 1: "Absorption affects 15–35\% of SAE latents on the first-letter spelling task across Gemma Scope 16k/65k SAEs \citep{karvonen2025saebench}, representing a substantial fraction of the feature pool used for mechanistic analysis."

---

## Major Issues

### Issue 3: Figure 1 Teaser Not Referenced in the Section

- **Location**: End of paragraph 4 or as a standalone visual anchor
- **Problem**: The outline specifies Figure 1 (teaser panel — OMP bar chart + ROC curve) as belonging to the Introduction. The Paper Output Contract requires figures to be referenced in text before they appear. The introduction contains zero figure references. This means Figure 1 will appear in the compiled paper with no callout in the surrounding text, which violates the style rules of every target venue (NeurIPS 2026 included).
- **Fix**: Add at the end of paragraph 4 (after the OMP result preview): "Figure~\ref{fig:teaser} illustrates these two results: absorption rates are identical under feedforward and OMP encoding (left panel), while encoder weight norm ROC curve dominates EDA (right panel)."

### Issue 4: Contribution 2 AUROC Range Inconsistency

- **Location**: Contribution list, item 2
- **Quote**: "Encoder weight norm, a weight-only absorption heuristic (AUROC $= 0.757$--$0.837$)"
- **Problem**: The upper bound of the AUROC range (0.837) comes from the TopK-32k architecture with resid_post labels, while the lower bound (0.757) comes from the Standard/L1 architecture with resid_pre labels. These are different hook points and different label types (IG gold vs. decoder-alignment proxy). The intro presents this range as if it represents a unified result on comparable experimental conditions. Both the notation.md and the experiments section itself (Section 3.4) explicitly warn: "AUROC values for the two architectures are not directly comparable." Presenting "0.757–0.837" as a continuous range misleads the reader about the comparability of these numbers.
- **Fix**: Separate the two AUROC values explicitly: "AUROC $= 0.757$ (Standard/L1, gold IG labels) and $0.837$ (TopK-32k, proxy labels; architectures not directly comparable due to hook confound — see Section~\ref{sec:method})"

### Issue 5: Notation Inconsistency — `$\|\mathbf{w}_{e,j}\|_2$` vs. `EncNorm`

- **Location**: End of paragraph 4, byproduct sentence
- **Quote**: "we introduce the \textbf{encoder weight norm} ($\|\mathbf{w}_{e,j}\|_2$) as a weight-only absorption indicator"
- **Problem**: The notation.md and glossary both define the encoder weight norm using the symbol `enc_norm(j) = ‖W_enc_j‖₂` where `W_enc_j` is the standard matrix row notation. The intro uses `$\|\mathbf{w}_{e,j}\|_2$`, a different variable name (`w_{e,j}` instead of the row `W_enc_j`). The method section uses `$\|\mathbf{w}_{e,j}\|_2$` consistently (same as intro), so the notation is internally consistent between intro and method — but both deviate from notation.md which registers the symbol as `W_enc_j`. This will cause confusion in the final paper if notation.md is the reference document.
- **Fix**: Either (a) update notation.md to register `\mathbf{w}_{e,j}` as the encoder row vector, or (b) change the intro and method to use `\mathbf{W}_{e,j}` (uppercase, consistent with the matrix row convention in notation.md). Option (a) is preferable since the lowercase boldface vector notation is clearer.

### Issue 6: Transition to Methods/Background Is Missing

- **Location**: After the contribution list (end of section)
- **Problem**: The outline specifies an explicit transition: "Before presenting the mechanistic test, we introduce the encoder weight norm detector that makes absorption auditing scalable." This transition is absent. The intro ends abruptly after the contribution list. In NeurIPS-format papers, the introduction should conclude with a roadmap sentence connecting to the next section, especially given that Section 2 is Related Work and Section 3 is Methods — the ordering can confuse a reader who expects to go directly to the experiment after reading the intro's experimental framing.
- **Fix**: Add a closing roadmap sentence: "Section~\ref{sec:related} reviews prior work on SAE absorption theory and detection. Section~\ref{sec:method} introduces encoder weight norm and the OMP oracle experiment design. Section~\ref{sec:experiments} reports all results. Section~\ref{sec:discussion} discusses implications for the SAE mitigation research program."

---

## Minor Issues

- **Paragraph 4, "byproduct" framing**: The phrase "As a byproduct of this investigation" undersells the encoder weight norm contribution. The outline identifies it as a co-primary contribution. Prefer: "Alongside the mechanistic experiment, we introduce..."
- **Paragraph 4, Jaccard description**: "$\rho = 0.044$" appears without defining $\rho$ as "Spearman rank correlation" at first use. Add "(Spearman $\rho$)" for clarity.
- **Contribution list, item 3**: "33\% of absorbed features require training-time interventions" — this claim depends on the F1 width recovery analysis which involves a hook confound (resid_pre vs. resid_post). The contribution as stated is overconfident given the confound. The discussion section explicitly flags this; the intro should add a hedge: "suggesting at least 33\% require training-time interventions (hook-confound caveat: Section~\ref{sec:experiments})".
- **Paragraph 1, "biconvex sparse dictionary learning (SDL) loss"**: SDL is expanded on first use here (good), but "biconvex" is not defined and appears only in this paragraph. A reader unfamiliar with Tang et al. does not know what biconvex means in this context. Add a brief parenthetical: "(biconvex: alternately convex in encoder and decoder directions)".
- **Paragraph 2 / 3, competing hypotheses**: The hypotheses are clearly presented, but both hypotheses mention "iterative solvers" — the amortization gap account says they help, the sparsity landscape account says they don't. Consider one sentence explicitly noting: "The two accounts thus make opposite empirical predictions, enabling a clean controlled test." This makes the falsification logic explicit before paragraph 4 introduces it.
- **"OMP oracle" terminology**: Glossary.md says "Orthogonal Matching Pursuit" should be used on first use and "OMP oracle" avoided as first mention. The intro introduces "Orthogonal Matching Pursuit" (OMP) in paragraph 2 (correct) but then jumps to "an OMP oracle" in paragraph 4. By glossary rules, this is fine for a second use — but confirm this is not the actual first full mention in the paper flow. If the reader encounters intro before related_work/method, this is correct. ✓

---

## Visual Element Assessment

- [ ] Figure 1 (teaser) is listed in the outline for the Introduction section but is **not referenced anywhere in the intro text** — this is a Critical gap per the Paper Output Contract
- [x] No other figures are expected in this section
- [ ] Caption for Figure 1 is defined in the outline but cannot be assessed without the actual figure file
- [ ] Needs: one callout sentence in paragraph 4 before any description of Figure 1's content

---

## What Works Well

1. **Hypothesis framing with falsification criterion**: Paragraph 3 ("Choosing between these accounts is not merely academic... computationally expensive but tractable... only changes to training objectives") is excellent. It explains the practical stakes, not just the intellectual stakes, and makes the two hypotheses decision-relevant for practitioners. This is exactly what a strong NeurIPS intro does.

2. **Leading with the decisive result**: Paragraph 4 opens "This paper reports a controlled experiment that resolves this debate" — direct, not hedged. The result ("OMP achieves a 0% absorption reduction") is stated before the methods. This is the correct order for a results-first narrative and will hold a reviewer's attention.

3. **Contribution list is specific and numbered**: Each of the three contributions names the metric (AUROC, DeLong p-value, percentage recovered), the comparison baseline (EDA), and the empirical domain (GPT-2-small layer 6). This is quantitatively honest and avoids the "first work to..." hype pattern. The mechanistic/methodological/practical taxonomy also cleanly maps contributions to audience segments.
