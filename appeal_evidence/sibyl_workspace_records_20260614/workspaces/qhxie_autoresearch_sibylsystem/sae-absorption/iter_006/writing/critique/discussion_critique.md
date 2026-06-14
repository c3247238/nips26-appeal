# Critique: Discussion

## Summary Assessment

The discussion section is well-structured across five subsections that map cleanly onto the paper's three pillars (metric audit, L0 phase transition, rate-distortion diagnostic) plus negative results and limitations. The section's greatest strength is its intellectual honesty---negative results and limitations are reported with specific numbers and without smoothing. Its greatest weakness is a tendency toward redundancy with the experiments section and a missed opportunity to connect the findings into a single, synthetic argument about what absorption actually is.

## Score: 7/10
**Justification**: A solid discussion that earns its place through honest negative results (Section 7.4) and actionable recommendations (Section 7.2). Reaching 8 requires: (a) eliminating redundancy with Sections 4--6, (b) adding a synthetic paragraph that unifies all three pillars into one coherent story, (c) tightening the metric-validity discussion with a more precise mechanistic account of *why* JumpReLU thresholds miscalibrate the metric, and (d) addressing two missing discussion topics from the outline (implications for safety-relevant features, connection to DeepMind's 2025 safety-team findings mentioned in the proposal).

## Critical Issues

### Issue 1: Section 7.1 re-states Section 4.2 results without adding new interpretive content
- **Location**: Section 7.1, paragraph 1, sentences 2--3
- **Quote**: "Our confound decomposition classifies 648 of 657 false negatives at $L_0$=22 as hedging (98.6%) and only 9 as hierarchy-driven (1.4%), with 0 attributable to reconstruction error (Section 4.2)."
- **Problem**: This sentence is a near-verbatim repetition of Section 4.2 (experiments: "657 false negatives decompose as follows: Hedging: 648 of 657 (98.6%)... Hierarchy-driven absorption: 9 of 657 (1.4%)... Reconstruction error: 0 of 657 (0.0%)"). A discussion section should add new interpretation, not repeat numbers already reported. The reader already encountered these numbers twice (Section 4.2 and Section 1).
- **Fix**: Replace the result recitation with a one-sentence cross-reference ("The hedging dominance at $L_0$=22 (98.6%; Section 4.2)") and immediately move to the new implication---what this means for how mitigations should be evaluated.

### Issue 2: No synthetic unifying argument across the three pillars
- **Location**: Section 7 overall; most apparent by its absence between 7.1 and 7.2
- **Problem**: The discussion treats each finding in isolation: 7.1 discusses mitigations, 7.2 discusses metric validity, 7.3 discusses rate-distortion theory. There is no paragraph that unifies the three into a single coherent narrative. The outline says the discussion should "integrate the implications -- the field should validate metrics before building mitigations, and the L0 operating point (not the encoder architecture) is the primary lever." This integration is gestured at but never fully articulated in one place. The reader must assemble the synthesis themselves.
- **Fix**: Add a brief opening paragraph (3--4 sentences) before 7.1 or between 7.2 and 7.3 that explicitly connects the findings: the metric does not transfer (7.2) -> most of what it measures is hedging, not competitive exclusion (7.1) -> L0 is the primary control parameter, and CMI may eventually predict which features are at risk (7.3) -> therefore validate metrics per architecture, adjust L0 before redesigning encoders, and invest in dimension-agnostic CMI estimators.

## Major Issues

### Issue 3: Section 7.2 does not distinguish the two proposed explanations with enough precision
- **Location**: Section 7.2, paragraph 3 ("Two explanations are compatible with the data")
- **Quote**: "First, the cosine similarity threshold ($\tau_{\cos} = 0.025$) and magnitude gap threshold ($\tau_{\text{mag}} = 1.0$) may be too permissive for JumpReLU SAEs..."
- **Problem**: This explanation is directionally plausible but imprecise. "Too permissive" could mean the thresholds admit too many latents as candidates (inflating the denominator) or that they classify non-firing latents as false negatives too easily (inflating the numerator). The second explanation ("may systematically overcount") is similarly vague about whether the overcounting comes from the threshold step or the attribution step. Without distinguishing these sub-mechanisms, the reader cannot design experiments to differentiate them. A top-venue reviewer would flag this as hand-waving.
- **Fix**: Make the mechanistic distinction explicit. For explanation 1: "Too permissive thresholds classify latents with incidental cosine alignment as candidate parent features, inflating the false-negative denominator." For explanation 2: "The attribution step counts every non-firing candidate as a false negative; on JumpReLU SAEs, where the hard threshold produces a larger fraction of near-miss non-activations than L1-ReLU's graded suppression, this step over-attributes." Then note which experiment (threshold sensitivity sweep from Section 3.2 / Appendix) speaks to which explanation.

### Issue 4: Section 7.3 does not acknowledge the circularity caveat prominently enough
- **Location**: Section 7.3, entire subsection
- **Problem**: The circularity issue with the phase transition scale prediction (lambda estimated from empirical half-max) is discussed in Section 6.3 but not mentioned at all in the discussion section's treatment of rate-distortion theory. Section 7.3 states "CMI-absorption correlation ($\rho_s = -0.383$, $p = 0.059$; Cohen's $d = -0.924$) at $d' = 10$ is directionally consistent with rate-distortion theory" without noting that the scale match is partly by construction. A reviewer familiar with Section 6.3 would notice the omission and question whether the discussion is cherry-picking the favorable framing.
- **Fix**: Add one sentence after the correlation report: "The scale match between predicted $L_{0,\text{crit}}$ and empirical half-maximum is partly by construction (Section 6.3); the non-circular prediction is the rank ordering of letters by absorption susceptibility."

### Issue 5: Section 7.4 H2 falsification language inconsistent with the conclusion section
- **Location**: Section 7.4, bullet 1 ("H2 falsified")
- **Quote**: "The pilot's 96.9% hierarchy-driven figure was a methodological artifact of single-$L_0$ classification that did not use cross-$L_0$ persistence as the classification criterion."
- **Problem**: The conclusion section states "Confound decomposition at $L_0$=22---where all 25 probes achieve F1=1.0---classifies 648 of 657 false negatives (98.6%) as hedging." But the discussion says the pilot's 96.9% hierarchy-driven figure was "a methodological artifact of single-$L_0$ classification." The conclusion does not mention this artifact story at all. A reader who reads only the conclusion would not understand why the pilot's result was inverted. The discussion and conclusion should be consistent in how they frame this reversal.
- **Fix**: This is primarily an issue for the conclusion, but in the discussion, add a parenthetical cross-reference: "(The classification criterion is defined in Section 3.4; the distinction from the pilot's single-$L_0$ method is critical.)"

### Issue 6: Section 7.5 omits the confound decomposition's own sensitivity to L0 selection
- **Location**: Section 7.5, all bullets
- **Problem**: The outline (Section 7.5) includes "Confound decomposition classification depends on which L0 values are included in the sweep" as a listed limitation. The actual section text does not contain this limitation. This is a significant omission: the entire hedging-vs-hierarchy classification rests on persistence across L0 values {22, 41, 82, 176}. If L0=176 were excluded, more tokens would be classified as hierarchy-driven. If an additional L0=500 were added, some currently "persistent" tokens might resolve. This sensitivity is not acknowledged.
- **Fix**: Add a limitation bullet: "**Confound decomposition sensitivity.** The hedging-vs-hierarchy classification depends on the specific L0 values tested ({22, 41, 82, 176}). Including higher L0 values could reclassify some of the 9 persistent core words as hedging; excluding L0=176 would increase the hierarchy-driven fraction. The decomposition provides a lower bound on hedging prevalence, not an absolute partition."

### Issue 7: Figure 5 is referenced before its appearance but the caption content is thin
- **Location**: Section 7.3, final paragraph and figure reference
- **Quote**: "As shown in Figure 5, the CMI-absorption correlation is restricted to $d' = 10$..."
- **Problem**: The figure caption ("Spearman rho between CMI and absorption rate as a function of subspace dimension d'. The negative correlation (theory-consistent) holds only at d'=10; at higher dimensions the sign reverses.") is adequate but this figure is buried in the discussion section rather than in Section 6.5 where the dimension sensitivity is first analyzed. The experiments section (6.5) discusses dimension sensitivity in detail with a full table but does not include Figure 5. Moving the figure to Section 6.5 and referencing it from 7.3 would be more natural and prevent the impression that the discussion is introducing new data.
- **Fix**: Move Figure 5 to Section 6.5 (where the dimension sensitivity analysis lives) and reference it from Section 7.3. If the figure must stay in Section 7, note explicitly that "the data underlying Figure 5 are reported in Table [Section 6.5's table]."

## Minor Issues

- **7.1, paragraph 2, sentence 1**: "This does not invalidate the mitigations themselves." -- The transition is abrupt. Add a brief connector: "Two caveats limit the strength of this conclusion."
- **7.1, paragraph 2**: "On L1-ReLU SAEs, where the metric was developed and validated, competitive exclusion may well dominate" -- the word "may well" is hedging without evidence. Either cite evidence that competitive exclusion dominates on L1-ReLU or use "may" alone.
- **7.2, paragraph 1**: "JumpReLU SAEs have fundamentally different activation dynamics: the hard threshold $\theta_j$ creates a discrete boundary between active ($z_j > \theta_j$) and inactive ($z_j = 0$) states." -- This repeats the method section (3.1) description almost verbatim. A brief reference ("As described in Section 3.1") would suffice.
- **7.3**: The formula "$L_{0,\text{crit}} \approx \lambda / \text{CMI}$" appears here for at least the third time in the paper (also in Section 6.4 and the conclusion). Consider whether the discussion instance adds value or is redundant.
- **7.4, H4 bullet**: "ITAC candidate pair median (1.35) does not separate from the random pair median (1.14; Mann-Whitney not significant)" -- the exact p-value should be reported per the paper's conventions (Section 10 of notation.md: "report... p-values as exact values").
- **7.4, H6 bullet**: "Bootstrap 95% CIs span $[-1.0, +1.0]$ for all predictors" -- this is a striking finding (CIs covering the entire range of possible correlations) that deserves one more sentence of interpretation: the sample size is simply too small to distinguish any signal.
- **7.5, "Metric dependence" bullet**: "Activation patching---zeroing child features and measuring parent recovery---would provide metric-independent validation but has not been performed." -- This exact sentence also appears in Section 5.4 and the conclusion. Consolidate: state it once in the discussion and reference from other sections.
- **7.5, "Cross-domain probe quality" bullet**: "The city-continent absorption signal (6.49%) is driven by a single continent (Asia, 21.62%)" -- the numbers are from Section 4.4 but the Asia-specific figure (21.62%) is not in the experiments section. Either add it to Section 4.4 or cite the source data here.
- **7.5, "Single task family" bullet**: "(where DeepMind's 2025 safety team reported SAE probe failures)" -- no citation is provided for this claim. Add the specific reference.

## Visual Element Assessment

- [ ] Figures/tables match outline plan -- **Partially.** The outline specifies "Figure A1: Dimension Sensitivity of CMI Correlation (Appendix C)" but the discussion includes it as Figure 5 in the main text. This discrepancy should be resolved: either promote it to a main-text figure (updating the outline) or move it to the appendix (updating the discussion).
- [x] All visuals referenced before appearance -- Figure 5 is referenced before it appears ("As shown in Figure 5...").
- [x] Captions are self-explanatory -- The Figure 5 caption conveys the key finding.
- [ ] No text-heavy sections that need visual support -- Section 7.4 (Negative Results) lists four falsified hypotheses in dense paragraph form. A summary table (Hypothesis | Predicted | Observed | Verdict) would improve scannability and reduce word count.

## What Works Well

- **Section 7.4 (Negative Results)** is the strongest part of the discussion. Each falsified hypothesis is reported with the specific numbers that falsify it, the source of the original prediction, and a diagnosis of why the prediction failed (e.g., "methodological artifact of single-$L_0$ classification"). This level of transparency is rare and valuable.
- **Section 7.2's concrete validation protocol** ("before measuring absorption on any new SAE architecture, researchers should verify that shuffled-label absorption is lower than measured absorption in at least three hierarchy domains") is specific, actionable, and immediately useful to practitioners. It elevates the paper from critique to constructive contribution.
- **Section 7.5's limitation on cross-architecture confound** ("Gemma 2 2B, $d_{\text{model}} = 2304$, $d_{\text{SAE}} = 16384$ vs. GPT-2 Small, $d_{\text{model}} = 768$, $d_{\text{SAE}} = 24576$") specifies the exact dimensions rather than vaguely noting "different models," giving the reader enough information to assess the confound severity themselves.
