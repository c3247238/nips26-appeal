# Critique: Discussion

## Summary Assessment

The Discussion section is generally well-structured and correctly interprets the experimental results. It navigates the mixed findings (one hypothesis supported, two falsified) with appropriate nuance and presents a coherent narrative arc from measurement confirmation to epistemic interpretation to safety implications. However, a misleading framing of the shuffled/permuted baseline results and an understatement of the H3 power problem constitute the most significant weaknesses.

## Score: 6/10

**Justification**: The section is competently written and internally consistent with the experiments section, earning it a solid floor. It reaches the next level (7-8) only if: (1) the "intermediate absorption" framing is corrected to reflect that shuffled/permuted baselines are near the trained SAE end, not the midpoint; and (2) the H3 power problem is elevated from a bullet in Limitations to a central interpretive constraint.

---

## Critical Issues

### Issue 1: Misleading "Intermediate Absorption" Framing

- **Location**: Section 7.1, paragraph 3
- **Quote**: "An unexpected finding was that shuffled features and permuted encoder baselines showed intermediate absorption (0.487 and 0.484, respectively), only slightly below trained SAEs."
- **Problem**: The word "intermediate" implies these values sit roughly between trained SAE (0.500) and random decoder (0.059). Geometrically, they do not. The gaps are:

  | Comparison | Gap from Trained SAE (0.500) | Gap from Random (0.059) |
  |------------|-------------------------------|-------------------------|
  | Trained SAE | 0.000 | +0.441 |
  | Shuffled | -0.013 | +0.428 |
  | Permuted | -0.016 | +0.425 |
  | Random | -0.441 | 0.000 |

  Shuffled and permuted baselines are 97.4% of the way from random toward trained SAE. They are not "intermediate" in any meaningful sense -- they are near-identical to trained SAE on this metric. The discussion's own conclusion ("absorption is partially driven by the decoder's geometric structure") actually supports this: if geometric structure alone produces near-maximal absorption, then calling 0.487 "intermediate" understates how close these baselines are to the trained condition. This framing could mislead reviewers into thinking the trained/geometry baselines are meaningfully distinguishable on absorption rate alone, when Table 1 shows the delta is only 0.013-0.016.

- **Fix**: Rewrite to acknowledge that shuffled and permuted baselines achieve near-maximal absorption, not merely intermediate absorption. E.g.: "Shuffled features and permuted encoder baselines achieved absorption rates of 0.487 and 0.484 -- nearly identical to trained SAEs -- suggesting that decoder geometry alone, without learned feature semantics, produces near-maximal absorption."

---

## Major Issues

### Issue 2: H3 Power Problem is Severely Understated

- **Location**: Section 7.3 and Section 7.5, bullet 4
- **Quote**: "Limited statistical power for H3: With only 7 absorbed features identified, the steering experiment had limited power to detect small effects. The non-absorbed control group (n=1014) also showed no steering response, suggesting the methodology may need refinement rather than more samples."
- **Problem**: The Discussion uses the H3 null result to support a substantive claim -- "absorption may be epistemic, not causal" -- while simultaneously acknowledging the experiment had "limited power." With n=7 absorbed features, the entire epistemic-vs-causal distinction rests on an observation with essentially zero statistical power. More critically, the non-absorbed group (n=1014) also showed zero steering response, which the Discussion uses to dismiss the H3 result ("suggesting the methodology may need refinement"), but this directly undermines using the absorbed group's zero response to argue for an epistemic interpretation. If the steering methodology fails for non-absorbed features too, the null result for absorbed features cannot distinguish between "absorption is epistemic" and "steering methodology is broken." The Discussion acknowledges this tension but does not resolve it, and the conclusion in 7.3 overstates what the data support.
- **Fix**: The epistemic-vs-causal interpretation should be framed as "consistent with" rather than "suggesting" the epistemic account, and the H3 power problem should be elevated to the first bullet of Limitations. The conclusion should explicitly acknowledge that the non-absorbed control group's identical null result means the steering methodology is the more parsimonious explanation for the null finding.

### Issue 3: Figure 5 Referenced in Discussion but Belongs in Methodology

- **Location**: Section 7.6, first bullet; outline Section 7.6
- **Quote**: "Figure 5: fig5_method_architecture_desc.md -- Architecture diagram of multi-child ablation procedure"
- **Problem**: The discussion's Figure & Table Plan lists Figure 5 as an architecture diagram of multi-child ablation, which is a methodological figure, not a discussion figure. The experiments section already contains sufficient narrative description of the multi-child ablation procedure. Listing this figure in the Discussion's artifact plan creates confusion about where it belongs. Moreover, Section 7.6's text ("replication on real language model features") does not reference Figure 5, confirming it is misplaced in the Discussion's figure plan.
- **Fix**: Remove Figure 5 from the Discussion's figure plan. If an architecture diagram is needed, it belongs in the Methodology or Experiments section where the multi-child ablation procedure is described.

### Issue 4: 7.4 Implies Safety Findings That Do Not Exist

- **Location**: Section 7.4 (entire section)
- **Quote**: "The positive frequency-absorption correlation adds a further concern: commonly activating features (which include many safety-relevant behaviors) may be more susceptible to absorption, not less."
- **Problem**: This section builds a safety concern on two unsupported premises: (1) that safety-relevant features are "commonly activating," and (2) that the positive frequency-absorption correlation generalizes to real language model features. Both are speculative. The correlation was measured on synthetic hierarchies with no safety relevance. Section 7.5 correctly notes "synthetic hierarchies may not capture real-world feature geometry," but 7.4 does not apply this caveat to the safety extrapolation. The section also admits H_Safe was "not implemented," yet draws safety conclusions anyway. This is a logical gap: you cannot claim your results have safety implications when the safety-relevant hypothesis was never tested.
- **Fix**: Either (a) remove the speculative safety extrapolation and replace with a statement that safety implications remain untested pending H_Safe implementation, or (b) add an explicit caveat that the positive correlation was measured only on synthetic hierarchies and should not be generalized to safety-critical features without empirical validation on real language models.

---

## Minor Issues

- **7.1, paragraph 2**: "prior work using single-feature ablation may have underestimated or mischaracterized absorption" -- "mischaracterized" is strong without specific citations to which prior work and how they mischaracterized absorption. Tone this down to "may not have fully characterized."

- **7.2, paragraph 3**: "Third, absorption may be driven more by geometric alignment than by competitive pressure" -- this third interpretation is speculative and unsupported by evidence presented in the paper. The geometric account is discussed more fully in 7.1; consider moving this sentence there or removing it.

- **7.5, bullet 2**: "Real transformer residual streams have different statistical properties than our synthetic data" -- this is true but vague. What specific properties differ (e.g., non-isotropic activation distributions, feature dependencies, hierarchy irregularity)? A more specific statement would strengthen the limitation.

- **7.6, fourth bullet**: "the relationship between absorption and downstream task performance requires characterization" -- this is stated as future work but is essentially what H3 attempted. The Discussion should acknowledge that H3 was an initial attempt that failed, and future work should build on lessons learned rather than treating it as unexplored territory.

- **7.7 conclusion**: "The negative results are scientifically valuable" -- this is good framing, but the conclusion does not summarize the key limitations that constrain how much weight these negative results should carry (especially H3's n=7 problem).

---

## Visual Element Assessment

- [x] Figures/tables match outline plan (Figures 1-4, Tables 1-2 are all referenced and appropriate for the Discussion's narrative)
- [ ] Figure 5 misplaced -- belongs in Methodology/Experiments, not Discussion
- [ ] No explicit "Figure X shows Y" references in running text of Discussion -- the section relies on the experiments section for figures and tables, but a brief forward reference (e.g., "As shown in Figure 2...") would strengthen the narrative integration
- [x] Figure captions in artifact plan are self-explanatory
- [ ] No visual elements within the Discussion text itself (appropriate -- Discussion interprets figures, does not present new data)

---

## What Works Well

1. **Section 7.2's treatment of the competitive exclusion falsification** (paragraphs 1-3): The discussion correctly identifies that a positive correlation contradicts the ecological Lotka-Volterra analogy, and offers three plausible interpretations without overclaiming. The treatment of negative results as informative rather than failures is exemplary.

2. **Section 7.3's nuanced epistemic-vs-causal framing** (paragraph 1): "Absorption may be epistemic: absorbed features may never have developed sensitivity to parent directions in the first place, rather than losing it through interference" -- this is a genuinely insightful reinterpretation that reframes a null result as theoretically informative.

3. **Section 7.5's Limitations structure**: The four limitations are clearly scoped and self-contained. Each one names the specific constraint (synthetic hierarchies, single architecture, steering methodology, H3 power) with enough specificity for readers to assess generalizability.

4. **Section 7.7's framing of negative results**: "The negative results are scientifically valuable. Falsifying the competitive exclusion hypothesis eliminates one theoretical account and constrains the space of plausible mechanisms." -- this is exactly the right attitude and would resonate well with a thoughtful reviewer.
