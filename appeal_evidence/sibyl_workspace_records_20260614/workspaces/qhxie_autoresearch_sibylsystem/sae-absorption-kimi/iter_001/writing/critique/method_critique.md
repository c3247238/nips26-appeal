# Critique: Methodology

## Summary Assessment
The Method section provides a clear, well-structured description of the training-free evaluation paradigm and the three experiments. However, it contains several critical inconsistencies between the proposal and the actual experiments, ambiguous notation, and incomplete reporting of implementation details that would make replication difficult.

## Score: 6/10
**Justification**: The section is readable and mostly accurate, but the mismatches between the proposed checkpoint corpus (Gemma-2-2B, Pythia-160M) and the actual one (GPT-2 Small only for E1/E3), the unresolved notation conflict for CE loss recovered, and the lack of detail on how hedging was actually computed all weaken it. Fixing these issues and adding more implementation specificity would bring it to an 8 or 9.

## Critical Issues

### Issue 1: Mismatched checkpoint corpus between proposal and actual experiments
- **Location**: Paragraph 2, Section 3.1
- **Quote**: "We evaluate 27 GPT-2 Small (117M) checkpoints released via SAELens."
- **Problem**: The proposal explicitly states E1 would use "Gemma-2-2B and Pythia-160M (the two most studied models in absorption research)" and E2 would use "Gemma-2-2B and Pythia-160M-deduped." The actual Method section reports GPT-2 Small for E1 and E3, with no mention of Pythia-160M at all. This is a major deviation that undermines the paper's stated scope.
- **Fix**: Either (a) add a clear justification for why GPT-2 Small replaced Gemma/Pythia in E1/E3, or (b) if Gemma/Pythia data was unavailable, reframe the contribution honestly and update the Intro/Abstract accordingly. Do not silently substitute the model.

### Issue 2: CE loss recovered notation is inconsistent and potentially inverted
- **Location**: Section 3.2, "Reconstruction fidelity"
- **Quote**: "$\text{CE}_{\text{recovered}} = \text{CE}_{\text{orig}} / \text{CE}_{\text{rec}}$."
- **Problem**: `notation.md` defines CE loss recovered identically, but this is mathematically suspicious. If reconstructed activations degrade performance, $\text{CE}_{\text{rec}} > \text{CE}_{\text{orig}}$, making the ratio $< 1$. If reconstruction is perfect, $\text{CE}_{\text{rec}} = \text{CE}_{\text{orig}}$, making the ratio $= 1$. But the Experiments section reports values like 1.172 for feature splitting, which implies $\text{CE}_{\text{rec}} < \text{CE}_{\text{orig}}$—i.e., reconstruction *improves* loss. This may be correct for some SAEs, but the notation and interpretation need much clearer explanation. More importantly, the glossary defines it as: "Values above 1.0 indicate that reconstruction preserves or improves next-token prediction; values below 1.0 indicate degradation." This is consistent with the formula, but readers will be confused because "recovered" usually means $(\text{CE}_{\text{orig}} - \text{CE}_{\text{rec}}) / \text{CE}_{\text{orig}}$ or similar.
- **Fix**: Add one sentence explicitly explaining that values $>1$ mean the SAE-reconstructed activations yield *lower* cross-entropy than the original activations. Consider renaming to "CE loss ratio" to avoid confusion, or clarify why "recovered" is used for a ratio that can exceed 1.

### Issue 3: Hedging metric description is too vague for replication
- **Location**: Section 3.2, "Hedging"
- **Quote**: "We compute a simplified feature-hedging score on correlated token pairs (antonyms). For each pair, we identify the top-1 SAE latent for both tokens. The hedging rate $h$ is the fraction of pairs where the same latent is the top feature for both tokens."
- **Problem**: The proposal says hedging is computed "on correlated token pairs (Chanin et al., 2025)" but does not specify which pairs, how many, what corpus they are drawn from, or how "top-1 SAE latent" is defined (max activation? max post-ReLU? on a specific token position?). The actual experiments report a hedging rate of 0.833 for Standard SAEs, but without knowing the token pair set, no one can replicate this.
- **Fix**: Specify the exact antonym pair list (e.g., "good/bad", "hot/cold"), the number of pairs, the corpus used to extract activations, and the precise definition of "top-1 latent" (e.g., latent with highest post-activation value averaged over the token position).

## Major Issues

### Issue 4: Task-agnostic absorption threshold lacks justification
- **Location**: Section 3.2, "Task-agnostic absorption"
- **Quote**: "Absorption is classified if the top ablation latent aligns with the probe direction at cosine similarity $> \tau$ (we set $\tau = 0.7$)."
- **Problem**: There is no justification for why $\tau = 0.7$ was chosen. Was this pre-registered? Is it a standard value from prior work? The arbitrary threshold undermines the metric's validity.
- **Fix**: Add a sentence explaining the rationale for $\tau = 0.7$ (e.g., "following Chanin et al., 2024" or "selected via pilot inspection of the alignment distribution"). If it was arbitrary, state that explicitly as a limitation.

### Issue 5: Pareto front objectives are internally inconsistent
- **Location**: Section 3.3, "Pareto front computation (E1)"
- **Quote**: "For each checkpoint we compute six normalized objectives: inverted absorption (so higher is better), inverted hedging, explained variance, CE loss recovered, inverted $L_0$ penalty, and inverted dead-neuron rate."
- **Problem**: The section says they compute "six normalized objectives" but then says CE loss recovered is included *as-is* (not inverted), despite the fact that higher CE loss recovered is better. This is fine directionally, but the list mixes raw metrics and inverted metrics without clarity. More importantly, the actual Pareto front in the Experiments section only discusses absorption, hedging, and explained variance—never mentioning the full 6-objective Pareto front. The reader cannot verify whether the 17/27 Pareto-optimal points were computed with all six objectives or a subset.
- **Fix**: Clarify exactly which six objectives were used and confirm that all 17 Pareto-optimal points were computed with the full 6-objective front. If the visualization only shows 2D projections, say so explicitly.

### Issue 6: Missing implementation details for E3 metric computation
- **Location**: Section 3.4, "Implementation Details"
- **Quote**: "Runtime for E1 is approximately 675 seconds for 27 checkpoints; E2 completes in under 10 seconds; E3 completes in approximately 24 seconds."
- **Problem**: While runtimes are reported, there is no mention of the LLM used for automated hierarchy discovery in E3, the prompt template, the number of top-N features labeled, or how the geography hierarchy was validated. The proposal mentions an "LLM judge" and "2–3 clean domains," but the Method section only mentions "geography" and gives no details on the LLM pipeline.
- **Fix**: Add a paragraph on E3 implementation: which LLM, how many features were labeled, the exact prompt, and how parent-child pairs were filtered.

### Issue 7: Limitation paragraph contradicts the outline
- **Location**: Section 3.4, final paragraph
- **Quote**: "Consequently, E1 is limited to Standard, TopK, and feature-splitting families on GPT-2 Small, while E2 covers the full architectural diversity at the cost of less controlled checkpoint selection."
- **Problem**: The outline's Figure & Table Plan (Table 1) lists rows for "Standard, TopK, TopK_MLP, TopK_Attn, feature_splitting"—four families, not three. The Method section says E1 is limited to "Standard, TopK, and feature-splitting," omitting TopK_MLP and TopK_Attn. Yet the Experiments section clearly reports results for TopK_MLP and TopK_Attn. This is a direct contradiction.
- **Fix**: Update the limitation paragraph to accurately reflect the families evaluated in E1: Standard, TopK, TopK_MLP, TopK_Attn, and feature-splitting.

## Minor Issues

- **Section 3.1, E2 description**: "All metrics---absorption, sparse probing F1, RAVEL Cause/Isolation, L0, and CE loss recovered---are drawn directly from the SAEBench dataset." The triple-dash `---` should be an em-dash with spaces or a colon for cleaner typesetting. → Replace with "All metrics—absorption, sparse probing F1, RAVEL Cause/Isolation, L0, and CE loss recovered—are drawn directly from the SAEBench dataset."

- **Section 3.2, "Absorption"**: The phrase "mean absorption rate, full absorption rate" appears in the outline but not in the Method section. The Method only defines "absorption rate $\alpha_{\text{FL}}$" as "the fraction of parent-feature tokens that are absorbed." If "full absorption rate" was also computed, it should be defined here. → Either define "full absorption rate" or remove it from the outline/abstract if it was not used.

- **Section 3.3, regression equation**: The equation uses $\text{outcome}_i = \beta_0 + \beta_1 \alpha_i + \dots$ but the text says "OLS with cluster-robust SEs confirms absorption as a significant negative predictor." The sign of $\beta_1$ is not pre-specified in the Method; this is fine, but it would be stronger to note that we expect $\beta_1 < 0$ under H2. → Add "We expect $\beta_1 < 0$ if absorption carries a unique negative causal cost."

- **Figure caption reference**: The pipeline figure is referenced at the end of the section, but there is no in-text call like "Figure 1 illustrates..." earlier in the section. → Add an explicit in-text reference before the figure appears.

## Visual Element Assessment
- [x] Figures/tables match outline plan
- [x] All visuals referenced before appearance
- [ ] Captions are self-explanatory
  - The pipeline figure caption is a full sentence, which is good, but it does not explain what the arrows or stages represent in enough detail for a reader to understand without the body text.
- [ ] No text-heavy sections that need visual support
  - The Pareto front description in Section 3.3 is very text-heavy and would benefit from a small diagram showing the six objectives and how they are inverted.

## What Works Well

1. **Training-free framing is clear and well-motivated.** The opening sentence explicitly states the paradigm and its rationale, which sets the paper apart from prior work. This is a strong, distinctive contribution.

2. **Metric definitions are precise and notation-aware.** With the exception of the CE loss recovered ambiguity, the mathematical notation is clean, consistent with `notation.md`, and easy to follow. The use of subscripts ($\alpha_{\text{FL}}$, $\alpha_{\text{TA}}$) is disciplined.

3. **The regression specification is fully written out.** Equation (1) in Section 3.3 gives the exact model, including controls and clustering, which is exactly the level of detail a reviewer wants to see in a Method section.
