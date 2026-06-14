# Critique: 10. Discussion

## Summary Assessment
The Discussion section provides a careful, multi-layered interpretation of predominantly negative results with appropriate epistemic hedging throughout. The structure from findings through failure analysis to limitations, implications, and open questions is logical and well-executed. Three factual errors reduce the section's credibility: the "sparsest layer" mischaracterization (layer 8 is least sparse, not most), the attribution of 8 latents to layer 4 when that count belongs to layer 8, and an unacknowledged quantitative inconsistency between 0.19% and 20.9% for layer 8 absorption.

## Score: 6/10
**Justification**: The Discussion demonstrates strong analytical reasoning about why hypotheses failed and provides genuinely useful practitioner guidance in Section 10.4. However, three factual errors constitute critical flaws that would draw immediate reviewer objections. Additionally, the anti-absorption mechanism narrative in Section 10.2 is stated as established fact when it should be explicitly speculative. Fixing these would raise the score substantially.

## Critical Issues

### Issue 1: "Sparsest layer" Mischaracterization
- **Location**: Section 10.1, second paragraph, line 9
- **Quote**: "layer 8 (mean L0 = 71.9, the sparsest)"
- **Problem**: Layer 8 has L0 = 71.9, the *highest* L0 in the dataset (18.9 at layer 0, up to 71.9 at layer 8). Higher L0 means more non-zero activations per token — layer 8 is the *least sparse* layer, not the sparsest. The experiments section (Section 5.2) correctly identifies layer 8 as "the sparsest layer" but then immediately clarifies "L0 = 71.9, the least sparse representation" — a clarifying phrase the Discussion omits. A reviewer will catch this immediately because L0 = 71.9 is orders of magnitude larger than the shallow layers.
- **Fix**: Change to "layer 8 (mean L0 = 71.9, the least sparse layer in our sample, with the most non-zero activations per token)". Apply this correction consistently throughout the discussion.

### Issue 2: "8 Latents" Incorrectly Attributed to Layer 4
- **Location**: Section 10.1, first paragraph
- **Quote**: "driven by a cluster of 8 latents with perfect Af = 1.0 scores"
- **Problem**: The 8-latent perfect-score cluster belongs to layer 8 (Section 5.1: "Eight latents achieve the maximum score of Af = 1.0"). Layer 4 has 6,170 latents (25.1%) with Af = 1.0 (Section 5.1, line 141). The Discussion attributes the wrong count to layer 4. This is a three-order-of-magnitude error (8 vs 6,170) that would immediately signal to a reviewer that the section conflates different experimental results.
- **Fix**: Replace with "6,170 latents (25.1%) with Af = 1.0" to match the experiments section's layer 4 reporting. Confirm all layer 4 perfect-score references in the Discussion use 6,170, not 8.

### Issue 3: Layer 8 Quantitative Inconsistency (0.19% vs 20.9%)
- **Location**: Section 10.1, first paragraph
- **Quote**: "At layer 8, the deepest layer measured, only 0.19% of latents have Af > 0.5, and 99.4% of all latents score exactly Af = 0.0"
- **Problem**: Table 2 (Section 5.7) and the layer sweep (Section 5.2, Table 2) show layer 8 has 20.9% with Af > 0.5 and 76.8% with Af = 0.0. The 0.19% and 99.4% figures come from the H1 pilot (Section 5.1), which used a stricter activating-token criterion. The Discussion never reconciles these two different measurements for the same layer. A reviewer comparing the Discussion to the experiments section will notice the discrepancy and suspect the authors are cherry-picking numbers.
- **Fix**: Either (a) cite both figures with explanation: "In the H1 pilot analysis (0.19%) and in the full layer sweep (20.9%), layer 8 shows low absorption — the discrepancy reflects different activating-token thresholds" or (b) use only the layer-sweep numbers (20.9%, 76.8%) consistently with Table 2. The Discussion should not use H1 pilot numbers for layer 8 statistics while using layer-sweep numbers for all other layers.

## Major Issues

### Issue 4: Anti-Absorption Mechanism Insufficiently Hedged
- **Location**: Section 10.2, first paragraph
- **Quote**: "The reconstruction pressure in SAE training appears to discourage redundant encoding... This anti-absorption pressure is not an explicit objective — it emerges from the reconstruction loss combined with sparsity regularization."
- **Problem**: This is a mechanistic story post-hoc, not a demonstrated causal finding. The paper measures absorption prevalence but provides no ablation varying reconstruction loss strength, no comparison across training runs, and no direct evidence that reconstruction pressure drives latents apart. The outline's critical correction explicitly flags this needs explicit hedging: "Restructure as explicitly speculative: 'One possible interpretation... our data cannot directly confirm...'"
- **Fix**: Restructure as: "One possible interpretation — which our data cannot directly confirm — is that SAE training's reconstruction objective implicitly discourages redundant encoding... An alternative explanation is that the 80% RVE threshold is too strict for subtler forms of feature overlap. Distinguishing these would require ablating reconstruction loss strength across training runs."

### Issue 5: "Worst Layer" Label Applied to Wrong Layer
- **Location**: Section 10.4, first paragraph
- **Quote**: "With 99.8% of latents showing Af <= 0.5 even at the worst layer (layer 8)"
- **Problem**: Layer 8 has 20.9% absorption (Table 2) — it is not the "worst" layer. The "worst" layer (highest absorption) is layer 4 at 49.3%. At layer 4, only 50.7% have Af <= 0.5, not 99.8%. The 99.8% figure (100% - 0.19% from H1 pilot, or 100% - 0.2% approximately) applies only to the H1 pilot's strict criterion, not to the layer-sweep measurements. The "worst" label is wrong and the percentage appears to conflate H1 pilot numbers with layer-sweep discussion.
- **Fix**: Remove "worst layer" entirely. Use: "At layer 8 (the deepest layer measured), only 20.9% of latents have Af > 0.5 per the layer sweep, and 76.8% score exactly 0.0" or simply drop the 99.8% figure since it mixes H1 pilot methodology with H3 layer-sweep context.

### Issue 6: H4 Confound Not Acknowledged
- **Location**: Section 10.2, H4 paragraph
- **Quote**: "the full SAE achieves 0.289 (Figure 3)"
- **Problem**: The conclusion that "absorption level does not predict causal importance" is confounded: the 10% subset destroys reconstruction capacity regardless of which latents are selected. The experiment cannot distinguish between (a) selecting absorption-irrelevant latents and (b) losing 90% of the dictionary's reconstruction capacity. The Discussion presents the 0.0 faithfulness of both subsets as evidence against absorption's causal relevance, but the design does not support this conclusion.
- **Fix**: Add: "Note that the 10% subset destroys 90% of reconstruction capacity regardless of latent selection criteria; the 0.0 faithfulness therefore cannot distinguish between absorption irrelevance and dictionary coverage loss. This confound is why H4 remains uninterpretable rather than falsified."

### Issue 7: H1 Failure Framing Conflates Layer 8 and Layer 4
- **Location**: Section 10, opening paragraph
- **Quote**: "H1 predicted absorption in over 20% of mid-layer latents; we found 0.19% at layer 8, though layer 4 exceeded the threshold at 49.3%."
- **Problem**: Layer 4 was an exploratory finding, not part of the pre-registered H1 falsification criterion. The pre-registered falsification criterion was "<10% at layer 8." The opening frames this as if H1 was tested across all layers and partially confirmed, when in fact H1 was falsified at its pre-registered layer (0.19% << 10%) and layer 4's 49.3% was a post-hoc discovery.
- **Fix**: Separate explicitly: "H1 was falsified at its pre-registered test layer (layer 8, 0.19% vs >20% predicted). Layer 4, an exploratory measurement, showed 49.3% — exceeding the H1 threshold but not part of the pre-registered falsification criterion."

### Issue 8: H5 Subsample Bias Not Acknowledged
- **Location**: Section 10.1, H5 paragraph
- **Problem**: The H5 result uses subselections of a single 24K SAE, prioritizing absorbable latents for inclusion. This is an upper-bound estimate, not an unbiased measurement of independent SAEs at different dictionary sizes. The Discussion does not acknowledge this.
- **Fix**: Add one sentence: "The dictionary-size comparison uses subselections of a single 24K SAE (prioritizing absorbable latents), giving an upper-bound estimate; independently trained SAEs of different sizes may show different relationships."

## Minor Issues
- **Section 10.1, second paragraph**: "does not compression-features" is ungrammatical. Change to "does not compress features into shared representations at higher rates."
- **Section 10.1, opening**: "at layer 8, the deepest layer measured" — layer 10 is deeper than layer 8. Change to "layer 8, the deepest layer in our primary analysis" or simply remove the superlative.
- **Figure references lack section numbers**: All figure references should include section numbers for navigability, e.g., "Figure 3 (Section 5.3)" instead of just "(Figure 3)".
- **Section 10.3**: Mentions only one factual recall circuit. Add: "Testing on only one factual recall circuit (France/Paris) limits generalizability to other circuit types."

## Visual Element Assessment
- [x] FIGURES block at end of section defines 4 figures with correct file references
- [x] Figure 3 (faithfulness) and Figure 4 (layer 4 histogram) are referenced before the FIGURES block appears
- [x] Captions in FIGURES block are self-explanatory
- [ ] No inline summary figure in the Discussion itself — Section 10.1 describes the inverted-U pattern in text but does not reproduce the key data table or reference it with a clear anchor
- [ ] Section 10.4 is text-heavy with four bullet-style implications — a summary infographic would improve clarity

## What Works Well

1. **Section 10.2 (Why Did All Hypotheses Fail)** is exemplary for its per-hypothesis mechanistic analysis. The H4 paragraph is particularly insightful: "the subset selection method selects latents that are corpus-wide absorbers or non-absorbers, not latents relevant to the specific circuit" — this correctly identifies the core design flaw.

2. **Section 10.4 (Implications for SAE-Based Interpretability)** provides four concrete, actionable takeaways directly supported by the data. Point 2 ("the full SAE dictionary is necessary for downstream causal validity") is a genuinely important finding that deserves emphasis.

3. **The honest acknowledgment of the H4 experiment design flaw** in Section 10.3 — "a better design would compare full SAEs at different layers (e.g., layer 4, which has the highest absorption, vs layer 8, which is sparser)" — without actually conducting the correct experiment is the kind of scientific integrity that strengthens rather than weakens the paper.

4. **Section 10.5 (Open Questions)** correctly identifies generalizability to larger models (GemmaScope) as the most urgent question, appropriately prioritizing beyond the GPT-2 small results.

---

## Summary of Required Fixes (Priority Order)

1. **[Critical]** Fix "sparsest layer" to "least sparse layer" — layer 8 (L0=71.9) is the most non-zero activations per token, not the sparsest
2. **[Critical]** Fix "8 latents" to "6,170 latents" for layer 4's perfect-score cluster
3. **[Critical]** Address layer 8 quantitative inconsistency: either reconcile 0.19% (H1 pilot) with 20.9% (layer sweep) or use only layer-sweep numbers
4. **[Critical]** Hedge the anti-absorption mechanism as explicitly speculative, not established fact
5. **[Major]** Remove "worst layer" label from layer 8 — layer 4 is the highest-absorption layer
6. **[Major]** Add H4 experiment confound acknowledgment (dictionary coverage vs. absorption selection)
7. **[Major]** Separate H1 layer 8 falsification (pre-registered) from layer 4 exploratory finding
8. **[Major]** Add H5 subsample bias acknowledgment
9. **[Minor]** Fix "does not compression-features" grammar error
10. **[Minor]** Add section numbers to all figure references
11. **[Minor]** Fix "deepest layer measured" — layer 10 is deeper than layer 8