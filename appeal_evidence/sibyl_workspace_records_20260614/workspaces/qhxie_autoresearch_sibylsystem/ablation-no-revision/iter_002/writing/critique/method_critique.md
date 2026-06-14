# Critique: The Absorption Score (Method Section)

## Summary Assessment

The method section is technically rigorous and well-structured, defining a clear, reproducible absorption metric with appropriate validation against random dictionary controls. The mathematical definitions are precise and the design rationale correctly distinguishes RVE from Pearson correlation. However, three critical issues undermine the section's completeness: the partial reconstruction formula omits the decoder bias term, the pipeline figure is not referenced in the body text despite being required by the outline, and a promised table (per-layer statistics) is entirely absent from the experimental setup section. The section also has terminology deviations from the glossary that need correction before finalization.

## Score: 7/10

**Justification**: The metric definition is sound and the validation strategy is exemplary. The section would score higher if the decoder bias were handled explicitly in the formula, the pipeline figure were properly referenced in-text, and the per-layer statistics table from the outline were present in Section 4. The writing quality is high but cross-section consistency has multiple failures (inconsistent formula with notation.md, missing layer 4 bimodality description, H4 labeled "falsified" when it should be "uninformative"). What prevents a higher score is not conceptual weakness but specific technical gaps that a reviewer would flag as blocking acceptance.

## Critical Issues

### Issue 1: Partial Reconstruction Formula Missing Decoder Bias Term

- **Location**: Section 3.1, Step 3 (line 17)
- **Quote**: "$x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t)$"
- **Problem**: The formula omits the decoder bias vector $b_{\text{dec}}$. According to `notation.md` line 28, the correct partial reconstruction is: $x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t) + b_{\text{dec}}$. The outline explicitly states "Note: b_dec cancels in the RVE ratio (subtraction of x_t removes the constant)" — but this cancellation is not explained in the method text. The formula as written is inconsistent with the notation table and misleading to readers who would implement it incorrectly.
- **Fix**: Either (a) add "$+ b_{\text{dec}}$" to the formula and add a sentence: "The bias term $b_{\text{dec}}$ appears identically in both $x_t$ and $x_t^{\text{partial}}$, so it cancels in the RVE ratio of Step 4," or (b) add a single sentence: "Note that $b_{\text{dec}}$ cancels in the RVE ratio since subtraction of $x_t$ removes the constant term, so it is omitted for brevity." Either approach makes the cancellation explicit rather than relying on reader inference.

### Issue 2: Pipeline Figure Promised by Outline but Not Referenced in Body Text

- **Location**: Section 3, no in-text reference to Figure 1
- **Quote**: "<!-- FIGURES: Figure: gen_pipeline.pdf — Architecture diagram of the experimental pipeline... -->"
- **Problem**: The outline explicitly specifies that "Figure 1: Experimental Pipeline Overview" should appear in Section 3 (Method) with content: "(1) Corpus to model activations, (2) SAE encoding, (3) Identify top-5 co-firers per token, (4) Partial reconstruction, (5) RVE computation, (6) Absorption score per latent." The method section contains only a comment block referencing the figure but no actual `![Figure 1...](figures/gen_pipeline.pdf)` embedding or in-text reference. A method section about a computational metric without a pipeline diagram is a significant visual gap — the text describes a multi-step process but provides no visual overview of that pipeline.
- **Fix**: Add before the horizontal rule at the end of Section 3: "Figure 1 illustrates the full absorption score computation pipeline." Then insert `![Figure 1: Experimental pipeline overview](figures/gen_pipeline.pdf)` with a caption: "The absorption score pipeline: (1) tokenize corpus and run model forward pass with SAE hook, (2) identify activating tokens per latent, (3) find top-5 co-firing latents per token, (4) compute partial reconstruction, (5) measure per-token RVE, (6) aggregate to absorption score per latent."

### Issue 3: Table 2 (Per-Layer Statistics) from Outline Missing from Section 4

- **Location**: Section 4 (Experimental Setup)
- **Quote**: The outline specifies: "Table 2: L0 and Absorption by Layer — inline table in Section 4 (Experimental Setup) showing mean L0, mean absorption, and % A_f > 0.5 for each of the six audited layers."
- **Problem**: The method section has no such inline table. The layer sweep statistics appear in experiments.md Section 5.7 (Table 2), but the outline explicitly places this table in Section 4 as part of the experimental setup description. This is a structural mismatch — a reader of Section 4 alone would not know the per-layer absorption statistics that the outline promises should be visible at the setup stage, not hidden in results.
- **Fix**: Either (a) add the per-layer table inline in Section 4.1 or 4.2 with the six-layer data, or (b) add a forward reference sentence: "Table 2 in Section 5.7 reports per-layer L0 and absorption statistics for all six audited layers." Option (a) is preferred as it makes the setup section complete without requiring cross-referencing.

## Major Issues

### Issue 4: Layer 4 Bimodality Promised in Outline but Absent from Section 3.3

- **Location**: Section 3.3 (Validation)
- **Quote**: The outline explicitly states: "**Layer 4 bimodality.** At layer 4, absorption scores are bimodal: 25.1% of latents score Af=1.0 (6,170 latents, fully absorbed) and 34.2% score Af=0.0 (fully independent), with 40.7% distributed between. This contrasts with layer 8 where 99.4% of latents score exactly 0.0."
- **Problem**: The current method section text (lines 35-44) describes only the random dictionary control and always-on feature exclusion. No mention of layer 4 bimodality. This creates an outline promise gap — the outline explicitly calls out this finding as part of Section 3's validation narrative. Additionally, the sharp clustering at boundary values (0.0 and 1.0) is flagged in the outline as "inconsistent with continuous absorption and suggests either genuine structural bifurcation or a threshold artifact" — this important caveat is absent.
- **Fix**: Add to Section 3.3 after the "Threshold sensitivity" paragraph: "**Layer 4 bimodality.** At layer 4, absorption scores are bimodal: 25.1% of latents (6,170 out of 24,576) score $A_f=1.0$ (fully absorbed) and 34.2% (8,400 latents) score $A_f=0.0$ (fully independent), with 40.7% distributed between. This contrasts with layer 8 where 99.4% of latents score exactly 0.0. The sharp clustering at boundary values is inconsistent with continuous absorption and may indicate a threshold artifact of the $A_f$ metric rather than genuine structural bifurcation."

### Issue 5: Layer 8 Selection for Patching Protocol Unjustified

- **Location**: Section 5 (Activation Patching Protocol), line 79
- **Quote**: "We patch four conditions at layer 8"
- **Problem**: The method section selects layer 8 for the patching protocol but provides no justification. Per the experiments section cross-reference, Table 2 shows layer 8 has 20.9% absorption (not the highest — layer 4 is 49.3%). The audited layers are $\{0, 2, 4, 6, 8, 10\}$ but the patching protocol uses only layer 8 without explanation. A reviewer will ask: why not layer 4 (highest absorption, where absorption would matter most for circuit analysis) or layer 0 (lowest absorption)? Layer 8 was the pre-registered H1 test layer, but the H4 experiment design flaw means layer 4 (49.3% absorption) was never tested.
- **Fix**: Add one sentence after "We patch four conditions at layer 8": "Layer 8 was selected as the pre-registered H1 test layer with moderate corpus-wide absorption (20.9% with $A_f > 0.5$) and well-characterized circuit structure for the France/Paris factual recall task. Note that layer 4 (49.3% absorption) was never tested in the H4 patching experiment due to the design flaw described in Section 5.3."

### Issue 6: H4 Labeled "Falsified" When It Should Be "Uninformative"

- **Location**: Section 5, line 91-93: "Verdict: H4 is falsified as an uninformative experiment."
- **Problem**: The experiments section says H4 is "falsified," but the glossary specifies "uninformative — 'falsified' when H4 experiment design was flawed (not a null result)." The proposal.md explicitly states H4 should be labeled "uninformative" not "falsified" because the experiment design was flawed (both subsets yielded 0.0, making any comparison impossible), not because data showed a null result. Additionally, the intro's second bullet point says "absorption level does not differentiate circuit-level causal importance" — but the H4 experiment does not support this conclusion since both subsets failed entirely. The experiment compared different-sized dictionary subsets (10% vs 100%), not different absorption levels at layers with different absorption profiles (layer 4 vs layer 8).
- **Fix**: Change to "Verdict: H4 is uninformative. The experiment design was flawed — both absorption subsets yielded 0.0, making the key comparison impossible. The experiment compared different-sized subsets of the same SAE (10% vs 100% of latents), testing dictionary completeness rather than absorption level. The correctly designed experiment (comparing full SAE at layer 4, 49.3% absorption, vs. layer 8, 0.19% absorption) was never conducted. The pilot data do not support any conclusion about whether absorption level predicts circuit-level causal importance."

## Minor Issues

- **"Fewer than 1%" vague**: Section 3.3 says "fewer than 1% of latents are always-on and excluded." The pilot has 24,576 latents per layer; 1% would be ~246. The outline specifies "38 latents." Fix to: "38 latents (0.15% of $d_{\text{sae}} = 24{,}576$) fire on more than 90% of corpus tokens and are excluded as bias-like."
- **"Simulated" in Section 4.1**: "sub-dictionaries ... are simulated by cumulatively subsampling" — "simulated" implies these are not real dictionaries. Use "generated by" or "created by" instead.
- **"The sparsest layer" mischaracterization**: If this phrase appears anywhere in cross-referenced sections, it is factually wrong — layer 8 has L0=71.9, the *highest* L0 (least sparse). Change to "the layer with the highest L0" or reference the specific value.
- **Figure 1 not referenced before appearance**: The figure exists in the figure plan but the body text never mentions it. Add an in-text reference before the figure appears in the text flow.

## Visual Element Assessment

- [ ] Figures/tables match outline plan — **PARTIAL**: Figure 1 (pipeline diagram) is in the figure plan but only in a comment block; Table 2 (per-layer stats) is missing from Section 4
- [ ] All visuals referenced before appearance — **NO**: Figure 1 is not referenced in body text at all
- [ ] Captions are self-explanatory — N/A for body text (no figures embedded)
- [ ] No text-heavy sections that need visual support — Section 3 would benefit from the pipeline diagram being referenced in-text as a reader aid for the multi-step process

## What Works Well

1. **The absorption score definition is precise and reproducible** (Section 3.1): The five-step computation is unambiguous, the mathematical notation is clean, and the 80% RVE threshold is operationally defined. A reader could implement this from the text alone.

2. **Random dictionary control is exemplary** (Section 3.3): The validation strategy is methodologically sound — random Gaussian decoders yield exactly 0.00% > 0.5 by construction, providing a definitive null baseline. This is exactly how metric validation should be done.

3. **Design rationale correctly distinguishes RVE from Pearson correlation** (Section 3.2): The explanation of why per-token RVE is used rather than linear correlation is clear and convincing. This is a subtle but important distinction that many reviewers would question without this explanation.

4. **Pre-registered hypotheses table with falsification criteria** (Section 4.3): Listing hypotheses with explicit falsification thresholds before any experiments are run is best practice. The table format is clean and the thresholds are specific.

5. **Patching protocol is reproducible** (Section 5): Clean/corrupted prompt structure, four patching conditions, and the faithfulness formula are all explicitly defined. Reproducible from the text alone.

## Cross-Section Consistency Notes

| Check | Status | Details |
|-------|--------|---------|
| Partial reconstruction formula | **FAIL** | method.md line 17 omits $b_{\text{dec}}$; notation.md line 28 includes it |
| Layer 4 bimodality in Section 3.3 | **FAIL** | Promised in outline Section 3.3, absent from method.md |
| Figure 1 in Section 3 | **FAIL** | Only HTML comment placeholder; figure not embedded or referenced |
| Table 2 in Section 4 | **FAIL** | Outline specifies Table 2 in Section 4; absent from method.md |
| H4 terminology | **FAIL** | "falsified" in experiments.md; glossary requires "uninformative" |
| Layer 8 selection justification | **MISSING** | No explanation for why layer 8 is used for patching (vs layer 4) |
| Always-on feature count | **PARTIAL** | "fewer than 1%" vs. outline's specific count of 38 latents |
| Notation consistency (RVE, $A_f$) | **OK** | RVE and $A_f$ formulas match notation.md |
| Per-layer vs. layer-wise terminology | **OK** | "per-layer" used correctly; no "layer-wise" found in method section |