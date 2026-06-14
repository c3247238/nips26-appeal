# Writing Quality Review

## Summary
This paper provides the first systematic empirical study of feature absorption in Sparse Autoencoders, introducing the absorption score $A_f$ as a training-free metric tested across five hypotheses on GPT-2 small SAEs. The central empirical finding is a layer-dependent inverted-U pattern: absorption peaks at layer 4 (49.3% of latents with $A_f > 0.5$) but is nearly absent at layer 8 (0.19%). Four of five hypotheses are falsified or uninformative; only dictionary size (H5) moves in the hypothesized direction. The writing is clear and the negative results are well-motivated. However, the abstract makes an assertion about corpus-level absorption scores predicting circuit-level causal importance that the experiments do not actually support -- both absorption subsets yielded identical 0.0 faithfulness, so no predictive relationship can be claimed. There are also three technical errors (decoder bias omission in the RVE formula, "sparsest layer" misdescription, H1 falsification scoping imprecision) that require correction.

## Detailed Assessment

### Structural Coherence: 8/10
The paper has a logical flow from problem (SAEs enable interpretability but absorption threatens reliability), through metric design, experimental setup, results for each hypothesis, and discussion of implications. The abstract accurately represents the paper's content and results. The roadmap at the end of Section 1 correctly previews all sections including Section 7 (Conclusion). The discussion usefully addresses why hypotheses failed, though the anti-absorption pressure mechanism in Section 6.2 is more post-hoc speculation than data-driven conclusion. Transitions between sections are generally smooth. One structural concern: Section 5.5 (H2 pending) is a placeholder that acknowledges the experiment was never run but provides a full pre-registered protocol -- this is actually the correct approach (honesty about what was not done) and represents good scientific practice.

### Notation & Terminology Consistency: 7/10
Key definitions from notation.md are consistently applied: $W_{\text{enc}}$, $W_{\text{dec}}$, $b_{\text{enc}}$, $b_{\text{dec}}$, $\text{act}_f(t)$, $\text{top5}(f,t)$, $A_f$. The glossary's preferred terminology is respected: "co-firing" (not "cofirer"), "GPT-2 small" (not "gpt2-small" in text), "activation patching" (not "activations"), "per-layer" (not "layer-wise"). However, there are three violations:

1. **Critical -- Section 3.1 formula**: The partial reconstruction formula in the paper's prose omits $b_{\text{dec}}$:
   - Paper: $x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t)$
   - notation.md: $x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t) + b_{\text{dec}}$
   The text says "where $W_{\text{dec}}[f]$ denotes column $f$ of the SAE decoder weight matrix" but $b_{\text{dec}}$ is missing. The paper later notes "$b_{\text{dec}}$ cancels in the RVE ratio" (Section 3.2) but this requires a proof, not an assertion, and the omission creates an inconsistency with notation.md that reviewers will flag.

2. **Critical -- Section 5.2**: The text states "layer 8 has the highest L0 (71.9), indicating the least sparse representation, yet it shows lower absorption than layer 4" -- this is internally consistent in this sentence but contradicts Section 5.1 which says "the sparsest layer (8, 10)" without the qualifier. The "sparsest" characterization is wrong: higher L0 = more non-zero activations = less sparse. Layer 8 is the least sparse layer, not the sparsest.

3. **Minor -- Section 1**: "feature collision" appears in Section 2.2 (correctly) but "superposition" is used without prior introduction in the abstract's background paragraph. The abstract's opening paragraph would benefit from one sentence framing superposition as a known dictionary learning phenomenon before contrasting with absorption.

### Claim-Evidence Integrity: 7/10
Most numerical claims are well-supported with specific data points from the experiments. The H1, H3, H4, H5 results are all correctly cited. The abstract's key finding #3 contains a precision problem:

- **Abstract**: "corpus-level absorption scores do not predict circuit-level causal importance -- both low-absorption and high-absorption latent subsets yield 0.0 faithfulness in activation patching experiments"

This phrasing implies that the experiment tested whether absorption predicts importance and found that it does not. But the experiment actually found that both subsets fail identically (0.0), making the comparison impossible. The correct interpretation is "the experiment was uninformative" -- not "it does not predict." This is an important distinction for a paper whose narrative is built on falsified/negative results. The abstract should say "the experiment was inconclusive" or "the comparison was impossible" rather than phrasing it as a confirmed negative finding.

The layer-dependence claim in abstract finding #1 is accurate: 0.19% at layer 8 (falsified) vs 49.3% at layer 4 (not falsified at that layer). The scoping is correctly handled in the body text (Section 5.1 explicitly notes layer 4 is exploratory and layer 8 is the primary test layer under the falsification criterion).

Table 2, Table 3, and Table 4 all match the pilot_summary.md data exactly. Figure references are consistent with the actual files in the figures directory (gen_pipeline.pdf, fig1_layer_absorption.pdf, fig4_layer4_histogram.pdf, fig3_faithfulness.pdf, fig2_dict_size.pdf all exist).

### Visual Communication: 9/10
The paper has all the required visual elements: one method pipeline diagram (Figure 1), one results table per hypothesis (Tables 1-4), and three analysis figures (Figures 2-5). All figures are referenced in the text before they appear. The figure captions are self-explanatory. The figure/table plan from the outline is fully realized. The one gap is that the abstract references "Table 1" as the hypothesis summary but does not include it inline in the abstract -- however, this is standard practice.

Figure 1 (gen_pipeline.pdf) appears twice in the paper (Sections 3 and 4.1) which is redundant but not incorrect.

### Writing Quality: 7/10
The paper is generally well-written with clear, direct sentences. Most banned patterns are avoided. However:

**Banned patterns found**:
- "has powered" in Section 1, paragraph 1: "This decomposition has powered circuit discovery" -- more concrete alternatives like "has enabled" or "has driven" would be stronger.
- "becomes unreliable" in Section 1: "threatening the reliability of SAE-based analyses" -- "may become unreliable" is more accurate given the paper establishes this is an open question.

**Hedging issues**:
- Section 1, paragraph 1: "a central concern in mechanistic interpretability" is stated as fact -- but the paper is actually investigating whether it IS a central concern by measuring prevalence. A hedged version: "a hypothesized failure mode" would be more accurate.
- Section 6.2: The "anti-absorption pressure" mechanism is presented as "the most straightforward interpretation" without sufficient hedging that this is post-hoc speculation with no ablation evidence. The outline review correctly flagged this.

**Redundancy**:
- The H4 section (5.3) restates the experimental setup that is already in Section 4.3. Some trimming would improve readability.

## Issues for the Editor

1. **Critical** **[Abstract misrepresents H4 result]**: Abstract Finding #3 states "corpus-level absorption scores do not predict circuit-level causal importance" as if this were a confirmed finding. The experiment actually produced 0.0 for both subsets, making the comparison impossible -- the result is "inconclusive," not "no effect confirmed." **Fix**: Change to "absorption level did not differentiate circuit importance -- both low-absorption and high-absorption subsets yielded 0.0 faithfulness, preventing comparison."

2. **Critical** **[RVE formula omits decoder bias]**: Section 3.1 defines $x_t^{\text{partial}}$ without $b_{\text{dec}}$, contradicting notation.md which includes it. While the paper claims $b_{\text{dec}}$ cancels in the RVE ratio, this cancellation is not proven in the text and reviewers will demand it. **Fix**: Either add $b_{\text{dec}}$ to the formula in the text to match notation.md, or add an explicit derivation showing it cancels in the $\text{Var}(x_t - x_t^{\text{partial}}) / \text{Var}(x_t)$ ratio.

3. **Critical** **["sparsest layer" mischaracterization]**: Section 5.2 states layer 8 "yet it shows lower absorption than layer 4" but the preceding sentence calls it "the sparsest layer." Layer 8 has L0=71.9, the highest in the dataset, meaning it has the most non-zero activations (least sparse). **Fix**: Change "the sparsest layer" to "the layer with the highest L0 (least sparse representation)" or "the densest layer" consistently throughout.

4. **Major** **[H1 falsification scoping imprecision]**: The abstract says H1 is "falsified" with the 0.19% figure, but layer 4 (49.3%) does NOT falsify H1 at that layer. The body correctly notes this in Section 5.1 ("H1 is falsified at layer 8" and "not falsifying H1 at that layer" for layer 4), but the abstract's framing is over-broad. **Fix**: Add "at layer 8" after "falsified" in the abstract's first key finding, or say "falsified at the primary test layer (layer 8, 0.19%)" to preserve the nuanced finding.

5. **Major** **[Anti-absorption pressure mechanism oversold]**: Section 6.2 presents the reconstruction-pressure interpretation as established finding rather than speculation. The text says "the most straightforward interpretation is that..." without clearly signaling this is a hypothesis. **Fix**: Add explicit hedging: "One possible interpretation... however our data cannot directly confirm this mechanism."

6. **Minor** **["has powered" wording]**: Section 1 says "This decomposition has powered circuit discovery" -- prefer "has enabled" or "has driven" per the banned patterns list. **Fix**: Replace with more specific verb.

7. **Minor** **[Section 6.2 verb error]**: "does not compression-features into shared representations" -- missing verb before "compression." **Fix**: "does not compress features into shared representations."

8. **Minor** **[Abstract H2 mention]**: The abstract mentions "H2 (token frequency correlation) remains pending" which is unusual for an abstract. **Fix**: Consider moving H2 to the limitations/open questions rather than the abstract, since it interrupts the summary of findings.

## What Works Well

1. **The abstract is comprehensive and accurate** (with the H4 caveat above): it correctly previews all five hypotheses, their outcomes, and the key quantitative findings. The structure (four findings plus one pending) is clear and gives reviewers a roadmap.

2. **Section 5.5 (H2 pending) handles omitted analysis correctly**: Rather than fabricating results, the authors transparently explain why H2 was not run (insufficient absorption variance at layer 8; layer 4 analysis was never executed), provide the full pre-registered protocol for future work, and acknowledge this as a limitation. This is the correct scientific practice for a negative results paper.

3. **The H4 design-flaw acknowledgment is exemplary**: The paper explicitly states "the correctly designed experiment was never conducted" and explains exactly what that experiment would be (compare full SAE at layers with different absorption profiles). This level of self-critique is rare and strengthens the paper's credibility.

4. **Figure references are consistently well-integrated**: Every figure is referenced in the text before it appears, and all captions are self-contained. The figure/table plan from the outline is fully realized.

5. **The random dictionary control validation is compelling**: The fact that random decoders yield exactly 0.00% absorption at all thresholds and co-firer counts convincingly establishes that the metric detects genuine structure rather than numerical artifacts.

SCORE: 6
