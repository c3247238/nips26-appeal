# Writing Quality Review

## Summary
The paper characterizes feature absorption in sparse autoencoders across four feature hierarchies and four model layers on Gemma 2 2B, finding a 15-fold layer-dependent variation (2.2% to 34.5%) and significant cross-domain differences (Kruskal-Wallis p=0.005). It provides the first interventional causal evidence for absorption via activation patching (32.5% recovery vs. 1.5% control, d=1.33), shows the widely cited ~98% hedging rate is near-tautological (strict hedging 7.9%), and reports definitive negative results for three unsupervised detectors. The paper is well-structured and its central claims are communicated clearly, though two missing figures and two broken cross-references need repair before compilation.

## Detailed Assessment

### Structural Coherence: 8/10
The paper follows a logical arc: problem statement (Section 1) -> background (Section 2) -> method (Section 3) -> primary results (Section 4) -> causal evidence (Section 5) -> architecture comparison (Section 6) -> negative results (Section 7) -> discussion (Section 8) -> conclusion (Section 9). Transitions between sections are explicit and well-motivated -- each section closes with a sentence that previews the next. The decision to lead with the unconfounded layer-dependence result (Section 4.1) before the confounded cross-domain result (Section 4.2) is structurally sound. Section 6 (architecture comparison, 0.5 pages) is appropriately compressed given its null result. Section 8 (Discussion) integrates findings into broader implications without merely repeating results.

Two structural issues: (1) The Discussion introduces "Table 7 in Section 4.4 of the extended results" (line 359), which does not exist anywhere in the manuscript. The paper jumps from Section 4.3 to Section 5. This phantom cross-reference disrupts the reader's ability to verify the claimed absorbed-vs-hedged proportions. (2) Section 3.5 references "Section 8.6" for future patching work, but Future Directions is Section 8.5 -- no Section 8.6 exists.

### Notation & Terminology Consistency: 9/10
Notation is consistent with `notation.md` throughout. All core symbols ($\mathbf{x}^{(L)}$, $\hat{\mathbf{x}}$, $\mathbf{a}$, $\mathbf{d}_i$, $\mathbf{w}_k$, $L_0$, $M$, AR, FN, IG, RR) are defined before first use and used consistently. The paper correctly uses "latent" rather than "neuron" or "atom." $L_0$ appears in math mode throughout. "First-letter" is consistently hyphenated. "Gemma 2 2B," "Gemma Scope," "SAEBench," and "SAELens" are formatted per glossary.

One minor deviation: the paper uses "absorption scores" on line 51 when describing ATM's published results. The glossary specifies "absorption rate" as the preferred term for our formulation but this instance describes another work's terminology, which is acceptable. No other terminology deviations detected against `glossary.md`.

### Claim-Evidence Integrity: 8/10
Every central claim is backed by specific numbers traceable to the consolidation summary:
- 15-fold layer variation: 2.2% (L18-16k) to 34.5% (L24-16k) -- matches `consolidation_summary.json` (0.022 and 0.345).
- Kruskal-Wallis p=0.005: matches source (hierarchy_p: 0.005).
- Activation patching: 32.5% vs. 1.5%, d=1.33, p=0.000218 -- all match source.
- Hedging decomposition: strict 7.9%, compensatory 86.2%, persistent 5.9% -- all match source.
- GAS rho=0.116, CMI rho=0.044, Tax rho=-0.20 -- all match source.

One discrepancy detected: the paper states throughout that the first-letter absorption measurements use probes with "F1 >= 0.97" (lines 5, 7, 25, 182, 197, etc.). The consolidation summary confirms F1=0.9711 for the sklearn probe at layer 24, but also records that the activation patching experiment used a probe with F1=0.883 (below the strict 0.90 gate). The paper does not disclose the activation patching probe's actual F1. Instead, Section 3.5 says "the `sae_spelling` ICL probe provides reliable ground truth for first-letter classification" -- implying the F1=0.97 probe was used. This should be clarified: either confirm that the ICL probe (F1=0.97) was the one used in patching, or disclose the 0.883 figure.

The reference to "Table 7 in Section 4.4 of the extended results" (line 359) is a claim-evidence failure: the data backing the absorbed-vs-hedged proportion at L24-16k (50.0%) vs. L6-16k (100.0%) cannot be verified by the reader because the cited table does not exist.

### Visual Communication: 5/10
The paper has 4 figures and 6 inline tables. The outline planned 6 figures and 6+ tables.

**Present and effective:**
- Figure 1 (heatmap): communicates the central finding clearly; referenced in Section 1 before appearance.
- Figure 3 (layer absorption bars): shows the L24 spike with bootstrap CIs; effective.
- Figure 4 (cross-domain heatmap): shows the layer-hierarchy interaction; effective.
- Tables 1-6 (all inline): well-formatted, self-explanatory captions, referenced before appearance.

**Critical gaps:**
- **Figure 2 (pipeline schematic):** exists only as a markdown description (`fig2_pipeline_desc.md`), not a rendered diagram. The paper includes an image reference to it, but a reader would see a broken image or placeholder. A method diagram is essential for a NeurIPS/ICML-caliber paper.
- **Figure 5 (activation patching paired dot plot):** completely absent. Not referenced in the text, not generated. The causal evidence in Section 5.1 -- the paper's second-strongest result (d=1.33) -- relies entirely on a truncated table. A visual showing the per-word treatment-vs-control separation would make the effect size immediately compelling. This is the most impactful missing figure.
- **Figure 6 (hedging decomposition stacked bar):** completely absent. Not referenced in the text. The "near-tautological hedging" argument in Section 5.2 is purely textual. A stacked bar (7.9% strict / 86.2% compensatory / 5.9% persistent) would communicate the key insight at a glance.

The paper's visual coverage is below the minimum standard specified in the outline (6 main-text figures). The absence of Figures 5 and 6 leaves Sections 5.1 and 5.2 -- which contain two of the paper's four contributions -- entirely text-heavy with no visual support. Table 2 (probe quality) has a redundant PDF rendering (`tab1_probe_quality.pdf`) that could be consolidated to save space for the missing figures.

### Writing Quality: 8/10
The prose is direct and evidence-forward. Sentences lead with concrete findings, numbers, or figure references rather than motivation. No banned patterns ("In recent years...", "To the best of our knowledge...", "Furthermore...", etc.) were detected. The paper consistently uses specific numbers instead of vague claims: "improves by 15-fold" rather than "significantly varies"; "32.5% vs. 1.5% (d=1.33)" rather than "substantially higher."

Specific strengths: the Introduction opens with the central quantitative finding (line 15: "Absorption rates on the same sparse autoencoder vary 15-fold..."), not with a generic motivation. The probe quality confound is front-and-center (Section 4.3), not buried in limitations. Negative results get a dedicated section (Section 7) with the same rigor as positive results.

Minor quality issues:
- Section 8.2 (layer-position mechanism hypothesis) is more speculative than the rest of the paper. The two "falsifiable predictions" (penultimate-layer spike, entropy/norm sharpening) are valuable but the paragraph preceding them could be tighter -- the phrase "This pattern is consistent with a representation-sharpening mechanism" followed by a multi-clause explanation is the paper's densest passage.
- The abstract at ~250 words is longer than the outline's 150-word target. It contains four distinct result paragraphs. While comprehensive, a reviewer scanning quickly may lose the thread.
- Table 5 is truncated ("Table truncated to top-recovery words; full results in Appendix") but no appendix exists yet in the manuscript.

## Issues for the Editor

1. **Critical** -- **Broken cross-reference: "Table 7 in Section 4.4"**: Section 8.2 (line 359) cites "Table 7 in Section 4.4 of the extended results" -- neither Table 7 nor Section 4.4 exists in the paper. **Fix**: Either add the absorbed-vs-hedged proportion data as a supplementary table (if extended results are planned), or replace the reference with inline numbers from the consolidation summary (L24-16k: 50.0% absorbed; L6-16k: 100.0% absorbed) and remove the dangling cross-reference.

2. **Critical** -- **Broken cross-reference: "Section 8.6"**: Section 3.5 (line 146) references "Section 8.6" for future patching work at L24, but Future Directions is Section 8.5. **Fix**: Change "Section 8.6" to "Section 8.5."

3. **Critical** -- **Two missing figures (5 and 6)**: The outline planned Figure 5 (activation patching dot plot) and Figure 6 (hedging decomposition stacked bar). Neither is generated, referenced, or included. Sections 5.1 and 5.2 -- containing two of the four contributions -- have zero visual support. **Fix**: Generate both figures from existing data (`phase0/activation_patching_full.json` and `pilots/phase0_tightened_hedging_full.json`), add image references in Sections 5.1 and 5.2, and reference them in text before they appear.

4. **Major** -- **Figure 2 is unrendered**: The pipeline schematic exists only as a markdown description (`fig2_pipeline_desc.md`). The paper includes an image tag pointing to it, but no PDF or rendered diagram exists. **Fix**: Render the description as a TikZ diagram or programmatic figure and replace `fig2_pipeline_desc.md` with a proper image file.

5. **Major** -- **Activation patching probe F1 ambiguity**: The consolidation summary records probe_f1=0.883 for the activation patching experiment, but the paper implies F1>=0.97 by saying the "ICL probe provides reliable ground truth." If the actual probe used was F1=0.883 (below the strict 0.90 gate), this should be stated transparently, as it is listed in the consolidation's "required caveats." **Fix**: Add the actual probe F1 for the patching experiment in Section 3.5 or 5.1. If it was indeed 0.883, acknowledge this as a caveat.

6. **Minor** -- **Abstract exceeds target length**: The abstract is ~250 words vs. the 150-word target in the outline. While comprehensive, it could be trimmed by removing the unsupervised detector details (GAS/CMI/Tax rho values) from the abstract and leaving them for the body. **Fix**: Consider trimming to ~200 words by moving detector details to the introduction or body.

7. **Minor** -- **Table 5 references nonexistent Appendix**: The truncation note says "full results in Appendix" but no appendix text is included in the manuscript. **Fix**: Either add the appendix with full per-word results, or remove the "in Appendix" reference.

## What Works Well

1. **Transparent limitation reporting (Sections 4.3, 8.4)**: The probe quality confound is presented with the same analytical rigor as positive results. Three specific confound mechanisms (denominator asymmetry, missed absorption, spurious false negatives) are enumerated with clear explanations. This preempts the obvious reviewer objection and establishes credibility.

2. **Evidence-forward paragraph structure throughout**: Nearly every paragraph opens with a quantitative finding or concrete observation. The Introduction leads with "Absorption rates on the same sparse autoencoder vary 15-fold" rather than generic motivation. Section 7 (Negative Results) reports failure metrics with the same precision as successes. This adherence to the "lead with the concrete thing" rule makes the paper efficient to read.

3. **Section transition sentences**: Each section ends with a sentence that bridges to the next (e.g., "Having established that absorption rates vary by layer and hierarchy, we now provide causal evidence..."). These match the outline's planned transitions nearly verbatim and create clear narrative momentum through the paper's nine sections.

SCORE: 7
