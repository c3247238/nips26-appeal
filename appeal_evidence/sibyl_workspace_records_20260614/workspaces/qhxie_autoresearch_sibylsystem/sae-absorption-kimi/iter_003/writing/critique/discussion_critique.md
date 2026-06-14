# Critique: Discussion

## Summary Assessment

The Discussion section offers a thoughtful interpretation of the three main findings and connects them to broader benchmark-design concerns. The hierarchy-specificity analysis (4.2) and Random-SAE interpretation (4.3) are particularly strong. However, the section contains a significant logical gap in 4.1 where a post-hoc power analysis is introduced without justification, overreaches in 4.3 by attributing a finding to Korznikov et al. that they did not report, and underdevelops the implications subsection (4.4) relative to the depth of the preceding analysis. The limitations (4.5) are comprehensive but include one that is overstated.

## Score: 6/10

**Justification**: The section demonstrates genuine interpretive insight---especially the geometric-basis explanation for the Random-SAE anomaly---and maintains appropriate epistemic humility about the inconclusive construct-validity result. To reach 7-8, it needs: (1) removal or reframing of the unsupported post-hoc power analysis, (2) correction of the Korznikov et al. citation overreach, (3) more substantive development of the implications subsection, and (4) a clearer distinction between what the data show and what the authors speculate. To reach 9-10, it would need to engage more directly with alternative interpretations of the Random-SAE result and situate the findings more precisely within the ongoing methodological debate in the SAE community.

---

## Critical Issues

### Issue 1: Unsupported Post-Hoc Power Analysis in 4.1

- **Location**: Section 4.1, second paragraph
- **Quote**: "A post-hoc power analysis shows that detecting $r = 0.6$ at $\alpha = 0.05$ with 80% power requires $n \approx 19$ SAEs."
- **Problem**: This post-hoc power analysis is methodologically inappropriate and potentially misleading. Post-hoc power calculations after observing data are widely criticized in the statistical literature (Hoenig & Heisey, 2001) because they are mathematically redundant with the p-value and can create a false sense of objectivity about sample-size needs. The observed correlation (r = 0.463) and its CI already communicate the uncertainty completely. Adding a power analysis based on an arbitrary effect-size threshold (r = 0.6, the H1 threshold) does not add information---it merely restates the width of the CI in a different format. Worse, it could mislead readers into thinking the study was underpowered for a specific effect size that was chosen after seeing the data.
- **Fix**: Remove the power analysis entirely. Replace with a forward-looking statement about sample size needs for future studies, framed as a design recommendation rather than a post-hoc calculation: "For future studies aiming to detect a correlation of r = 0.6 with 80% power at alpha = 0.05, a sample of approximately 19 architectures would be needed." Alternatively, simply report the observed CI and let it speak for itself.

### Issue 2: Overreaching Citation of Korznikov et al. (2025)

- **Location**: Section 4.3, paragraph 2
- **Quote**: "This finding aligns with Korznikov et al. (2025), who showed that orthogonal constraints on SAE decoders reduce absorption by 65%---suggesting that decoder geometry is a major determinant of absorption scores."
- **Problem**: The proposal.md (line 18) states that Korznikov et al. (2025) showed "OrtSAE: 65% absorption reduction"---but this refers to their *architecture's* absorption reduction relative to a baseline, not a finding about decoder geometry as a "major determinant of absorption scores." The Discussion reframes their architectural contribution as empirical evidence for a geometric explanation, which overreaches what Korznikov et al. actually demonstrated. Their paper introduced orthogonal constraints as a training objective; it did not systematically decompose absorption into geometric vs. learned components. The 65% figure is a comparison between architectures, not evidence that geometry dominates.
- **Fix**: Reattribute the claim. Replace with: "This finding is consistent with the intuition behind Korznikov et al. (2025), who introduced orthogonal decoder constraints and reported 65% absorption reduction---suggesting that decoder geometry may influence absorption. However, their work does not establish that geometry is the primary determinant, and our Random-SAE result provides more direct evidence for this interpretation." This preserves the connection while accurately representing the source.

---

## Major Issues

### Issue 3: Underdeveloped Implications Subsection (4.4)

- **Location**: Section 4.4
- **Quote**: "Our findings carry four implications for SAE benchmark design and community practice." [followed by four bullet-style paragraphs]
- **Problem**: After the depth and nuance of 4.1-4.3, the implications subsection reads like a checklist rather than a developed argument. Each "implication" is stated in a single paragraph with minimal engagement with counterarguments or trade-offs. For example, implication 2 notes that Matryoshka SAEs report ~10x absorption reduction on first-letter tasks, but does not address whether the community should *stop* using absorption for architecture comparison entirely, or merely add caveats. Implication 3 calls for random-baseline correction but does not discuss the practical cost (running random baselines for every new metric) or whether PCA baselines might be preferable. The subsection feels like it is going through motions rather than wrestling with the hard questions the findings raise.
- **Fix**: Expand each implication to 2-3 sentences that acknowledge trade-offs. For implication 2, add: "This does not mean absorption should be abandoned as an architecture criterion---first-letter absorption may still capture meaningful structural properties---but claims of absorption reduction should be accompanied by evidence of generalization." For implication 3, add: "While random-baseline correction adds computational overhead, the marginal cost is small relative to SAE training, and the diagnostic value is substantial." Develop the tension between the recommendations rather than presenting them as independent bullet points.

### Issue 4: Overstated "Probe Ceiling Effects" Limitation

- **Location**: Section 4.5, paragraph 5
- **Quote**: "All hierarchies achieved perfect probe AUROC ($= 1.0$) on residual activations, leaving no headroom for SAE latents to match or exceed residual performance. This ceiling effect means the absorption formula's numerator ($\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}$) is bounded above by the residual accuracy, which is already at maximum."
- **Problem**: This limitation is logically confused. The absorption formula divides by $\text{acc}_{\text{resid}}$, so a ceiling at 1.0 actually *maximizes* the denominator and thus *minimizes* the absorption score for any given accuracy drop. If residual AUROC were lower (e.g., 0.7), the same accuracy drop would produce a *larger* absorption score. The "ceiling effect" described here actually works against finding high absorption, making the observed absorption scores (0.235 mean) more rather than less concerning. The authors seem to confuse "ceiling effect" (a measurement problem where scores cluster at the maximum) with "high baseline," which is a different issue. A true ceiling effect would occur if SAE latents also scored 1.0, forcing absorption to 0---but they do not.
- **Fix**: Remove this as a limitation, or reframe it correctly: "The perfect residual AUROC (1.0) sets a high baseline that minimizes the absorption score for any given accuracy drop. Hierarchies with lower residual AUROC might show larger absolute absorption scores, though the relative ranking of architectures may be stable." Alternatively, note that the ceiling effect is actually on the *SAE* side: if SAE latents also achieved AUROC = 1.0, absorption would be zero, but this does not occur.

### Issue 5: Missing Engagement with Alternative Explanations for Random-SAE Result

- **Location**: Section 4.3
- **Quote**: "What could these artifacts be? The permuted decoder preserves the geometric properties of the activation space---the angles between decoder directions, their norms, and their coverage of the residual-stream manifold."
- **Problem**: The section presents the geometric-basis explanation as the primary interpretation of the Random-SAE anomaly but does not engage with alternative explanations that a skeptical reviewer would raise. Three alternatives are worth at least brief mention: (1) The probe may be learning a simple linear classifier that works on any basis with sufficient dimensionality, not because of geometry but because of overparameterization. (2) The synthetic sentence template may be so trivial that any basis with 2048 dimensions can discriminate "house" from "building" with similar accuracy. (3) The permutation may not be as destructive as assumed---the encoder is still trained, and the permuted decoder may preserve enough structure for the specific task. The section's confidence in the geometric explanation exceeds what the data can support.
- **Fix**: Add a paragraph or sentence acknowledging these alternatives: "We favor the geometric-basis explanation, but alternative interpretations are possible. The probe may exploit overparameterization rather than basis geometry; the synthetic template may be too simple to stress the representation; or the trained encoder may compensate for decoder permutation. Distinguishing these explanations requires additional controls (e.g., varying latent dimension, natural-language templates, fully random encoder-decoder pairs)."

---

## Minor Issues

- **Section 4.1, line 7**: "The conservative interpretation is that the current evidence base is insufficient for confident claims about construct validity." The phrase "conservative interpretation" is slightly hedged. The data *require* this interpretation---it is not merely conservative. Fix: "The only warranted interpretation is that..."

- **Section 4.2, line 13**: "This contradicts the theoretical motivation for the absorption metric." This is strong but accurate. However, the next sentence attributes the theoretical motivation specifically to Chanin et al. (2024), who proved sparsity loss incentivizes absorption for hierarchical features. The contradiction is clear, but the Discussion could note that Chanin et al.'s proof applies to their specific construction---it does not necessarily generalize to all hierarchical features or all SAE architectures.

- **Section 4.2, line 19**: "The fixed sentence template ('The [concept] is [property].') may introduce spurious correlations." This is a good point, but it should be cross-referenced to the same concern raised in Limitations (4.5, paragraph 4) to avoid reader confusion about whether this is a new point or a restatement.

- **Section 4.3, line 29**: "The Random-SAE control---constructed by permuting the decoder directions of the Standard SAE while retaining its trained encoder---yields first-letter absorption of 0.030, near zero as expected." The phrase "as expected" is used without prior justification in the Discussion. The Method section explains the construction but not why near-zero first-letter absorption is expected. Add: "because permuting the decoder destroys the letter-specific features learned during training."

- **Section 4.4, implication 1**: "The SAEBench absorption metric should not be extended to semantic hierarchies without substantial modification." This recommendation is sound, but "substantial modification" is vague. What kind of modification? The Discussion should be more specific: e.g., "modification that demonstrates hierarchy specificity and training sensitivity."

- **Section 4.5, paragraph 1**: "With seven trained SAEs plus one random control, statistical power is limited." This is accurate but repetitive with 4.1, which already discussed sample size. The Limitations section should focus on limitations not already covered in depth.

- **Section 4.5, paragraph 4**: "Synthetic sentence templates may not reflect natural language distribution." This limitation is valid but should note that the *first-letter benchmark also uses synthetic templates* (albeit different ones), so this is not a unique weakness of the semantic-hierarchy adaptation.

- **Missing transition to Conclusion**: The Discussion ends with "From these implications, we distill concrete recommendations for the field." but the outline specifies the transition should be "From these implications, we conclude with concrete recommendations." The current phrasing is close but slightly awkward. The Conclusion section begins with "This paper presents..." which is a hard cut. A softer transition would help.

---

## Visual Element Assessment

- [x] Figures/tables match outline plan
  - The Discussion section correctly references Figures 2, 3, and 4 from the Results section without introducing new visuals, consistent with the outline's Figure & Table Plan (no new figures planned for Discussion).
- [x] All visuals referenced before appearance
  - All figure references are to Results-section figures that appear earlier in the paper. The Discussion does not introduce any new visuals.
- [x] Captions are self-explanatory
  - Not applicable---no new captions in this section.
- [ ] No text-heavy sections that need visual support
  - Section 4.3 (Random-SAE Anomaly) is text-dense and would benefit from a conceptual diagram showing: (1) trained decoder directions, (2) permuted decoder directions, (3) the geometric properties preserved under permutation. The outline does not plan such a figure, but adding one would significantly improve clarity for readers who struggle to visualize what "permuted decoder" means.

---

## What Works Well

1. **Epistemic calibration in 4.1**: The section correctly resists the temptation to overinterpret the point estimate (r = 0.463) and instead centers the CI width as the key message. The sentence "We cannot conclude that first-letter absorption generalizes to semantic hierarchies, nor should we dismiss the possibility outright" is a model of appropriate uncertainty communication.

2. **Geometric-basis explanation in 4.3**: The interpretation that "the semantic-hierarchy absorption score depends primarily on how well the probe can discriminate concepts given the geometric structure of the basis" is intellectually satisfying and connects the empirical anomaly to a mechanistic explanation. The reasoning from preserved angles and norms to preserved scores is clear and testable.

3. **Integration of findings across subsections**: The Discussion does not treat the three findings as isolated results. Instead, it weaves them together: the inconclusive H1 (4.1) sets up the specificity failure (4.2), which is then explained by the Random-SAE anomaly (4.3), which then motivates the benchmark-design recommendations (4.4). This cumulative argument structure is effective and easy to follow.
