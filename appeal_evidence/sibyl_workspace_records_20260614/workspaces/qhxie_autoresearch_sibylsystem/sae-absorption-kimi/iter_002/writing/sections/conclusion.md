# 5 Conclusion

## 5.1 Summary

This study reports the first construct-validity test of the SAEBench feature absorption metric on real semantic hierarchies. We evaluated eight SAE architectures---seven trained and one random control---on first-letter classification, WordNet semantic hierarchies, and a non-hierarchy correlated-feature control, all on Pythia-160M layer 8.

Our analysis yields three main findings, each answering one of the research questions posed in Section 1.3. **RQ1 (construct validity):** The Pearson correlation between first-letter and semantic-hierarchy absorption is $r = 0.463$ (95% CI: $[-0.389, 0.981]$) across the seven trained SAEs. The point estimate suggests a moderate positive relationship, but the confidence interval spans from negative to near-perfect correlation, rendering the construct-validity test statistically inconclusive. **RQ2 (hierarchy specificity):** The metric fails hierarchy specificity. Non-hierarchy correlated features yield significantly higher absorption scores than semantic hierarchies (paired $t$-test testing semantic $>$ non-hierarchy: $t = -4.748$, $p = 0.0032$). **RQ3 (robustness):** The inconclusiveness of the construct-validity test is stable across feature-splitting thresholds ($\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$), with $r$ ranging from 0.468 to 0.471 and all CIs remaining wide.

A fourth finding cuts across all three questions. A Random-SAE control---constructed by permuting the decoder directions of the Standard SAE---achieves semantic-hierarchy absorption of 0.352, identical (to three decimal places) to the Standard SAE, despite scoring near-zero (0.030) on first-letter absorption. This dissociation indicates that the metric on semantic tasks captures artifacts unrelated to learned SAE structure, while the original first-letter metric behaves as theory predicts.

The GPT-2 small replication reinforces the model-specificity concern: absolute semantic-hierarchy scores on GPT-2 are near-zero compared to Pythia-160M, and the pattern of hierarchy-specificity differences does not replicate cleanly. This suggests that construct-validity conclusions drawn from a single base model may not generalize.

## 5.2 Recommendations

These results carry direct implications for the SAE research community.

**For benchmark designers:** Treat the semantic-hierarchy adaptation of the absorption metric as degenerate---it captures artifacts unrelated to learned SAE structure, as demonstrated by identical scores from trained and Random-SAE controls. Do not extend first-letter absorption to semantic tasks without a validated metric that satisfies three criteria: (1) higher scores on parent-child pairs than on semantically correlated non-hierarchical pairs, (2) distinguishable scores between trained and random SAEs, and (3) robust correlation with first-letter absorption across diverse architectures.

**For architecture researchers:** Claims of "absorption reduction" should specify the task domain and provide evidence across multiple task types. An architecture that reduces first-letter absorption may or may not reduce semantic-hierarchy absorption. The ranking inversions observed in this study (e.g., GatedSAE lowest on first-letter but fourth-lowest on semantic hierarchies) demonstrate that absorption performance is not a single scalar property.

**For the community:** Invest in domain-specific absorption metrics with demonstrated hierarchy specificity, rather than assuming that a single synthetic benchmark generalizes across semantic domains. A valid semantic-hierarchy absorption metric must be shown to measure the phenomenon it claims to measure before it can guide architecture development.

## 5.3 Future Work

Several directions warrant follow-up work.

A larger SAE cohort (15--20 architectures) would tighten the confidence interval and resolve the construct-validity question. Power analysis suggests that $n = 15$ would narrow the 95% CI to a width of approximately 0.6 correlation units under the observed variance.

Deeper WordNet hierarchies (3--4 levels, e.g., "animal" $\supset$ "mammal" $\supset$ "dog") would test whether the hierarchy-specificity failure is an artifact of shallow hierarchies or a broader limitation of the metric.

Multiple base models beyond Pythia-160M and GPT-2 small---including larger models (Pythia-1.4B, Llama-3-8B) and different architectures---would test whether the observed patterns are model-specific or generalize across the model landscape.

Causal ablation studies---systematically ablating parent-feature latents and measuring child-feature reconstruction---on 3--4 hierarchies would distinguish "truly missing" from "merely hidden" absorbed features, clarifying whether the metric measures the phenomenon it claims to measure.

Until then, the first-letter absorption benchmark should be treated as a useful but narrow diagnostic, not a universal proxy for feature absorption in sparse autoencoders.

<!-- FIGURES
- None
-->
