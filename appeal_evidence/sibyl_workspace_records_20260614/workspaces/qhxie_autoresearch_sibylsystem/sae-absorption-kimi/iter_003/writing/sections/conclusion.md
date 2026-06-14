# 5. Conclusion

## 5.1 Summary

This paper presents the first construct-validity study of the SAEBench feature absorption metric, testing whether first-letter absorption scores generalize to real semantic hierarchies drawn from WordNet. Across eight SAE architectures on Pythia-160M layer 8, three findings emerge.

First, construct validity is inconclusive. The Pearson correlation between first-letter and semantic-hierarchy absorption is $r = 0.463$ ($95\%$ CI: $[-0.389, 0.981]$). The point estimate suggests a moderate positive relationship, but the confidence interval spans from moderately negative to near-perfect correlation. With only seven trained SAEs, the evidence neither supports nor rejects the hypothesis that the metric generalizes.

Second, hierarchy specificity fails. Non-hierarchy correlated features show significantly higher absorption ($\bar{A}_{\text{NH}} = 0.331$) than semantic hierarchies ($\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.003$). Every architecture except TopK shows this reversal. The metric detects absorption-like behavior in semantically correlated pairs that lack parent-child structure, contradicting the theoretical motivation that absorption is specific to hierarchical features.

Third, the Random-SAE control reveals degeneracy. A randomized SAE with permuted decoder directions achieves semantic-hierarchy absorption of $0.352$, identical to the trained Standard SAE ($0.352$), and non-hierarchy control absorption of $0.416$, also identical. The first-letter task correctly distinguishes trained from randomized structure ($0.030$), but the semantic adaptations of the metric do not. This indicates that semantic-hierarchy absorption scores capture artifacts of basis geometry rather than learned SAE structure.

## 5.2 Recommendations

These findings yield three concrete recommendations for the SAE research community.

**For benchmark designers:** Do not extend the first-letter absorption metric to semantic hierarchies without substantial modification. The current adaptation---replacing first-letter hierarchies with WordNet hypernyms while retaining the same probe protocol---produces scores that are insensitive to learned structure and fail hierarchy specificity. Any semantic-hierarchy absorption metric should demonstrate both hierarchy specificity (hierarchy $>$ non-hierarchy) and sensitivity to training (trained $>$ random) before community adoption.

**For architecture researchers:** Absorption-reduction claims should be validated on multiple task types. Matryoshka SAEs report order-of-magnitude absorption reductions relative to Standard SAEs on first-letter tasks (Bussmann et al., 2025). Our results do not challenge this claim for first-letter hierarchies, but they do challenge its generalization to semantic tasks. Architecture comparisons that rank SAEs by absorption alone may be optimizing for a metric that does not reflect behavior on real conceptual hierarchies.

**For the community:** Invest in domain-specific absorption metrics with validated hierarchy specificity. A metric that cannot distinguish hierarchies from correlated pairs, and cannot distinguish trained from randomized SAEs, is not ready for benchmark status. Future metrics should explicitly test for hierarchical structure, perhaps using causal ablations or interventional probes that go beyond passive probe accuracy.

## 5.3 Future Work

Four directions would strengthen and extend this study.

**Larger SAE cohorts.** A cohort of 15--20 architectures would provide adequate statistical power for definitive construct-validity testing. Our post-hoc power analysis shows that detecting $r = 0.6$ at $\alpha = 0.05$ with $80\%$ power requires $n \approx 19$ SAEs. The current sample of seven trained architectures yields a bootstrap distribution too diffuse for confident inference.

**Deeper hierarchies and multiple base models.** Our WordNet hierarchies are single-level parent-child relationships. Deeper hierarchies (3--4 levels, e.g., "animal" $\to$ "mammal" $\to$ "dog") may exhibit different absorption patterns, and the sparsity-loss incentive may compound across levels. The GPT-2 replication shows model-dependent effects, with near-zero absolute scores that differ from Pythia-160M. A broader model sweep---spanning scales from GPT-2 small to Gemma-2-2B and layers from early to late---would test whether semantic-hierarchy absorption is a stable phenomenon or a model-specific artifact.

**Alternative hierarchy sources.** WordNet is one of many lexical ontologies. ConceptNet (Speer et al., 2017) provides broader relational coverage; BabelNet (Navigli & Ponzetto, 2012) integrates multilingual hierarchies. Testing absorption on hierarchies from multiple sources would guard against ontology-specific artifacts and increase ecological validity.

**Causal validation.** The current metric is correlational: it measures probe accuracy drops, not whether parent-feature information is truly missing or merely hidden in distributed form. Causal ablation studies---following the methodology of Chanin et al. (2024)---could distinguish "truly absorbed" from "merely distributed" parent features. Activation patching or interchange intervention (Geiger et al., 2023) could test whether suppressing child latents restores parent-feature accessibility. Such experiments would move absorption measurement from diagnostic probing to causal validation, aligning the metric more closely with the theoretical construct it claims to measure.

<!-- FIGURES
- None
-->
