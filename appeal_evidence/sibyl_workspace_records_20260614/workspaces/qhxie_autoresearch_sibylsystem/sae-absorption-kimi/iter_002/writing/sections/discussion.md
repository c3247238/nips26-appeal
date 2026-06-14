# 4 Discussion

## 4.1 Interpreting the Inconclusive Construct Validity

The point estimate $r = 0.463$ between first-letter and semantic-hierarchy absorption suggests a moderate positive relationship. Across the seven trained SAEs, architectures with lower first-letter absorption tend---on average---to show lower semantic-hierarchy absorption. BatchTopK and TopK, the two highest first-letter absorbers, also rank above the median on semantic hierarchies. PAnneal and GatedSAE, the two lowest first-letter absorbers, rank first and fourth respectively on semantic hierarchies.

Yet the bootstrap 95% CI $[-0.389, 0.981]$ spans from a moderate negative correlation to near-perfect positive correlation. With only $n = 7$ trained SAEs, the sampling variance is too large to support confident claims. The interval includes values below $0.3$ (which would indicate weak or no relationship) and values above $0.8$ (which would indicate strong generalization). Including the Random SAE drops the correlation to $r = 0.281$ (CI: $[-0.578, 0.901]$), further eroding any claim of generalization.

**Key takeaway:** The current evidence base is insufficient to conclude whether first-letter absorption predicts semantic-hierarchy absorption. A moderate positive point estimate is compatible with both genuine generalization and sampling noise. The field needs larger SAE cohorts---we estimate 15--20 architectures would be required to narrow the CI to a width of approximately $0.6$---before construct-validity claims can be evaluated with confidence.

## 4.2 The Hierarchy Specificity Failure

The paired t-test result ($t = -4.748$, $p = 0.0032$) is unambiguous: non-hierarchy correlated features produce *higher* absorption scores than genuine semantic hierarchies. Mean non-hierarchy absorption ($\bar{A}_{\text{NH}} = 0.331$) exceeds mean semantic-hierarchy absorption ($\bar{A}_{\text{SH}} = 0.235$) by 0.096 points, a 41% relative increase. This rejects H2 and contradicts the theoretical motivation for absorption.

If the metric were hierarchy-specific, parent-child pairs such as "building--house" should show the highest information loss, because the parent feature is most directly competing with its children for representational capacity. Instead, semantically correlated but non-hierarchical pairs such as "doctor--hospital" and "big--large" produce higher scores. Three explanations are plausible:

1. **Synthetic template artifacts.** The sentence templates used to generate training data (e.g., "The {concept} is a place with rich history.") may introduce spurious lexical correlations that inflate absorption for certain word pairs. Words that co-occur in similar template contexts may activate overlapping SAE latents regardless of their semantic relationship.

2. **Semantic relatedness without hierarchy.** The SAE encoder may treat any pair of semantically related words as competing for the same latent dimensions, regardless of whether the relationship is hierarchical. If "doctor" and "hospital" activate overlapping feature sets, the encoder's sparsity constraint may suppress one when the other is active---producing an absorption-like signal that has nothing to do with parent-child structure.

3. **Coarse k-sparse threshold.** The k-sparse probing protocol ($k = 10$) may be too coarse to distinguish fine-grained semantic distinctions. If the top-10 latents for "building" and "house" overlap substantially, the probe may fail to discriminate them even when finer-grained latents (say, the 11th--20th) contain the distinguishing information. A hierarchy-specific metric might require adaptive $k$ or per-hierarchy threshold tuning.

The hierarchy specificity failure is the most consequential negative result in this study. It indicates that the semantic-hierarchy adaptation of the SAEBench metric does not isolate the phenomenon it was designed to measure.

## 4.3 The Random-SAE Anomaly

The Random-SAE control yields the most striking finding. A structurally destroyed SAE---decoder directions permuted at random---achieves semantic-hierarchy absorption of 0.352 and non-hierarchy control absorption of 0.416, identical (to three decimal places) to the Standard SAE. The first-letter absorption of the Random SAE is 0.030, near zero as expected.

This dissociation is critical. On first-letter tasks, the metric behaves as theory predicts: trained SAEs show variable absorption, random SAEs show none. On semantic tasks, the metric is completely insensitive to whether the SAE has learned anything at all. The implication is direct: **the semantic-hierarchy adaptation of the absorption metric is degenerate.** It captures statistical artifacts---likely related to the geometry of the decoder directions and the distribution of the probe inputs---that persist even when learned structure has been eliminated.

Why does the degeneracy affect semantic tasks but not first-letter tasks? One hypothesis is that first-letter hierarchies are structurally simple (binary parent-child splits on a single character) and the SAE must learn specific directional structure to represent them. Semantic hierarchies are more complex, and the logistic probe may be sensitive to directions in activation space that are preserved under random permutation of decoder weights. Testing this hypothesis would require analyzing which decoder directions the probe relies on and whether those directions align with learned vs. random features.

## 4.4 Implications for Benchmark Design

These findings carry direct implications for how the SAE community measures and optimizes absorption.

**Do not extend first-letter absorption to semantic tasks without validation.** The SAEBench absorption metric was designed and validated on first-letter classification. Our results show that applying the same formula to semantic hierarchies produces a metric that (a) correlates inconclusively with the original, (b) fails hierarchy specificity, and (c) is insensitive to learned structure. Any study reporting "absorption" on semantic tasks using the SAEBench formula should be interpreted with extreme caution.

**Architecture comparisons may be task-dependent.** The ranking inversions observed in Table 1 (e.g., GatedSAE lowest on first-letter but fourth-lowest on semantic) suggest that an architecture's absorption performance is not a single scalar property. An architecture that reduces first-letter absorption may or may not reduce semantic-hierarchy absorption. Claims of "absorption reduction" should specify the task domain and provide evidence across multiple task types.

**Future work: domain-specific absorption metrics.** Rather than assuming a single absorption metric generalizes across all hierarchy types, the community should develop domain-specific metrics with demonstrated hierarchy specificity. A valid semantic-hierarchy absorption metric would need to (1) score higher on parent-child pairs than on semantically correlated non-hierarchical pairs, (2) distinguish trained from random SAEs, and (3) correlate robustly with first-letter absorption across diverse architectures. None of these properties hold for the current adaptation.

## 4.5 Limitations

Our study has several limitations that bound the scope of the conclusions.

**Small SAE sample.** With $n = 7$ trained SAEs, the correlation analysis is underpowered. The bootstrap CI width of 1.37 correlation units reflects this directly. A cohort of 15--20 architectures would be needed for adequate statistical power.

**Single layer and model size.** All experiments use Pythia-160M layer 8. Absorption patterns may differ at other layers (earlier layers encode more lexical information; later layers encode more semantic information) and at other model scales. The GPT-2 replication shows model-dependent effects, but the near-zero absolute scores on GPT-2 make interpretation difficult.

**Shallow hierarchies.** Our WordNet hierarchies are two-level parent-child pairs. Deeper hierarchies (3--4 levels, e.g., "animal" $\supset$ "mammal" $\supset$ "dog") might exhibit stronger absorption signals if the sparsity constraint compounds across levels. Testing deeper hierarchies is an important direction for future work.

**Synthetic sentence templates.** The balanced datasets use simple templates that may not reflect natural language distribution. More varied or naturally sampled sentences could alter the absorption patterns, though the frequency-matching constraint would be harder to satisfy.

**Probe training variance.** Logistic regression probes are trained with 200 Adam epochs and a fixed learning rate. While all hierarchies achieved perfect AUROC ($1.0$), probe training dynamics could introduce variance in the accuracy values that feed into the absorption formula. We did not evaluate probe training stability across random seeds.

**Single threshold for k-sparse probing.** We use $k = 10$ for all hierarchies and architectures. An adaptive $k$ (e.g., proportional to the number of children in the hierarchy) might yield different results, though the $\tau_{\text{fs}}$ robustness analysis suggests the correlation is stable across thresholds.

From these implications, we conclude with concrete recommendations for the field.

<!-- FIGURES
- None
-->
