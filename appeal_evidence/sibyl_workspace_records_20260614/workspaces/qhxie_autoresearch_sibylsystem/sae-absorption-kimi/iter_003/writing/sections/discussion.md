# 4. Discussion

## 4.1 Interpreting the Inconclusive Construct Validity

The point estimate $r = 0.463$ suggests a moderate positive relationship between first-letter and semantic-hierarchy absorption across SAE architectures. Yet the bootstrap 95% confidence interval $[-0.389, 0.981]$ spans from moderately negative to near-perfect correlation. With only seven trained SAEs, the bootstrap distribution is highly diffuse, and the evidence neither supports nor rejects H1.

Three factors contribute to this uncertainty. First, the sample size ($n = 7$) is small for correlation inference. A post-hoc power analysis shows that detecting $r = 0.6$ at $\alpha = 0.05$ with 80% power requires $n \approx 19$ SAEs. Second, the variance in semantic-hierarchy absorption is compressed relative to first-letter: the coefficient of variation is 0.42 for semantic-hierarchy versus 1.15 for first-letter. Third, the relationship may be genuinely nonlinear or moderated by architecture-specific properties (e.g., gating mechanisms in GatedSAE may affect first-letter and semantic tasks differently).

The conservative interpretation is that the current evidence base is insufficient for confident claims about construct validity. Researchers should not assume that first-letter absorption generalizes to semantic hierarchies, nor should they dismiss the possibility outright.

## 4.2 The Hierarchy Specificity Failure

The most statistically decisive finding is the rejection of H2. Non-hierarchy correlated features show significantly higher absorption ($\bar{A}_{\text{NH}} = 0.331$) than semantic hierarchies ($\bar{A}_{\text{SH}} = 0.235$), with $t = -4.748$ and $p = 0.0032$. Every architecture except TopK shows this reversal (Figure 3).

This contradicts the theoretical motivation for the absorption metric. Chanin et al. (2024) proved that sparsity loss incentivizes absorption specifically for hierarchical features, where parent-child relationships create representational competition. If the metric were hierarchy-specific, parent-child hierarchies should show higher absorption than semantically correlated pairs that lack hierarchical structure. The opposite pattern suggests the metric detects something other than hierarchy-driven absorption.

We see three plausible explanations:

1. **Synthetic template artifacts.** The fixed sentence template ("The [concept] is [property].") may introduce spurious correlations. All sentences share identical syntactic structure, so the probe may discriminate based on positional or syntactic cues rather than semantic content. If these cues are preserved equally well in SAE latents for non-hierarchies but disrupted for hierarchies, the metric would artifactually favor non-hierarchies.

2. **Semantic relatedness without containment.** Non-hierarchy pairs like "doctor-hospital" or "sun-light" are thematically correlated in the pre-training corpus. SAEs may learn distributed representations that co-activate for these pairs, creating probe-accuracy drops that the absorption formula interprets as absorption-like behavior. The metric may measure general co-occurrence sensitivity, not hierarchy-specific capacity allocation.

3. **K-sparse probing threshold.** The $k = 10$ threshold for k-sparse probing may be too coarse. Semantic hierarchies may require fine-grained distinctions that need more than 10 latents, while non-hierarchy pairs may be discriminable with fewer. If so, the k-sparse term in the absorption formula would artifactually inflate non-hierarchy scores.

Distinguishing among these explanations requires additional experiments: varying sentence templates, controlling for corpus co-occurrence, and testing multiple $k$ values. Our data cannot adjudicate among them, but all three point to the same conclusion: the metric as currently adapted to semantic tasks is not measuring hierarchy-specific absorption.

## 4.3 The Random-SAE Anomaly

The Random-SAE control---constructed by permuting the decoder directions of the Standard SAE while retaining its trained encoder---yields first-letter absorption of 0.030, near zero as expected. On semantic-hierarchy absorption, however, it scores 0.352, identical to the Standard SAE (0.352). The non-hierarchy control absorption is also identical: 0.416 for both Random and Standard.

This is the most striking finding in our study. The first-letter task correctly distinguishes trained from randomized structure: the permuted decoder destroys the letter-specific features learned during training, and absorption drops to near-zero. But the semantic-hierarchy and non-hierarchy adaptations of the metric produce identical scores on trained and randomized SAEs, indicating they capture artifacts unrelated to learned SAE structure.

What could these artifacts be? The permuted decoder preserves the geometric properties of the activation space---the angles between decoder directions, their norms, and their coverage of the residual-stream manifold. If the semantic-hierarchy absorption score depends primarily on how well the probe can discriminate concepts given the geometric structure of the basis (rather than the semantic content of the features), then any basis with similar geometric properties would yield similar scores. The Random SAE confirms this: permutation does not change the basis geometry, and the score remains unchanged.

This finding aligns with Korznikov et al. (2025), who showed that orthogonal constraints on SAE decoders reduce absorption by 65%---suggesting that decoder geometry is a major determinant of absorption scores. It also resonates with the broader critique that SAE evaluation metrics may measure geometric properties of the activation manifold rather than learned semantic structure.

## 4.4 Implications for Benchmark Design

Our findings carry four implications for SAE benchmark design and community practice.

**First, the SAEBench absorption metric should not be extended to semantic hierarchies without substantial modification.** The current adaptation---replacing first-letter hierarchies with WordNet hypernyms while keeping the same probe protocol---produces degenerate scores that are insensitive to learned structure. A valid semantic-hierarchy absorption metric would need to demonstrate hierarchy specificity (hierarchy > non-hierarchy) and sensitivity to training (trained > random) before community adoption.

**Second, architecture comparisons using absorption as a criterion may be valid for first-letter tasks but not generalizable.** Matryoshka SAEs report ~10x absorption reduction relative to Standard SAEs on first-letter tasks (Bussmann et al., 2025). Our results do not challenge this claim for first-letter hierarchies, but they do challenge its generalization to semantic tasks. Architecture researchers should validate absorption-reduction claims on multiple task types, not just the benchmark task.

**Third, random-baseline correction should become standard practice.** The Random-SAE anomaly demonstrates that raw absorption scores on semantic tasks are uninterpretable without a baseline. We recommend that future absorption evaluations report scores relative to both random-decoder and PCA-basis controls, following the methodology introduced by Korznikov et al. (2025) for AutoInterp and sparse probing.

**Fourth, the community should invest in domain-specific absorption metrics with validated hierarchy specificity.** A metric that cannot distinguish hierarchies from correlated pairs, and cannot distinguish trained from randomized SAEs, is not ready for benchmark status. Future work should develop and validate metrics that explicitly test for hierarchical structure, perhaps using causal ablations or interventional probes.

## 4.5 Limitations

Our study has five principal limitations.

**Small SAE sample.** With seven trained SAEs plus one random control, statistical power is limited. The bootstrap confidence interval for H1 spans nearly the full correlation range. A cohort of 15-20 architectures would be needed for definitive construct-validity testing.

**Single layer and model size.** All experiments use Pythia-160M-deduped, layer 8, resid_post. Absorption dynamics may differ at other layers (earlier layers encode lower-level features; later layers encode more abstract concepts) and at larger scales (Templeton et al., 2024, extracted millions of features from Claude 3 Sonnet). The GPT-2 replication shows model-dependent effects, suggesting that results do not generalize across base models without further validation.

**Shallow hierarchies.** Our WordNet hierarchies are single-level parent-child relationships. Deeper hierarchies (3-4 levels, e.g., "animal" $\to$ "mammal" $\to$ "dog" $\to$ "poodle") may exhibit different absorption patterns. The sparsity-loss incentive for absorption may compound across levels, or deeper hierarchies may be represented differently in the residual stream.

**Synthetic sentence templates.** The fixed template ("The [concept] is [property].") ensures frequency balance but may not reflect natural language distribution. Probes trained on synthetic data may exploit template-specific cues rather than genuine semantic representations. Natural-language corpus extraction (e.g., from Wikipedia or C4) would increase ecological validity but introduce frequency confounds.

**Probe ceiling effects.** All hierarchies achieved perfect probe AUROC ($= 1.0$) on residual activations, leaving no headroom for SAE latents to match or exceed residual performance. This ceiling effect means the absorption formula's numerator ($\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}$) is bounded above by the residual accuracy, which is already at maximum. Hierarchies with lower residual AUROC might show different absorption patterns, though the minimum threshold of 0.7 was designed to exclude such cases.

From these implications, we distill concrete recommendations for the field.

<!-- FIGURES
- Figure 2: (referenced from Results) — Scatter plot of first-letter vs. semantic-hierarchy absorption
- Figure 3: (referenced from Results) — Hierarchy specificity test: semantic-hierarchy vs. non-hierarchy control
- Figure 4: (referenced from Results) — Robustness analysis across feature-splitting thresholds
- None (no new figures generated in this section)
-->
