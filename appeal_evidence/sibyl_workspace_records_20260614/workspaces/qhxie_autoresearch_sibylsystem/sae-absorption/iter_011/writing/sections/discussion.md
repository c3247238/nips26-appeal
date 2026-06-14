# 7 Discussion

Sections 4--6 established three empirical results: absorption rates vary across hierarchy types with probe quality as a major confound (Section 4), activation patching confirms a universal competitive exclusion mechanism (Section 5), and architecture choice has no detected effect relative to hierarchy type (Section 6). This section synthesizes these findings into implications for the field, addresses failed approaches, and catalogues limitations.

## 7.1 Causal Methods Succeed Where Correlational Methods Fail

Four independent correlational and statistical approaches to predicting absorption all failed:

- **Geometric Absorption Score (GAS):** $\rho = 0.116$, AUROC $= 0.571$, bootstrap 95% CI $[-0.333, 0.536]$ (Appendix B). GAS measures decoder-activation geometry---the angular mismatch between a feature's decoder direction and its typical activation pattern. The 25-word full-scale evaluation confirms the pilot null: decoder geometry does not predict which features will be absorbed.

- **Conditional mutual information (CMI):** $\rho = 0.044$, $p = 0.84$ at $L_0 = 22$ (Appendix C). At this sparsity level, all 25 first-letter probes achieve $F_1 = 1.0$, eliminating probe quality as a confound. The bootstrap CI $[-0.41, 0.47]$ includes zero. Information-theoretic dependence between parent and child activations does not predict absorption.

- **Absorption Tax $T(G)$:** Ranking $\rho = -0.20$, concordance 50%---chance level (Appendix D). The minimum additional $L_0$ cost for absorption-free representation produces quantitative predictions no better than random ordering. The qualitative insight that absorption imposes a sparsity cost is retained; the quantitative framework is not supported.

- **Rate-distortion three-factor model:** $\rho = 0.286$, $R^2 = 0.104$ across 131 parent-child pairs (Appendix F). All three individual predictors---decoder cosine similarity ($\rho = -0.090$), co-occurrence ($\rho = -0.189$), and reconstruction importance ($\rho = -0.239$)---have the wrong sign relative to the hypothesis. The direction reversal from the 20-pair pilot ($\rho > 0$) to the 131-pair full evaluation demonstrates small-sample instability.

This quadruple negative establishes a methodological boundary. Absorption is driven by encoder competitive dynamics during inference, not by static geometric or information-theoretic properties of the trained SAE. Decoder cosine similarity, co-occurrence frequency, and reconstruction importance---the most natural candidate predictors---all fail. Only activation patching, which intervenes on the causal pathway by zeroing the child feature and measuring parent recovery, successfully characterizes absorption (Section 5). Future absorption research should prioritize interventional methods over correlational proxies.

## 7.2 Universal Competitive Exclusion with Hierarchy-Dependent Recovery

Activation patching confirms that competitive exclusion---the suppression of parent feature activation by the child feature during SAE encoding---operates across all three tested hierarchy types (Section 5.1--5.2). The effect sizes range from $d = 0.75$ (city-language) to $d = 1.50$ (city-continent), all with $p < 10^{-3}$. Control features matched by activation magnitude produce recovery rates below 7%. The mechanism is universal, not specific to the first-letter domain.

Recovery magnitude varies across hierarchies in a pattern consistent with hierarchy structure. City-continent, with $K = 6$ coarse-grained parent classes, shows the highest recovery (61.9%): removing a single child feature releases a large fraction of the parent information encoded in the decoder. First-letter ($K = 25$) and city-language ($K \approx 20$) show lower recovery (32.5% and 34.2%), consistent with parent information being distributed across more latents in finer-grained hierarchies. The lower effect size for city-language ($d = 0.75$) may additionally reflect its many-to-many mapping structure, where the parent concept (language) is not cleanly encoded as a single residual stream direction.

Decoder information entanglement is consistent with this account: child decoders carry 3.98--6.16 nats of parent-direction information across hierarchies, versus 0.01--0.12 nats for random-direction controls (Section 5.3). The consistency across hierarchy types reinforces the universal mechanism conclusion, with the circularity caveat that this diagnostic shares the probe direction with the false-negative classification and therefore measures decoder geometry rather than computational redundancy.

The retraction of the concentrated-versus-distributed dichotomy from iteration 9---based on a buggy pilot that produced $d = -0.91$ for city-continent, subsequently corrected to $d = +1.50$ and verified by spot-check (Section 4.7)---illustrates a practical lesson. Activation patching results are sensitive to implementation details (feature selection, mode of intervention, data provenance), and independent verification is essential before interpreting effect sizes or signs.

## 7.3 Probe Quality as a Major Confound

The probe degradation ablation (Section 4.6) demonstrates that probe $F_1$ and measured absorption rate are monotonically related ($\rho = -1.0$, $p = 0.009$; linear $R^2 = 0.777$). A 0.30-point drop in probe $F_1$ inflates measured absorption by approximately 14.5 pp. Two of three RAVEL hierarchies---city-continent ($\Delta = +0.6$ pp from curve) and city-country ($\Delta = +8.5$ pp)---have their elevated absorption rates largely or entirely explained by their lower probe quality relative to first-letter ($F_1 = 1.0$).

City-language is the exception. At $F_1 = 0.82$, the degradation curve predicts 32.9% absorption; the observed rate is 11.6%, a residual of $\Delta = -21.3$ pp. Probe quality cannot explain why city-language has the lowest absorption rate of any hierarchy tested. This hierarchy-specific suppression effect is the strongest evidence that genuine hierarchy structure modulates absorption beyond measurement artifacts.

The many-to-many mapping structure of city-language---where a single city associates with multiple languages (e.g., Brussels with French, Dutch, and German) and a single language spans cities across continents---is qualitatively distinct from the one-to-one structures of first-letter, city-continent, and city-country. This structural difference is a candidate explanation for the absorption suppression: if the parent concept (language) is not encoded as a single clean direction in the residual stream, competitive exclusion pressure may be weaker. The mechanism connecting many-to-many structure to absorption suppression is untested and constitutes future work.

The recommendation for the field is concrete: any cross-domain absorption study must include probe degradation controls. Without them, apparent cross-domain differences may reflect probe quality variation rather than genuine hierarchy effects. First-letter, with its $F_1 = 1.0$ probes, is the only hierarchy in our study where the measured absorption rate is free of this confound.

Two caveats temper this result. The degradation curve is estimated from first-letter probes (binary classification, 26 balanced classes) and extrapolated to RAVEL probes (multi-class, imbalanced). This cross-domain extrapolation is not validated experimentally. The 7-point curve with 2 free parameters has limited degrees of freedom; the perfect monotonicity ($\rho = -1.0$) may partly reflect small sample size. We present the linear fit as the primary result and the quadratic fit ($R^2 = 0.942$) as exploratory (Section 4.6).

## 7.4 Implications for SAE Reliability

Absorption rates of 11--45% at layer 24 (the final Gemma Scope layer, 2 layers before the model's output) are directly relevant to safety applications. Google DeepMind deprioritized SAE research after finding 10--40% degradation in safety-relevant feature detection using SAE-reconstructed activations (Smith et al., 2025). Our cross-domain results provide a mechanistic account: competitive exclusion at the prediction layer causes parent features to systematically fail when more specific child features fire.

Three aspects of the findings compound this concern. First, the layer concentration effect---absorption at L24 is 15--26$\times$ higher than at earlier layers (Section 4.2)---means the failure disproportionately affects layers where task-specific computation occurs and safety-relevant features are most needed. Second, per-class variance is extreme: Europe at 90.2% versus Africa at 3.9% for city-continent (Section 4.4). Absorption creates blind spots for specific entity subsets, not uniform degradation. A safety monitor relying on SAE features would systematically miss European-city-related signals while detecting African-city-related ones. Third, the quadruple negative for correlational predictors (Section 7.1) means there is currently no unsupervised method to detect which features are absorbed; detection requires supervised probes with known ground truth, limiting scalability.

The width effect offers partial mitigation: 65k-feature SAEs reduce city-country absorption from 45.1% to 32.9% ($-12.2$ pp; Section 4.3). Matryoshka SAEs achieve absorption rates of approximately 0.03 on the first-letter benchmark (Bussmann et al., 2025). Whether these architectural improvements generalize beyond first-letter to entity-attribute hierarchies remains untested, and our architecture comparison (Section 6) is underpowered to resolve this.

## 7.5 Layer Dependence

Absorption concentrating at layer 24---with 15--26$\times$ higher rates than at layers 6, 12, and 18 across all hierarchies (Section 4.2)---suggests absorption arises from the model's task-specific computation at the final prediction layer, not from generic feature representation at intermediate layers. Gemma 2 2B has $L = 26$ layers; layer 24 is the last for which Gemma Scope provides SAEs.

The non-monotonic profile at intermediate layers (first-letter dips from 4.7% at L12 to 2.0% at L18 before jumping to 27.1% at L24) is consistent with intermediate layers performing computations that are structurally different from the token-prediction computation at L24. Probe quality is not the driver of the layer effect: first-letter probes achieve $F_1 \geq 0.96$ at all four tested layers.

The layer concentration has a practical implication. If absorption is specific to late layers, interventions that target only the final SAE (e.g., patching parent features at L24 based on child feature detection) could recover parent information without modifying the full SAE stack. Whether such targeted interventions are effective is untested.

## 7.6 Hypothesis Verdicts

Table 4 summarizes all hypothesis outcomes.

| Hypothesis | Verdict | Key metric | Confidence | Section |
|-----------|---------|-----------|-----------|---------|
| H1: Cross-domain variation | Supported with nuance | KW $p = 7.4 \times 10^{-66}$ (within-RAVEL) | High | 4.1 |
| H2': Semantic > first-letter at L24 | Refuted | First-letter 27.1% is between city-language 11.6% and city-country 45.1% | High | 4.5 |
| H3: Hedging decomposition varies | Partially supported | Strict 0--22.6% vs. loose 92.6% | Medium | 5.4 |
| H4: GAS unsupervised detector | Definitive negative | $\rho = 0.116$, AUROC $= 0.571$ | High | App. B |
| H5: Absorption Tax $T(G)$ | Not supported | Ranking $\rho = -0.20$ | High | App. D |
| H6: Architecture generalization | Not detected (underpowered) | ANOVA $p = 0.50$--$0.53$; 12 observations | Low | 6 |
| H7: Causal absorption (first-letter) | Supported | $d = 1.33$, $p = 0.000218$ | High | 5.1 |
| H7-cross: Causal absorption (cross-domain) | Supported | $d = 0.75$--$1.50$, all $p < 10^{-17}$ | High | 5.2 |
| H8: Decoder entanglement (cross-hierarchy) | Consistent | 6.16 nats (first-letter) vs. 3.98 nats (city-continent) | High | 5.3 |
| H9: Rate-distortion predictor | Not supported | $\rho = 0.286$, $R^2 = 0.104$; all predictors wrong sign | High | App. F |
| H10: Probe quality confound | Mixed | $R^2 = 0.777$; city-language outlier $\Delta = -21.3$ pp | High | 4.6 |

**Table 4.** Hypothesis verdicts. Of 11 hypotheses, 4 are supported (H1, H7, H7-cross, H8), 1 is partially supported (H3), 1 is mixed (H10), 1 is refuted (H2'), 3 are negative results (H4, H5, H9), and 1 is underpowered (H6). The quadruple failure of correlational predictors (H4, H5, H9, plus the rate-distortion model) is itself a finding: absorption requires causal methods.

## 7.7 Limitations

**Single model.** All experiments use Gemma 2 2B. Generalization to other architectures (Llama, Pythia, GPT-2), model scales (2B to 70B), and training paradigms (instruction-tuned, RLHF) is untested. The layer-dependent absorption profile and per-class patterns may differ for models with different depth, width, or training data.

**Probe quality confound is partially resolved.** The probe degradation ablation ($R^2 = 0.777$) demonstrates that probe quality is a major confound, but the curve is estimated from first-letter probes and extrapolated to multi-class RAVEL probes without experimental validation of this cross-domain transfer. The 7-point curve with 2 free parameters has limited degrees of freedom.

**Token position asymmetry.** First-letter probes evaluate at token position $-6$; RAVEL probes at position $-2$. Within-RAVEL comparisons are unaffected (all share position $-2$), but first-letter versus RAVEL comparisons carry this uncontrolled confound.

**Decoder entanglement circularity.** The decoder information entanglement diagnostic (Section 5.3) shares the probe direction with the false-negative classification. It establishes that child decoders carry large-magnitude parent information but cannot determine whether this information is computationally utilized by the model. A genuine test requires activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits.

**Architecture comparison is underpowered.** Twelve observations across four architectures, with a width mismatch confound (Matryoshka 32k versus others at 16k or 65k). The null result ($p = 0.50$--$0.53$) should be interpreted as "effect not detected," not "no effect exists."

**City-country probe quality.** City-country ($F_1 = 0.73$) falls below both quality gates. Its 45.1% absorption rate is partially explained by the probe degradation curve (residual $+8.5$ pp). Results for this hierarchy are exploratory.

**Scope of hierarchy types.** Four hierarchy types (one syntactic, three entity-attribute) cover a narrow slice of the hierarchies relevant to mechanistic interpretability. Part-of-speech, taxonomic (animal $\to$ mammal $\to$ dog), syntactic constituency, and factual knowledge (person $\to$ profession) hierarchies remain unexplored. The city-language anomaly demonstrates that hierarchy structure matters for absorption; extending to structurally diverse hierarchies is essential for determining which structural properties drive the variation.

<!-- FIGURES
- Table 4: inline --- Hypothesis verdict summary across all 11 tested hypotheses
- None
-->
