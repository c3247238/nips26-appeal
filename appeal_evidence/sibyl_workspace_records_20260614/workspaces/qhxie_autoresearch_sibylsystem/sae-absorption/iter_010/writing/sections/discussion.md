# Discussion

The preceding sections established three empirical results: absorption rates vary across hierarchy types (Section 4), activation patching confirms a universal competitive exclusion mechanism (Section 5), and architecture choice has no detected effect relative to hierarchy type (Section 6). This section synthesizes these findings into five implications for the field, explicitly addresses failed approaches, and catalogues limitations.

## 7.1 Causal Methods Succeed Where Correlational Methods Fail

Four independent correlational and statistical approaches to predicting absorption all failed:

- **Geometric Absorption Score (GAS):** $\rho = 0.116$, AUROC $= 0.571$, bootstrap 95% CI $[-0.333, 0.536]$ (Appendix B). GAS measures decoder-activation geometry---the angular mismatch between a feature's decoder direction and its typical activation pattern. The 25-word full-scale evaluation confirms the pilot null: decoder geometry does not predict which features will be absorbed.

- **Conditional mutual information (CMI):** $\rho = 0.044$, $p = 0.84$ at $L_0 = 22$ (Appendix C). At this sparsity level, all 25 first-letter probes achieve $F_1 = 1.0$, eliminating probe quality as a confound. The bootstrap CI $[-0.41, 0.47]$ firmly includes zero. Information-theoretic dependence between parent and child activations does not predict absorption.

- **Absorption Tax $T(G)$:** Ranking $\rho = -0.20$, concordance 50% (chance level; Appendix D). The theoretical construct---the minimum additional $L_0$ cost for absorption-free representation of a hierarchy---produces quantitative predictions that are no better than random ordering. The qualitative insight (absorption imposes a sparsity cost) is retained; the quantitative framework is not supported.

- **Rate-distortion three-factor model:** $\rho = 0.286$, $R^2 = 0.104$ across 131 parent-child pairs from all hierarchies (Appendix F). All three individual predictors---decoder cosine similarity ($\rho = -0.090$), co-occurrence ($\rho = -0.189$), and reconstruction importance ($\rho = -0.239$)---have the wrong sign relative to the hypothesis. Statistical significance ($p < 0.001$) reflects large sample size, not meaningful effect. The direction reversal from the 20-pair pilot ($\rho > 0$) to the 131-pair full evaluation demonstrates small-sample instability.

This quadruple negative establishes a methodological boundary. Absorption is driven by encoder competitive dynamics during inference, not by static geometric or information-theoretic properties of the trained SAE. Decoder cosine similarity, co-occurrence frequency, and reconstruction importance---the most natural candidate predictors---all fail. Only activation patching, which intervenes on the causal pathway (zeroing the child feature and measuring parent recovery), successfully characterizes absorption. The implication for the field is direct: future absorption research should prioritize interventional methods over correlational proxies.

## 7.2 Universal Competitive Exclusion with Hierarchy-Dependent Recovery

Activation patching confirms that the same mechanism---competitive exclusion of the parent feature by the child feature during SAE encoding---operates across all three tested hierarchy types:

- First-letter: recovery 32.5% vs. control 1.5%, Cohen's $d = 1.33$, $p = 0.000218$
- City-continent: recovery 61.9% vs. control 5.2%, $d = 1.50$, $p < 10^{-20}$
- City-language: recovery 34.2% vs. control 6.8%, $d = 0.75$, $p < 10^{-18}$

All three effects are large and unambiguous: zeroing the child feature partially restores parent probe accuracy, while zeroing a control feature of matched activation magnitude does not. The mechanism is universal, not first-letter-specific.

Recovery magnitude varies by hierarchy (61.9% for city-continent versus 32.5% for first-letter), but this likely reflects hierarchy-dependent information distribution rather than distinct mechanisms. City-continent has $K = 6$ coarse-grained parent classes, so removing a single child feature can release a large fraction of the parent information encoded in the decoder. First-letter has $K = 25$ classes with finer-grained parent features, plausibly distributing parent information across more latents. The lower recovery for city-language ($d = 0.75$) may additionally reflect its many-to-many mapping structure, where multiple cities share a language and multiple languages share a city.

Decoder information entanglement is consistent with this account. Child decoder vectors carry $|\Delta_{\text{logit}}| = 6.16$ nats (first-letter, $N = 158$) and 3.98 nats (city-continent, $N = 1{,}464$) of parent-direction information, versus 0.012 nats for random-direction controls. The 1.55$\times$ magnitude ratio between hierarchies is modest; both show 100% of instances exceeding all classification thresholds (0.05, 0.1, 0.2 nats). This consistency across hierarchy types reinforces the universal mechanism conclusion. We note that this diagnostic shares the probe direction with the false-negative classification, so it measures decoder geometry rather than computational redundancy. A genuine test of whether the absorbed information is computationally recoverable would require activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits.

## 7.3 Probe Quality as a Major Confound

The probe degradation ablation (Section 4.6) is the paper's most important methodological contribution. Degrading first-letter probe $F_1$ from 1.0 to 0.69 via weight noise injection produces a well-fitted absorption curve: $R^2 = 0.777$ (linear), $R^2 = 0.942$ (quadratic), Spearman $\rho = -1.0$ (perfect monotonic), $p = 0.009$. A 0.30-point drop in probe $F_1$ inflates measured absorption by approximately 14.5 percentage points.

Overlaying RAVEL absorption rates on this curve decomposes the cross-domain variation into two sources:

**Probe quality effect.** City-continent absorption (31.4% at $F_1 = 0.87$) falls within 0.6 percentage points of the degradation curve prediction (30.8%). City-country absorption (45.1% at $F_1 = 0.73$) is within 8.5 pp of the prediction (36.6%). For these hierarchies, the elevated absorption rates are largely---or entirely---explained by their lower probe quality relative to first-letter.

**Genuine hierarchy-specific effect.** City-language absorption (11.6% at $F_1 = 0.82$) sits 21.3 pp below the curve prediction (32.9%). Probe quality alone cannot explain why city-language has the lowest absorption of any hierarchy tested. This suppression effect is the strongest evidence that genuine hierarchy-specific factors modulate absorption beyond probe measurement artifacts.

The recommendation for the field is concrete: any cross-domain absorption study must include probe degradation controls. Without them, apparent cross-domain differences may reflect probe quality variation rather than genuine hierarchy effects. The first-letter task, with its $F_1 = 1.0$ probes, is the only hierarchy in our study where the measured absorption rate is free of this confound.

The city-language anomaly warrants investigation. The many-to-many mapping structure (multiple cities share a language; a city may have multiple official languages) is qualitatively different from the many-to-one structures of first-letter, city-continent, and city-country. This structural difference may reduce the competitive pressure between parent and child features---if the parent concept (language) is not cleanly encoded as a single direction in the residual stream, competitive exclusion may be weaker. This hypothesis is untested and constitutes future work.

## 7.4 Implications for SAE Reliability

Absorption rates of 11--45% at the final prediction layer (layer 24 of 26 in Gemma 2 2B) are directly relevant to SAE deployment for safety applications. Google DeepMind deprioritized SAE research after finding 10--40% degradation in safety-relevant feature detection using SAE-reconstructed activations (Smith et al., 2025). Our cross-domain results provide a mechanistic account of this degradation: competitive exclusion at the prediction layer causes parent features (e.g., "harmful intent," "deceptive reasoning") to systematically fail when more specific child features fire.

Three aspects of our findings amplify this concern. First, the layer concentration effect (absorption at L24 is 15$\times$ higher than at earlier layers) means absorption disproportionately affects the layers where task-specific computation occurs---precisely where safety-relevant features are most needed. Second, the per-class variance is extreme (Europe 90.2% vs. Africa 3.9% for city-continent), implying that absorption creates blind spots for specific entity subsets rather than uniform degradation. Third, the quadruple negative for correlational predictors means there is currently no unsupervised method to detect which features are absorbed; detection requires supervised probes with known ground truth.

The width effect offers partial mitigation. Wider SAEs (65k vs. 16k) reduce absorption for city-country from 45.1% to 32.9% (a 12.2 pp reduction). Scaling SAE dictionary size provides more features to represent parent concepts, reducing the sparsity pressure that drives absorption. The Matryoshka SAE architecture (Bussmann et al., 2025), which achieves absorption rates of approximately 0.03 on the first-letter benchmark versus 0.29 for BatchTopK, represents a more targeted architectural intervention. Whether these architectural improvements generalize beyond first-letter to entity-attribute hierarchies remains untested.

## 7.5 Hypothesis Verdicts

Table 4 summarizes the outcome of each hypothesis tested in this study.

| Hypothesis | Verdict | Key metric | Confidence | Section |
|-----------|---------|-----------|-----------|---------|
| H1: Cross-domain variation | Supported with nuance | KW $p = 7.4 \times 10^{-66}$ (within-RAVEL) | High | 4.1 |
| H2': Semantic > first-letter at L24 | Refuted | First-letter 27.1% < city-country 45.1% but > city-language 11.6% | High | 4.5 |
| H3: Hedging decomposition varies | Partially supported | Strict 0--22.6% vs. loose 92.6% | Medium | 5.4 |
| H4: GAS unsupervised detector | Definitive negative | $\rho = 0.116$, AUROC $= 0.571$ | High | App. B |
| H5: Absorption Tax $T(G)$ | Not supported | Ranking $\rho = -0.20$ | High | App. D |
| H6: Architecture generalization | Not detected (underpowered) | ANOVA $p = 0.50$--$0.53$; 12 obs. | Low | 6 |
| H7: Causal absorption (first-letter) | Supported | $d = 1.33$, $p = 0.000218$ | High | 5.1 |
| H7-cross: Causal absorption (cross-domain) | Supported | $d = 0.75$--$1.50$, all $p < 10^{-17}$ | High | 5.2 |
| H8: Decoder entanglement (cross-hierarchy) | Consistent | 6.16 nats (first-letter) vs. 3.98 nats (city-continent) | High | 5.3 |
| H9: Rate-distortion predictor | Not supported | $\rho = 0.286$, $R^2 = 0.104$; all predictors wrong sign | High | App. F |
| H10: Probe artifact vs. hierarchy effect | Mixed | $R^2 = 0.777$; city-language outlier $\Delta = -21.3$ pp | High | 4.6 |

**Table 4.** Hypothesis verdicts. Multiple negative results (H4, H5, H9) are reported alongside positive findings (H7, H7-cross). The mixed verdict on H10---probe quality explains most but not all variation---is the study's most nuanced finding.

Of the 11 hypotheses, 4 are supported (H1, H7, H7-cross, H8), 1 is partially supported (H3), 1 is mixed (H10), 1 is refuted (H2'), 3 are negative (H4, H5, H9), and 1 is underpowered (H6). The honest reporting of multiple negative results is deliberate: the quadruple failure of correlational predictors is itself a finding that shapes the field's methodological choices.

## 7.6 Limitations

**Single model.** All experiments use Gemma 2 2B. Generalization to other architectures (Llama, Pythia, GPT-2), model scales (2B to 70B), and training paradigms (instruction-tuned, RLHF) is untested. The layer-dependent absorption profile and per-class patterns may differ for models with different depth, width, or training data composition.

**Probe quality confound is partially but not fully resolved.** The probe degradation ablation ($R^2 = 0.777$) shows probe quality is a major confound, but the control condition ($F_1 = 1.0$ absorption rate of 21.6%) falls below the iter\_009 baseline CI $[26.3\%, 34.7\%]$ due to per-token versus per-word aggregation differences. The trend is consistent (slope $= -0.398$, $p = 0.009$), but absolute calibration between aggregation methods introduces uncertainty.

**Token position asymmetry.** First-letter experiments use token position $-6$ while RAVEL hierarchies use position $-2$. Within-RAVEL comparisons are unaffected (all share position $-2$), but first-letter versus RAVEL comparisons carry this uncontrolled confound. Position $-6$ and $-2$ may differ in the degree to which hierarchical information is encoded in the residual stream.

**Decoder entanglement circularity.** The decoder information entanglement diagnostic (Section 5.3) shares the probe direction with the false-negative classification. It establishes that child decoders carry large-magnitude parent information, but does not answer whether this information is computationally utilized by the model. A genuine computational-redundancy test would require activation-level ablation or path patching through independent circuits.

**Architecture comparison is underpowered.** The ANOVA across four architectures (JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, Matryoshka 32k) has 12 total observations. The width mismatch (Matryoshka 32k versus others 16k or 65k) is an uncontrolled confound. The null result ($p = 0.50$--$0.53$) should be interpreted as "effect not detected," not "no effect exists."

**City-country probe quality.** The city-country probe ($F_1 = 0.73$) falls below both quality gates. Its 45.1% absorption rate is partially but not fully explained by the probe degradation curve (residual $+8.5$ pp). Results for this hierarchy should be treated as exploratory.

**Scope of hierarchy types.** Four hierarchy types (one syntactic, three entity-attribute) were tested. Other important hierarchy types---part-of-speech, taxonomic (animal $\to$ mammal $\to$ dog), syntactic constituency, factual knowledge (person $\to$ profession $\to$ field)---remain unexplored. The city-language anomaly (Section 7.3) demonstrates that hierarchy structure matters, so extending to structurally diverse hierarchies is essential.

<!-- FIGURES
- Table 4: inline — Hypothesis verdict summary across all 11 tested hypotheses
- None (no code-generated figures in this section)
-->
