# 7. Discussion

Table 4 summarizes the verdicts for all nine hypotheses tested in this study: two are strongly supported, two are falsified, and five are not supported or only partially supported. This pattern of results -- where the majority of pre-registered hypotheses fail while a smaller set of robust findings emerges -- carries significant implications for the field's approach to understanding and mitigating feature absorption in SAEs.

| Hyp. | Name | Verdict | Key Metric | Conf. | Sec. |
|------|------|---------|------------|-------|------|
| H1 | Cross-Domain Variation | **SUPPORTED** | Kruskal-Wallis p=7.4e-66 | HIGH | 4 |
| H2' | Semantic > First-Letter | **REFUTED** | No simple ordering; 4x range | HIGH | 4 |
| H3 | Hedging Decomposition | **SUPPORTED** | $\chi^2$=91.5, p=1.0e-19 | HIGH | 5 |
| H4 | GAS Detector | **NEGATIVE** | $\rho$=0.116, AUROC=0.571 | HIGH | App. B |
| H5 | Absorption Tax $T(G)$ | NOT SUPPORTED | $\rho$=-0.20, 50% concordance | HIGH | App. D |
| H6 | Architecture Generalization | PARTIAL | Arch p=0.50; Hier p=0.005 | MED | 6 |
| H7 | Causal Absorption | **SUPPORTED (FL)** | $d$=1.33, p=0.000218 | HIGH/LOW | 5 |
| H8 | Benign Absorption | **FALSIFIED** | 0% benign; $|\Delta_{\text{logit}}|$=3.98 | HIGH | 5 |
| H9 | Rate-Distortion Predictors | NOT SUPPORTED | $\rho$=0.250, $R^2$=0.088 | HIGH | App. E |

**Table 4.** Hypothesis verdict summary. FL = first-letter. Confidence reflects statistical power and consistency across iterations. Two hypotheses (H1, H3) are strongly supported; two (H2', H8) are refuted/falsified with the refutations themselves constituting positive findings; five fail or receive only partial support.

## 7.1 Correlational Methods Fail; Causal Methods Partially Succeed

Five correlational or statistical approaches attempted to predict or detect absorption without interventional experiments. All five failed:

- **GAS** (H4): $\rho$=0.116, AUROC=0.571, bootstrap 95% CI [--0.333, 0.536]. A 25x scale-up from pilot (200 to 5,000 sequences) did not improve the correlation. GAS measures decoder geometry, but absorption is driven by encoder competitive exclusion dynamics -- a decoder-encoder gap that purely geometric approaches cannot bridge.
- **CMI** (Appendix C): $\rho$=0.044, $p$=0.83 at $L_0$=22. At this sparsity, all 25 letter probes achieve $F_1$=1.0, eliminating the probe quality confound entirely. The information-theoretic measure fails to capture absorption.
- **Absorption Tax** $T(G)$ (H5): Ranking $\rho$=-0.20, pairwise concordance at 50% (chance). The minimum additional $L_0$ cost for absorption-free representation does not predict which hierarchies suffer more absorption.
- **Rate-distortion predictors** (H9): Model $\rho$=0.250, $R^2$=0.088 with $n$=262 feature pairs. Individual predictors correlate in the *opposite* direction to the hypothesis: $\cos(\mathbf{d}_p, \mathbf{d}_c)$ $\rho$=-0.108, $P(c \mid p)$ $\rho$=-0.173 ($p$=0.005), $R(p)$ $\rho$=-0.203 ($p$=0.0009). The sign reversal between pilot ($n$=20) and full run ($n$=262) exposes the instability of small-sample correlational analyses.
- **Competition coefficients** (Appendix D): Non-significant for first-letter ($\rho$=0.182) and negative for city-continent ($\rho$=-0.486). The ecological competition analogy does not yield quantitative predictions.

The only approach that produced a positive result is activation patching (H7): zeroing child features recovers parent probe predictions for first-letter at 32.5% vs. 1.5% control ($p$=0.000218, Cohen's $d$=1.33). This is the first interventional -- not merely correlational -- evidence for competitive exclusion in SAEs. The result holds across 19 of 25 words and across 200 contexts per word.

The consistent failure of correlational approaches suggests that absorption is not a simple function of readily computable feature statistics. Decoder similarity, co-occurrence, reconstruction importance, information-theoretic dependence, and geometric mismatch scores all fail to capture the dynamics that produce absorption during encoding. This motivates a methodological shift: the field should prioritize causal, interventional methods (activation patching, circuit analysis) over statistical predictors when studying SAE failure modes.

## 7.2 Concentrated vs. Distributed Absorption Mechanisms

Activation patching reveals a sharp mechanistic divide between hierarchy types. For first-letter spelling, zeroing a single child feature (e.g., the feature for "Saturday") recovers the parent probe's prediction (e.g., "starts with S") in 32.5% of false negative cases (Cohen's $d$=1.33). The mechanism is concentrated: one child feature suppresses one parent feature through competitive exclusion in the SAE encoder.

For city-continent, the same intervention fails entirely. Zeroing the identified child feature recovers the parent probe in 0.05% of cases -- compared to 14.5% for the random control ($d$=-0.91, $n$=93 entities, 3,751 FN instances). The effect is reversed: random feature zeroing is more likely to help than targeted child zeroing.

This contrast points to two distinct absorption regimes:

1. **Concentrated absorption** (first-letter): A single child feature captures and suppresses the parent. The SAE encoder routes the input to the child feature at the expense of the parent. Zeroing the child releases the parent.
2. **Distributed absorption** (semantic hierarchies): Parent information is spread across multiple features. No single child feature is responsible for the parent's failure. Zeroing any one feature has negligible effect because the absorption is a collective property of the SAE's representation of the hierarchy.

The first-letter task's clean structure -- 26 letters, near-uniform distribution, 100% parent-child co-occurrence by construction -- produces concentrated absorption that is amenable to single-feature interventions. Semantic hierarchies (80 countries, 6 continents, 23 languages) have richer structure, and the model distributes continent/country/language information across many features rather than concentrating it in a single child.

This mechanistic divide has practical implications. Absorption mitigation strategies that target individual features (e.g., post-hoc adjustment of child feature activations) may work for concentrated absorption but will fail for distributed absorption. Addressing distributed absorption likely requires modifications to the SAE training objective or architecture that encourage the encoder to maintain parent feature activations even when child features are present.

## 7.3 Absorption Is Always Pathological

The contrarian hypothesis (H8) proposed that a substantial fraction of absorption might be benign -- reflecting computational redundancy rather than information loss. If the model does not independently use the parent feature when the child is active, absorption would be a faithful representation of the model's computation, not a failure.

This hypothesis is decisively falsified. Across 1,471 false negative instances from 50 city-continent entities, 0% are benign at all three thresholds tested ($\tau$=0.05, 0.1, 0.2). Ablating the parent direction $\hat{\mathbf{w}}_p$ from the child feature's decoder vector $\mathbf{d}_c$ produces a mean $|\Delta_{\text{logit}}|$=3.98 nats. The control (ablating a random direction) produces $|\Delta_{\text{logit}}|$=0.004 nats. The effect ratio is approximately 1,000x ($t$=-365.3, $p$<$10^{-100}$). Even the minimum observed $|\Delta_{\text{logit}}|$ across all 1,471 instances is 2.34 nats -- more than 10x above the strictest threshold.

The model relies on the parent direction carried by the child feature's decoder for downstream predictions. When absorption occurs, the SAE reconstruction loses this information, and the model's output degrades. There is no regime in which absorption is harmless.

This result converts what began as a philosophical objection into a quantitative finding with immediate practical consequences. SAE-based mechanistic interpretability cannot tolerate 11--45% absorption rates at the prediction layer (L24) if every absorbed instance distorts model output. For safety-critical applications -- where feature reliability determines whether harmful behaviors are detected -- absorption rates at these levels represent a serious threat to SAE deployment.

## 7.4 Probe Quality as a Fundamental Limitation

All cross-domain results depend on probe quality, and only the first-letter hierarchy passes the strict quality gate ($F_1$=0.97 at L24). RAVEL probes achieve $F_1$=0.87 for city-continent (relaxed pass), $F_1$=0.82 for city-language (relaxed pass), and $F_1$=0.73 for city-country (below gate). Measured absorption rates for RAVEL hierarchies are therefore upper bounds: some fraction of apparent false negatives may reflect probe errors rather than genuine absorption.

Three considerations mitigate this limitation. First, the cross-domain variation finding (Kruskal-Wallis $p$=7.4$\times 10^{-66}$, $N$=3,566) is robust to probe quality uncertainty -- the 4x range in absorption rates (11.6% to 45.1%) far exceeds the magnitude of probe error. Second, the first-letter task ($F_1$=1.0 for binary probes at all layers) serves as a gold-standard positive control where absorption measurements are uncontaminated by probe imperfection. Third, the pathological absorption finding ($|\Delta_{\text{logit}}|$=3.98, tested on city-continent at $F_1$=0.87) would remain qualitatively unchanged even if 13% of instances were misclassified false negatives, since the minimum observed logit change (2.34 nats) exceeds all thresholds by an order of magnitude.

Better probes would sharpen the cross-domain estimates. Contrastive learning approaches, richer prompt templates, and larger entity sets could improve RAVEL probe quality. The city-country hierarchy ($F_1$=0.73) would benefit most: its measured absorption rate of 45.1% at L24 should be treated with particular caution.

## 7.5 Layer Dependence and Computational Implications

Absorption concentrates at the final prediction layer. For first-letter spelling on the 16k SAE, absorption rises from 2.4% at L6 to 42.9% at L24 -- an 18x increase. At L18, the rate drops to 2.2%, making the L24 spike discontinuous rather than gradual. This pattern suggests absorption is not a generic property of SAE reconstruction at any layer, but is tied to the model's task-specific computation at the prediction layer.

At L24, the model's residual stream carries features that directly influence the next-token logits. Parent and child features compete for representation in this final bottleneck. At earlier layers, the model has not yet committed to task-specific feature assignments, and parent-child competition is less intense. The 15--18x ratio between L24 and earlier layers implies that absorption is primarily a phenomenon of the model's output computation, not of its internal representation formation.

This layer dependence has methodological implications. Studies that measure absorption only at intermediate layers (e.g., L12) will systematically underestimate the severity of the problem. The field should standardize absorption measurement at the model's prediction-relevant layers.

## 7.6 Implications for SAE-Based Mechanistic Interpretability

Three findings from this study bear directly on the viability of SAEs for mechanistic interpretability:

**First**, absorption is not an architecture-specific artifact. The architecture comparison (Section 6) shows no significant effect of SAE type on absorption rates (Kruskal-Wallis $p$=0.50 at L24 across JumpReLU, Matryoshka; $p$=0.75 at L12 across JumpReLU, BatchTopK, Matryoshka). The hierarchy type -- what the SAE must represent -- determines absorption, not how the SAE enforces sparsity. Switching from JumpReLU to BatchTopK or Matryoshka will not fix absorption.

**Second**, the 11--45% absorption rates at L24 represent a substantial fraction of parent feature predictions lost to SAE encoding. For safety applications that rely on feature activation to detect specific behaviors (e.g., deception, harmful content), a 45% false negative rate for parent features is operationally unacceptable. The always-pathological nature of absorption (mean $|\Delta_{\text{logit}}|$=3.98 nats, Section 5.3) means these failures have real downstream consequences.

**Third**, current absorption detection methods require supervised labels. All successful absorption measurements in this paper use probes trained on known hierarchies. No unsupervised approach (GAS, CMI, competition coefficients) achieved meaningful predictive power. Scaling absorption characterization to new domains requires either developing better unsupervised detectors -- which must account for encoder dynamics, not just decoder geometry -- or accepting the cost of supervised probe development for each hierarchy of interest.

These findings do not invalidate SAEs as interpretability tools. Anthropic's circuit tracing in Claude 3.5 Haiku demonstrates that SAE features, when reliable, enable powerful mechanistic understanding. But absorption represents a systematic failure mode that the field must address before SAE-based analyses can be trusted at scale -- particularly for safety-critical applications where false negatives carry asymmetric costs.

<!-- FIGURES
- Table 4: gen_table4_hypothesis_verdicts.py, table4_hypothesis_verdicts.pdf — Hypothesis verdict summary with color-coded verdicts for all 9 hypotheses
-->
