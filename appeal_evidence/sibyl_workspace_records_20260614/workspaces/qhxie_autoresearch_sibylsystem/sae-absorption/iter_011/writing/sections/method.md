# 3 Methodology

This section describes the measurement pipeline for cross-domain absorption characterization. We define four feature hierarchies with varying structural properties (Section 3.1), specify probe training and quality gating (Section 3.2), formalize the absorption measurement procedure (Section 3.3), describe the activation patching protocol for causal validation (Section 3.4), introduce a decoder information entanglement diagnostic with an explicit circularity caveat (Section 3.5), present the hedging decomposition that disentangles absorption subtypes (Section 3.6), and detail the probe degradation ablation that decomposes probe quality confounds from genuine hierarchy effects (Section 3.7).

All experiments use Gemma 2 2B (`google/gemma-2-2b`) with Gemma Scope JumpReLU SAEs (Lieberum et al., 2024) at layers 6, 12, 18, and 24, with dictionary widths of $m = 16{,}384$ and $m = 65{,}536$. No SAE training is performed; all SAEs are pre-trained checkpoints from Gemma Scope and SAEBench. Table 1 summarizes probe quality across all hierarchy-layer combinations.

## 3.1 Feature Hierarchies

We study four feature hierarchies that span syntactic and semantic domains, varying in class count, class balance, probe quality, and co-occurrence structure.

**First-letter spelling.** Each word has exactly one first letter, producing 26 parent classes with near-uniform frequency. Test set: 500 words $\times$ 3 prompt contexts = 1,500 tokens (drawn from the `sae-spelling` vocabulary; Chanin et al., 2024). Probes achieve $F_1 = 1.0$ at layer 24, making this hierarchy a positive control with no probe quality confound. Parent-child co-occurrence is 100% by construction: every token of "Saturday" co-occurs with letter "S."

**City-continent.** Six continent classes (Africa, Asia, Europe, North America, Oceania, South America), 173 city entities from RAVEL (Huang et al., 2024), moderate class balance. Probe $F_1 = 0.87$ at layer 24. The hierarchy is shallow (city $\to$ continent) with deterministic one-to-one mapping.

**City-language.** Approximately 20 language classes, 201 city entities from RAVEL. Probe $F_1 = 0.82$ at layer 24. This hierarchy has a many-to-many structure: cities in multilingual countries (e.g., Brussels) may associate with multiple languages, and a single language (e.g., Spanish) spans cities across multiple continents. This structural difference from the one-to-one hierarchies is a candidate explanation for the anomalous absorption rate observed in Section 4.6.

**City-country.** Approximately 80 country classes, 1,405 city entities from RAVEL. Probe $F_1 = 0.73$ at layer 24---below the strict quality gate ($F_1 > 0.90$) and the relaxed gate ($F_1 \geq 0.80$). Class balance is highly skewed: USA has 176 entities while Chad has 5. We include city-country results with an explicit caveat that absorption rates are confounded by probe quality, and verify via the probe degradation ablation (Section 3.7) that this confound accounts for most of the observed rate.

| Hierarchy | $K$ (classes) | $N$ (entities) | Probe $F_1$ (L24) | Gate status | Co-occurrence | Balance |
|-----------|:---:|:---:|:---:|---|---|---|
| First-letter | 26 | 500 words | **1.00** | Strict pass | 1-to-1, 100% | Near-uniform |
| City-continent | 6 | 173 | 0.87 | Relaxed pass | 1-to-1 | Moderate |
| City-language | ~20 | 201 | 0.82 | Relaxed pass | Many-to-many | Moderate |
| City-country | ~80 | 1,405 | 0.73 | Below gate | 1-to-1 | Highly skewed |

**Table 1.** Feature hierarchy properties and probe quality at layer 24. Gate status indicates whether probe $F_1$ passes the strict ($> 0.90$) or relaxed ($\geq 0.80$) quality threshold for absorption measurement.

## 3.2 Probe Training and Quality Gates

Linear probes (logistic regression, scikit-learn `LogisticRegression` with regularization $C$) are trained on Gemma 2 2B residual stream activations at layers $\ell \in \{6, 12, 18, 24\}$. For first-letter probes, 3,517 words $\times$ 4 prompt contexts = 14,068 training tokens, with $C = 0.001$. For RAVEL probes, training follows the RAVEL protocol (Huang et al., 2024) with standard train-test splits. Test tokens are disjoint from training tokens.

**Quality gates.** We define two quality thresholds:
- *Strict gate*: $F_1 > 0.90$. Results passing this gate are reported without caveat.
- *Relaxed gate*: $F_1 \geq 0.80$. Results passing this gate are reported with an explicit note that probe quality may inflate absorption measurements.
- Results below the relaxed gate ($F_1 < 0.80$) are included with a documented caveat that probe quality is a major confound.

**Aggregation method.** All absorption rates use per-token aggregation: $\alpha = N_{\text{FN}} / N_{\text{probe-correct-raw}}$, where each token (word $\times$ prompt context) is an independent observation. This choice is the most defensible statistically---each token represents a separate forward pass through the model. Per-word aggregation (averaging absorption rates across unique words, giving equal weight to each word regardless of prompt count) yields systematically higher values because rare, high-absorption words receive disproportionate weight. We report per-word sensitivity analysis in Appendix A.

**Bootstrap confidence intervals.** We compute 95% CIs using percentile bootstrap with 10,000 resamples over token observations. The bootstrap resamples individual tokens, ensuring that CIs are consistent with the per-token point estimates. Earlier iterations used per-word bootstrap resampling that produced CIs inconsistent with per-token point estimates (lower bounds exceeding point estimates in 6 of 7 rows); the per-token bootstrap resolves this.

**Token position.** First-letter probes evaluate at token position $-6$ (the last token before the target word, following Chanin et al., 2024). RAVEL probes evaluate at position $-2$ (standard RAVEL protocol). This asymmetry is a limitation: different token positions access different contextual information, and position-dependent effects on absorption are uncontrolled. We note this as a caveat throughout.

## 3.3 Absorption Measurement Pipeline

Our measurement pipeline adapts the procedure of Chanin et al. (2024) to entity-attribute hierarchies.

**Step 1: False negative identification.** For each test token, apply the trained probe to (a) the raw residual stream activation $\mathbf{x}$ and (b) the SAE-reconstructed activation $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$. A false negative (FN) occurs when the probe classifies the raw activation correctly ($\hat{y}_{\text{raw}} = y$) but misclassifies the SAE reconstruction ($\hat{y}_{\text{SAE}} \neq y$). The absorption rate is $\alpha = N_{\text{FN}} / N_{\text{probe-correct-raw}}$.

**Step 2: Child feature identification.** For each FN token, compute integrated-gradients attribution (Sundararajan et al., 2017) over the SAE latent activations $\mathbf{z}$ with respect to the probe's output for the correct class. The feature with the most negative attribution (largest magnitude contribution pushing the probe prediction away from the correct class) is identified as the primary child feature.

**Step 3: Absorption candidate validation.** Compute cosine similarity between the child feature's decoder vector $\mathbf{d}_c$ and the parent probe direction $\mathbf{w}_p$: $\cos(\mathbf{d}_c, \mathbf{w}_p)$. A threshold $\tau_{\cos} = 0.30$ classifies candidates as absorption instances. This threshold is robust across the range 0.20--0.40: a sensitivity analysis over 5 thresholds $\times$ 2 SAE configurations yields CV = 0.077, and the late > early layer ordering holds in all 10 cells (Appendix E). The data-driven p95 of random cosine similarities is 0.044--0.049, confirming all tested thresholds are well above chance.

**Controls.** Two controls validate the measurement:
1. *Random direction baseline.* Replace the probe direction $\mathbf{w}_p$ with a random unit vector and measure false negatives. Establishes that absorption is specific to the probe direction.
2. *Shuffled hierarchy control.* Permute entity-attribute labels and re-measure absorption. Tests whether absorption is hierarchy-specific or an artifact of the SAE reconstruction.

## 3.4 Activation Patching Protocol

Activation patching provides interventional evidence for competitive exclusion: if zeroing the child feature recovers the parent probe's correct prediction, then the child feature was causally responsible for the parent's failure to fire.

For each FN token:
1. Identify the primary child feature (Section 3.3, Step 2).
2. Set $z_c = 0$ in the SAE latent vector, producing $\mathbf{z}^{(c \to 0)}$.
3. Reconstruct: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$.
4. Apply the parent probe to $\hat{\mathbf{x}}^{(c \to 0)}$. If the prediction reverts to the correct class, the token is *recovered*.

The recovery rate $R_c$ is the fraction of FN tokens recovered by zeroing the primary child feature. The control recovery rate $R_{\text{ctrl}}$ is computed by zeroing 10 random non-child features (matched by activation magnitude) per FN token and averaging across control features.

**Statistical tests.** Wilcoxon signed-rank test compares per-entity recovery rates (child-zeroed vs. control). Bootstrap 95% CIs (10,000 resamples) quantify uncertainty. Cohen's $d$ measures effect size. For first-letter patching, the unit of analysis is the word ($n = 25$ words, 200 prompt contexts each). For RAVEL patching, the unit is the city entity ($n = 128$ for city-continent, $n = 201$ for city-language, 50 contexts each).

## 3.5 Decoder Information Entanglement Diagnostic

This diagnostic measures how much parent-relevant information is encoded in child decoder vectors, providing a geometric characterization of absorption.

For each absorption instance:
1. Project the child decoder $\mathbf{d}_c$ onto the parent probe direction $\hat{\mathbf{w}}_p$: compute the component $(\mathbf{d}_c \cdot \hat{\mathbf{w}}_p) \hat{\mathbf{w}}_p$.
2. Remove this component: $\mathbf{d}_c^{(\neg p)} = \mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p) \hat{\mathbf{w}}_p$.
3. Reconstruct using the modified decoder and measure the downstream logit change $\Delta_{\text{logit}}$ for parent-relevant tokens.
4. Classify: if $|\Delta_{\text{logit}}| > \tau$ (tested at $\tau \in \{0.05, 0.1, 0.2\}$ for robustness), the absorption instance involves substantial parent information in the child decoder.
5. *Control*: ablate a random direction of the same norm instead of $\hat{\mathbf{w}}_p$ and measure $\Delta_{\text{logit}}^{\text{ctrl}}$.

**Circularity caveat.** This diagnostic shares the probe direction $\hat{\mathbf{w}}_p$ with the FN classification in Step 1 (Section 3.3). It therefore measures *decoder geometry* relative to the probe---not computational redundancy in the model's forward pass. A token classified as FN because the probe direction is missing from the SAE reconstruction will, by construction, show a large logit change when that direction is ablated from the child decoder. The diagnostic establishes that child decoders carry large-magnitude parent information (informative about absorption mechanics) but cannot distinguish whether this information is computationally redundant. A genuine test of computational redundancy would require activation-level ablation ($z_p = 0$, measuring downstream task performance) or path patching through separate circuits.

## 3.6 Hedging Decomposition

Not all false negatives are absorption. We decompose FN tokens into three categories:

1. **Strict absorbed** ($\text{FN}_{\text{strict}}$): The main parent feature (highest-ranked by activation for the correct class) does *not* fire. The SAE dictionary lacks a dedicated feature for this parent concept in this context---a genuine feature gap.
2. **Compensatory** ($\text{FN}_{\text{comp}}$): The main parent feature *does* fire, but other features in the SAE reconstruction interfere with the probe's prediction. The parent information is present in the latent vector but not correctly reconstructed.
3. **Persistent** ($\text{FN}_{\text{persist}}$): Residual false negatives that do not fit either category, attributable to probe boundary errors or reconstruction noise.

This classification tightens the original hedging criterion of Chanin et al. (2024), which counts *any* FN where *any* parent-related feature fires as "hedged." That loose criterion yields 92.6% hedging---near-tautological, since SAEs with $L_0 > 20$ will almost always have some parent-related feature active. Our strict criterion asks whether the *main* parent feature fires, yielding 0--22.6% strict absorption across hierarchies.

For first-letter hierarchies, we additionally analyze hedging across multiple $L_0$ values (from 22 to 176) to characterize how sparsity level affects the strict-vs-compensatory balance.

## 3.7 Probe Degradation Ablation

Cross-domain absorption comparison requires controlling for probe quality, since RAVEL probes ($F_1 = 0.73$--$0.87$) are substantially worse than first-letter probes ($F_1 = 1.0$). An imperfect probe classifies some tokens incorrectly on raw activations, potentially inflating the measured false negative rate on SAE reconstructions.

The probe degradation ablation quantifies the relationship between probe $F_1$ and measured absorption rate by intentionally degrading a near-perfect probe:

1. Start with the trained first-letter probe at layer 24 ($F_1 \approx 1.0$).
2. Inject Gaussian noise into the probe weight matrix: $\mathbf{W}_p^{(\epsilon)} = \mathbf{W}_p + \epsilon \cdot \mathcal{N}(0, \mathbf{I})$, where $\epsilon$ controls the noise scale.
3. Calibrate $\epsilon$ to achieve 7 target $F_1$ levels: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, and 1.0.
4. At each degraded $F_1$ level, run the full absorption measurement pipeline (Section 3.3) on the same 2,345 test words $\times$ 5 prompts = 11,725 tokens.
5. Average across 3 noise seeds per $F_1$ level to reduce noise-realization variance.
6. Fit a linear regression: $\hat{\alpha}_{\text{lin}}(F_1) = \beta_0 + \beta_1 F_1$.

**Confound decomposition.** Overlaying RAVEL hierarchies on the fitted curve decomposes each hierarchy's absorption rate into two components:
- *Probe-explained component*: the absorption rate predicted by the degradation curve at that hierarchy's probe $F_1$.
- *Residual (curve delta)*: $\Delta = \alpha_{\text{observed}} - \hat{\alpha}_{\text{lin}}(F_1)$. Positive $\Delta$ indicates excess absorption beyond what probe quality explains; negative $\Delta$ indicates absorption suppression.

A hierarchy whose observed absorption falls on the degradation curve ($|\Delta| < 5$ pp) is consistent with the probe quality confound explanation. A hierarchy with $|\Delta| > 10$ pp is a genuine outlier requiring a hierarchy-specific explanation.

**Limitations.** The degradation curve is estimated from first-letter probes (binary classification, 26 balanced classes) and extrapolated to RAVEL probes (multi-class, imbalanced). This cross-domain extrapolation is not validated; the relationship between probe degradation and absorption may differ for multi-class probes. The 7-point curve with 2 free parameters (slope, intercept) has limited degrees of freedom, and the perfect monotonicity ($\rho = -1.0$) may partly reflect the limited sample size rather than a true deterministic relationship. We present the linear fit as the primary result and a quadratic fit ($R^2 = 0.942$) as exploratory.

<!-- FIGURES
- Table 1: inline --- Feature hierarchy properties and probe quality at layer 24
- None (no generated figures in this section)
-->
