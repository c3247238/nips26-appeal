# 3 Method

We measure feature absorption across four feature hierarchies, four model layers, and three SAE architectures on Gemma 2 2B, using an adapted version of the Chanin et al. (2024) absorption measurement pipeline. Figure 2 illustrates the full pipeline from residual stream extraction through absorption detection and causal validation via activation patching.

![Experimental pipeline from residual stream extraction through probe classification, false negative detection, integrated-gradients attribution, and activation patching.](figures/fig2_pipeline_desc.md)

## 3.1 Model and SAEs

All experiments use Gemma 2 2B (`unsloth/gemma-2-2b`; $d = 2304$, 26 layers) at inference time. No SAE training is performed; we evaluate pre-trained SAEs only.

**Gemma Scope SAEs.** We use eight JumpReLU SAEs from Google DeepMind's Gemma Scope collection (Lieberum et al., 2024) spanning four layers ($L \in \{6, 12, 18, 24\}$) at two dictionary sizes ($M \in \{16384, 65536\}$). These SAEs operate at $L_0 \approx 75$--$87$ depending on the layer.

**SAEBench SAEs.** At layer 12, we additionally evaluate two SAEs from SAEBench (Karvonen et al., 2025): a BatchTopK SAE with $M = 16384$ ($L_0 = 20$) and a Matryoshka SAE with $M = 32768$. These enable cross-architecture comparison at a single layer.

The 10 SAE configurations are summarized below:

| SAE Family | Architecture | Layer(s) | $M$ | $L_0$ |
|---|---|---|---|---|
| Gemma Scope | JumpReLU | 6, 12, 18, 24 | 16k | 75--87 |
| Gemma Scope | JumpReLU | 6, 12, 18, 24 | 65k | 75--87 |
| SAEBench | BatchTopK | 12 | 16k | 20 |
| SAEBench | Matryoshka | 12 | 32k | -- |

## 3.2 Feature Hierarchies

We define four feature hierarchies $G = (V, E)$, each mapping tokens to a categorical attribute through parent-child relationships where the child feature logically implies the parent.

**First-letter (syntactic).** Each token maps to its initial letter: $K = 26$ classes with near-uniform distribution. We use the `sae_spelling` pipeline (Chanin et al., 2024) with in-context learning prompts to extract first-letter representations at the residual stream. This hierarchy serves as the baseline and replication target.

**City-continent (factual, coarse).** Each city entity maps to its continent: $K = 6$ classes (Africa, Asia, Europe, North America, Oceania, South America). Data comes from the RAVEL dataset ($n = 2039$ cities; Huang et al., 2024). We use in-context learning prompts with last-token ($\text{position} = -1$) extraction.

**City-country (factual, fine-grained).** Each city maps to its country: $K = 80$ classes with highly imbalanced distribution (the United States, Russia, and India dominate). $n = 1881$ cities from RAVEL after filtering entities with missing country labels.

**City-language (factual, medium granularity).** Each city maps to its primary official language: $K = 50$ classes. $n = 1859$ cities from RAVEL.

Table 1 and Figure tab1_probe_quality report probe quality across all hierarchy-layer combinations.

| | Layer 6 | Layer 12 | Layer 18 | Layer 24 |
|---|---|---|---|---|
| **First-letter** | 0.69 | 0.31 | **0.94** | **0.97** |
| **City-continent** | 0.65 | 0.79 | 0.84 | 0.84 |
| **City-country** | 0.35 | 0.62 | 0.78 | 0.79 |
| **City-language** | 0.52 | 0.69 | 0.81 | 0.82 |

**Table 1.** Weighted F1 scores for linear probes (sklearn logistic regression, position $-1$) across hierarchies and layers. Bold entries pass the strict quality gate (F1 $\geq$ 0.90). Only first-letter probes at layers 18 and 24 pass the strict gate. RAVEL probes peak at layer 24 (F1 = 0.79--0.84) but remain below the strict gate; we include them with this caveat documented. For first-letter absorption measurement, the `sae_spelling` pipeline (position $-2$, ICL prompts) achieves F1 $\approx$ 1.0 on its own evaluation protocol; the lower sklearn values at layers 6 and 12 reflect the cross-hierarchy-consistent probing setup and are not used for first-letter absorption experiments at those layers.

![Probe quality heatmap. Checkmarks indicate strict gate passage (F1 >= 0.90); tilde indicates relaxed gate (0.85--0.90).](figures/tab1_probe_quality.pdf)

## 3.3 Probe Training

For each hierarchy-layer combination, we train an $L_2$-regularized logistic regression probe $f_{\text{probe}}$ (scikit-learn, `LogisticRegression` with `solver='lbfgs'`):

- **Regularization:** $C \in \{0.01, 0.1, 1.0, 10.0\}$ selected by 5-fold stratified cross-validation.
- **Split:** 80/20 stratified train-test split, `seed=42`.
- **Input:** Residual stream activation $\mathbf{x}^{(L)}$ at the last token position ($\text{position} = -1$) for RAVEL hierarchies; position $-1$ (sklearn) or $-2$ (`sae_spelling` LinearProbe) for first-letter.
- **Metric:** Weighted macro-F1.

**Quality gates.** We enforce two thresholds: a strict gate (F1 $\geq$ 0.90) and a relaxed gate (F1 $\geq$ 0.85). Only first-letter probes at layers 18 (F1 = 0.94) and 24 (F1 = 0.97) pass the strict gate. RAVEL probes achieve F1 = 0.79--0.84 at layer 24, their best layer. The probe quality confound is analyzed in Section 4.3: probe quality correlates with false negative rate at $\rho = -0.756$ ($p < 0.001$), meaning lower-quality probes miss more correct raw-condition classifications, potentially masking or inflating absorption rates.

**First-letter probes** additionally use the `sae_spelling` pipeline (Chanin et al., 2024), which trains a `LinearProbe` on ICL-formatted spelling prompts at position $-2$. At layers where both methods are available, the `sae_spelling` probe achieves F1 = 0.87 (L24) while the sklearn probe achieves F1 = 0.97 (L24). We use the higher-quality sklearn probes for absorption measurement to maximize the denominator of false-negative detection.

## 3.4 Absorption Measurement

Our absorption measurement pipeline adapts Chanin et al. (2024) to support arbitrary feature hierarchies.

**Step 1: False negative detection.** For each token $t$ in the evaluation set, we compute:
$$\hat{y}_{\text{raw}} = f_{\text{probe}}(\mathbf{x}^{(L)}), \quad \hat{y}_{\text{sae}} = f_{\text{probe}}(\hat{\mathbf{x}})$$
where $\hat{\mathbf{x}} = \text{SAE}(\mathbf{x}^{(L)})$. A token is a false negative if $\hat{y}_{\text{raw}} = y$ (probe correctly classifies raw activation) but $\hat{y}_{\text{sae}} \neq y$ (probe misclassifies SAE-reconstructed activation). The set $\text{FN}$ contains all such tokens.

**Step 2: Integrated-gradients attribution.** For each false negative, we compute the attribution of each active SAE latent $i$ to the probe's classification change using integrated gradients (Sundararajan et al., 2017) with 10 interpolation steps between a zero baseline and the actual latent activation vector $\mathbf{a}$:
$$\text{IG}_i(t) = a_i \cdot \int_0^1 \frac{\partial f_{\text{probe}}(\text{SAE}_{\text{dec}}(\alpha \cdot \mathbf{a}))}{\partial a_i} \, d\alpha$$

**Step 3: Absorption detection.** A false negative is classified as absorbed if the highest-attribution latent $i^*$ satisfies two conditions simultaneously:
1. **Cosine alignment:** $\cos(\mathbf{d}_{i^*}, \mathbf{w}_y) > \tau_{\cos}$ (default $\tau_{\cos} = 0.025$), where $\mathbf{d}_{i^*}$ is the decoder direction and $\mathbf{w}_y$ is the probe weight for the true class.
2. **Magnitude gap:** The IG attribution of $i^*$ exceeds the second-highest attribution by a factor $\geq \tau_{\text{gap}}$ (default $\tau_{\text{gap}} = 1.0$).

**Absorption rate.** $\text{AR}$ is the fraction of $K$ classes for which at least one absorbed false negative exists:
$$\text{AR} = \frac{|\{k : \exists t \in \text{FN}_k \text{ classified as absorbed}\}|}{K}$$

**Threshold robustness.** A 5$\times$4 sensitivity grid ($\tau_{\cos} \in \{0.01, 0.02, 0.025, 0.03, 0.05\}$, $\tau_{\text{gap}} \in \{0.5, 1.0, 1.5, 2.0\}$) confirms that false negatives remain constant at $n = 87$ across all 20 cells, with AR varying from 11.8% to 15.1% (CV = 0.077). This establishes that absorption is structural, not a detection-threshold artifact (Section 7.4).

**Statistical tests.** All absorption rates are accompanied by bootstrap 95% confidence intervals (10,000 resamples, `seed=42`). Cross-domain comparisons use paired permutation tests (10,000 permutations) with Cohen's $d$ effect sizes. The overall hierarchy effect is assessed via a Kruskal-Wallis test.

## 3.5 Activation Patching

Activation patching provides interventional causal evidence for feature suppression, complementing the correlational IG-based absorption detection.

**Procedure.** For each word $t$ with detected absorption at layer 12 (Gemma Scope JumpReLU 16k):

1. Encode $\mathbf{x}^{(12)}$ through the SAE to obtain latent activations $\mathbf{a}$ and reconstructed activation $\hat{\mathbf{x}}$.
2. Identify the highest-IG child latent $c$ responsible for the false negative.
3. **Treatment:** Set $a_c := 0$ (zero the child feature), recompute $\hat{\mathbf{x}}' = \text{SAE}_{\text{dec}}(\mathbf{a}')$, and evaluate $f_{\text{probe}}(\hat{\mathbf{x}}')$.
4. **Control:** Zero 15 magnitude-matched random latents (matched on $|a_i|$ to within 10%) and evaluate recovery.
5. Repeat across 200 contexts per word.

**Recovery rate.** $\text{RR}_{\text{child}}$ is the fraction of absorbed tokens (those with $\geq 3$ absorbed instances) where zeroing the child feature causes the probe to recover the correct parent prediction. The control recovery rate $\text{RR}_{\text{ctrl}}$ is the mean recovery across 15 random-feature ablations.

**Word selection.** We tested 25 words total: 7 pilot core words selected from high-frequency tokens in the `sae_spelling` evaluation set, and 18 additional words discovered via IG-guided search for tokens exhibiting large attribution gaps between the top latent and the rest. Of 25 tested words, 19 exhibited $\geq 3$ absorbed instances and entered the statistical analysis. The discovery procedure biases toward finding absorption; the 58.8% absorption rate among tested words reflects this selection, not a corpus-wide prevalence estimate.

**Statistical tests.** Wilcoxon signed-rank test (one-sided: child recovery $>$ control recovery), Mann-Whitney $U$ test, paired $t$-test, and bootstrap confidence interval on $\Delta\text{RR} = \text{RR}_{\text{child}} - \text{RR}_{\text{ctrl}}$.

## 3.6 Hedging Decomposition

We decompose false negatives into three categories to distinguish genuine absorption from resolution at higher sparsity:

1. **Strict hedging ($H_{\text{strict}}$):** The parent feature $p$ itself reactivates when the SAE operates at a higher $L_0^{\text{target}}$, recovering the correct probe prediction.
2. **Compensatory resolution ($H_{\text{comp}}$):** The false negative resolves at $L_0^{\text{target}}$, but through other (non-parent) latents adding sufficient information -- the parent feature does not reactivate.
3. **Persistent ($H_{\text{persist}}$):** The false negative persists even at $L_0^{\text{target}}$.

We test at $L_0^{\text{base}} = 22$ with $L_0^{\text{target}} = 176$ (8$\times$ multiplier), and additionally at $L_0^{\text{base}} = 82$ with $L_0^{\text{target}} = 176$ for sensitivity analysis. The widely-cited "hedging rate" (Chanin et al., 2025) corresponds to our loose rate: $H_{\text{strict}} + H_{\text{comp}} = 1 - H_{\text{persist}}$. Our decomposition reveals what fraction of this loose hedging is attributable to the parent feature itself versus unrelated compensatory features.

<!-- FIGURES
- Figure 2: fig2_pipeline_desc.md — Experimental pipeline schematic showing absorption measurement and activation patching workflow
- Table 1: gen_tab1_probe_quality.py, tab1_probe_quality.pdf — Probe quality heatmap across hierarchies and layers with quality gate annotations
-->
