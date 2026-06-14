# Methodology

This section describes the four feature hierarchies under study, the probe training and quality gating procedure, the absorption measurement pipeline, the activation patching protocol for causal validation, the decoder information entanglement diagnostic, the hedging decomposition, and the probe degradation ablation that controls for the probe quality confound.

## 3.1 Feature Hierarchies

We measure absorption on four feature hierarchies that span syntactic and semantic domains (Table 1).

**First-letter spelling.** Following Chanin et al. (2024), the parent concept is the first letter of a word (e.g., "starts with S") and the child is the word itself (e.g., "Saturday"). The hierarchy has $K = 25$ parent classes (letter "x" excluded for insufficient test examples), approximately 2,345 test words drawn from the `sae-spelling` library, and 100% parent-child co-occurrence by construction. Each word is evaluated in 5 prompt contexts using the template "The word {WORD} begins with the letter," with the target word at token position $-6$. First-letter probes achieve $F_1 = 1.0$ at all tested layers, making this hierarchy a positive control with maximal probe quality.

**City-continent.** Drawn from RAVEL (Huang et al., 2024), the parent concept is the continent in which a city is located (e.g., "in Europe") and the child is the city entity (e.g., "Paris"). $K = 6$ continent classes, 1,567 entities, moderate class balance. Probe $F_1 = 0.87$ at layer 24. Entities are evaluated at token position $-2$ using RAVEL's attribute-specific prompt templates.

**City-language.** Also from RAVEL, the parent concept is the primary language spoken in a city (e.g., "speaks French"). $K = 23$ language classes (including compound labels such as "Arabic,Kurdish,Turkish"), 1,229 entities, intermediate class balance. Probe $F_1 = 0.82$ at layer 24. Same token position ($-2$) and prompt templates as city-continent.

**City-country.** From RAVEL, the parent concept is the country in which a city is located. $K = 77$ country classes (3 classes with $< 5$ entities excluded from the original 80), 1,405 entities, highly imbalanced (USA: 176 entities, Chad: 5). Probe $F_1 = 0.73$ at layer 24---below the strict quality gate. This hierarchy is included with a documented caveat: its absorption rate may be inflated by probe imperfection (quantified in Section 4.6 via the probe degradation ablation).

**Token position asymmetry.** First-letter experiments use token position $-6$ (the `sae-spelling` convention), while RAVEL hierarchies use position $-2$ (the RAVEL convention). Both positions were chosen to maximize probe quality for their respective tasks. Within-RAVEL comparisons (the primary cross-domain analysis in Section 4) are unaffected because all three RAVEL hierarchies share position $-2$. First-letter versus RAVEL comparisons should be interpreted with this confound in mind.

## 3.2 Probe Training and Quality Gates

We train logistic regression probes on Gemma 2 2B residual stream activations at layers 6, 12, 18, and 24. First-letter probes use L2-regularized one-vs-rest logistic regression ($C = 0.01$, selected via cross-validation) on 4,132 training tokens, evaluated on 1,033 test tokens. RAVEL probes use cross-validated logistic regression on all RAVEL city entities at each layer and hierarchy, evaluated by weighted-macro $F_1$.

Table 1 reports probe quality across all 16 hierarchy-layer combinations with quality gate status.

| Hierarchy | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Gate (L24) |
|-----------|---------|----------|----------|----------|------------|
| First-letter | $F_1 = 1.00$ | $F_1 = 1.00$ | $F_1 = 1.00$ | $F_1 = 1.00$ | **Strict pass** |
| City-continent | $F_1 = 0.61$ | $F_1 = 0.74$ | $F_1 = 0.81$ | $F_1 = 0.87$ | Relaxed pass |
| City-language | $F_1 = 0.53$ | $F_1 = 0.68$ | $F_1 = 0.76$ | $F_1 = 0.82$ | Relaxed pass |
| City-country | $F_1 = 0.41$ | $F_1 = 0.56$ | $F_1 = 0.67$ | $F_1 = 0.73$ | **Below gate** |

**Table 1.** Probe $F_1$ scores across four hierarchies and four layers on Gemma 2 2B. Quality gate thresholds: strict $> 0.90$; relaxed $\geq 0.80$. First-letter achieves perfect $F_1$ at all layers. RAVEL probes improve monotonically with depth, reaching their best quality at layer 24. City-country falls below the relaxed gate and is included with a documented caveat.

**Quality gates.** We define two thresholds: a strict gate ($F_1 > 0.90$) and a relaxed gate ($F_1 \geq 0.80$). Only first-letter passes the strict gate at layer 24. City-continent and city-language pass the relaxed gate. City-country fails both gates. Results from hierarchies passing only the relaxed gate are reported with the caveat that probe imperfection may inflate measured absorption (Section 4.6 quantifies this effect). Results from city-country are included as exploratory data with explicit uncertainty.

**Aggregation method.** Absorption rates are computed per-token: $\alpha = n_{\text{FN}} / n_{\text{probe-correct-raw}}$, where each token occurrence is one observation. For first-letter, each of the 2,345 test words appears in 5 prompt contexts, yielding 11,725 token observations. For RAVEL hierarchies, each entity appears once. We also computed per-unique-word and per-letter aggregations as robustness checks; the qualitative finding of significant cross-domain variation holds under all three aggregation methods.

## 3.3 Absorption Measurement Pipeline

The absorption measurement pipeline adapts Chanin et al. (2024) to arbitrary feature hierarchies:

1. **Encode.** Pass input tokens through Gemma 2 2B and extract residual stream activations $\mathbf{x} \in \mathbb{R}^d$ at layer $\ell$.

2. **Probe on raw activations.** Apply the linear probe to $\mathbf{x}$. Record the predicted parent class $\hat{y}_{\text{raw}}$. If $\hat{y}_{\text{raw}} \neq y$ (probe incorrect on raw), skip this token.

3. **SAE encode.** Pass $\mathbf{x}$ through the SAE encoder: $\mathbf{z} = \sigma(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}})$, where $\sigma$ is JumpReLU.

4. **SAE reconstruct.** Compute $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$.

5. **Probe on SAE activations.** Apply the same probe to $\hat{\mathbf{x}}$. Record $\hat{y}_{\text{SAE}}$.

6. **Classify false negatives.** If $\hat{y}_{\text{raw}} = y$ but $\hat{y}_{\text{SAE}} \neq y$, this token is a false negative (FN)---the SAE reconstruction lost parent information that the raw activations carried.

7. **Identify parent-child pairs.** For each FN, use integrated-gradients attribution to identify which active SAE features contributed most to the probe prediction change. Features with high attribution and cosine similarity $\cos(\mathbf{d}_i, \mathbf{w}_p) > 0.30$ to the parent probe direction $\mathbf{w}_p$ are candidate child features.

**Cosine similarity threshold.** We use a threshold of 0.30 for parent-child identification. This threshold was validated as robust across the range 0.20--0.40 (coefficient of variation = 0.077 across 5 thresholds $\times$ 2 SAE configurations; subtype ordering is preserved in 10/10 cells). The data-driven 95th-percentile of random cosine similarity is 0.044--0.049, confirming all tested thresholds are well above chance (Appendix E).

**Controls.** Two controls validate the pipeline:
- *Random direction baseline:* Replace the parent probe direction with a random unit vector. If the pipeline still reports high "absorption," the measurement is an artifact.
- *Shuffled hierarchy control:* Randomly reassign parent labels across entities and re-measure absorption. Genuine absorption requires the correct hierarchy structure.

## 3.4 Activation Patching Protocol

Activation patching provides causal---not merely correlational---evidence for competitive exclusion. For each absorbed instance (probe correct on raw, incorrect on SAE):

1. Identify the primary child feature $c$ via integrated-gradients attribution on the SAE latent activations.
2. Zero the child feature: $\mathbf{z}^{(c \to 0)}_c = 0$, keeping all other latents unchanged.
3. Reconstruct from the modified latents: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$.
4. Apply the parent probe to $\hat{\mathbf{x}}^{(c \to 0)}$. If the probe now predicts correctly ($\hat{y} = y$), this instance shows *recovery*---the child feature was suppressing the parent.

**Control condition.** For each word/entity, zero a randomly selected non-child feature matched by activation magnitude rather than the identified child feature. The control recovery rate estimates the baseline rate of probe prediction changes from single-feature zeroing.

**Statistics.** Recovery rates are compared between child-zeroed and control conditions via Wilcoxon signed-rank test (paired by word/entity). Effect sizes are reported as Cohen's $d$. Confidence intervals use the bootstrap percentile method with 10,000 resamples.

**Scale.** For first-letter, 25 words (19 with absorption) are tested, each in 200 prompt contexts. For city-continent, 128 entities across 4,902 absorbed contexts. For city-language, 201 entities across 7,814 absorbed contexts.

## 3.5 Decoder Information Entanglement Diagnostic

This diagnostic measures the magnitude of parent-direction information encoded in child feature decoder vectors:

1. For each absorbed instance with identified child feature $c$ and parent class $p$:
2. Compute the modified decoder: $\mathbf{d}_c^{(\neg p)} = \mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p)\hat{\mathbf{w}}_p$, removing the parent probe direction.
3. Measure the logit change $\Delta_{\text{logit}}$ for parent-relevant tokens when using $\mathbf{d}_c^{(\neg p)}$ instead of $\mathbf{d}_c$.
4. An instance is classified as "entangled" at threshold $\tau$ if $|\Delta_{\text{logit}}| > \tau$.
5. **Control:** Ablate a random direction of the same norm as $\hat{\mathbf{w}}_p$; measure the same logit change.

We evaluate at three thresholds ($\tau \in \{0.05, 0.1, 0.2\}$) for robustness.

**Circularity caveat.** This diagnostic shares the parent probe direction $\hat{\mathbf{w}}_p$ with the FN classification itself. It measures decoder geometry---specifically, how much parent-direction information the child decoder carries---not whether the absorbed information is computationally redundant. A genuine computational-redundancy test would require activation-level ablation ($z_p = 0$) or path patching through separate circuits. We report the decoder entanglement magnitude as informative about absorption mechanics, with this circularity limitation acknowledged throughout.

## 3.6 Hedging Decomposition

Not all false negatives represent the same failure mode. We decompose FNs into three categories:

- **Strict absorbed** ($\text{FN}_{\text{strict}}$): The main parent feature does not fire ($z_p = 0$). The SAE dictionary has a genuine gap---no feature represents the parent concept for this input.
- **Compensatory** ($\text{FN}_{\text{comp}}$): The main parent feature fires ($z_p > 0$) but the probe still classifies the SAE reconstruction incorrectly. Other features interfere with the probe direction.
- **Persistent** ($\text{FN}_{\text{persist}}$): Residual false negatives that fit neither category (probe boundary errors).

This three-way classification refines the original hedging analysis of Chanin et al. (2024), which used a *loose* criterion: any FN where *any* parent-related feature fires was classified as "hedged." That loose classification yields 92.6%---near-tautological, since high-$L_0$ SAEs almost always have some parent-related feature active. Our *strict* classification uses only the main (highest-ranked) parent feature, yielding 0--22.6% across hierarchies.

For first-letter, we additionally report a multi-$L_0$ analysis ($L_0$ from 22 to 176) to examine how the strict/compensatory decomposition varies with sparsity.

## 3.7 Probe Degradation Ablation

The probe degradation ablation is a control experiment that disentangles the probe quality confound from genuine hierarchy effects. First-letter probes achieve $F_1 = 1.0$; RAVEL probes range from $F_1 = 0.73$ to $0.87$. If cross-domain absorption variation merely reflects this difference in probe quality, then degrading the first-letter probes to match RAVEL $F_1$ levels should reproduce RAVEL absorption rates.

**Protocol.** Starting from trained first-letter probes at layer 24 ($F_1 = 1.0$):

1. Inject Gaussian noise into the probe weight matrix: $\mathbf{W}_p^{(\epsilon)} = \mathbf{W}_p + \epsilon \cdot \mathcal{N}(0, I)$, calibrating $\epsilon$ to degrade test-set $F_1$ to seven target levels: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, and 1.0 (undegraded control).
2. For each degradation level, run the full absorption measurement pipeline (Section 3.3) at layer 24 with the 16k JumpReLU SAE.
3. Average absorption rates across 3 noise seeds per degradation level for stability.
4. Fit a regression of absorption rate $\alpha^{(\epsilon)}$ on degraded probe $F_1^{(\epsilon)}$ to estimate the slope $\beta_1$ of the probe degradation curve.
5. Overlay the three RAVEL hierarchies at their actual $F_1$ levels. The vertical distance from each RAVEL point to the curve (the *curve delta*) estimates the hierarchy-specific effect that probe quality alone cannot explain.

**Interpretation.** If a RAVEL hierarchy falls on or near the curve, its absorption rate is explained by probe quality. If it deviates by more than 10 percentage points, the hierarchy has a genuine hierarchy-specific absorption effect. The probe degradation ablation was conducted with 11,725 tokens per degradation level.

<!-- FIGURES
- Table 1: inline — Probe quality (F1) across 4 hierarchies × 4 layers with quality gate status
- None (no generated figures in this section)
-->
