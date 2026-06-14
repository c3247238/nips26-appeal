# Section 3: Methodology

## 3.1 Feature Hierarchies

We measure feature absorption across four hierarchy types that vary in class count, balance, and semantic structure. Each hierarchy defines a set of parent classes $p$ and child entities $c$ such that every child belongs to exactly one parent in $\mathcal{C}(p)$.

**First-letter spelling.** 26 parent classes (letters A--Z), approximately 500 child entities (common English words), with binary probes. Parent--child co-occurrence is 100% by construction: every word starting with "S" activates the "starts with S" parent. This hierarchy, introduced by Chanin et al. (2024), serves as our positive control because probes achieve perfect F1 = 1.0 at all layers (trained at token position $-6$ following the `sae-spelling` protocol).

**City-continent.** 6 parent classes (Africa, Asia, Europe, North America, Oceania, South America), 1,567 child entities drawn from the RAVEL dataset (Huang et al., 2024). Class distribution is moderately imbalanced: Asia has 377 entities while Oceania has 78.

**City-language.** 23 parent classes (English, Spanish, Chinese, etc.), 1,229 entities from RAVEL. Intermediate balance, with English (388 entities) dominating and several classes containing fewer than 15 entities.

**City-country.** 80 parent classes, 1,405 entities from RAVEL. Highly imbalanced: the United States contributes 176 entities while 30 countries contribute fewer than 10 each. This hierarchy's large class count and imbalance make it the most challenging for probe training.

## 3.2 Probe Training and Quality Gates

We train linear probes (logistic regression classifiers) on Gemma 2 2B residual stream activations $\mathbf{x}^{(\ell)}$ at layers $\ell \in \{6, 12, 18, 24\}$.

**First-letter probes** follow the `sae-spelling` pipeline: activations are extracted at token position $-6$ (the position encoding the first character of the next word). We train one-vs-all logistic regression probes with L2 regularization ($C = 0.01$, selected via cross-validation) on 4,132 training examples and evaluate on 1,033 held-out examples. These probes achieve $F_1 = 1.0$ (weighted) at all four layers, reflecting that Gemma 2 2B encodes first-letter information robustly throughout its residual stream.

**RAVEL probes** use standard logistic regression (scikit-learn) at token position $-2$ (the final position before the entity token that most reliably encodes attribute information). We apply 4- or 5-fold stratified cross-validation over a regularization grid ($C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$), selecting the $C$ value that maximizes weighted $F_1$. Training sets range from 1,350 (city-language) to 1,632 (city-continent) examples; test sets contain 338--409 examples. For city-country (80 classes), frequency-balanced sampling ensures minority classes appear in training.

**Quality gates** partition results into three reliability tiers:

- **Strict pass** ($F_1 \geq 0.90$): results are fully reliable. Only first-letter probes achieve this gate at all layers.
- **Relaxed pass** ($F_1 \geq 0.80$): results are interpretable but absorption rates should be treated as upper bounds, since probe errors inflate false negative counts. City-continent ($F_1 = 0.87$ at L24) and city-language ($F_1 = 0.82$ at L24) fall in this tier.
- **Below gate** ($F_1 < 0.80$): results are reported with explicit caveats. City-country ($F_1 = 0.73$ at L24) is the only L24 hierarchy in this tier; its 80-class prediction task inherently limits probe accuracy.

Table 1 presents the complete probe quality matrix. All four hierarchies achieve their best $F_1$ at layer 24, consistent with factual knowledge concentrating in later transformer layers. The first-letter hierarchy serves as a gold-standard anchor: with $F_1 = 1.0$, its absorption rates are uncontaminated by probe error.

![Table 1: Probe quality (weighted F1) across four feature hierarchies and four transformer layers. Checkmark: strict gate (F1 >= 0.90). Tilde: relaxed gate (F1 >= 0.80). Cross: below gate.](figures/table1_probe_quality.pdf)

## 3.3 Absorption Measurement Pipeline

Our absorption measurement adapts the protocol of Chanin et al. (2024) to handle multi-class hierarchies and multiple SAE configurations.

**Definition.** For a given hierarchy $\mathcal{H}$, SAE, and input, we classify an instance as a **false negative** (FN) when the probe predicts the correct parent class from the raw activation $\mathbf{x}$ but the wrong class from the SAE reconstruction $\hat{\mathbf{x}}$:
$$\text{FN}: \quad \hat{y}_{\text{raw}} = y \;\;\text{and}\;\; \hat{y}_{\text{SAE}} \neq y$$
The **absorption rate** $\alpha(\mathcal{H}, \text{SAE})$ is the fraction of probe-correct instances that become false negatives after SAE encoding:
$$\alpha = \frac{|\{\text{FN}\}|}{|\{\hat{y}_{\text{raw}} = y\}|}$$

**Parent--child pair identification.** For each false negative, integrated-gradients attribution identifies which active SAE features $z_i > 0$ contribute most to the probe's changed prediction. The child feature is the active feature with highest attribution magnitude whose decoder vector $\mathbf{d}_i$ has non-trivial cosine similarity with the parent probe direction $\mathbf{w}_p$.

**SAE configurations.** We evaluate Gemma Scope JumpReLU SAEs (Lieberum et al., 2024) at two dictionary widths ($m = 16{,}384$ and $m = 65{,}536$) across all four layers. At layer 12, we additionally evaluate SAEBench BatchTopK ($m = 16{,}384$) and Matryoshka ($m = 32{,}768$) SAEs (Karvonen et al., 2025).

**Controls.** Three baselines bound the measurement:
1. *Random direction baseline:* replace $\mathbf{w}_p$ with a random unit vector in $\mathbb{R}^d$ and re-measure the false negative rate. This quantifies the chance FN rate for an arbitrary linear direction.
2. *Shuffled hierarchy control:* randomly permute parent labels across entities and re-measure absorption. This controls for hierarchy-independent SAE reconstruction error.
3. *Probe-only baseline:* measure the probe's FN rate on raw activations alone (the floor set by probe imperfection).

**Statistics.** All absorption rates are accompanied by bootstrap 95% confidence intervals (10,000 resamples). Cross-hierarchy comparisons use the Kruskal-Wallis test with pairwise permutation tests and Bonferroni correction for 6 pairwise comparisons.

## 3.4 Activation Patching Protocol

Activation patching provides causal (not merely correlational) evidence for competitive exclusion by intervening on specific SAE features and measuring downstream effects.

**Procedure.** For each absorption instance (entity, context, identified child feature $c$):
1. Encode the residual stream activation through the SAE to obtain latents $\mathbf{z}$.
2. Zero the child feature: $\mathbf{z}^{(c \to 0)} = \mathbf{z}$ with $z_c = 0$.
3. Reconstruct: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$.
4. Apply the parent probe to $\hat{\mathbf{x}}^{(c \to 0)}$ and record whether the correct parent class is recovered.

**Control condition.** For each patched instance, we also zero a control feature matched by activation magnitude ($|z_{\text{ctrl}}| \approx |z_c|$) but semantically unrelated to the hierarchy. The **recovery rate** $R_c$ is the fraction of false negatives where zeroing the child restores the correct probe prediction; $R_{\text{ctrl}}$ is the corresponding rate for the control feature. The recovery difference $\Delta R = R_c - R_{\text{ctrl}}$ isolates the causal contribution of the child feature.

**Statistics.** We report the Wilcoxon signed-rank test (paired, per-entity) on recovery rates, bootstrap 95% CI on $\Delta R$, and Cohen's $d$ for effect size.

**Scope.** First-letter patching uses 25 words with 200 contexts each. Cross-domain patching (city-continent) uses 93 entities with 50 contexts each (3,751 FN instances total).

## 3.5 Benign vs. Pathological Diagnostic

This diagnostic tests whether absorption causes genuine information loss or merely reflects computational redundancy -- i.e., whether the model would use the parent feature independently even when the child is active.

**Procedure.** For each false negative instance:
1. Compute the parent direction in activation space: $\hat{\mathbf{w}}_p = \mathbf{w}_p / \|\mathbf{w}_p\|$.
2. Ablate the parent direction from the child feature's decoder vector: $\mathbf{d}_c^{(\neg p)} = \mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p)\hat{\mathbf{w}}_p$.
3. Reconstruct the activation using the modified decoder and measure the logit change $\Delta_{\text{logit}}$ for parent-relevant output tokens.

**Classification.** An instance is **benign** if $|\Delta_{\text{logit}}| \leq \tau$ (the model's output is unaffected by losing parent information) and **pathological** if $|\Delta_{\text{logit}}| > \tau$. We evaluate at three thresholds $\tau \in \{0.05, 0.1, 0.2\}$ for robustness, reporting the full distribution of $|\Delta_{\text{logit}}|$ rather than only the binary classification.

**Control.** We ablate 5 random directions (matched norm) per instance and measure the corresponding logit changes. The ratio of parent-ablation effect to random-ablation effect quantifies how specifically the parent direction contributes to model output.

**Scope.** 50 city-continent entities, 30 contexts each, yielding 1,471 false negative instances.

## 3.6 Hedging Decomposition

False negatives can arise from distinct mechanisms. We decompose each FN into three categories to distinguish genuine feature gaps from representational interference.

**Strict absorbed** (FN$_{\text{strict}}$): the main parent feature -- the SAE feature with highest cosine similarity to the parent probe direction $\mathbf{w}_p$ -- does not fire ($z_{\text{main}} = 0$). This indicates a genuine gap in the SAE dictionary: no feature captures the parent concept for this input.

**Compensatory** (FN$_{\text{comp}}$): the main parent feature fires ($z_{\text{main}} > 0$), but other features in the SAE reconstruction interfere with the probe's prediction. The parent information is present in the SAE latent space but the aggregate reconstruction distorts the activation enough to flip the probe's output.

**Persistent** (FN$_{\text{persist}}$): residual false negatives that fit neither category, typically arising from probe boundary effects on marginal inputs.

**Multi-$L_0$ analysis.** For first-letter, we extend the decomposition across SAEs with $L_0$ values ranging from 22 to 176 (8 Gemma Scope JumpReLU configurations at layers 6, 12, 18, 24 with two dictionary widths each). This reveals how the strict-to-compensatory ratio shifts as the SAE allocates more active features per input.

**Statistics.** Cross-hierarchy variation in the decomposition is tested with the chi-square test on contingency tables (hierarchy $\times$ decomposition category), with Fisher's exact test for pairwise comparisons and Bonferroni correction.

## 3.7 Architecture Comparison

We compare four SAE architectures on their absorption rates: JumpReLU 16k, JumpReLU 65k (Gemma Scope), BatchTopK 16k, and Matryoshka 32k (SAEBench). All four architectures are available at layer 12; at layer 24, only JumpReLU (both widths) and Matryoshka are available.

The analysis uses a two-factor design: architecture $\times$ hierarchy, with absorption rate as the dependent variable. We report Kruskal-Wallis tests separately for the architecture factor and the hierarchy factor at each layer. The width mismatch between Matryoshka ($m = 32{,}768$) and the other architectures ($m = 16{,}384$ or $65{,}536$) is documented as a confound.

## 3.8 Implementation

All experiments run on Gemma 2 2B (`google/gemma-2-2b`) loaded via TransformerLens with pre-trained SAEs from Gemma Scope and SAEBench loaded via SAELens. Inference uses a single GPU with $\geq 24$ GB VRAM. Random seed is fixed at 42 for all stochastic operations. Activation caching and probe training require approximately 10 GB of storage.

Code for all experiments, including probe training, absorption measurement, activation patching, benign/pathological diagnostic, and hedging decomposition, will be released publicly upon publication.

<!-- FIGURES
- Table 1: gen_table1_probe_quality.py, table1_probe_quality.pdf — Probe quality (weighted F1) heatmap across 4 hierarchies x 4 layers with quality gate annotations
-->
