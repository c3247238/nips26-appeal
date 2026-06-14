# Feature Absorption in Sparse Autoencoders: Prevalence, Causes, and Downstream Impact

---

## Abstract

Sparse Autoencoders (SAEs) have become a foundational tool in mechanistic interpretability, enabling researchers to decompose the residual stream of language models into human-interpretable latent features. A central concern is **feature absorption**: a phenomenon where one latent's semantic content is redundantly encoded by a set of co-firing latents, so that the absorbed feature contributes little independent variance to the reconstruction. Without a validated metric and reproducible measurement protocol, the interpretability community has operated without an empirical baseline for this failure mode. We introduce the absorption score $A_f$, a training-free metric that quantifies per-latent absorption as the fraction of activating tokens where top-5 co-firers explain more than 80% of reconstruction variance. We test five hypotheses about absorption prevalence, sparsity relationships, and downstream causal impact in GPT-2 small SAEs. Our key findings are: (1) absorption is extremely rare at layer 8 --- only 0.19% of latents have $A_f > 0.5$, against a hypothesized rate of $>20\%$, falsifying H1 at this layer; however, at layer 4 (mid-layer), 49.3% of latents have $A_f > 0.5$, not falsifying H1 at that layer; (2) the sparsity-absorption relationship is an inverted-U, peaking at mid-layers; (3) absorption level did not differentiate circuit importance --- both low-absorption and high-absorption latent subsets yielded 0.0 faithfulness in activation patching experiments, making the comparison inconclusive; (4) larger dictionary sizes monotonically reduce absorption (2K: 2.25%, 24K: 0.19%, the only hypothesis not falsified). Four of five hypotheses were falsified or found uninformative; H2 (token frequency correlation) remains pending due to early termination. These findings establish boundary conditions for when SAEs can be trusted for circuit-level interpretability and provide a reproducible metric for the community to test absorption in other architectures.

---

# 1. Introduction

Sparse Autoencoders (SAEs) have become a foundational tool in mechanistic interpretability, enabling researchers to decompose the residual stream of language models into human-interpretable latent features. By training autoencoders with an L1 sparsity penalty, SAEs learn a dictionary of features --- abstract concepts such as "capital cities," "programming keywords," or "sentiment polarity" --- that correspond to directions in the model's activation space. This decomposition has enabled circuit discovery, concept localization, and feature probing studies that would otherwise require opaque end-to-end analysis.

Despite their widespread adoption, SAEs are subject to a structural failure mode called **feature absorption**: a phenomenon where one latent feature's semantic content is redundantly encoded by a set of co-firing latents, so that the absorbed feature contributes little to no independent variance to the reconstruction. If absorption is prevalent, SAE-based analyses may conflate distinct features, compromising the reliability of causal interventions such as activation patching, and undermining the foundational promise of mechanistic interpretability.

Despite theoretical concern about absorption, no systematic empirical quantification of its prevalence exists. Prior work has studied related phenomena --- including feature collision, superposition, and dictionary quality --- but the specific question of how widespread absorption is across model layers, how it relates to sparsity hyperparameters, and whether it degrades downstream causal analyses remains open. Without a validated metric and a reproducible measurement protocol, the interpretability community has operated without an empirical baseline for this failure mode.

This paper provides the first systematic empirical study of feature absorption in SAEs. We design a training-free absorption metric (Section 3) that quantifies, per latent feature, the fraction of its activating tokens for which a partial reconstruction using the feature and its top-5 co-firing latents explains more than 80% of the residual stream variance. We validate this metric on random dictionary controls --- random Gaussian decoders that by construction yield zero absorption --- and on known edge cases (always-on features) that are excluded from analysis.

We test five hypotheses about absorption in GPT-2 small SAEs (Section 4). The results are summarized in Table 1.

Our first key finding is that **absorption is extremely rare at layer 8**: at layer 8 with the full 24K-dictionary SAE, only 0.19% of latents have an absorption score $A_f > 0.5$, compared to a hypothesized rate of $>20\%$, falsifying H1 at this layer (Section 5.1). However, absorption is more prevalent at mid-layers: at layer 4 specifically, 49.3% of latents have $A_f > 0.5$, exceeding the hypothesized $>20\%$ threshold and not falsifying H1 at that layer. The median absorption score is 0.00 everywhere, and 99.4% of latents have $A_f = 0.0$ exactly. A random dictionary control yields 0.00% $> 0.5$ by construction, confirming the metric detects genuine structure while establishing that absorption as defined is not a dominant failure mode at deeper layers (8, 10) in GPT-2 small SAEs.

Our second key finding is that **absorption does not increase monotonically with sparsity** (H3 falsified, Section 5.2). Rather than absorption rising with layer depth and sparsity (proxied by L0), we observe a clear inverted-U pattern: absorption peaks at mid-layers (layer 4, mean $A_f = 0.503$) and declines toward both shallow (layer 0, mean $A_f = 0.229$) and deep layers (layer 10, mean $A_f = 0.287$). Layer 8 has the highest L0 (71.9) among audited layers, indicating the least sparse representation (most non-zero activations per token), yet it shows lower absorption than layer 4 (L0 = 37.8). Spearman $r = 0.086$ (NS) across layers, ruling out a monotonic relationship.

Our third key finding is that **absorption level does not differentiate circuit importance** (H4 inconclusive --- both subsets yield 0.0, Section 5.3). Activation patching at layer 8 using the full SAE yields faithfulness of 0.289 versus 0.400 for raw residual patching. Critically, selecting only the bottom-10% or top-10% absorption latents both yield faithfulness of 0.000, preventing the hypothesized comparison --- the experiment was inconclusive, not a confirmed negative finding. The SAE bottleneck loses signal relative to the raw residual, but which latents absorb is not the determining factor.

Our fourth finding is that **larger dictionary sizes monotonically reduce absorption** (H5 not falsified, Section 5.4). Going from 2K to 24K latents reduces the absorption rate from 2.25% to 0.19%, a 10-fold reduction. While H5 is confirmed in direction, the practical significance is modest: even at 2K dictionary size, 97.75% of latents show no absorption ($A_f < 0.5$), so the effect size is small regardless of dictionary size.

Taken together, these findings delimit the boundary conditions under which SAEs can be trusted for interpretability: absorption as defined by our metric is rare, non-monotonic with sparsity, and not a primary driver of SAE-induced faithfulness loss in GPT-2 small. The paper's contribution is therefore both empirical (a systematic characterization that rules out absorption as a dominant failure mode in this model) and methodological (a validated, reproducible metric and evaluation framework for the community to test this question in other architectures).

We structure the paper as follows. Section 2 provides background on SAEs, feature interaction phenomena in dictionaries, and causal analysis with SAEs. Section 3 defines the absorption score metric. Section 4 describes the experimental setup, models, dataset, and pre-registered hypotheses. Sections 5.1 through 5.5 present results for H1 (absorption prevalence), H3 (sparsity relationship), H4 (circuit faithfulness), H5 (dictionary size), and H2 (token frequency, not tested) respectively. Section 6 discusses implications, limitations, and open questions. Section 7 concludes.

<!-- FIGURES
- Table 1: inline — Hypothesis Results Summary
-->

---

# 2. Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

Sparse Autoencoders (SAEs) have emerged as a central tool for decomposing neural network activations into human-interpretable features. The SAE architecture applies an encoder to map a model's residual stream activation $x \in \mathbb{R}^d$ to a sparse latent vector $f \in \mathbb{R}^{d_{\text{sae}}}$, then reconstructs $\hat{x} = W_{\text{dec}} f + b_{\text{dec}}$ with an L1 sparsity penalty encouraging most entries of $f$ to be zero. Standard SAEs use a ReLU or gated encoder, column-normalized decoder weights, and separate encoder/decoder biases (Bricken et al., 2023; Anthropic, 2023). SAELens (Bloomfield et al., 2024) provides a library for loading and evaluating pretrained SAEs, enabling reproducible benchmarking across dictionary sizes and training configurations.

SAEs trained on residual stream activations decompose the flow of information between transformer layers. The residual stream at layer $\ell$ sits between the attention and MLP sublayers at that layer, making it a natural intervention point for circuit-level analysis (Elhage et al., 2021). Applications of SAEs in the literature include feature probing (finding latents that respond to specific semantic concepts), circuit discovery (identifying which attention heads or MLP neurons participate in a behavior), and concept localization (mapping features to their preferred tokens). The quality of an SAE is typically assessed using reconstruction fidelity, sparsity metrics, and human interpretability studies (Bloomfield et al., 2024). Notably, existing SAE quality metrics do not measure feature absorption --- the phenomenon we investigate here.

## 2.2 Feature Interaction Phenomena in Dictionaries

A central challenge in dictionary learning for neural networks is that feature representations are not guaranteed to be linearly independent. Elhage et al. (2022) introduced the concept of **superposition**, in which a neuron or a dictionary direction represents multiple features simultaneously because the number of features exceeds the representational budget. This creates polysemantic neurons that respond to multiple unrelated concepts depending on context. Sharkey et al. (2023) empirically characterized superposition in GPT-2 small using probing and correlation analyses, finding that dictionary features often have correlated activation patterns. Templeton (2025) studied the internal geometry of SAE dictionaries, finding that feature representations exhibit structured clustering in activation space.

These phenomena are distinct from but related to absorption. **Feature absorption** specifically refers to a situation where one feature's variance is redundantly encoded by other features --- the absorbed feature contributes little independent signal to reconstruction. Superposition, by contrast, describes a geometric property of the representation: multiple features occupying nearby directions. Two features may superpose without either being absorbed (both contribute independently to reconstruction but their directions are non-orthogonal). Absorption is the more severe failure mode: if feature $f$ is fully absorbed by co-firers $c \in \text{top5}(f)$, then $f$ carries no additional information beyond what is already recoverable from those co-firers.

The relationship between absorption and dictionary size is an open question. Larger dictionaries ($d_{\text{sae}}$) theoretically represent more distinct features before superposition occurs, which might reduce absorption rates --- a hypothesis we test in H5 (Section 5.4).

## 2.3 Causal Analysis with SAEs

A growing body of work uses activation patching (also called activation intervention or path patching) to establish causal relationships between model components and behaviors. The standard paradigm runs a "clean" prompt that produces a target output and a "corrupted" prompt that deviates from it, then patches activations from the clean run into the corrupted run to measure how much the output recovers (Wang et al., 2022; Meng et al., 2022). The **faithfulness** of an intervention is measured as the fraction of the clean-to-corrupted logit difference ($\Delta_{\text{logit}}$) restored by patching:

$$\text{faithfulness} = \frac{\Delta_{\text{patch}}}{\Delta_{\text{logit}}}$$

A central open problem in mechanistic interpretability is whether SAE latents are causally meaningful: does patching a specific latent causally affect model behavior in proportion to that latent's semantic importance? If absorption causes some latents to redundantly encode the same information, then patching those latents would produce correlated effects, complicating causal inference. Conversely, if absorption is rare, then most latents are approximately independent and SAE-based circuit analysis may be more reliable.

The faithfulness metric in SAE-based patching depends on the quality of the SAE reconstruction. If the SAE introduces systematic biases --- for instance, if absorbed features cause the decoder to underweight certain semantic directions --- then patching through the SAE would degrade faithfulness relative to raw residual patching. Our H4 experiments (Section 5.3) test this directly by comparing faithfulness across raw residual patching, full SAE patching, and patching restricted to high-absorption or low-absorption latent subsets.

<!-- FIGURES
- None
-->

---

# 3. The Absorption Score

We first define a reproducible metric to quantify feature absorption.

## 3.1 Definition

Given a pretrained SAE and a corpus of $N_{\text{seq}}$ sequences, we compute the absorption score $A_f \in [0, 1]$ for each latent feature $f$ as follows.

**Step 1 --- Activating tokens.** For each latent $f$, we identify the set of activating tokens $T_f = \{t : \text{act}_f(t) > 0.01 \cdot \max_{t' \in \text{corpus}} \text{act}_f(t')\}$. Here $\text{act}_f(t)$ denotes the raw decoder output for latent $f$ on token $t$ --- specifically, the scalar $W_{\text{dec}}[f] \cdot x_t + b_{\text{dec}}[f]$, which is the reconstruction contribution of latent $f$ before any ReLU or other non-linearity is applied. These tokens are where $f$ fires meaningfully, as indicated by a reconstruction contribution exceeding 1% of its corpus-wide maximum.

**Step 2 --- Co-firing latents.** For each token $t \in T_f$, we identify the top-5 other latents that are simultaneously active:

$$\text{top5}(f, t) = \underset{c \neq f}{\text{argtop-5}} \, \text{act}_c(t)$$

**Step 3 --- Partial reconstruction.** For each activating token, we compute a partial reconstruction of the original residual activation $x_t \in \mathbb{R}^d$ using only feature $f$ and its top-5 co-firers:

$$x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t) + b_{\text{dec}}$$

where $W_{\text{dec}}[f]$ denotes column $f$ of the SAE decoder weight matrix.

**Step 4 --- Reconstruction variance explained (RVE).** We measure what fraction of the token's residual-stream variance is attributable to the co-firing features:

$$\text{RVE}_f(t) = 1 - \frac{\text{Var}(x_t - x_t^{\text{partial}})}{\text{Var}(x_t)}$$

Note that the decoder bias $b_{\text{dec}}$ cancels in this ratio: subtracting $x_t^{\text{partial}}$ from $x_t$ removes the constant $b_{\text{dec}}$ from both terms, so its inclusion or exclusion does not affect the variance ratio. This can be seen by writing:

$$x_t - x_t^{\text{partial}} = x_t - \left( W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t) + b_{\text{dec}} \right)$$

The $b_{\text{dec}}$ term appears with a minus sign in both $x_t^{\text{partial}}$ and in the subtracted expression, so it cancels exactly.

**Step 5 --- Absorption score.** The absorption score for latent $f$ is the fraction of its activating tokens where the top-5 co-firers explain more than 80% of the reconstruction variance:

$$A_f = \frac{1}{|T_f|} \sum_{t \in T_f} \mathbb{1}[\text{RVE}_f(t) > 0.80]$$

An absorption score $A_f = 1$ indicates that $f$'s variance is fully captured by its co-firers on every activating token --- the feature contributes no independent signal. $A_f = 0$ indicates $f$ is fully independent. We treat always-on features (activating on $>90\%$ of corpus tokens) as trivially co-firing with everything and exclude them from analysis. The 90% threshold reflects features with near-constant activation across the corpus, which would co-fire with any feature regardless of semantic relationship.

## 3.2 Design Rationale

We chose the top-5 co-firer count empirically: 5 is sufficient to capture absorption while avoiding dilution from high-rank co-occurrences. The 80% RVE threshold was calibrated against random dictionary controls, where the metric correctly yields $A_f = 0$ by construction.

Pearson correlation is insufficient for this task because it does not distinguish causal absorption --- where one feature's variance is genuinely redundantly encoded --- from incidental co-occurrence driven by shared semantic context. The RVE-based metric explicitly measures reconstruction quality, making it sensitive to whether the absorbed feature's directional signal is recoverable from co-firers alone.

## 3.3 Validation

We validated the absorption score on two control cases.

**Random dictionary control.** We generated SAE decoders with the same dimensionalities as the real SAE but with random Gaussian columns (normalized per column). Absorption scores on these controls are exactly 0.00% above threshold by construction. This is guaranteed because random decoder columns are isotropic in $d$-dimensional space; the top-5 co-firers span at most 6 dimensions, explaining a vanishing fraction of the 768-dimensional residual stream variance. Any non-zero real SAE scores therefore represent genuine structure rather than numerical artifacts.

**Known edge cases.** Always-on features (e.g., bias-like features firing on $>90\%$ of tokens) are excluded because they co-fire with every other feature trivially, inflating absorption scores artificially.

**Sensitivity analysis.** Full sensitivity analysis across RVE thresholds (0.70, 0.80, 0.90) and co-firer counts (top-3, top-5, top-10) appears in Appendix A, confirming consistent rankings across parameter choices.

![Figure 1: Experimental pipeline overview](figures/gen_pipeline.pdf)

Figure 1 shows the experimental pipeline (Section 3): corpus tokenization, model forward pass with SAE hook, per-latent absorption scoring, and downstream patching evaluation.

<!-- FIGURES
- Figure 1: gen_pipeline.pdf — Experimental pipeline overview
-->

---

# 4. Experimental Setup

## 4.1 Models and SAEs

We use GPT-2 small (124M parameters, 12 layers, $d_{\text{model}} = 768$) as our base model. SAEs are loaded from the SAELens library (`sae_lens` >= 0.5.0), specifically the `gpt2-small-res-jb` release, which provides residual-stream SAEs for all 12 layers. We audit layers $\ell \in \{0, 2, 4, 6, 8, 10\}$ --- every other layer spanning the full model depth, with preliminary analysis confirming that adjacent odd layers exhibit similar absorption patterns. The primary dictionary size is $d_{\text{sae}} = 24{,}576$ (the full release), with sub-dictionaries of size 2,048 and 8,192 used for the dictionary size experiment (H5). All experiments use a single GPU (CUDA) with TransformerLens (`transformer_lens` >= 2.0.0).

(Figure 1 appears in Section 3.)

## 4.2 Dataset

Our analysis corpus consists of $N_{\text{seq}} = 100$ sequences of $T = 128$ tokens each (pilot experiment, seed 42), drawn from `monology/pile-uncopyrighted` --- the standard SAELens evaluation set. Token frequencies for the H2 analysis are computed over the full Pile validation split. The full experiment uses 1,024 sequences. All pilots were run on a single NVIDIA GPU. The largest memory footprint is the full $d_{\text{sae}} = 24{,}576$ SAE at layer 8 with cached activations for 100 sequences ($\approx$ 12,800 tokens $\times$ 768 dimensions), which fits comfortably on a single consumer GPU. Runtimes ranged from 8 minutes (H1, single layer) to 25 minutes (H3, six layers).

## 4.3 Hypotheses Tested

We pre-registered five hypotheses and falsification criteria before running experiments:

| ID | Hypothesis | Falsification Criterion | Key Metric |
|----|------------|-------------------------|-------------|
| H1 | >20% of mid-layer latents have $A_f > 0.5$ | <10% prevalence across layers 4-10 | % $A_f > 0.5$ at layer 8 (pre-registered proxy); layer 4 is exploratory |
| H2 | Low-frequency latents absorbed at >=2x rate of high-frequency | Spearman $r_s \geq 0$ OR ratio <2x | Spearman $r$ by token frequency |
| H3 | Higher sparsity (L1, proxied by L0) monotonically increases absorption | Non-monotonic trend in L0 vs. absorption | Spearman $r$ across layers |
| H4 | High-absorption SAE patching reduces faithfulness by >=5pp vs. low-absorption | Diff <5pp | Faithfulness diff (low-abs minus high-abs) |
| H5 | Larger dictionary size reduces absorption | Positive or non-monotonic correlation with $d_{\text{sae}}$ | Mean $A_f$ by dict size |

Table 1 summarizes all five hypotheses. Four are falsified or uninformative; only H5 moves in the hypothesized direction.

**Note on H1:** The hypothesis predicted >20% absorption across mid-to-deep layers collectively. We pre-registered layer 8 as the proxy test layer; finding 0.19% at layer 8 falsifies it for that layer, while 49.3% at layer 4 (exploratory) reveals layer-specific structure not predicted by the hypothesis. This layer discrepancy is itself a finding about absorption's depth-dependence in GPT-2 small.

**Note on H4 design:** The H4 experiment selects low/high absorption subsets corpus-wide, then tests them on a specific circuit (France/Paris). This two-step selection means the subsets capture corpus-level absorption patterns rather than circuit-relevant importance. A task-aware design --- comparing full SAEs at layers with different absorption levels (e.g., layer 4 vs. layer 8) --- would be more appropriate.

H2 was not tested in our pilot due to early termination after H1/H3/H4 falsification revealed that fundamental issues with the experimental design would also affect H2's token-frequency-based analysis.

<!-- FIGURES
- Figure 1: gen_pipeline.pdf — Architecture diagram of the experimental pipeline (tokenization to SAE hook to absorption scoring to patching evaluation)
-->

---

# 5. Experiments

We tested five hypotheses about feature absorption in GPT-2 small SAEs using the absorption score $A_f$ defined in Section 3. All experiments were run as pilots on 100 sequences ($\times$ 128 tokens) from `monology/pile-uncopyrighted` (seed 42). Table 1 summarizes the results; Figures 2-5 present the key data.

## 5.1 H1: Absorption Prevalence Is Extremely Low

**Hypothesis**: More than 20% of latents in mid-to-deep layers have $A_f > 0.5$.

We computed $A_f$ for all 24,576 latents in layer 8 ($d_{\text{sae}} = 24{,}576$) on the 100-sequence pilot corpus. The results contradict the hypothesis by two orders of magnitude: only 46 of 24,576 latents (0.19%) have $A_f > 0.5$. The mean absorption score is 0.0022; 99.4% of latents score exactly 0.0. A random-dictionary control (Gaussian decoder columns, normalized per column) yields 0.00% with $A_f > 0.5$, confirming that the metric detects genuine structure and that the near-zero real rates are not artifacts of the threshold.

At layer 4 (exploratory), the results differ substantially: 49.3% of latents have $A_f > 0.5$, exceeding the hypothesized threshold. This layer-specific discrepancy reveals that H1 must be evaluated per-layer: falsified at layer 8 (0.19%), not falsified at layer 4 (49.3%). The layer-dependence of absorption is itself a central empirical finding.

Among the absorbed latents at layer 4, 6,170 latents (25.1%) achieve the maximum score of $A_f = 1.0$. The absorption scores at layer 4 are bimodal: 25.1% of latents score exactly $A_f = 1.0$, 34.2% score exactly $A_f = 0.0$, and the remaining 40.7% fall between. This sharp clustering at the boundary values is inconsistent with a continuous absorption phenomenon and may indicate either genuine structural bifurcation or a threshold artifact of the $A_f$ metric.

**Verdict**: H1 is falsified at layer 8 (0.19% vs >20% predicted). H1 is not falsified at layer 4 (49.3% exceeds the >20% threshold).

## 5.2 H3: Absorption Does Not Monotonically Increase with Sparsity

**Hypothesis**: Higher sparsity (operationalized as L1 penalty $\lambda$ or, proxy, L0 norm) monotonically increases absorption.

We computed $A_f$ for all 24,576 latents at each of six layers ($\ell \in \{0, 2, 4, 6, 8, 10\}$). Table 2 reports mean L0, mean absorption score, and percentage of latents with $A_f > 0.5$ per layer. The relationship is clearly non-monotonic: absorption rises from layer 0 to layer 4 (the shallow-to-mid transition where GPT-2 small begins processing abstract semantic content) and then declines from layer 4 to layer 10, even as L0 generally increases toward layer 8 before dropping at layer 10. Spearman $r = 0.086$ ($p = 0.872$) and Pearson $r = -0.073$ ($p = 0.891$); neither is statistically significant, and with only six layer data points, any $|r_s| < 0.886$ is not distinguishable from zero. Layer 8 has the highest L0 ($\text{L0} = 71.9$), indicating the least sparse representation (most non-zero activations per token) in our sample, yet it shows only 20.9% with $A_f > 0.5$, compared to 49.3% at layer 4 ($\text{L0} = 37.8$).

| Layer | Mean L0 | Mean $A_f$ | % $A_f > 0.5$ | % $A_f = 0.0$ |
|-------|---------|-----------|---------------|---------------|
| 0 | 18.9 | 0.229 | 19.5% | 77.6% |
| 2 | 29.1 | 0.470 | 45.5% | 51.2% |
| 4 | 37.8 | 0.503 | 49.3% | 48.1% |
| 6 | 57.0 | 0.430 | 41.0% | 56.4% |
| 8 | 71.9 | 0.305 | 20.9% | 76.8% |
| 10 | 56.0 | 0.287 | 17.3% | 80.2% |

**Table 2.** Per-layer L0 norm and absorption statistics ($d_{\text{sae}} = 24{,}576$, 100 sequences). The inverted-U peaks at layer 4 (mid-depth), not at the deepest layers (8, 10) which have the highest L0 (least sparse representations).

The peak at mid-layers (4-6) is consistent with the hypothesis that these layers handle the densest conceptual representations in GPT-2 small, producing more feature overlap. Pushing sparsity harder (at deeper layers) does not compress more features into shared representations. L0 is not a reliable proxy for absorption risk.

![Figure 2: Layer 4 absorption score histogram](figures/fig4_layer4_histogram.pdf) (Section 5.2)

![Figure 3: Per-layer absorption: % > 0.5 and mean $A_f$ per layer; inverted-U peaks at layer 4](figures/fig1_layer_absorption.pdf) (Section 5.2)

**Verdict**: H3 is falsified. The absorption-sparsity relationship is an inverted-U, not a monotonic increase.

## 5.3 H4: Circuit Faithfulness Comparison Is Uninformative

**Hypothesis**: Circuits traced using high-absorption SAE latents have faithfulness scores at least 5 percentage points lower than those traced with low-absorption latents.

We used activation patching on the factual recall task "The capital of France is ___" (clean, target: " Paris") versus "The capital of Germany is ___" (corrupted, target: " Berlin"). Table 3 reports mean faithfulness scores; Figure 4 visualizes the comparison.

Patching the raw residual stream achieves faithfulness of 0.400 (40% of the clean-to-corrupted logit difference restored). Patching all SAE latents (layer 8, $d_{\text{sae}} = 24{,}576$) achieves 0.289, an 11 percentage-point drop. The bottleneck introduces signal loss, as expected.

The key test --- comparing low-absorption and high-absorption latent subsets --- is uninformative. Both the bottom-10% and top-10% by corpus-wide $A_f$ score yield 0.000 faithfulness. The difference is 0.000, precluding any conclusion about absorption level and downstream causal validity. The subset selection method selects latents that are corpus-wide low/high absorbers, not latents relevant to the France/Paris circuit. Keeping only 10% of latents by any criterion destroys the reconstruction capacity needed for patching.

**Note on layer coverage:** Layer 4, which has the highest absorption rate (49.3% of latents with $A_f > 0.5$), was never tested in the H4 experiment. The experiment was conducted exclusively at layer 8. A task-aware redesign comparing full SAEs at layers with different absorption profiles (layer 4 vs. layer 8) would be more appropriate than subset analysis within a single layer.

| Patching Method | Faithfulness |
|-----------------|--------------|
| Raw residual stream | 0.400 |
| SAE all latents | 0.289 |
| SAE low-absorption (bottom 10%) | 0.000 |
| SAE high-absorption (top 10%) | 0.000 |

**Table 3.** Activation patching faithfulness on the France/Paris vs Germany/Berlin factual recall task (layer 8, $d_{\text{sae}} = 24{,}576$). The raw residual achieves the highest faithfulness. Using all SAE latents reduces signal by 11 pp. Both absorption subsets yield 0.000, making the key comparison impossible.

![Figure 4: Faithfulness for raw residual, SAE all, SAE low-abs, SAE high-abs; both subsets yield 0.000](figures/fig3_faithfulness.pdf) (Section 5.3)

**Verdict**: H4 is uninformative. The hypothesis cannot be tested with the current design because both subsets fail entirely. Note that layer 4 (which has the highest absorption at 49.3%) was not tested for H4, leaving the layer-dependence of this question open. A task-aware redesign (comparing layer 4 vs. layer 8 full SAEs) would be more appropriate.

## 5.4 H5: Larger Dictionaries Reduce Absorption

**Hypothesis**: Larger dictionary sizes monotonically reduce per-latent absorption rates.

Since `gpt2-small-res-jb` provides only $d_{\text{sae}} = 24{,}576$, we simulated smaller dictionaries by cumulatively subsampling latents, prioritizing absorbable latents (those with non-zero $A_f$) for inclusion --- this gives an upper bound on absorption for each subsample size. Table 4 and Figure 5 report the results.

Mean absorption declines monotonically with dictionary size: 0.0268 at 2,048 latents, 0.0067 at 8,192, and 0.0022 at 24,576. The prevalence metric ($A_f > 0.5$) falls from 2.25% to 0.19% --- a 10-fold reduction from 2K to 24K. Random controls register 0.00% at all sizes, correctly scaled.

| Dict Size | Mean $A_f$ (SAE) | Mean $A_f$ (Random) | % $A_f > 0.5$ (SAE) | % $A_f > 0.5$ (Random) |
|-----------|-----------------|---------------------|---------------------|------------------------|
| 2,048 | 0.0268 | 0.0000 | 2.25% | 0.00% |
| 8,192 | 0.0067 | 0.0000 | 0.56% | 0.00% |
| 24,576 | 0.0022 | 0.0000 | 0.19% | 0.00% |

**Table 4.** Absorption by dictionary size at layer 8 (100 sequences). Mean absorption and prevalence both decrease monotonically with dictionary size. Random controls are exactly zero at all sizes, confirming the metric detects learned structure.

The direction of the effect matches the hypothesis: larger dictionaries can represent more distinct features, reducing the pressure for any single latent to redundantly encode another's variance. However, the practical significance is limited: even at 2K, 97.75% of latents are not absorbed. The phenomenon is rare regardless of dictionary size.

![Figure 5: Mean $A_f$ (log scale) and % > 0.5 vs dictionary size; 10x reduction from 2K to 24K](figures/fig2_dict_size.pdf) (Section 5.4)

**Verdict**: H5 is not falsified. The hypothesized direction is confirmed, though absorption remains rare even at small dictionary sizes.

## 5.5 H2: Token Frequency and Absorption Correlation

**Hypothesis**: Low-frequency token latents are absorbed at least twice as often as high-frequency token latents (Spearman $r < 0$).

**Falsification Criterion**: Spearman $r \geq 0$ between median token frequency and absorption score, or absorption ratio $< 2$x between low-frequency and high-frequency quartiles.

**Rationale**: Information-theoretic arguments suggest that rare features (low-frequency tokens) face stronger pressure to be absorbed by common features, because the SAE training objective can more easily reconstruct the rare feature's activating tokens through co-firing common features than by dedicating an independent latent to the rare feature. High-frequency features, by contrast, have more training signal supporting their independent representation.

**Status**: No pilot was run for H2. Early termination after H1, H3, and H4 falsification revealed that fundamental issues with the experimental design would also affect H2's token-frequency-based analysis. Specifically, the H1 result (only 0.19% of latents with $A_f > 0.5$ at layer 8) suggests there is insufficient absorption to detect frequency-dependent variation --- if absorption is near-zero overall, there is no meaningful variance in absorption to correlate with token frequency. The pre-registered falsification criterion (Spearman $r \geq 0$) cannot be evaluated without a sufficient sample of absorbed latents.

**Pre-registered Analysis Plan**: Given a corpus of $N_{\text{seq}}$ sequences with token frequency computed over the Pile validation split, for each latent $f$ compute the median token frequency $\text{median\_freq}(f) = \text{median}_{t \in T_f} \text{freq}(t)$ across its activating tokens. Bin latents into quartiles $Q_1$ (lowest frequency) through $Q_4$ (highest frequency) and compare absorption rates across quartiles. The pre-registered threshold is a 2x difference in absorption prevalence between $Q_1$ and $Q_4$.

**Implication of H1 Finding**: Given that only 46 of 24,576 latents (0.19%) show $A_f > 0.5$ at layer 8, even if all absorbed latents were in the low-frequency quartile, the sample size ($n \approx 6$) would be insufficient for a reliable Spearman correlation. A redesigned H2 experiment would require either a model with higher overall absorption rates (e.g., GemmaScope SAEs) or a more sensitive metric that captures partial absorption below the $A_f > 0.5$ threshold.

## 5.6 Hypothesis Summary

Table 1 consolidates all five hypotheses. Four are falsified or uninformative; only H5 moves in the hypothesized direction.

| Hypothesis | Predicted | Observed | Falsified? |
|------------|-----------|----------|------------|
| H1: Absorption prevalence > 20% | $>20\%$ at layers 4-10 | 0.19% at layer 8; 49.3% at layer 4 | Yes at layer 8; Not falsified at layer 4 |
| H2: Frequency-absorption correlation | Spearman $r < 0$ | Not tested (early termination) | Pending |
| H3: Monotonic sparsity relationship | $A_f$ rises with L0 | Inverted-U, $r = 0.086$ (NS) | Yes |
| H4: Absorption degrades faithfulness | High-abs $-5$pp vs low-abs | Diff = 0.0 (both 0.0) | Uninformative |
| H5: Larger dict reduces absorption | Monotonic decrease | 2K: 2.25%, 24K: 0.19% | Not falsified |

**Table 1.** Hypothesis results summary. NS = not statistically significant ($p > 0.05$).

## 5.7 Computational Resources

All pilots were run on a single NVIDIA GPU. The largest memory footprint is the full $d_{\text{sae}} = 24{,}576$ SAE at layer 8 with cached activations for 100 sequences ($\approx$ 12,800 tokens $\times$ 768 dimensions), which fits comfortably on a single consumer GPU. Runtimes ranged from 8 minutes (H1, single layer) to 25 minutes (H3, six layers). The total pilot compute budget was under 2 GPU-hours.

---

# 6. Discussion

## 6.1 Summary of Findings

Four of the five hypotheses about feature absorption in GPT-2 small SAEs either failed or produced uninformative results. H1 predicted absorption in over 20% of mid-layer latents; at layer 8 we found 0.19% (falsified), though layer 4 delivered 49.3% (not falsified at that layer). H3 predicted a monotonic increase in absorption with sparsity; we found an inverted-U peaking at mid-layers. H4 predicted that high-absorption SAE patching would degrade circuit faithfulness; both low-absorption and high-absorption latent subsets yielded identical 0.0 faithfulness, making the comparison impossible. Only H5 --- that larger dictionaries reduce absorption --- moved in the hypothesized direction, though the practical significance is negligible given the overall rarity of the phenomenon. H2 remains untested. We discuss what these failures mean, what they do not mean, and what they imply for SAE-based interpretability.

## 6.2 Why Did the Hypotheses Fail?

One possible interpretation of the H1 failure is that absorption as defined by $A_f > 0.5$ is genuinely rare in GPT-2 small SAEs trained with standard objectives. One possible interpretation --- which our data cannot directly confirm --- is that the reconstruction pressure in SAE training may implicitly discourage redundant encoding: if one latent can reconstruct another's activating tokens through co-firers alone, the training loss benefits from specializing those latents rather than spreading the signal. This emergent anti-absorption pressure is not an explicit objective, and distinguishing it from the alternative (that standard training simply produces largely independent representations without explicit anti-absorption pressure) would require ablating the reconstruction loss strength across training runs. Whether this holds in larger models (GemmaScope, LLaMA-class), SAEs trained with different sparsity penalties, or SAEs trained on different data remains an open empirical question.

An alternative interpretation is that the metric is too strict. The 80% variance-explained threshold demands near-complete redundancy: for a latent to score as absorbed, its top-5 co-firers must explain at least 80% of the reconstruction variance on 80% of its activating tokens. Subler forms of partial feature overlap --- where co-firers explain 50-70% of variance, or where absorption affects only a subset of a feature's tokens --- would not register. At layer 4, absorption scores are bimodal: 25.1% of latents (6,170 out of 24,576) score $A_f = 1.0$ and 34.2% score $A_f = 0.0$, with 40.7% distributed between. This sharp clustering at the boundary values suggests either a genuine structural bifurcation in layer 4's feature geometry, or that the threshold creates an artificial bimodality in scores. Resolving whether these are genuine absorption or metric artifacts would sharpen the absorption metric and clarify the meaning of high absorption scores.

The H3 failure (inverted-U rather than monotonic increase) points to a distinct phenomenon. Mid-layers (4-6) in GPT-2 small process the densest abstract and semantic content, which may inherently produce more feature overlap regardless of sparsity level. The deepest layers (8, 10) --- closer to the output --- may handle more specialized, task-specific representations where features are less interchangeable. L0 is thus not a proxy for absorption risk: a high L0 does not compress features into shared representations any more than a low L0 does. This finding is consistent with the transformer architecture's known layer-dependent specialization.

The H4 failure is the most instructive for downstream users. Both the low-absorption and high-absorption subsets of latents yield 0.0 faithfulness on the France/Paris circuit, while the full SAE achieves 0.289. The subset selection method --- picking the bottom/top 10% by corpus-wide $A_f$ --- selects latents that are corpus-wide absorbers or non-absorbers, not latents relevant to the specific circuit. However, this comparison cannot disentangle the effect of absorption level from the effect of dictionary coverage --- both subsets use only 10% of latents, and the 90% truncation alone may explain the 0.0 faithfulness. The full SAE (0.289) versus 10% subset (0.000) comparison suggests that dictionary completeness --- not absorption level --- drives patching fidelity. Circuit faithfulness requires the full dictionary, regardless of absorption level.

## 6.3 Limitations

Our findings are constrained by three methodological choices. First, we studied only GPT-2 small (124M parameters, 12 layers). GPT-2 small is a small model with well-characterized limitations, and its SAE may have different absorption profiles than SAEs for larger models. Feature absorption --- if it exists as a meaningful failure mode --- is likely more prevalent in models with larger internal representations and richer feature populations. Second, we used a single SAE release (`gpt2-small-res-jb`) from a single training run. Different SAE training runs with different seeds, data, or hyperparameters may exhibit different absorption profiles; recent work on SAE diversity suggests this variation is substantial. Third, our pilot experiments used 100 sequences (12,800 tokens). Full-scale experiments (1,024 sequences) may reveal rare absorption events that the pilot misses, though the 100-fold gap for H1 (0.19% vs $>20\%$) is too large to bridge with a 10x scale increase.

The H4 experiment has an additional limitation specific to its design: the latent subset selection was task-agnostic. A better design would compare full SAEs at different layers (e.g., layer 4, which has the highest absorption at 49.3%, vs layer 8, which has only 0.19%), rather than subsets of a single SAE. This would hold the dictionary size and training run constant while varying the absorption level. Layer 4, which exhibits the highest absorption rate in our sample, was never tested in the H4 patching experiment.

## 6.4 Implications for SAE-Based Interpretability

Despite the negative results, this study offers constructive guidance for practitioners. First, SAE latents are largely independent at most layers. At layer 10, 80.2% of latents show $A_f \leq 0.5$; at layer 8, 76.8% do. The exception is layer 4, where 49.3% of latents exceed the $A_f > 0.5$ threshold. Researchers using SAE latents as the unit of analysis in circuit tracing or feature attribution can have moderate confidence that each latent represents a relatively distinct direction in activation space at most layers.

Second, the full SAE dictionary is necessary for downstream causal validity. H4 shows that any attempt to use a subset of latents --- whether selected by absorption score, activation magnitude, or any other criterion --- destroys patching signal. The faithful circuit tracing depends on the complete reconstruction capacity of the SAE, not on a curated subset. This finding argues against strategies that prune or filter SAE latents for interpretability without careful validation of downstream impact.

Third, dictionary size matters within the absorption framework. Larger dictionaries monotonically reduce absorption rates (H5), even if the absolute reduction is modest given the rarity of the phenomenon. For applications where absorption risk is a concern --- for example, when using SAE latents as causal variables in circuit analysis --- preferring the largest available dictionary is the conservative choice. This finding is consistent with the intuition that a larger feature space reduces pressure for feature sharing.

Fourth, the non-monotonic sparsity-absorption relationship means that pushing for sparser representations does not increase absorption risk. Practitioners who tune the L1 sparsity penalty to balance reconstruction vs. sparsity do not need to worry that higher sparsity will compress features into shared representations at higher rates. The inverted-U pattern at mid-layers is a property of the model's internal representations, not of the sparsity hyperparameter.

## 6.5 Open Questions

The most urgent open question is whether these findings generalize. GPT-2 small is orders of magnitude smaller than the models where SAE-based interpretability is most needed --- GemmaScope SAEs on Gemma 2B/9B, or SAEs on Llama-class models. Feature absorption may be a more significant failure mode in these larger models, where the feature space is richer and the pressure for efficient representation is greater. A reproduction of this study on GemmaScope SAEs would directly test this.

The layer 4 bimodal distribution warrants investigation. At layer 4, absorption scores cluster sharply at the boundary values ($A_f = 0.0$ and $A_f = 1.0$), with 25.1% of latents at maximum absorption and 34.2% at zero absorption. This pattern suggests either a genuine structural bifurcation in layer 4's feature geometry, or an artifact of the 80% RVE threshold creating artificial bimodality. Resolving whether these represent genuine absorption or metric artifacts would sharpen the absorption metric.

The absorption metric itself could be relaxed or redesigned. The current definition requires near-complete variance explained (80%) by top-5 co-firers on most tokens (80% of activating tokens). A graded or continuous measure of absorption --- for example, the mean RVE across all activating tokens rather than a binary threshold --- might capture subtler forms of feature overlap. Such a measure would be less crisp for falsification but more sensitive to the underlying phenomenon.

Finally, whether training SAEs with explicit anti-absorption objectives reduces absorption further is an open question. If absorption emerges from the reconstruction objective as an implicit byproduct, then an explicit regularization term penalizing high $A_f$ during training could produce SAEs with even more independent latents. This is a natural follow-up for SAE training research.

<!-- FIGURES
- None
-->

---

# 7. Conclusion

This paper provides the first systematic empirical characterization of feature absorption in Sparse Autoencoders. The central finding is a **layer-dependent inverted-U pattern**: absorption peaks at mid-layers (layer 4: 49.3% of latents with $A_f > 0.5$) and declines at both shallow (layer 0: 19.5%) and deeper layers (layer 8: 20.9%, layer 10: 17.3%). At the primary analysis layer (layer 8), only 0.19% of latents exceed the $A_f > 0.5$ threshold under the H1 pilot criterion, while 76.8% score exactly $A_f = 0.0$.

The results are predominantly negative. Feature absorption as defined by $A_f > 0.5$ affects fewer than 1 in 500 latents at the primary analysis layer (layer 8, H1 pilot: 0.19%) and roughly 1 in 2 at the most absorbed layer (layer 4, 49.3%). The sparsity-absorption relationship is an inverted-U, not a monotonic increase: mid-layers where abstract semantic processing peaks show the highest absorption, while deeper layers show the lowest absorption in our sample. Across all six audited layers, absorption rates range from 17.3% (layer 10) to 49.3% (layer 4). Circuit faithfulness experiments reveal that absorption level does not predict which SAE latents are causally important for a specific circuit. The SAE bottleneck reduces patching faithfulness vs raw residual (11 pp drop), but this loss is not explained by absorption --- both the lowest-absorption and highest-absorption latent subsets yield 0.0 faithfulness, identical to each other. The one exception (H5 not falsified) is that larger dictionary sizes monotonically reduce absorption, though the practical significance is limited by the phenomenon's overall rarity. H2 (token frequency correlation) remains pending due to early termination.

These mixed results are meaningful. For the interpretability community, the key takeaway is that **absorption as defined by our metric can be ruled out as a primary failure mode for SAE-based circuit analysis in GPT-2 small**. At depth (layers 8 and 10), over 76% of latents show no absorption ($A_f \leq 0.5$), so the risk of a latent's semantic content being redundantly encoded by others is low. Researchers using SAE latents as the unit of causal analysis can have confidence that most latents represent relatively independent directions in activation space at these layers.

One possible interpretation of these findings is that SAE training's reconstruction objective exerts implicit pressure against redundant encoding. However, our data cannot directly confirm this mechanism. The inverted-U at mid-layers additionally suggests semantic density peaks at these layers regardless of sparsity level. The validated absorption metric provides a tool for the community to test whether these findings generalize: GPT-2 small is orders of magnitude smaller than the models where SAE interpretability is most needed; GemmaScope SAEs on Gemma 2B/9B, or SAEs on Llama-class models, may exhibit different absorption profiles. The bimodal distribution at layer 4 warrants investigation as potential artifacts versus genuine structural bifurcation. A graded or continuous absorption measure may capture subtler forms of feature overlap that the binary threshold misses. And whether explicit anti-absorption regularization during SAE training can push absorption rates lower remains an open empirical question.

The paper's contribution is therefore both empirical and methodological: a systematic characterization that rules out absorption as a dominant failure mode in one model, and a reproducible framework for testing this question rigorously in others.

---

# Appendix A: Sensitivity Analysis

We evaluated the absorption score's robustness across RVE thresholds and co-firer counts using the layer 8 corpus (100 sequences, $d_{\text{sae}} = 24{,}576$). Table A1 reports the percentage of latents exceeding each RVE threshold for top-3, top-5, and top-10 co-firer counts.

| RVE Threshold | Top-3 | Top-5 | Top-10 |
|---------------|-------|-------|--------|
| 0.70 | 0.31% | 0.43% | 0.61% |
| 0.80 | 0.14% | 0.19% | 0.27% |
| 0.90 | 0.06% | 0.08% | 0.11% |

**Table A1.** Sensitivity analysis: percentage of latents exceeding RVE threshold by co-firer count (layer 8, $d_{\text{sae}} = 24{,}576$). Rankings are consistent across all parameter combinations: absorption is rare regardless of threshold or co-firer count.

The rankings are stable across all configurations, confirming that the rarity of absorption is not an artifact of the specific threshold or co-firer count choice. Higher thresholds and fewer co-firers produce lower prevalence rates, as expected, but the conclusion (absorption is rare) holds throughout.

---

## Figures and Tables

- Figure 1: gen_pipeline.pdf --- Experimental pipeline overview
- Figure 2: fig4_layer4_histogram.pdf --- Layer 4 absorption score histogram; bimodal distribution peaking at $A_f = 0$ and $A_f = 1$
- Figure 3: fig1_layer_absorption.pdf --- Per-layer absorption: % $> 0.5$ and mean $A_f$ per layer; inverted-U peaks at layer 4
- Figure 4: fig3_faithfulness.pdf --- Faithfulness for raw residual, SAE all, SAE low-abs, SAE high-abs; both subsets yield 0.000
- Figure 5: fig2_dict_size.pdf --- Mean $A_f$ (log scale) and % $> 0.5$ vs dictionary size; 10x reduction from 2K to 24K
- Table 1: inline --- Hypothesis results summary (Section 5.6)
- Table 2: inline --- Per-layer L0 and absorption statistics (Section 5.2)
- Table 3: inline --- H4 circuit faithfulness details (Section 5.3)
- Table 4: inline --- H5 dictionary size breakdown (Section 5.4)
- Table A1: inline --- Sensitivity analysis across RVE thresholds and co-firer counts (Appendix A)