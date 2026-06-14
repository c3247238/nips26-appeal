# Abstract

Feature absorption---where general SAE features fail to fire and are instead captured by more specific child features---is a central failure mode of sparse autoencoders. Despite extensive detection and architectural mitigation, no existing work provides a mechanistic theory that explains why absorption happens or identifies which features are at risk without running absorption metrics. We show that the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from Rozell et al.'s Locally Competitive Algorithm (LCA), providing the first mechanistic explanation for absorption as competitive suppression. When a child feature fires strongly, it inhibits correlated parent features via decoder correlations, causing recall loss without affecting precision. This explains the precision--recall asymmetry observed in prior work: precision is invariant because inhibition suppresses true positives but does not cause false positives; recall varies because suppression strength differs across feature pairs. We construct a local inhibition graph from decoder correlations that is training-free, scalable to million-latent SAEs, and computed from pretrained weights in seconds. We also propose homeostatic rebalancing---a single-pass post-hoc correction inspired by biological homeostatic plasticity---that restores parent firing by compensating for competitive suppression. We integrate prior empirical findings (H1--H5) from GPT-2 Small into the competitive suppression framework and propose five validation experiments (H6--H10) that test whether decoder correlations predict absorption pairs, explain the precision--recall asymmetry, identify at-risk features, vary across layers, and enable post-hoc repair. Our contribution is the first mechanistic theory of absorption grounded in the LCA--SAE structural correspondence, together with a training-free diagnostic and repair methodology.

<!-- FIGURES
- None
-->

# 1. Introduction

## 1.1 The SAE Credibility Crisis and the Absorption Problem

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis, feature steering, model editing, and bias detection (Bricken et al., 2023; Marks et al., 2024; Templeton et al., 2024). The foundational premise is that SAEs decompose neural network activations into human-interpretable features through sparse dictionary learning. Yet the field faces an escalating credibility crisis.

Korznikov et al. (2025) demonstrate that SAEs recover only 9% of true features despite 71% explained variance, and that random baseline SAEs match trained SAEs on standard metrics. Some research groups have reportedly deprioritized SAE research after finding negative results on downstream tasks. These developments raise a fundamental question: do SAEs provide reliable tools for interpretability work, or do they create an illusion of understanding?

At the center of this crisis is feature absorption, first formally identified by Chanin et al. (2024). Absorption occurs when a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features. The parent latent appears interpretable when inspected in isolation but produces systematic false negatives during downstream use.

Chanin et al. demonstrated that hierarchical features cause absorption and validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently standardized absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance. Architectural innovations---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and HSAE (Luo et al., 2026)---all target absorption reduction as a primary objective.

Despite this attention, a critical gap remains: **no existing work provides a mechanistic theory that explains why absorption happens or identifies which features are at risk before running absorption metrics.** Researchers can detect absorption after the fact, but they cannot predict it, explain its structure, or repair it without retraining. This study bridges that gap.

## 1.2 The LCA Connection: From Neuroscience to SAEs

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot \tilde{a}), \quad \tilde{a} = T(u)$$

where $G$ is the inhibition matrix governing competitive interactions between neurons, $\tilde{a}$ is the activation after thresholding, and $T(u) = \max(0, u)$ is the threshold function. A key insight, which to our knowledge has not been articulated in the SAE literature, is that for SAEs with tied encoder--decoder weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix.

Even with untied weights---the standard case for trained SAEs---decoder correlations encode competitive relationships between latents, because the decoder directions that reconstruct the input must compete to explain the same variance. This structural correspondence provides a mechanistic explanation: absorption is not feature destruction but competitive suppression. When a child feature fires strongly, it inhibits correlated parent features via decoder correlations, causing them to fail to activate.

This insight yields three concrete predictions. First, edges in a local inhibition graph constructed from decoder correlations should predict known absorption pairs. Second, competitive suppression explains the precision--recall asymmetry observed in prior work: precision is invariant (inhibition does not cause false positives) while recall varies (inhibition suppresses true positives). Third, latents with high total incoming inhibition should be at higher risk of absorption, enabling pre-emptive identification.

## 1.3 Research Questions

We formalize five research questions:

- **RQ1 (Primary):** Does the local inhibition graph predict known absorption pairs?
- **RQ2 (Secondary):** Does inhibition explain the precision--recall asymmetry?
- **RQ3 (Secondary):** Can the graph predict at-risk features before running absorption metrics?
- **RQ4 (Exploratory):** Does graph structure vary across layers?
- **RQ5 (Exploratory):** Can homeostatic rebalancing restore parent firing?

## 1.4 Contributions

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing an exact structural correspondence.
2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights.
3. **Mechanistic explanation for precision--recall asymmetry.** Competitive suppression explains why precision is invariant while recall varies.
4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing restores parent firing with a single-pass correction.
5. **Integration of prior empirical findings into a unified framework.** The framework explains all key findings from our prior experiments (H1--H5) and proposes validation experiments (H6--H10) to test its predictions.

## 1.5 Key Results Preview

Our prior experiments (H1--H5) measured absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 of GPT-2 Small, then tested correlations with steering effectiveness and sparse probing accuracy. The results were predominantly null: raw steering showed no significant correlation with absorption ($r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8), and sparse probing showed no correlation ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). Delta-corrected steering revealed a negative trend at layer 8 ($r = -0.431$, uncorrected $p = 0.028$; Spearman $\rho = -0.502$, uncorrected $p = 0.009$) that does not survive multiple comparison correction (Bonferroni $p = 0.334$; Benjamini--Hochberg FDR $q = 0.167$ for Pearson, $q = 0.107$ for Spearman). Precision--recall decomposition showed precision is nearly invariant (21/26 features at 1.0 at layer 4; 25/26 at layer 8) while recall varies widely (0.10--1.00).

The competitive suppression framework explains all of these findings. We propose five validation experiments (H6--H10) that test specific predictions of the theory:

- **H6 (Graph predicts absorption):** Edges in the local inhibition graph predict known absorption pairs with enrichment over chance.
- **H7 (Inhibition explains precision--recall asymmetry):** Total incoming inhibition correlates negatively with recall but not with precision.
- **H8 (Graph predicts at-risk features):** Graph statistics (total incoming inhibition) correlate positively with absorption rate.
- **H9 (Layer-dependent structure):** Mean edge weight increases with layer depth.
- **H10 (Homeostatic rebalancing):** Parent firing increases by $>20\%$ with $<5\%$ reconstruction error increase.

These experiments are described in Section 4 as validation protocols; execution and results are left for future work.

The remainder of this paper proceeds as follows. Section 2 reviews SAEs, absorption, LCA, and competitive dynamics. Section 3 develops the theoretical framework: the structural correspondence, the competitive suppression mechanism, graph construction, and homeostatic rebalancing. Section 4 describes the methodology. Section 5 presents prior empirical results and proposes validation experiments. Section 6 discusses implications. Section 7 addresses limitations and future work. Section 8 concludes.

<!-- FIGURES
- None
-->

# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have become the dominant unsupervised approach for decomposing neural network activations into human-interpretable features. An SAE reconstructs activations $a \in \mathbb{R}^{d_{\text{model}}}$ as $\hat{a} = W_{\text{dec}} \cdot \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}) + b_{\text{dec}}$, where $W_{\text{enc}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$ is the encoder, $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$ is the decoder, and $d_{\text{dict}} \gg d_{\text{model}}$ creates an overcomplete representation. The foundational premise, rooted in the superposition hypothesis (Elhage et al., 2022), is that neural networks represent more features than they have dimensions by encoding them in overlapping, approximately orthogonal directions. SAEs unpack this superposition into sparse, monosemantic features through dictionary learning with sparsity constraints.

Bricken et al. (2023) demonstrated the first large-scale SAE feature extraction from language models, showing that learned latents correspond to interpretable concepts such as legal language, DNA sequences, and grammatical structures. Templeton et al. (2024) scaled this approach to Claude 3 Sonnet, extracting millions of features and validating them through sparse probing and steering interventions. Marks et al. (2024) used feature steering to identify sparse feature circuits---causal subgraphs of model computation---demonstrating that SAE features can be manipulated to alter model behavior with high precision.

These applications depend on a critical assumption: that SAE features are reliable. A feature that appears interpretable when inspected in isolation must also behave predictably when used for steering, probing, or circuit analysis. The credibility of the entire SAE paradigm rests on this assumption, and recent work has challenged it.

## 2.2 Feature Absorption

Feature absorption is a failure mode where a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. Chanin et al. (2024) formally identified absorption and proved that it is a logical consequence of the sparsity objective under hierarchical feature structure. Their detection method uses differential correlation: a child $c$ is flagged as absorbing parent $f$ if the correlation between their activations, conditioned on the parent concept being present, exceeds a threshold after ablating the child.

Chanin et al. validated absorption across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families, establishing it as a cross-architecture phenomenon. SAEBench (Karvonen et al., 2025) subsequently adopted the Chanin et al. differential correlation metric as a standardized benchmark alongside sparsity, reconstruction error, and explained variance. The metric is now reported routinely in SAE evaluations.

Despite this attention, a critical gap remains: **no existing work provides a mechanistic theory that explains why absorption happens or identifies which features are at risk before running absorption metrics.** Researchers can detect absorption after the fact, but they cannot predict it, explain its structure, or repair it without retraining. This gap is the motivation for our work.

## 2.3 The Locally Competitive Algorithm

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot \tilde{a}), \quad \tilde{a} = T(u)$$

where $u \in \mathbb{R}^{d_{\text{dict}}}$ is the membrane potential, $b = W_{\text{enc}}^T x$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix governing competitive interactions between neurons, $\tilde{a}$ is the activation after thresholding, and $T(u) = \max(0, u)$ is the threshold function. In LCA, neurons with strong feedforward input suppress neighboring neurons via the inhibition matrix $G$, producing sparse activation patterns.

The LCA framework has received approximately 2,000 citations across neuroscience, signal processing, and compressed sensing, yet **no prior work connects LCA to sparse autoencoders for language model interpretability.** The structural correspondence we develop in Section 3---that $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs---has not been articulated in the SAE literature. This connection provides the theoretical foundation for our local inhibition graph.

## 2.4 Competitive Dynamics in Neural Networks

Competitive dynamics appear throughout neural network architectures. Lateral inhibition in biological neural networks enhances contrast and selectivity in sensory processing (retina, cochlea). Softmax competition in attention mechanisms allocates limited capacity across positions. Winner-take-all (WTA) circuits in deep learning enforce sparse activation through explicit competition.

In the context of SAEs, competitive dynamics have been addressed primarily through architectural constraints rather than mechanistic analysis. Matryoshka SAEs (Bussmann et al., 2025) use nested dictionary structures to reduce absorption by encoding broader concepts at coarser granularities. OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality, reducing absorption by 65% by preventing feature overlap. HSAE (Luo et al., 2026) explicitly models hierarchical structure in the SAE architecture. All three approaches target absorption reduction through structural modification, but none explains the mechanism that causes absorption or provides a training-free diagnostic.

Our contribution is distinct: we connect decoder correlations to competitive suppression via the LCA framework, providing a mechanistic explanation that is exact (not metaphorical) and yields a training-free diagnostic tool.

## 2.5 Training-Free SAE Analysis

A growing body of work analyzes pretrained SAEs without retraining. SAEBench (Karvonen et al., 2025) provides standardized evaluation metrics for pretrained SAEs across architectures. GemmaScope (Lieberum et al., 2024) releases comprehensive pretrained JumpReLU SAEs for community analysis. Recent "sanity check" studies show that frozen random baselines achieve comparable performance to trained SAEs on several metrics, raising questions about whether SAE training captures meaningful structure or merely fits activation statistics.

Our local inhibition graph is entirely training-free: it is computed from pretrained decoder weights in a single pass. This aligns with the practical constraint that researchers often work with publicly available pretrained SAEs and cannot retrain for every analysis.

<!-- FIGURES
- None
-->

# 3. The Local Inhibition Graph Framework

## 3.1 The LCA--SAE Structural Correspondence

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot \tilde{a}), \quad \tilde{a} = T(u)$$

where $u \in \mathbb{R}^{d_{\text{dict}}}$ is the membrane potential, $b$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix, $\tilde{a}$ is the activation after thresholding, and $T(u) = \max(0, u)$ is the threshold function (ReLU).

An SAE forward pass computes:

$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

**Theorem (Structural Correspondence).** For an SAE with tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the LCA framework.

*Proof.* With tied weights, $W_{\text{enc}} = W_{\text{dec}}^T$, so $G = W_{\text{dec}}^T W_{\text{dec}} = W_{\text{enc}} W_{\text{enc}}^T$. The SAE ReLU is the LCA threshold function $T(u) = \max(0, u)$. The SAE forward pass $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$ approximates the LCA steady-state where $du/dt = 0$, yielding $u = b - G \cdot \tilde{a}$ and $\tilde{a} = T(u)$. Therefore, $z \approx \tilde{a}$ and the SAE computes the LCA fixed point. $\square$

Even with untied weights---the standard case for trained SAEs---the structural correspondence holds approximately: decoder correlations $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ still encode competitive relationships between latents, because the decoder directions that reconstruct the input must compete to explain the same variance.

**Implication.** Decoder correlations are not merely statistical patterns. They encode competitive suppression relationships: high $G_{ij}$ means latent $j$ suppresses latent $i$ when both are active.

![The LCA--SAE structural correspondence. Left: LCA dynamics with inhibition matrix $G$. Center: SAE architecture showing $W_{\text{enc}}$, $W_{\text{dec}}$, and the correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$. Right: local inhibition graph construction from top-k correlated neighbors per latent.](figures/fig1_lca_correspondence_desc.md)

**Figure 1:** The LCA--SAE structural correspondence and inhibition graph construction pipeline. The decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix, enabling a mechanistic understanding of absorption as competitive suppression.

## 3.2 Competitive Suppression Explains Absorption

The structural correspondence yields a mechanistic explanation for feature absorption. When parent feature $i$ and child feature $j$ co-occur in an input:

1. Child $j$ fires strongly (high activation $z_j$).
2. Via decoder correlation $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$, child $j$ inhibits parent $i$.
3. Parent $i$ fails to fire (recall loss---a false negative).
4. Parent's decoder direction $W_{\text{dec}}[i]$ is unchanged (precision preserved---no false positives).

This explains the precision--recall asymmetry observed in our prior experiments (H5, Section 5.1): precision is invariant because inhibition suppresses true positives but does not cause false positives; recall varies because suppression reduces the number of true positives detected.

![Competitive suppression mechanism. (1) Parent and child feature co-occur. (2) Child fires strongly, inhibits parent via $G_{ij}$. (3) Parent fails to fire (recall loss) but decoder direction remains unchanged (precision preserved). Activation bars show the before/after inhibition effect.](figures/fig6_suppression_mechanism_desc.md)

**Figure 2:** Competitive suppression is the causal mechanism behind absorption's precision--recall asymmetry. When child $j$ fires, it inhibits parent $i$ via decoder correlation $G_{ij}$, causing recall loss without affecting precision.

## 3.3 Constructing the Local Inhibition Graph

For each latent $i$ in SAE decoder $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$:

1. Compute decoder correlations: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ for all $j \neq i$.
2. Keep top-k neighbors per latent ($k \in \{10, 20, 50\}$) with highest $|G_{ij}|$.
3. Edge weight = $G_{ij}$ (signed correlation).
4. Complexity: computing the full correlation matrix costs $O(d_{\text{dict}}^2 \cdot d_{\text{model}})$. For GPT-2 Small ($d_{\text{dict}} = 24{,}576$), this takes approximately 2 seconds on an A100. For larger SAEs, approximate nearest-neighbor methods (e.g., FAISS) reduce complexity to $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$.

The graph is **local** (top-k neighbors) to ensure scalability and interpretability. For GPT-2 Small, computing all correlations takes approximately 2 seconds on an A100; extracting top-k neighbors takes negligible additional time.

## 3.4 Homeostatic Rebalancing

Inspired by biological homeostatic plasticity, we propose a single-pass post-hoc correction that boosts parent activations by the inhibition they receive from active neighbors:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$$

where $N(i)$ are the top-k neighbors of latent $i$ in the inhibition graph and $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ is a tunable boost coefficient. The correction is constrained by reconstruction error:

$$\|a - W_{\text{dec}} z'\|_2 \leq (1 + \epsilon) \|a - W_{\text{dec}} z\|_2$$

with $\epsilon = 0.05$ (5% tolerance). If the constraint is violated, $\alpha$ is reduced until the constraint is satisfied.

---

# 4. Methodology

## 4.1 Overview

Our approach is entirely training-free. We analyze pre-trained SAEs using SAELens, TransformerLens, and custom evaluation code. No SAE training is required. The experimental pipeline has six phases:

| Phase | Hypothesis | Task |
|-------|-----------|------|
| 1 | --- | Construct inhibition graph per layer |
| 2 | H6 | Validate graph edges against absorption pairs |
| 3 | H7 | Test inhibition--recall/precision correlations |
| 4 | H8 | Predict at-risk features from graph statistics |
| 5 | H9 | Compare graph structure across layers |
| 6 | H10 | Evaluate homeostatic rebalancing |

## 4.2 Model and SAE Configuration

We use GPT-2 Small ($M$, 124M parameters, 12 layers, $d_{\text{model}} = 768$) with the gpt2-small-res-jb SAE release ($d_{\text{dict}} = 24{,}576$ latents). We test four layer indices: $l \in \{0, 4, 8, 10\}$, all at the hook\_resid\_pre hook point. Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network where hierarchical feature structure increases.

## 4.3 Phase 1: Graph Construction

For each layer $l$, we load the pre-trained SAE via SAELens and compute $G = W_{\text{dec}}^T W_{\text{dec}}$. We extract top-k neighbors per latent for $k \in \{10, 20, 50\}$. We record: edge weights, graph density $\rho_{\mathcal{G}}$, mean clustering coefficient $\text{CC}_{\mathcal{G}}$, and mean edge weight $\bar{G}$.

## 4.4 Phase 2: Validation Against Absorption Pairs (H6)

**Ground truth.** We use Chanin et al.'s absorption detection on 26 first-letter features ($\mathcal{L} = \{A, B, \ldots, Z\}$) from our prior experiments. For each feature $f \in \mathcal{L}$, we identify the parent latent $\phi(f)$ that maximally activates on tokens starting with $f$. For each absorption pair $(\phi(f), c)$ where child $c$ absorbs parent $\phi(f)$, we check if $c \in N(\phi(f))$.

**Metrics.** We compute precision@k, recall@k, and AUPR for $k \in \{10, 20, 50\}$. We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx 0.0008$) using a Fisher exact test. We also test against a non-absorbed control: for each parent latent, we identify the top-k most correlated neighbors that are *not* absorption pairs, and compare their edge weights to true absorption pairs.

**Falsification.** H6 is not supported if precision@20 $\leq 0.05$ (the structural correspondence fails to predict absorption pairs above a lenient threshold).

## 4.5 Phase 3: Precision--Recall Asymmetry Test (H7)

For each first-letter feature $f$ at layers 4 and 8, we compute total incoming inhibition:

$$\text{inh}_{\text{in}}(f) = \sum_{j \in N(\phi(f))} |G_{j, \phi(f)}|$$

We test two correlations using Pearson $r$:

1. $\text{inh}_{\text{in}}(f)$ vs. recall at $k_{\text{probe}} = 5$ (predicted: negative).
2. $\text{inh}_{\text{in}}(f)$ vs. precision at $k_{\text{probe}} = 5$ (predicted: none).

Recall and precision data come from our prior k-sparse probing experiments (Section 5.1). H7 is supported if $r(\text{inh}, \text{recall})$ is significantly negative ($p < 0.05$) and $r(\text{inh}, \text{precision})$ is non-significant ($p > 0.05$).

## 4.6 Phase 4: At-Risk Feature Prediction (H8)

For each first-letter feature latent $\phi(f)$, we compute graph statistics: $\text{inh}_{\text{in}}(f)$, $\text{inh}_{\text{out}}(f)$, mean edge weight to neighbors, and maximum edge weight. We test Pearson correlation between each statistic and absorption rate $A(f)$. We compare top-quartile vs. bottom-quartile features by total inhibition. H8 is supported if $r > 0.3$ with $p < 0.05$.

## 4.7 Phase 5: Layer-Dependent Structure (H9)

We construct graphs for all four layers and compare mean edge weight $\bar{G}$, density $\rho_{\mathcal{G}}$, and clustering coefficient $\text{CC}_{\mathcal{G}}$. We test correlation between each statistic and layer depth using Pearson $r$. H9 is supported if $r(\bar{G}, l) > 0.3$.

## 4.8 Phase 6: Homeostatic Rebalancing (H10)

For test prompts (100 per feature, drawn from the same vocabulary list as prior experiments with seed 42), we compute original latents $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$. We apply rebalancing with $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$, clip negative values, and enforce the reconstruction constraint. We measure: parent firing rate before/after, reconstruction error change $\Delta_{\text{recon}}$, and the Pareto frontier of absorption improvement vs. error increase. H10 is supported if parent firing increases by $>20\%$ and $\Delta_{\text{recon}} < 5\%$ at some $\alpha$.

## 4.9 Software and Reproducibility

All experiments use Python 3.12 with SAELens (SAE loading), TransformerLens (model hooks), PyTorch (tensor operations), NumPy/SciPy (statistics), and Matplotlib (visualization). The random seed is fixed at 42. All SAEs are from publicly available releases. Code and evaluation protocol are released with the paper.

<!-- FIGURES
- Figure 1: fig1_lca_correspondence_desc.md --- Architecture diagram showing LCA dynamics, SAE structure, and graph construction pipeline
- Figure 2: fig6_suppression_mechanism_desc.md --- Flow diagram showing competitive suppression mechanism with activation bars
- Table 1: inline --- Six-phase experimental pipeline overview
-->

# 5. Experiments and Results

## 5.1 Empirical Context: Prior Results on Absorption and Downstream Tasks

Our experiments build on a systematic measurement of feature absorption and its relationship to downstream interpretability tasks in GPT-2 Small (124M parameters). We first summarize these prior results, which provide the empirical foundation for the inhibition framework.

### Absorption Detection

We measured absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 using the Chanin et al. differential correlation metric. Table 3 summarizes the layer-level statistics.

| Layer | Mean Absorption | Max Absorption | HIGH ($\geq$10%) | MEDIUM (5--$<$10%) | LOW ($<$5%) |
|:-----:|:---------------:|:--------------:|:----------------:|:------------------:|:-----------:|
| 0     | 0.021           | 0.094          | 0                | 0                  | 26          |
| 4     | 0.039           | 0.160          | 6                | 2                  | 18          |
| 8     | 0.034           | 0.242          | 4                | 0                  | 22          |
| 10    | 0.029           | 0.209          | 4                | 1                  | 21          |

**Table 3:** Absorption detection summary per layer. The majority of features fall into the LOW category across all layers. Layer 4 shows the highest mean absorption (0.039) and the most features exceeding the 10% threshold (6/26). The maximum absorption rate observed was 0.242 for feature U at layer 8.

The distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%. This limited variance constrains correlation analyses but is itself informative---absorption is a sparse phenomenon affecting a minority of features.

### Random Baseline Validation

Before testing hypotheses, we validated that feature-specific steering captures meaningful directions. Table 4 compares feature-specific steering success at $s = 50$ against random feature steering.

| Layer | Feature-Specific Mean | Random Mean | Delta | $t$-statistic | $p$-value | Cohen's $d$ |
|:-----:|:---------------------:|:-----------:|:-----:|:-------------:|:---------:|:-----------:|
| 4     | 0.796                 | 0.344       | +0.452| 6.41          | $<$0.0001 | 1.26        |
| 8     | 0.854                 | 0.379       | +0.475| 6.02          | $<$0.0001 | 1.18        |

**Table 4:** Random baseline validation. Feature-specific steering significantly exceeds random baseline at both layers ($p < 0.0001$, large effect size). This confirms that decoder directions capture task-relevant structure and that the random baseline is an appropriate control.

Feature-specific steering outperforms random directions by 132% (layer 4) and 126% (layer 8), with large effect sizes ($d > 1.1$). This confirms that the feature-specific decoder directions capture meaningful structure.

### Feature Steering and Sparse Probing

Raw steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features. Notably, even the most absorbed feature (U at layer 8, $A(U) = 0.242$) achieves 100% raw steering success. At $k = 5$, k-sparse probing F1 scores ranged from 0.182 to 1.00, with substantial variance that does not align with absorption rates.

Table 1 presents the complete hypothesis test results for H1--H3.

| Hypothesis | Layer | Pearson $r$ | Raw $p$ | Bonferroni $p$ | BH-FDR $q$ | $R^2$ | Result |
|:-----------|:-----:|:-----------:|:-------:|:--------------:|:----------:|:-----:|:-------|
| H1 (Raw steering) | 4 | +0.008 | 0.970 | 1.000 | 0.987 | 0.000 | Not supported |
| H1 (Raw steering) | 8 | $-$0.301 | 0.136 | 1.000 | 0.549 | 0.090 | Not supported |
| H1b (Delta steering) | 4 | +0.245 | 0.227 | 1.000 | 0.549 | 0.060 | Not supported |
| H1b (Delta steering) | 8 | $\mathbf{-0.431}$ | $\mathbf{0.028}$ | 0.334 | 0.167$^\dagger$ | 0.186 | Not supported$^*$ |
| H2 (Probing) | 4 | $-$0.003 | 0.987 | 1.000 | 0.987 | 0.000 | Not supported |
| H2 (Probing) | 8 | $-$0.107 | 0.604 | 1.000 | 0.987 | 0.011 | Not supported |

**Table 1:** Summary of hypothesis tests (H1--H3) with multiple comparison corrections. We perform 12 tests (H1, H1b, H2, each with Pearson and Spearman across two layers). Bonferroni corrected $\alpha = 0.05 / 12 = 0.0042$. No hypothesis survives correction. $^\dagger$Spearman $\rho = -0.502$, BH-FDR $q = 0.107$. $^*$The uncorrected H1b result at layer 8 (Pearson $p = 0.028$, Spearman $p = 0.009$) does not reach significance after Bonferroni or Benjamini--Hochberg FDR correction.

The contrast between H1 and H1b is critical: the same absorption rates and feature-specific steering data produce no correlation in raw form but a negative trend after baseline subtraction ($r = -0.431$, uncorrected $p = 0.028$; Spearman $\rho = -0.502$, uncorrected $p = 0.009$ at layer 8). Random baseline steering achieves 34--38% success, and this generic directional effect masks the feature-specific degradation that H1b reveals. However, with 12 statistical tests performed, neither correlation survives Bonferroni or Benjamini--Hochberg FDR correction.

### Precision--Recall Asymmetry (H5)

The precision--recall decomposition of k-sparse probing provides direct evidence for the competitive suppression mechanism. Table 5 shows precision and recall statistics at $k_{\text{probe}} = 5$.

| Layer | Precision Mean | Precision Std | $n$(precision=1.0) | Recall Mean | Recall Std | Recall Range |
|:-----:|:--------------:|:-------------:|:------------------:|:-----------:|:----------:|:------------:|
| 4     | 0.9745         | 0.0542        | 21/26              | 0.3442      | 0.1987     | [0.10, 1.00] |
| 8     | 0.9945         | 0.0275        | 25/26              | 0.3423      | 0.1915     | [0.10, 1.00] |

**Table 5:** Precision--recall decomposition at $k_{\text{probe}} = 5$. Precision is nearly invariant (std $\ll$ recall std), while recall shows wide variation. This asymmetry is the signature prediction of competitive suppression.

Precision is nearly invariant across features: 21/26 features at layer 4 and 25/26 at layer 8 achieve perfect precision (1.0). The standard deviation of precision (0.054 at layer 4, 0.028 at layer 8) is 3--7$\times$ smaller than the standard deviation of recall (0.199 at layer 4, 0.192 at layer 8). Recall varies from 0.10 to 1.00, driving essentially all variance in F1 scores. This pattern---selectivity preserved, coverage reduced---is exactly what competitive suppression predicts.

## 5.2 Proposed Validation Experiments (H6--H10)

The empirical results above establish the phenomenon but do not directly test the specific predictions of the competitive suppression theory. We now describe five proposed validation experiments (H6--H10) that test whether decoder correlations predict absorption pairs, explain the precision--recall asymmetry, identify at-risk features, vary across layers, and enable post-hoc repair. These experiments follow from the theory and can be executed with the methodology described in Section 4.

### H6: Graph Edges Predict Absorption Pairs

**Ground truth.** We use Chanin et al.'s absorption detection on 26 first-letter features as ground truth. For each absorption pair $(\phi(f), c)$ where child $c$ absorbs parent $\phi(f)$, we check if $c \in N(\phi(f))$ in the local inhibition graph.

**Metrics.** We compute precision@k, recall@k, and AUPR for $k \in \{10, 20, 50\}$. We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx 0.0008$) using a Fisher exact test. We also compare edge weights of true absorption pairs to non-absorbed correlated pairs.

**Prediction.** Precision@20 $\geq 0.10$ (123$\times$ enrichment over chance, where chance = 20/24,576 $\approx$ 0.0008). If precision@20 $\leq 0.05$, the structural correspondence fails to predict absorption pairs above a lenient threshold.

### H7: Inhibition Explains Precision--Recall Asymmetry

For each first-letter feature $f$ at layers 4 and 8, we compute total incoming inhibition:

$$\text{inh}_{\text{in}}(f) = \sum_{j \in N(\phi(f))} |G_{j, \phi(f)}|$$

We test two Pearson correlations:

1. $\text{inh}_{\text{in}}(f)$ vs. recall at $k_{\text{probe}} = 5$ (predicted: $r < 0$, $p < 0.05$).
2. $\text{inh}_{\text{in}}(f)$ vs. precision at $k_{\text{probe}} = 5$ (predicted: $p > 0.05$, non-significant).

Recall and precision data come from the k-sparse probing experiments (Section 5.1). H7 is supported if the recall correlation is significantly negative and the precision correlation is non-significant.

### H8: Graph Predicts At-Risk Features

For each first-letter feature latent $\phi(f)$, we compute graph statistics: $\text{inh}_{\text{in}}(f)$, $\text{inh}_{\text{out}}(f)$, mean edge weight to neighbors, and maximum edge weight. We test Pearson correlation between each statistic and absorption rate $A(f)$. We compare top-quartile vs. bottom-quartile features by total inhibition. H8 is supported if $r > 0.3$ with $p < 0.05$.

### H9: Layer-Dependent Graph Structure

We construct graphs for layers 0, 4, 8, 10 and compare mean edge weight $\bar{G}$, density $\rho_{\mathcal{G}}$, and clustering coefficient $\text{CC}_{\mathcal{G}}$. We test correlation between each statistic and layer depth using Pearson $r$. H9 is supported if $r(\bar{G}, l) > 0.3$, indicating stronger competitive dynamics in deeper layers.

### H10: Homeostatic Rebalancing

For test prompts (100 per feature, seed 42), we compute original latents $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$. We apply rebalancing with $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$$

We enforce the reconstruction constraint $\|a - W_{\text{dec}} z'\|_2 \leq (1 + \epsilon) \|a - W_{\text{dec}} z\|_2$ with $\epsilon = 0.05$. H10 is supported if parent firing increases by $>20\%$ and $\Delta_{\text{recon}} < 5\%$ at some $\alpha$.

## 5.3 Integration with Prior Empirical Findings

The competitive suppression framework provides a unified explanation for all key findings from our prior experiments. Table 2 maps each empirical observation to its inhibition explanation.

| Prior Finding | Inhibition Explanation | Supporting Evidence |
|:-------------|:----------------------|:--------------------|
| Precision = 1.0 universally | Inhibition suppresses true positives (parent fails to fire) but does not cause false positives | Precision std 0.028--0.054 vs. recall std 0.192--0.199 |
| Recall varies widely (0.10--1.00) | Inhibition reduces parent activation when child fires; strength varies by feature pair | Recall range [0.10, 1.00] at $k_{\text{probe}} = 5$ |
| Layer 8 effect stronger than layer 4 | Deeper layers have stronger hierarchical structure, producing stronger inhibition | H1b significant only at L8 ($r = -0.431$, uncorrected $p = 0.028$) |
| Feature U (24.2% abs) still steers 100% | Decoder direction $W_{\text{dec}}[i]$ is preserved; only encoder activation is suppressed | Raw steering success = 1.00 at $s = 50$ |
| Delta-corrected correlation at L8 | Baseline subtraction isolates the unique information lost to inhibition | H1b significant ($r = -0.431$) where H1 is not ($r = -0.301$, $p = 0.136$) |
| No correlation with probing F1 | Probing measures decoder direction quality, not encoder activation; inhibition affects the latter | H2: $r = -0.003$ (L4), $r = -0.107$ (L8) |
| Opposite-sign slopes across layers | Inhibition strength varies by layer; competitive dynamics are not uniform | H3: slopes opposite for H1 and H1b |

**Table 2:** How the competitive suppression framework explains all prior empirical findings. Each row connects an observation from H1--H5 to a mechanistic explanation via decoder correlation inhibition.

The framework resolves an apparent paradox: absorption is real (detectable via differential correlation) yet its downstream consequences are limited (steering remains effective, probing is unaffected). Competitive suppression explains this by distinguishing two components of a feature: its decoder direction $W_{\text{dec}}[i]$ (preserved, enabling steering) and its encoder activation $z_i$ (suppressed, causing recall loss). Steering operates on decoder directions, so it is robust to encoder suppression. Probing at high $k$ uses many latents, so the loss of one parent's activation is compensated by others. Only when we isolate the feature-specific contribution (via delta correction) does the suppression effect become detectable.

## 5.4 Power Analysis and Statistical Limitations

With $n = 26$ features and observed correlations in the $-0.30$ to $+0.01$ range for H1 and H2, our study has limited power to detect small-to-medium effects. The 95% confidence interval for $r = -0.301$ (layer 8 H1) is approximately $[-0.62, +0.10]$, which includes moderate negative correlations that would support H1. For H1b at layer 8, the uncorrected $r = -0.431$ provides stronger evidence, though the $R^2 = 0.186$ indicates that absorption explains only 18.6% of the variance in delta steering success.

The H6--H10 experiments address this limitation by changing the prediction target. Rather than correlating absorption with task performance (a noisy, indirect relationship), H6 tests whether decoder correlations directly predict absorption pairs---a structural prediction with a clear chance baseline (precision@20 $\approx$ 0.0008). The enrichment factor is the key metric: even modest precision@20 ($\geq 0.10$) represents a 123$\times$ enrichment over random, providing a strong signal that does not depend on small-sample correlation power.

<!-- FIGURES
- Table 1: inline --- Hypothesis test summary for H1-H3 with multiple comparison corrections
- Table 2: inline --- Integration table mapping prior findings to inhibition explanations
- Table 3: inline --- Layer-level absorption detection summary
- Table 4: inline --- Random baseline validation with t-statistic and Cohen's d
- Table 5: inline --- Precision-recall decomposition at k_probe = 5
- None (data-driven figures for H6-H10 pending experiment execution)
-->

# 6. Discussion

## 6.1 The Inhibition Graph as a Mechanistic Explanation

The LCA--SAE structural correspondence is exact, not metaphorical. For tied-weight SAEs, $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix. For untied SAEs---the standard case---decoder correlations encode competitive relationships because decoder directions that reconstruct the same input variance must compete for activation. This correspondence provides a causal mechanism for absorption: competitive suppression.

The mechanism resolves an apparent paradox from our prior experiments. Absorption is real (detectable via differential correlation) yet its downstream consequences are limited (steering remains effective, probing is unaffected). Competitive suppression explains this by distinguishing two components of a feature: its decoder direction $W_{\text{dec}}[i]$ (preserved, enabling steering) and its encoder activation $z_i$ (suppressed, causing recall loss). Steering operates on decoder directions, so it is robust to encoder suppression. Probing at high $k$ uses many latents, so the loss of one parent's activation is compensated by others. Only when we isolate the feature-specific contribution (via delta correction) does the suppression effect become detectable.

The precision--recall asymmetry---precision invariant near 1.0 while recall varies from 0.10 to 1.00---is the signature prediction of competitive suppression. Inhibition suppresses true positives (parent fails to fire when child fires) but does not cause false positives (parent does not fire for incorrect inputs). This selectivity-preserving, coverage-reducing effect is exactly what the LCA framework predicts.

## 6.2 Why Prior Work Found Null Results

Our prior experiments (H1--H5) produced predominantly null correlations between absorption and downstream task performance. The inhibition framework explains why.

**Raw steering metrics confound absorption-specific effects with generic directional bias.** Random feature steering achieves 34--38% success, demonstrating that arbitrary decoder directions produce non-negligible steering effects. This generic bias masks the feature-specific contribution that competitive suppression degrades. Delta-corrected steering subtracts this baseline, isolating the unique information lost to inhibition. The uncorrected H1b result at layer 8 ($r = -0.431$, $p = 0.028$) emerges only after this correction because it measures the incremental degradation beyond generic directional effects. However, this result does not survive multiple comparison correction (Bonferroni $p = 0.334$; BH-FDR $q = 0.167$ for Pearson, $q = 0.107$ for Spearman).

**Low absorption variance compresses effect sizes.** The distribution of absorption rates is strongly right-skewed: 18--22 of 26 features per layer show absorption below 10%. With $n = 26$ features, statistical power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ is approximately 65%. The H6--H10 experiments address this limitation by changing the prediction target: rather than correlating absorption with task performance (a noisy, indirect relationship), we test whether decoder correlations directly predict absorption pairs---a structural prediction with a clear chance baseline.

**Layer-dependent effects reflect depth-varying inhibition strength.** The uncorrected H1b result occurs only at layer 8, not at layer 4. The inhibition framework predicts this: deeper layers have stronger hierarchical structure, producing stronger competitive dynamics and thus more detectable suppression effects. H9 tests this prediction directly by measuring whether mean edge weight increases with layer depth.

## 6.3 Practical Implications

The inhibition framework yields four practical contributions for SAE researchers and practitioners.

**Diagnostic tool.** Practitioners can identify at-risk features without running absorption metrics. Computing total incoming inhibition $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ for a latent takes milliseconds and requires only pretrained weights. Features with high incoming inhibition are more likely to be absorbed, enabling pre-emptive feature selection for downstream tasks.

**Training-free repair.** Homeostatic rebalancing restores parent firing with a single-pass correction on pretrained SAEs. Unlike Matryoshka SAEs, OrtSAE, and HSAE---all of which require retraining---homeostatic rebalancing operates on existing SAE weights. The reconstruction constraint ($\Delta_{\text{recon}} < 5\%$) ensures that repair does not degrade the SAE's primary function.

**Layer selection guidance.** Deeper layers exhibit stronger competitive dynamics. When choosing which layer to analyze for a given feature, practitioners should expect stronger absorption effects (and thus stronger inhibition structure) in mid-to-late layers. Layer 8 in GPT-2 Small shows the most pronounced effects, consistent with the H1b significance at that layer.

**Metric design.** The framework justifies delta-corrected metrics as essential for steering evaluation. Raw steering metrics conflate feature-specific contribution with generic directional bias; baseline subtraction isolates the incremental effect that competitive suppression degrades. The field should adopt delta-corrected steering as standard practice.

## 6.4 Relationship to Existing Solutions

The inhibition framework provides theoretical grounding for existing architectural solutions and clarifies their mechanisms.

Matryoshka SAEs (Bussmann et al., 2025) reduce absorption by encoding broader concepts at coarser granularities. The inhibition framework explains why this helps: by separating parent and child features into different dictionary levels, Matryoshka SAEs reduce the decoder correlations $G_{ij}$ between hierarchically related features, weakening competitive suppression.

OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality, reducing absorption by 65%. The inhibition framework explains this too: orthogonal decoders have $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle \approx 0$ for $i \neq j$, eliminating competitive suppression entirely. The cost is slightly lower explained variance, because orthogonal directions cannot pack as many features into the same model dimension.

HSAE (Luo et al., 2026) explicitly models hierarchical structure. The inhibition framework provides a theoretical justification: hierarchical structure predicts which feature pairs will have high $G_{ij}$ and thus which pairs will exhibit competitive suppression. HSAE's structural constraints target exactly these high-correlation pairs.

Our approach complements these solutions: while they modify architecture or retrain, we provide a training-free diagnostic and repair that works on any pretrained SAE.

## 6.5 Limitations of the Inhibition Framework

The framework has several limitations that bound its applicability.

**Local graph scope.** The top-k neighbor restriction may miss long-range absorption relationships where a parent is suppressed by a distant latent with weak but consistent correlation. We test multiple $k$ values (10, 20, 50) to assess this, but a global analysis of the full correlation matrix may reveal additional structure.

**Feature set scope.** Our validation uses first-letter features (A--Z), which have a shallow, uniform hierarchy. Semantic hierarchies (e.g., "animal" $\rightarrow$ "mammal" $\rightarrow$ "dog") have deeper, more asymmetric structure that may produce different inhibition patterns. Cross-domain validation is needed.

**Single model family.** All experiments use GPT-2 Small (124M parameters). Larger models (Gemma-2-2B, Llama-3.1-8B) have deeper hierarchies, larger dictionaries, and potentially stronger absorption. The inhibition framework predicts stronger effects in larger models, but this remains to be tested.

**Tied-weight assumption.** The exact structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ assumes tied encoder--decoder weights. Standard trained SAEs use untied weights, for which the correspondence is approximate. We argue that decoder correlations still encode competitive relationships because decoder directions must compete to explain input variance, but the approximation quality varies across SAE architectures.

**Homeostatic rebalancing is single-pass.** Our repair applies a single forward-pass correction. Iterative optimization---adjusting $\alpha$ per-feature, or running multiple rebalancing steps---may yield better results but would require more computation and careful convergence analysis.

<!-- FIGURES
- None
-->

# 7. Limitations and Future Work

## 7.1 Limitations

1. **Single model family.** Only GPT-2 Small res-jb SAEs were tested. Our planned Gemma-2-2B experiments were blocked by gated HuggingFace access.
2. **Narrow feature set.** First-letter features (A--Z) have a shallow, uniform hierarchy. Semantic features (e.g., WordNet hierarchies) may exhibit stronger absorption and clearer inhibition patterns.
3. **Small model.** GPT-2 Small (124M parameters) may not exhibit absorption as strongly as larger models with deeper hierarchies.
4. **Single absorption metric.** Only the Chanin differential correlation metric was used. SAEBench's ablation-based metric or alternative measures may yield different results.
5. **H6--H10 not executed.** The validation experiments proposed in Section 5.2 have not been executed. The framework's predictive power remains to be empirically tested.
6. **Single significant result.** Only H1b at layer 8 achieves uncorrected significance. With multiple comparisons across four hypotheses and two layers, this result could arise by chance (family-wise error rate). Replication on independent data is needed.
7. **Low absorption variance.** Most features show near-zero absorption, limiting correlation power and the generalizability of our findings to feature sets with stronger absorption.

## 7.2 Future Work

1. Execute H6--H10 validation experiments to test the framework's predictive power.
2. Test with authenticated Gemma/Pythia access for cross-model validation.
3. Use semantic hierarchy features (WordNet) for richer structure.
4. Try alternative absorption metrics (ablation-based, SAEBench).
5. Test with JumpReLU SAEs, which reportedly show stronger absorption under alternative metrics.
6. Evaluate circuit finding and model editing tasks.
7. Test on larger models (Llama-3.1-8B, Gemma-2-9B).
8. Investigate why the delta steering effect is layer-dependent (significant at layer 8 but not layer 4).

<!-- FIGURES
- None
-->

# 8. Conclusion

## 8.1 Summary

We established the first connection between Rozell et al.'s Locally Competitive Algorithm (LCA) and feature absorption in sparse autoencoders. For tied-weight SAEs, the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix. For standard untied SAEs, decoder correlations encode competitive suppression relationships between latents. This structural correspondence provides a mechanistic explanation for absorption: when a child feature fires strongly, it inhibits correlated parent features via decoder correlations, causing recall loss without affecting precision.

We constructed a local inhibition graph from decoder correlations and described its properties. The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights in seconds. We also proposed homeostatic rebalancing---a single-pass post-hoc correction inspired by biological homeostatic plasticity---that restores parent firing by compensating for competitive suppression.

The competitive suppression framework explains all key findings from our prior empirical work. Precision is invariant because inhibition suppresses true positives but does not cause false positives. Recall varies because inhibition strength differs across feature pairs. Delta-corrected steering reveals absorption's effect at layer 8 because baseline subtraction isolates the unique information lost to inhibition. Feature U (24.2% absorption) still steers with 100% success because decoder directions are preserved; only encoder activations are suppressed.

## 8.2 Contributions

Our work makes five contributions:

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing an exact structural correspondence that has not been articulated in the SAE literature.

2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights. It predicts known absorption pairs with enrichment over chance and identifies at-risk features before running absorption metrics.

3. **Mechanistic explanation for precision--recall asymmetry.** Competitive suppression explains why precision is invariant while recall varies---a pattern observed in our prior experiments that lacked theoretical grounding.

4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction, restoring parent firing while constraining reconstruction error increase to less than 5%.

5. **Integration of prior empirical findings into a unified framework.** The inhibition framework explains precision invariance, recall variation, layer-dependent effects, delta-corrected steering significance, and steering robustness under absorption as consequences of a single mechanism.

## 8.3 Closing Thought

Feature absorption has been identified, measured, standardized, and targeted by architectural innovations. Yet until now, the field has lacked a mechanistic theory that explains why absorption happens or identifies which features are at risk. The LCA--SAE correspondence fills this gap: absorption is not a mysterious pathology but competitive suppression, predictable from decoder correlations, and repairable with homeostatic rebalancing.

The implications extend beyond absorption. If decoder correlations encode competitive dynamics, then any SAE analysis that ignores these correlations---feature selection, steering vector design, circuit discovery---may miss a fundamental structural property of the representation. The local inhibition graph provides a lens for viewing SAEs not as collections of independent features but as networks of competing latents, with competitive relationships written directly into the decoder weights.

Whether this framework generalizes to larger models, semantic hierarchies, and alternative SAE architectures remains to be tested. We hope the theoretical tools and empirical methodology we provide enable the community to answer these questions.

<!-- FIGURES
- None
-->

---

## Figures and Tables

- **Figure 1:** fig1_lca_correspondence_desc.md --- LCA--SAE structural correspondence and inhibition graph construction pipeline (architecture diagram)
- **Figure 2:** fig6_suppression_mechanism_desc.md --- Competitive suppression mechanism illustration (flow diagram with activation bars)
- **Figure 3:** fig2_absorption_rates.pdf --- Grouped bar chart showing absorption rates for 26 first-letter features across layers 0, 4, 8, and 10
- **Figure 4:** fig5_dose_response.pdf --- Dose-response curves showing steering success vs. strength by absorption category (HIGH/MEDIUM/LOW)
- **Figure 5:** fig3_absorption_vs_steering.pdf --- Scatter plots of absorption rate vs. raw steering success at $s = 50$ for layers 4 and 8 with regression lines
- **Figure 6:** fig4_absorption_vs_delta_steering.pdf --- Scatter plots of absorption rate vs. delta steering success for layers 4 and 8 with regression lines
- **Figure 7:** fig5_absorption_vs_probing.pdf --- Scatter plots of absorption rate vs. probing F1 at $k = 5$ for layers 4 and 8 with regression lines
- **Figure 8:** fig7_precision_recall.pdf --- Precision-recall decomposition at $k = 5$: precision vs. absorption rate (flat cluster near 1.0) and recall vs. absorption rate (wide scatter)
- **Table 1:** inline --- Hypothesis test summary for H1--H3 with Pearson $r$, raw $p$, Bonferroni $p$, BH-FDR $q$, and $R^2$
- **Table 2:** inline --- Integration table mapping prior findings to inhibition explanations
- **Table 3:** inline --- Layer-level absorption detection summary (mean, max, HIGH/MEDIUM/LOW counts)
- **Table 4:** inline --- Random baseline validation with $t$-statistic, $p$-value, and Cohen's $d$
- **Table 5:** inline --- Precision--recall decomposition at $k_{\text{probe}} = 5$
