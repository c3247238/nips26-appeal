# 2. Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

Sparse Autoencoders (SAEs) have emerged as a foundational tool in mechanistic interpretability, enabling researchers to decompose neural network activations into interpretable, sparse feature representations. Anthropic's foundational work demonstrated that SAEs trained on residual stream activations could identify human-interpretable features, with 70% of features judged genuinely monosemantic by human evaluators [@chanin_absorption]. Features discovered include DNA sequences, HTTP requests, legal language, and emotional sentiment.

The SAE training objective combines reconstruction loss with an L1 sparsity penalty:

$$L = \|\mathbf{x} - \mathbf{\hat{x}}\|^2 + \lambda \|\mathbf{z}\|_1$$

where $\mathbf{x} \in \mathbb{R}^d$ is the original activation, $\mathbf{\hat{x}} \in \mathbb{R}^d$ is the reconstruction, $\mathbf{z} \in \mathbb{R}^n$ is the sparse feature activation, and $\lambda$ controls sparsity. This objective creates a bottleneck that forces the SAE to learn efficient, distributed representations.

Formally, an SAE consists of an encoder network $f(\cdot)$ with weights $W_e \in \mathbb{R}^{n \times d}$ and bias $b_e \in \mathbb{R}^n$ that maps input activations to sparse latent representations, and a decoder network $g(\cdot)$ with weights $W_d \in \mathbb{R}^{d \times n}$ and bias $b_d \in \mathbb{R}^d$ that reconstructs the input as $\mathbf{\hat{x}} = g(\mathbf{z})$.

The appeal of SAEs for interpretability stems from the hypothesis that individual features in the learned dictionary correspond to semantically meaningful concepts. When this correspondence holds, researchers can intervene on specific features to test causal hypotheses about model behavior using steering interventions or activation patching. However, this intervention-based workflow relies on a critical assumption: that the features identified by the SAE actually correspond to the intended semantic structure of the model.

## 2.2 The Feature Absorption Problem

Feature absorption is a phenomenon where child features in an SAE learned hierarchy substitute for their parent features in sparse representations [@chanin_absorption]. When a parent feature $p$ would otherwise activate strongly, its child features $c_1, c_2, \ldots, c_k$ instead absorb the incoming activation, causing the parent's reconstructed activation to decrease even though the underlying model computation still depends on the parent concept.

Concretely, consider a 3-level feature hierarchy where a parent feature detects "political statements" and its children detect subcategories like "policy arguments" and "political commentary." When processing text that strongly invokes the parent concept, absorption causes the child features to fire instead, and the parent's reconstructed activation drops. The SAE's internal representation no longer matches the intended hierarchical feature structure.

The **multi-child proportional absorption rate** quantifies this phenomenon:

$$A_{multi}(p) = \frac{a_p^{after}}{a_p^{before}}$$

where $a_p^{before}$ is the parent activation before ablating child features and $a_p^{after}$ is the parent activation after ablating the top $k$ child features. Values of $A_{multi}(p) > 1.0$ indicate the parent was being suppressed by its children; ablating the children releases the parent's true activation.

Absorption threatens the reliability of SAE-based interpretability for critical tasks. If safety-critical features (detecting deception, jailbreaking, harm) are disproportionately absorbed, then SAE-based safety assessments may systematically underestimate risk.

## 2.3 Prior Explanations of Absorption

Prior work has offered two main hypotheses for the origin of absorption.

**Decoder geometry hypothesis.** Chanin et al. [@chanin_absorption] documented absorption and hypothesized that it arises from decoder geometry --- specifically, that the decoder's learned weight structure causes child feature directions to overlap with parent directions in activation space. Under this account, absorption is fundamentally a geometric artifact of how the decoder represents features.

**Sparsity optimization hypothesis.** Korznikov et al. [@korznikov_sanity] proposed that absorption results from sparsity optimization pressures during training. Under this account, the SAE learns to compress redundant activation patterns by representing parent activations through child features when they are co-active, trading hierarchical structure for compression efficiency.

Both hypotheses share a common assumption: that the decoder plays a primary role in absorption. The encoder is implicitly treated as a passive feature extractor, with absorption arising from how the decoder reconstructs the encoded features.

A critical gap in prior work is the lack of controlled factorial experiments isolating encoder versus decoder contributions. If absorption were purely decoder-driven, then randomizing the decoder while keeping the encoder fixed should eliminate absorption; conversely, randomizing the encoder while keeping the decoder fixed should have no effect on absorption.

## 2.4 Factorial Decomposition: Isolating Encoder vs Decoder

The experimental design in this paper tests these predictions directly using a $2 \times 2$ factorial design crossing encoder (random/trained) and decoder (random/trained) configurations, first proposed by Tang et al. [@tang_sdl] from an information geometry perspective:

| Condition | Encoder | Decoder | Interpretation |
|-----------|---------|---------|----------------|
| $C_A$ | Random | Random | Baseline geometry only |
| $C_B$ | Trained | Random | Encoder alignment only |
| $C_C$ | Random | Trained | Decoder geometry only |
| $C_D$ | Trained | Trained | Full training |

If the decoder geometry hypothesis holds, Condition C should show elevated absorption relative to Condition A (decoder randomization matters). If the encoder alignment hypothesis holds, Condition B should show absorption comparable to Condition D (encoder is sufficient).

This factorial approach reveals a more nuanced picture than prior work assumed: the encoder's learned alignment with hierarchical structure is a primary driver of absorption, but the decoder's contribution is **configuration-dependent** rather than uniformly zero. The pilot experiments suggested decoder irrelevance ($C_C \approx C_A$), but the full 5-seed experiment reveals that Condition C has extreme seed-dependent variance (std = 17.13, range 0--43.84), indicating the decoder does contribute in some configurations. This finding means prior work attributing absorption to decoder geometry is not wrong, but incomplete.

## 2.5 Feature Sensitivity and the Pareto Frontier Question

A parallel line of work has examined feature sensitivity, measuring how much a feature's output changes in response to steering interventions [@hu_sensitivity]. Hu et al. introduced steering coefficient variance as a measure of feature sensitivity and raised the possibility of an irreducible trade-off between sensitivity and absorption --- improving one might necessarily degrade the other, forming a Pareto frontier.

This theoretical prediction motivates the hypothesis (H_Pareto) tested in the Experiments section: whether a sensitivity-absorption Pareto frontier exists in practice. The absorption-sensitivity uncertainty relation from information geometry theory predicts an irreducible trade-off. However, the empirical results in this paper suggest this frontier, if it exists, is not easily detectable using current measurement approaches on synthetic hierarchies.

Safety-critical applications of SAEs require understanding whether absorbed features retain useful steering signal. Basu et al. [@basu_actionability] highlighted the tension between interpretability and actionability: even if SAEs identify concerning features, interventions based on those features may not reliably change model behavior. If absorption merely compresses redundant representations without destroying causal information, then interventions on absorbed features might still be effective.

## 2.6 Relationship to Hierarchical Feature Structures

Tang et al. [@tang_sdl] provided theoretical grounding for sparse dictionary learning in mechanistic interpretability. Their analysis suggests that hierarchical feature structures emerge naturally from training on data with inherent compositional structure. The presence of hierarchical structure raises the question of how hierarchy strength --- the cosine similarity between parent and child feature directions --- affects absorption.

Cunningham et al. [@cunningham_mi] and subsequent work in circuit-level interpretability have documented that features form hierarchies at multiple scales in transformer models. Understanding absorption therefore requires understanding how hierarchical structure interacts with the SAE training dynamics.

---

<!-- FIGURES
- None
-->
