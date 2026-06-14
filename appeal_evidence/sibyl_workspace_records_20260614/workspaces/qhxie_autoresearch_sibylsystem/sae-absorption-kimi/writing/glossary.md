# Glossary

## Core Technical Terms

**Feature Absorption**
: The subsumption of parent-feature information by child features in a sparse autoencoder. When parent and child features co-occur, the SAE's sparsity loss incentivizes allocating representational capacity to the children at the parent's expense, causing the parent's information to be actively suppressed. First characterized analytically by Chanin et al. (2024).

**Feature Hedging**
: The opposite failure mode to absorption, where reconstruction loss drives correlated features to share latents rather than split them. Matryoshka SAEs reduce absorption but increase hedging in inner dictionary levels (Chanin et al., 2025).

**Sparse Autoencoder (SAE)**
: An autoencoder with an overcomplete hidden layer and sparsity constraints, trained to reconstruct input activations while using only a small fraction of latent dimensions per sample. Used in mechanistic interpretability to decompose neural network activations into interpretable features.

**Superposition**
: The hypothesis that neural networks represent more features than they have dimensions by encoding them as non-orthogonal linear combinations of activations (Elhage et al., 2022). This overcomplete representation creates polysemanticity, motivating the use of SAEs.

**Polysemanticity**
: The phenomenon where individual neurons or directions in a neural network respond to multiple unrelated stimuli, making direct interpretation intractable. SAEs aim to recover monosemantic features from polysemantic representations.

**Monosemanticity**
: The property of a feature (or latent) that encodes a single, human-interpretable concept. The goal of SAE training is to produce monosemantic latents.

## SAE Architectures

**Standard ReLU SAE**
: The baseline SAE architecture using ReLU activation and L1 sparsity penalty. The encoder computes $z = \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$, and the loss includes a reconstruction term plus $\lambda_1 \|z\|_1$.

**TopK SAE**
: An SAE variant that replaces the L1 sparsity penalty with a hard top-k activation: only the $k$ largest-magnitude latent activations are retained, all others are set to zero. This enforces exact sparsity without a learnable threshold.

**Matryoshka SAE**
: An SAE architecture that learns nested dictionaries of increasing size, where each inner dictionary is a subset of the next. Introduced by Bussmann et al. (2025), combining multi-scale decomposition, batch TopK sparsity, and hierarchical loss weighting.

**MultiScale SAE**
: A component of Matryoshka SAEs referring specifically to the nested dictionary decomposition. In our component-isolated study, +MultiScale tests only the multi-scale dictionary structure without the full Matryoshka loss.

**OrtSAE / Orthogonality SAE**
: An SAE variant that adds an orthogonality penalty on the decoder matrix, encouraging decoder directions to be near-orthogonal. Introduced by Korznikov et al. (2025).

**Gated SAE**
: An SAE architecture that decouples the detection path (which features are active) from the magnitude path (how strongly they fire). Uses a gating mechanism to control activation independently of reconstruction.

**JumpReLU SAE**
: An SAE variant that uses a thresholded ReLU with a learned threshold per feature, allowing features to have different activation thresholds.

## Metrics and Evaluation

**Absorption Rate (Ground-Truth)**
: In our synthetic benchmark, the fraction of parent features that are subsumed by their child features, computed directly from known ground-truth parent-child relationships. No probes or AUROCs are used. Formula: $A = \frac{1}{|\mathcal{H}|} \sum_{(p,c) \in \mathcal{H}} \mathbb{1}[\text{parent } p \text{ absorbed by child } c]$.

**Absorption Rate (SAEBench / Probe-Based)**
: The dominant community metric based on first-letter classification tasks. Uses logistic regression probes trained on base-model residuals, full SAE latents, and k-sparse latents. Formula: $A_{\text{full}} = \max(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}})$.

**Feature Recovery MCC**
: Matthews correlation coefficient between learned latent assignments and ground-truth feature assignments. Measures how well the SAE recovers the true feature structure. Computed via Hungarian matching between latent directions and ground-truth features.

**L0 Sparsity**
: The average number of active (non-zero) latent dimensions per input sample. A hard measure of sparsity (as opposed to L1, which is a soft proxy). For TopK SAEs, L0 is exactly $k$ by construction. Our results show absorption rate strongly correlates with L0 (r = 0.87), suggesting sparsity level is the operative mechanism.

**Dead Latents**
: Latent dimensions that never activate during training or evaluation. A known pathology in SAEs, particularly severe in TopK architectures. We report dead latent rates as a secondary metric to ensure absorption reductions are not artifacts of feature suppression.

**Additive Expectation**
: In component interaction analysis, the predicted outcome if components combine linearly. For Full Matryoshka, the additive expectation is computed from the individual reductions of TopK and MultiScale. The observed value deviating from this expectation indicates non-additive (synergistic or antagonistic) interaction.

**Hedging Score**
: The fraction of latents that incorrectly mix correlated features, as defined by Chanin et al. (2025). Measures the opposite failure mode to absorption.

**Cohen's d**
: A standardized effect size measure: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$. Conventional thresholds: small (0.2), medium (0.5), large (0.8).

## Dataset and Benchmark Terms

**SynthSAEBench-16k**
: A synthetic benchmark for SAE evaluation with 16,384 ground-truth features (10,884 hierarchical), organized into 128 root trees of depth 3 and branching factor 4. Built into SAELens (Chanin & Garriga-Alonso, 2026).

**First-Letter Hierarchy**
: The artificial hierarchies used in SAEBench's absorption evaluation, defined by character-level properties (e.g., parent: "starts with S", children: "short", "small", "smart").

**Semantic Hierarchy**
: Real-world conceptual hierarchies derived from WordNet (e.g., parent: "animal", children: "dog", "cat").

**Co-occurrence Confound**
: The finding that probe-based absorption metrics score higher on semantically correlated non-hierarchical pairs than on true hierarchies, proving the metric detects correlation rather than containment.

**Ceiling Effect**
: A measurement artifact where all values hit the maximum possible score, rendering the metric unable to discriminate conditions. In our prior work, all probe AUROCs = 1.0, collapsing the absorption formula.

## Statistical Terms

**Component-Isolated Design**
: An experimental design where variants differ by exactly one component, enabling causal attribution of effects to that component. The core methodology of this paper.

**Construct Validity**
: The degree to which a measurement instrument captures the theoretical construct it claims to measure. From psychometric theory (Cronbach & Meehl, 1955).

**One-Way ANOVA**
: Analysis of variance testing whether means differ across three or more independent groups. Used to test whether absorption rates differ across SAE variants.

**Tukey HSD**
: Tukey's Honestly Significant Difference test, a post-hoc procedure following ANOVA that controls the family-wise error rate across all pairwise comparisons.

**Holm-Bonferroni Correction**
: A sequential procedure for adjusting p-values across multiple comparisons that is less conservative than standard Bonferroni while maintaining family-wise error control.

**Antagonistic Interaction**
: An interaction effect where combining two components produces a worse outcome than either component alone. In our study, Full Matryoshka (TopK + MultiScale + hierarchical loss) shows an antagonistic interaction: its observed absorption (0.066) is worse than either TopK (0.056) or MultiScale (0.055) individually, contradicting the additive expectation.

**Component Interaction Analysis**
: A method for testing whether combining architectural components produces additive, synergistic, or antagonistic effects. We compute the expected absorption under additivity and compare it to the observed absorption of the combined architecture.

**Mixed-Effects Model**
: A statistical model with both fixed effects (variant, which we manipulate) and random effects (replicate seed, which we sample from a population).

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| LLM | Large Language Model |
| MCC | Matthews Correlation Coefficient |
| MSE | Mean Squared Error |
| AUROC | Area Under the Receiver Operating Characteristic curve |
| ANOVA | Analysis of Variance |
| HSD | Honestly Significant Difference |
| CI | Confidence Interval |
| ReLU | Rectified Linear Unit |
| TopK | Top-K (hard sparsity selecting k largest activations) |
| L1 | L1 norm (sum of absolute values, used as sparsity penalty) |
| L0 | L0 "norm" (count of non-zero elements, true sparsity measure) |
| PCA | Principal Component Analysis |
| SAEBench | Standardized benchmark suite for SAE evaluation |
| SAELens | Library for training and analyzing SAEs |

## Preferred Phrasing

- "fine-tuning" (not "finetuning")
- "few-shot" (not "few shot")
- "ground truth" (not "ground-truth" when used as noun; "ground-truth" as adjective)
- "sparsity" (not "sparseness")
- "overcomplete" (not "over-complete")
- "latent" (not "hidden unit" or "neuron" when referring to SAE dictionary elements)
- "decoder" (not "dictionary" when referring to $W_{\text{dec}}$)
- "encoder" (not "analysis matrix" when referring to $W_{\text{enc}}$)
- "absorption rate" (not "absorption score" when referring to the ground-truth metric; "absorption score" reserved for SAEBench probe-based metric)
- "component-isolated" (hyphenated when used as adjective)
- "multi-scale" (hyphenated)
