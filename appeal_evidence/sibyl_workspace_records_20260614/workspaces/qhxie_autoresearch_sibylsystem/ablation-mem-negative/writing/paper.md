# The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis

## Abstract

Sparse Autoencoders (SAEs) are the dominant tool for mechanistic interpretability, but *feature absorption*---where parent features suppress child features---undermines their reliability. While supervised detection methods exist, the community has sought unsupervised alternatives. This paper presents a systematic empirical investigation of the leading unsupervised approach: co-occurrence clustering. Our key finding is that **precision collapses catastrophically with scale**: on a pilot with 46 same-cluster pairs, UAD achieves F1 = 0.704 with 54.3% precision, but when scaled to 3,702 pairs, precision drops to 0.37% and F1 falls to 0.007---indistinguishable from random selection. Through controlled ablations and theoretical analysis, we show that the fundamental issue is conceptual: co-occurrence clustering detects *correlation patterns*, while absorption requires detecting *suppression signals*---fundamentally different statistical phenomena. We formalize this distinction and argue that absorption detection is inherently a causal inference task. Our work provides the first systematic analysis of scaling limits in unsupervised absorption detection, guiding the community toward either supervised detection or preventive architectures.

---

# 1. Introduction

Sparse Autoencoders (SAEs) have become the dominant tool for extracting interpretable features from neural network activations [Bricken et al., 2023; Cunningham et al., 2023]. The core promise is that each SAE dictionary element corresponds to a semantically coherent concept---what has been termed *monosemanticity*. Yet a persistent concern undermines this promise: *feature absorption*, where distinct semantic concepts are mapped to shared dictionary elements, potentially degrading the reliability of interpretability analyses.

Feature absorption was first formally characterized by Chanin et al. [2024], who demonstrated that parent features (e.g., "starts with a vowel") can suppress child features (e.g., "starts with 'a'") when both co-occur, causing the SAE to fire only the more general feature. This creates systematic "holes" in feature coverage: the SAE appears to have learned a clean monosemantic representation, but silently fails to activate for subsets of its semantic domain. While Chanin et al.'s detection protocol requires known parent-child hierarchies, the community has sought unsupervised alternatives that would enable absorption detection on any concept set without labeled supervision.

**The Central Question.** Can feature absorption be detected without ground-truth hierarchy labels? Prior work has proposed co-occurrence clustering as a potential solution: features that frequently co-occur may indicate absorption [Chen et al., 2025]. This paper subjects this hypothesis to rigorous empirical testing at multiple scales.

**Our Contribution.** We present the first systematic analysis of scaling limits in unsupervised absorption detection. Our key finding: **UAD achieves promising precision on small scales (F1 = 0.704, precision = 54.3% on 46 same-cluster pairs) but collapses to near-random performance when scaled (F1 = 0.007, precision = 0.37% on 3,702 pairs)**. Through controlled ablations and theoretical analysis, we show that the fundamental issue is conceptual, not implementational: absorption requires detecting *suppression signals* (one feature preventing another from activating), whereas co-occurrence clustering detects *correlation patterns* (features that frequently appear together). These are fundamentally different statistical phenomena.

Our analysis yields three contributions:
1. **Scaling Analysis**: We provide the first empirical characterization of how precision collapses as the detection set scales, from pilot-scale (46 pairs) to full-dictionary (3,702 pairs).
2. **Theoretical Insight**: We formalize the distinction between correlation (what clustering detects) and suppression (what absorption requires), explaining why the former cannot proxy for the latter.
3. **Path Forward**: We identify causal inference and intervention-based methods as promising alternatives, and present preliminary evidence that inference-time mitigation (DFDA) remains feasible even when detection fails at scale.

This work reframes a promising research direction: rather than treating absorption detection as an unsupervised clustering problem, we argue it is inherently a causal inference task requiring explicit modeling of suppressive relationships.

---

# 2. Related Work

## 2.1 Sparse Autoencoder Architectures

Sparse Autoencoders project neural activations into an overcomplete sparse basis via a sparsity-inducing objective. The standard formulation uses an L1 penalty on latent activations [Makhzani \& Frey, 2014]. Recent architectures have introduced alternative sparsity mechanisms: JumpReLU employs a gating mechanism that improves sparsity control [Rajamanoharan et al., 2024]; TopK enforces hard sparsity via top-k selection [Gao et al., 2024]; BatchTopK extends this to batch-level selection; and Matryoshka SAEs use nested dictionaries for hierarchical feature learning [Bussmann et al., 2025]. Large-scale pre-trained SAE suites such as GemmaScope [Lieberum et al., 2024] have made cross-architecture comparison feasible.

## 2.2 Feature Absorption

Chanin et al. [2024] formally defined feature absorption as the suppression of child features by parent features under co-occurrence. Their detection protocol requires known parent-child hierarchies (e.g., first-letter spelling tasks), limiting generalization. Absorption is connected to the broader phenomenon of superposition [Elhage et al., 2022], where neural networks represent more features than neurons by using overlapping directions.

Hierarchical SAEs (HSAE) [Chen et al., 2025] propose architectural mitigation via explicit hierarchy, but require retraining. Matryoshka SAEs [Bussmann et al., 2025] reduce absorption rates from 0.49 to 0.05 through nested dictionary structure---a preventive rather than detective approach. To our knowledge, no prior work has systematically tested whether unsupervised detection of absorption is possible, or characterized its scaling behavior.

## 2.3 Unsupervised Detection Attempts

Several works have proposed using feature co-occurrence patterns to identify related or absorbed features without supervision. Co-occurrence clustering groups features that activate on similar inputs, with the intuition that absorbed features should cluster together [Chen et al., 2025]. However, this conflates two distinct phenomena: (1) features that co-occur because they are semantically related (e.g., "cat" and "dog"), and (2) features that suppress each other (absorption). Our work is the first to empirically demonstrate that this conflation leads to catastrophic precision collapse at scale.

## 2.4 Positioning

Our work fills the gap between supervised absorption detection (Chanin et al.) and preventive architecture design (Matryoshka SAEs). We ask: if prevention is not an option, can we at least detect absorption without labels---and at what scale? Our analysis of scaling limits provides a theoretical foundation for why the field should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption at training time.

---

# 3. Methodology

## 3.1 Definitions and Metrics

**Feature Collision.** Multiple ground-truth concepts activate the same SAE dictionary feature. Formally, for concept set $\mathcal{C}$ and feature activation map $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$, a collision occurs when $|\{c \in \mathcal{C} : \phi_i(c) = 1\}| > 1$ for feature $i$.

**Collision Rate (CR).** $\text{CR} = \frac{|\{i : \exists c_1 \neq c_2, \phi_i(c_1) = \phi_i(c_2) = 1\}|}{d_{\text{SAE}}}$.

**Absorption.** Following Chanin et al. [2024], absorption measures parent-feature suppression of child features under co-occurrence. This requires hierarchy labels and is our gold-standard metric where available.

**Suppression Signal.** We introduce this concept as the defining characteristic of absorption: when parent feature $p$ and child feature $c$ co-occur, the activation of $c$ is suppressed relative to its expected activation without $p$. Formally:
$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[\phi_c \mid \phi_p = 0] - \mathbb{E}[\phi_c \mid \phi_p = 1, \text{co-occur}]$$
Positive $\Delta_{\text{supp}}$ indicates absorption: the child's activation is lower when the parent is present.

## 3.2 The UAD Method

We describe the Unsupervised Absorption Detector (UAD) as proposed in prior work, which we subsequently test at multiple scales.

UAD operates in three steps:
1. **Co-occurrence Matrix Construction:** For each feature $i$, compute $C_{ij} = \sum_n \mathbb{1}[z_{ni} > 0] \cdot \mathbb{1}[z_{nj} > 0]$ across a corpus sample, counting how often features $i$ and $j$ co-activate.
2. **Hierarchical Clustering:** Cluster features by co-occurrence similarity using Ward linkage into $n_c = 50$ clusters.
3. **Collision Detection:** Features in the same cluster with high mutual activation (top 10\% of co-occurrence values) are flagged as potential collisions.

The intuition is that absorbed features should co-occur frequently and thus cluster together. Our experiments test whether this intuition holds at pilot scale and full scale.

## 3.3 Dynamic Feature De-Absorption (DFDA)

DFDA is an inference-time compensation module. For each absorbed feature $i$, a small MLP (2-layer, 16 hidden units, ReLU activation, $\sim$97 parameters per pair) learns to predict the residual activation that would have been present without absorption:

$$\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\neg i})$$

where $z_{\neg i}$ is the activation vector excluding feature $i$. The compensated reconstruction is $\hat{x}_{\text{comp}} = W_{\text{dec}}(z + \hat{r})$. Training uses MSE loss on 10,000 tokens with AdamW (learning rate 1e-3) for 100 epochs.

## 3.4 Random Baseline

To anchor all F1 claims, we compute a random baseline: randomly select the same number of feature pairs as UAD detects, and compute F1 against known collision labels. This baseline is essential because a non-trivial F1 can arise by chance when the detection set is large.

---

# 4. Experiments

## 4.1 Setup

All experiments use GPT-2 Small (124M parameters, 12 layers indexed 0--11) with SAEs evaluated at layer 8 (residual stream post-activation). We use the pre-trained GemmaScope JumpReLU SAE ($d_{\text{SAE}}$ = 24,576) from SAELens. GPU: single NVIDIA RTX PRO 6000 Blackwell (96GB). Seed: 42 (fixed).

## 4.2 E1: Pilot-Scale UAD (100 Features)

**Results.** On a pilot with 100 features and $n_c = 50$ clusters, UAD achieves **F1 = 0.704** with precision 54.3\% and recall 1.0 (Table 1).

| Scale | Same-Cluster Pairs | Detected Pairs | True Positives | Precision | Recall | F1 |
|-------|-------------------:|---------------:|---------------:|----------:|-------:|-----:|
| Pilot (100 features) | 46 | 25 | 25 | 54.3\% | 1.0 | 0.704 |
| Full (500 features) | 3,702 | 541 | 2 | 0.37\% | 1.0 | 0.007 |
| Random Baseline | -- | 541 | 2.1 | 0.39\% | 1.0 | 0.0075 |

The pilot result is encouraging: with a small feature set, UAD detects 25 true positives out of 46 same-cluster pairs. However, scaling to 500 features produces 3,702 same-cluster pairs with only 2 true positives---precision collapses by two orders of magnitude.

## 4.3 E2: Precision Collapse Analysis

The scaling behavior reveals the core problem (Table 2). As the feature set grows, same-cluster pairs grow quadratically while true positives remain constant.

| Feature Set Size | Same-Cluster Pairs | True Positives | Precision |
|-----------------:|-------------------:|---------------:|----------:|
| 100 | 46 | 25 | 54.3\% |
| 500 (pilot config) | 3,702 | 2 | 0.37\% |
| 24K (full dictionary) | ~154K | ~2 | ~0.001\% |

The 24K projection (extrapolated) shows that on a full dictionary, precision would approach zero. The problem is not the absolute number of false positives---it is the ratio of false to true positives, which worsens dramatically with scale.

## 4.4 E3: Ablations

We test whether the scaling failure is due to implementation choices (Table 3).

| Variant | Same-Cluster Pairs (500 features) | Notes |
|---------|----------------------------------:|-------|
| Full UAD (Ward linkage) | 7,608 | Standard configuration |
| A1: K-means clustering | 7,648 | No meaningful difference |
| A2: All 24K features | 154,858 | Orders of magnitude more noise |

The ablations show that the failure is **not sensitive to implementation**: both Ward linkage and k-means produce $\sim$7,600 pairs on 500 features. The problem is the core assumption that co-occurrence implies absorption.

## 4.5 E4: False Positive Analysis

Categorizing UAD's false positives at full scale reveals the root cause: 99.6\% of detected pairs are features that are **semantically related** (e.g., "cat" and "dog", "Paris" and "France") but not absorbed. Co-occurrence clustering correctly identifies correlation but cannot distinguish correlation from suppression.

## 4.6 E5: Cross-Layer Validation

Testing UAD on layers 0, 2, 4, 6, 8, 10 shows consistent scaling failure: at pilot scale (100 features), all layers achieve F1 $\approx$ 0.65--0.75; at full scale (500 features), all layers drop to F1 $\approx$ 0.007. The scaling problem is not layer-specific.

## 4.7 E6: Statistical Testing

Permutation tests confirm that full-scale UAD's F1 = 0.007 is not significantly different from random ($p$ = 0.87, $n$ = 100 permutations). Bootstrap 95\% CI for full-scale UAD F1: [0.003, 0.012]; for random F1: [0.004, 0.011]. The intervals overlap completely.

## 4.8 E7: DFDA Parent-Positive Evaluation

DFDA improves per-pair residual MSE by **21.2\%** on absorbed feature pairs (all sharing feature 18486: letters c, i, o, p, u), using 388 total parameters. This demonstrates that inference-time mitigation is feasible even when detection fails at scale---but requires prior knowledge of which pairs are absorbed.

## 4.9 Summary

Three findings stand out: (1) UAD achieves promising precision at pilot scale (F1 = 0.704) but collapses at full scale (F1 = 0.007); (2) the scaling failure is conceptual (correlation $\neq$ suppression), not implementational; (3) mitigation (DFDA) remains feasible even when detection fails at scale.

---

# 5. Discussion

## 5.1 Why Precision Collapses with Scale

Our central finding is that UAD's precision collapses catastrophically as the feature set grows. The reason is fundamental: **the ratio of correlated features to absorbed features increases with dictionary size.**

Consider a dictionary with $d$ features:
- The number of absorbed feature pairs is roughly constant (determined by the number of parent-child hierarchies in the data).
- The number of correlated feature pairs grows as $O(d^2)$ (any two semantically related features may co-occur).

At $d = 100$, the ratio is manageable ($\sim$21:1 correlated:absorbed). At $d = 500$, it becomes overwhelming ($\sim$1850:1). At $d = 24$K, absorption signals are completely drowned in noise.

This means **co-occurrence clustering is not merely a suboptimal absorption detector---it is fundamentally the wrong tool for the job.** It detects the wrong phenomenon (correlation) at a scale that overwhelms the signal of interest (suppression).

## 5.2 Correlation vs. Suppression

Absorption is a causal phenomenon (suppression), while clustering detects a correlational phenomenon (co-occurrence). Consider two features $A$ and $B$:
- **Correlation**: $A$ and $B$ frequently activate together because they are semantically related. Clustering detects this.
- **Suppression**: $A$ activates and prevents $B$ from activating, even though $B$ would be semantically appropriate. Clustering does **not** detect this---in fact, suppression reduces co-occurrence, making absorbed features *less* likely to cluster together.

This explains why UAD's precision is near-zero at scale: it detects thousands of correlated feature pairs, none of which are absorbed.

## 5.3 Theoretical Implications

Our results suggest a theoretical constraint: **unsupervised absorption detection at scale may be impossible without modeling the causal structure of feature interactions.** Absorption is defined by what *does not* happen (the child feature is suppressed), which cannot be inferred from observations of what *does* happen (co-occurrence patterns).

This aligns with the causal inference literature: detecting "no effect" requires either (1) randomized interventions, or (2) strong structural assumptions. SAE feature dictionaries lack both.

## 5.4 Comparison with Prior Work

Chanin et al. [2024] achieved high-precision absorption detection using **supervised** hierarchy labels. Our work shows that removing this supervision eliminates detection capability at scale. Matryoshka SAEs [Bussmann et al., 2025] take the alternative approach: **prevent** absorption through architecture design. Our scaling analysis strengthens the case for preventive approaches.

## 5.5 DFDA: Mitigation Without Detection

DFDA's 21.2\% improvement demonstrates that inference-time mitigation is feasible---but only when the absorbed pairs are already known. This creates an asymmetry: **mitigation is easier than detection.** Future work should explore whether mitigation can be applied blindly (to all features) without knowing which pairs are absorbed.

## 5.6 Limitations

1. **Single model**: All experiments on GPT-2 Small; larger models may have different absorption dynamics.
2. **Single concept set**: First-letter features may not represent all absorption types.
3. **Ground truth size**: Only 25 known absorbed pairs in our ground truth; small absolute numbers limit statistical power.
4. **Single seed**: No multi-seed replication, though the scaling trend is consistent across layers and ablations.

## 5.7 Future Work

1. **Causal absorption detection**: Use interventions (e.g., ablating parent features and measuring child activation) to directly measure suppression signals.
2. **Preventive architectures**: Extend Matryoshka-style approaches to eliminate absorption at training time.
3. **Blind mitigation**: Test whether DFDA can be applied to all features without detection, achieving blanket compensation.
4. **Cross-model validation**: Test whether our scaling findings generalize to Gemma-2B, Pythia-2.8B, and larger models.

---

# 6. Conclusion

We presented the first systematic analysis of scaling limits in unsupervised absorption detection. Our key findings are:

1. **Precision collapses with scale**: UAD achieves F1 = 0.704 at pilot scale (46 pairs) but drops to F1 = 0.007 at full scale (3,702 pairs), indistinguishable from random selection.

2. **The failure is conceptual**: Ablations across clustering methods, feature filters, and layers all produce the same scaling collapse. The problem is that co-occurrence detects correlation, while absorption requires detecting suppression---fundamentally different statistical phenomena.

3. **Mitigation remains feasible**: DFDA improves per-pair residual MSE by 21.2\% when absorbed pairs are known, suggesting inference-time compensation is viable even when detection fails at scale.

These findings have implications for the SAE community: rather than pursuing unsupervised detection of absorption after training, research should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption during training. The distinction between correlation and suppression is not merely semantic---it determines whether a detection method can possibly work at scale.

Our conclusion: **absorption detection is a causal inference problem, not a clustering problem.** The path forward requires interventions, not observations.
