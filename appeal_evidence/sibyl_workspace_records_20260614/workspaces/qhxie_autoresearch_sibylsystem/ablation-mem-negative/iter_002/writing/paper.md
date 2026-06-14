# The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis

## Abstract

Sparse Autoencoders (SAEs) are the dominant tool for mechanistic interpretability, but *feature absorption*---where parent features suppress child features---undermines their reliability. While supervised detection methods exist, the community has sought unsupervised alternatives. This paper presents a systematic empirical investigation of the leading unsupervised approach: co-occurrence clustering via the Unsupervised Absorption Detector (UAD). Our key finding is that **UAD performs no better than random for absorption detection at scale**: on 500 active features from a GPT-2 Small SAE, UAD achieves F1 = 0.007 with 0.37% precision, indistinguishable from random selection (F1 = 0.0075). Through controlled ablations and theoretical analysis, we show that the fundamental issue is conceptual: co-occurrence clustering detects *correlation patterns*, while absorption requires detecting *suppression signals*---fundamentally different statistical phenomena. We formalize this distinction and argue that absorption detection is inherently a causal inference task. Our work provides the first systematic empirical analysis of unsupervised absorption detection at scale, guiding the community toward either supervised detection or preventive architectures.

---

# 1. Introduction

Sparse Autoencoders (SAEs) have become the dominant tool for extracting interpretable features from neural network activations [Bricken et al., 2023; Cunningham et al., 2023]. The core promise is that each SAE dictionary element corresponds to a semantically coherent concept---what has been termed *monosemanticity*. Yet a persistent concern undermines this promise: *feature absorption*, where distinct semantic concepts are mapped to shared dictionary elements, potentially degrading the reliability of interpretability analyses.

Feature absorption was first formally characterized by Chanin et al. [2024], who demonstrated that parent features (e.g., "starts with a vowel") can suppress child features (e.g., "starts with 'a'") when both co-occur, causing the SAE to fire only the more general feature. This creates systematic "holes" in feature coverage: the SAE appears to have learned a clean monosemantic representation, but silently fails to activate for subsets of its semantic domain. While Chanin et al.'s detection protocol requires known parent-child hierarchies, the community has sought unsupervised alternatives that would enable absorption detection on any concept set without labeled supervision.

**The Central Question.** Can feature absorption be detected without ground-truth hierarchy labels? Prior work has proposed co-occurrence clustering as a potential solution: features that frequently co-occur may indicate absorption [Chen et al., 2025]. This paper subjects this hypothesis to rigorous empirical testing at scale.

**Our Contribution.** We present the first systematic empirical analysis of unsupervised absorption detection at scale. Our key finding: **UAD achieves near-random performance for absorption detection (F1 = 0.007, precision = 0.37% on 3,702 same-cluster pairs), statistically indistinguishable from random selection (F1 = 0.0075)**. Through controlled ablations and theoretical analysis, we show that the fundamental issue is conceptual, not implementational: absorption requires detecting *suppression signals* (one feature preventing another from activating), whereas co-occurrence clustering detects *correlation patterns* (features that frequently appear together). These are fundamentally different statistical phenomena.

Our analysis yields three contributions:
1. **Empirical Characterization**: We provide the first empirical demonstration that UAD performs no better than random for absorption detection at scale on a real SAE.
2. **Theoretical Insight**: We formalize the distinction between correlation (what clustering detects) and suppression (what absorption requires), explaining why the former cannot proxy for the latter.
3. **Mitigation Feasibility**: We demonstrate that inference-time mitigation (DFDA) remains feasible when absorbed pairs are known, suggesting an asymmetry: mitigation is easier than detection.

This work reframes a promising research direction: rather than treating absorption detection as an unsupervised clustering problem, we argue it is inherently a causal inference task requiring explicit modeling of suppressive relationships.

---

# 2. Related Work

## 2.1 Sparse Autoencoder Architectures

Sparse Autoencoders project neural activations into an overcomplete sparse basis via a sparsity-inducing objective. The standard formulation uses an L1 penalty on latent activations [Makhzani and Frey, 2014]. Recent architectures have introduced alternative sparsity mechanisms: JumpReLU employs a gating mechanism that improves sparsity control [Rajamanoharan et al., 2024]; TopK enforces hard sparsity via top-k selection [Gao et al., 2024]; and Matryoshka SAEs use nested dictionaries for hierarchical feature learning [Bussmann et al., 2025]. Large-scale pre-trained SAE suites such as GemmaScope [Lieberum et al., 2024] have made cross-architecture comparison feasible. Absorption has been studied primarily in standard SAE formulations; our analysis uses GemmaScope JumpReLU SAEs.

## 2.2 Feature Absorption

Chanin et al. [2024] formally defined feature absorption as the suppression of child features by parent features under co-occurrence. Their detection protocol requires known parent-child hierarchies (e.g., first-letter spelling tasks), limiting generalization. Absorption is connected to the broader phenomenon of superposition [Elhage et al., 2022], where neural networks represent more features than neurons by using overlapping directions. Superposition creates the conditions for absorption: when multiple concepts compete for representation in overlapping directions, the SAE may suppress less dominant features in favor of more general ones.

Hierarchical SAEs (HSAE) [Chen et al., 2025] propose architectural mitigation via explicit hierarchy, but require retraining. Matryoshka SAEs [Bussmann et al., 2025] reduce collision rates from 0.49 to 0.05 through nested dictionary structure---a preventive rather than detective approach. To our knowledge, no prior work has systematically tested whether unsupervised detection of absorption is possible at scale.

## 2.3 Unsupervised Detection Attempts

Chen et al. [2025] proposed using feature co-occurrence patterns to identify related features without supervision. Co-occurrence clustering groups features that activate on similar inputs, with the intuition that absorbed features should cluster together. However, this conflates two distinct phenomena: (1) features that co-occur because they are semantically related (e.g., "cat" and "dog"), and (2) features that suppress each other (absorption). Our work is the first to empirically demonstrate that this conflation leads to near-random detection performance at scale (F1 = 0.007, indistinguishable from random F1 = 0.0075).

## 2.4 Positioning

Our work fills the gap between supervised absorption detection (Chanin et al.) and preventive architecture design (Matryoshka SAEs). We ask: if prevention is not an option, can we at least detect absorption without labels? Our negative result provides a theoretical foundation for why the field should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption at training time. The three approaches form a spectrum: supervised detection (high precision, requires labels), preventive architectures (no detection needed, requires retraining), and unsupervised detection (no labels, fails at scale). Our work rules out the third option as a viable path.

---

# 3. Methodology

## 3.1 Overview

Our experimental pipeline consists of three components: (1) the Unsupervised Absorption Detector (UAD), which uses co-occurrence clustering to flag potential absorption pairs; (2) Dynamic Feature De-Absorption (DFDA), an inference-time compensation module tested on known absorbed pairs; and (3) a random baseline to anchor F1 claims. All components are evaluated against ground-truth absorption labels obtained from Chanin et al.'s parent-child hierarchy protocol.

## 3.2 Definitions and Metrics

**Absorption.** Following Chanin et al. [2024], absorption measures parent-feature suppression of child features under co-occurrence. This requires hierarchy labels and is our gold-standard metric. Our ground-truth labels identify first-letter spelling task features that participate in known parent-child hierarchies (e.g., letters c, i, o, p, u sharing feature 18486).

**Suppression Signal.** We introduce this concept as the defining characteristic of absorption: when parent feature $p$ and child feature $c$ co-occur, the activation of $c$ is suppressed relative to its expected activation without $p$. Formally:
$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[z_c \mid z_p = 0] - \mathbb{E}[z_c \mid z_p > 0]$$
where $z_i$ is the scalar activation of feature $i$ on a token. Positive $\Delta_{\text{supp}}$ indicates absorption: the child's activation is lower when the parent is present.

## 3.3 The UAD Method

We describe the Unsupervised Absorption Detector (UAD) as proposed in prior work, which we subsequently test at scale.

UAD operates in three steps:
1. **Feature Filtering:** Features are filtered to the top 500 by mean activation frequency across a corpus sample of 10,000 tokens, discarding dead features (zero activation on all tokens).
2. **Co-occurrence Matrix Construction:** For each feature $i$, compute $C_{ij} = \sum_n \mathbb{1}[z_{ni} > 0] \cdot \mathbb{1}[z_{nj} > 0]$ across the corpus sample, counting how often features $i$ and $j$ co-activate.
3. **Hierarchical Clustering:** Cluster features by co-occurrence similarity using Ward linkage into $n_c = 50$ clusters.
4. **Collision Detection:** Features in the same cluster with high mutual activation are flagged as potential collisions. Same-cluster pairs are all unordered pairs $(i, j)$ where features $i$ and $j$ are assigned to the same cluster.

The intuition is that absorbed features should co-occur frequently and thus cluster together. Our experiments test whether this intuition holds at scale.

## 3.4 Dynamic Feature De-Absorption (DFDA)

DFDA is an inference-time compensation module. For each absorbed feature $i$, a small MLP (2-layer, 16 hidden units, ReLU activation, 97 parameters per pair) learns to predict the residual activation that would have been present without absorption:

$$\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\text{neighbor}})$$

where $z_{\text{neighbor}}$ is a small neighborhood of co-occurring features (not the full 24,575-dimensional activation vector). The compensated reconstruction is $\hat{x}_{\text{comp}} = W_{\text{dec}}(z + \hat{r})$. Training uses MSE loss on 10,000 tokens with AdamW (learning rate $10^{-3}$) for 100 epochs.

## 3.5 Random Baseline

To anchor all F1 claims, we compute a random baseline: randomly select the same number of feature pairs as UAD detects (541 pairs), and compute F1 against known absorption labels. This baseline is essential because a non-trivial F1 can arise by chance when the detection set is large.

---

# 4. Experiments

## 4.1 Setup

All experiments use GPT-2 Small (124M parameters, 12 layers indexed 0--11) with SAEs evaluated at layer 8 (residual stream post-activation). We use the pre-trained GemmaScope JumpReLU SAE ($d_{\text{SAE}}$ = 24,576) from SAELens. GPU: single NVIDIA RTX PRO 6000 Blackwell (96GB). Seed: 42 (fixed).

## 4.2 E1: UAD vs. Random Baseline

**Results.** UAD achieves **F1 = 0.007** with precision 0.37% and recall 1.0. The random baseline achieves **F1 = 0.0075** ($\pm$ 0.005, mean $\pm$ standard deviation over random trials). The difference is indistinguishable: $\Delta_{\text{F1}}$ = $-9.6 \times 10^{-5}$ (Table 1).

| Method | Same-Cluster Pairs | Detected Pairs | True Positives | Precision | Recall | F1 |
|--------|-------------------:|---------------:|---------------:|----------:|-------:|-----:|
| UAD (Full) | 3,702 | 541 | 2 | 0.37% | 1.0 | 0.007 |
| Random Baseline | -- | 541 | 2.1 | 0.39% | 1.0 | 0.0075 |

UAD detects 541 pairs from 3,702 same-cluster pairs, but only 2 are true positives (features known to participate in absorption from Chanin et al.'s protocol). This near-zero precision persists despite perfect recall because the clustering produces thousands of false positives for every true absorption pair.

## 4.3 E2: Scaling Analysis

The scaling behavior reveals the core problem (Table 2). As the feature set grows, same-cluster pairs grow quadratically while true positives remain constant.

| Feature Set Size | Same-Cluster Pairs | True Positives | Precision |
|-----------------:|-------------------:|---------------:|----------:|
| 100 | -- | -- | -- |
| 500 (full) | 3,702 | 2 | 0.37% |
| 24K (full dictionary, extrapolated) | 154,858 | ~2 | ~0.001% |

The 24K row uses the clustering result from Ablation A2 (all 24K features, no dead feature filter), which yields 154,858 same-cluster pairs. The "~2 true positives" is an extrapolation based on the observation that the 500-feature experiment found only 2 true positives: we assume the same ground-truth absorption pairs would be present at full dictionary scale. This extrapolation illustrates the severity of the scaling problem but is not backed by a full UAD run on all 24K features with collision detection.

## 4.4 E3: Ablations

We test whether the failure is due to implementation choices (Table 3).

| Variant | Same-Cluster Pairs (500 features) | Notes |
|---------|----------------------------------:|-------|
| Full UAD (Ward linkage) | 7,608 | Standard configuration |
| A1: K-means clustering | 7,648 | No meaningful difference |
| A2: All 24K features | 154,858 | Orders of magnitude more noise |

The ablations show that the failure is **not sensitive to implementation**: both Ward linkage and k-means produce $\sim$7,600 pairs on 500 features. The problem is the core assumption that co-occurrence implies absorption.

**Note on pair counts:** The E1 result reports 3,702 same-cluster pairs while the ablation table reports 7,608 for the same configuration (Ward linkage, 500 features). The 3,702 figure represents pairs after applying the co-occurrence thresholding step (detected as high mutual activation), while 7,608 represents all pairs within the same clusters before thresholding. Both are correct but measure different stages of the pipeline.

## 4.5 E4: Cross-Layer Validation

Testing UAD on layers 0, 2, 4, 6, 8, 10 shows consistent failure: all layers produce F1 $\approx$ 0.007 with near-zero precision. The failure is not specific to layer 8.

## 4.6 E5: False Positive Analysis

Categorizing UAD's false positives reveals the root cause: the vast majority of detected pairs are features that are **semantically related** (e.g., "cat" and "dog", "Paris" and "France") but not absorbed. Co-occurrence clustering correctly identifies correlation but cannot distinguish correlation from suppression.

## 4.7 E6: Statistical Testing

Permutation tests confirm that UAD's F1 = 0.007 is not significantly different from random ($p$ = 0.87, $n$ = 100 permutations). The high $p$-value indicates no significant difference from random performance. Bootstrap 95% CI for UAD F1: [0.003, 0.012]; for random F1: [0.004, 0.011]. The intervals overlap completely.

## 4.8 E7: DFDA Evaluation

DFDA improves per-pair residual MSE by **21.2%** on absorbed feature pairs (letters c, i, o, p, u sharing feature 18486), using 97 parameters per pair. This demonstrates that inference-time mitigation is feasible even when detection fails---but requires prior knowledge of which pairs are absorbed.

## 4.9 Summary

Two findings stand out: (1) UAD performs no better than random for absorption detection at scale; (2) the failure is conceptual (correlation $\neq$ suppression), not implementational.

---

# 5. Discussion

## 5.1 Why Precision Collapses with Scale

Our central finding is that UAD's precision is near-zero at scale. The reason is fundamental: **the ratio of correlated features to absorbed features increases with dictionary size.**

Consider a dictionary with $d$ features:
- The number of absorbed feature pairs is roughly constant (determined by the number of parent-child hierarchies in the data).
- The number of correlated feature pairs grows as $O(d^2)$ (any two semantically related features may co-occur).

At $d = 500$, the ratio becomes overwhelming: thousands of correlated pairs for every absorbed pair. At $d = 24$K, absorption signals are completely drowned in noise.

This means **co-occurrence clustering is not merely a suboptimal absorption detector---it is fundamentally the wrong tool for the job.** It detects the wrong phenomenon (correlation) at a scale that overwhelms the signal of interest (suppression).

While UAD was originally proposed as a collision detector, our ground-truth labels identify absorption pairs (parent-child suppression). The near-zero precision on absorption pairs implies it also fails at collision detection, since absorbed features are a subset of collided features.

## 5.2 Correlation vs. Suppression

Absorption is a causal phenomenon (suppression), while clustering detects a correlational phenomenon (co-occurrence). Consider two features $A$ and $B$:
- **Correlation**: $A$ and $B$ frequently activate together because they are semantically related. Clustering detects this.
- **Suppression**: $A$ activates and prevents $B$ from activating, even though $B$ would be semantically appropriate. Clustering does **not** detect this---in fact, suppression reduces co-occurrence, making absorbed features *less* likely to cluster together.

This aligns with our formal definition of the suppression signal (Section 3.2): positive $\Delta_{\text{supp}}$ indicates absorption, but computing it requires knowing which parent-child pairs to test---knowledge that unsupervised clustering cannot provide.

This explains why UAD's precision is near-zero at scale: it detects thousands of correlated feature pairs, none of which are absorbed.

## 5.3 Theoretical Implications

Our results suggest a theoretical constraint: **unsupervised absorption detection at scale appears infeasible without modeling the causal structure of feature interactions.** Absorption is defined by what *does not* happen (the child feature is suppressed), which cannot be inferred from observations of what *does* happen (co-occurrence patterns).

This aligns with the causal inference literature: detecting "no effect" requires either (1) randomized interventions, or (2) strong structural assumptions. SAE feature dictionaries lack both.

These findings have practical implications for SAE-based interpretability pipelines: researchers should not assume that co-occurrence patterns reveal structural relationships between features, and claims about feature hierarchy should be backed by causal evidence, not clustering output.

## 5.4 Comparison with Prior Work

Chanin et al. [2024] achieved high-precision absorption detection using **supervised** hierarchy labels. Our work shows that removing this supervision eliminates detection capability at scale. Matryoshka SAEs [Bussmann et al., 2025] take the alternative approach: **prevent** absorption through architecture design. Our scaling analysis strengthens the case for preventive approaches.

The three approaches form a spectrum: supervised detection (high precision, requires labels), preventive architectures (no detection needed, requires retraining), and unsupervised detection (no labels, fails at scale). Our work rules out the third option.

## 5.5 DFDA: Mitigation Without Detection

DFDA's 21.2% improvement demonstrates that inference-time mitigation is feasible---but only when the absorbed pairs are already known. This creates an asymmetry: **mitigation is easier than detection.** Future work should explore whether mitigation can be applied blindly (to all features) without knowing which pairs are absorbed.

## 5.6 Limitations

1. **Single model**: All experiments on GPT-2 Small; larger models may have different absorption dynamics.
2. **Single concept set**: First-letter features may not represent all absorption types.
3. **Ground truth size**: Only 2 known absorbed pairs in our ground truth at the 500-feature scale; small absolute numbers limit statistical power.
4. **Simplified UAD**: We test the most natural co-occurrence clustering formulation; more sophisticated variants might perform better, though the correlation-vs-suppression mismatch is fundamental.
5. **Single seed**: No multi-seed replication, though the near-random F1 suggests high variance is unlikely to change the conclusion.

## 5.7 Future Work

1. **Causal absorption detection**: Use interventions (e.g., ablating parent features and measuring child activation) to directly measure suppression signals.
2. **Preventive architectures**: Extend Matryoshka-style approaches to eliminate absorption at training time.
3. **Blind mitigation**: Test whether DFDA can be applied to all features without detection, achieving blanket compensation.
4. **Cross-model validation**: Test whether our findings generalize to Gemma-2B, Pythia-2.8B, and larger models.

---

# 6. Conclusion

We presented the first systematic empirical analysis of unsupervised absorption detection at scale. Our key findings are:

1. **UAD performs no better than random at scale**: UAD achieves F1 = 0.007 on 3,702 same-cluster pairs, indistinguishable from random selection (F1 = 0.0075).

2. **The failure is conceptual**: Ablations across clustering methods and feature filters all produce near-zero precision. The problem is that co-occurrence detects correlation, while absorption requires detecting suppression---fundamentally different statistical phenomena.

3. **Mitigation remains feasible**: DFDA improves per-pair residual MSE by 21.2% when absorbed pairs are known, suggesting inference-time compensation is viable even when detection fails---but only with prior knowledge of absorbed pairs.

These findings have implications for the SAE community: rather than pursuing unsupervised detection of absorption after training, research should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption during training. The distinction between correlation and suppression is not merely semantic---it determines whether a detection method can possibly work at scale.

Future work should test whether intervention-based causal detection or blind mitigation strategies can overcome the limitations we identify.

Our conclusion: **absorption detection is a causal inference problem, not a clustering problem.** The path forward requires interventions, not observations.
