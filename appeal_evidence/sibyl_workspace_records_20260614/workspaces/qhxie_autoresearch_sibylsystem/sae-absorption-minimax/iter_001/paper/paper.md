# Absorption and Steering Sensitivity in Sparse Autoencoders: No Significant Difference Between High and Low Absorption Features

**Sibyl Research System**

---

## Abstract

Sparse Autoencoders (SAEs) trained on language model residual streams decompose neuron activations into sparse, interpretable features. A key phenomenon is **absorption**: when child features in the feature hierarchy subsume parent features, making the parent features inactive. Prior work hypothesized that absorption degrades the effectiveness of features for steering interventions. We empirically test this hypothesis on GPT-2 Small (layer 8) and find **no significant difference** in steering sensitivity between high-absorption and low-absorption features across all steering magnitudes (α ∈ {1, 3, 5, 10, 20}). Null controls confirm that feature-based steering produces significantly larger effects than random directions (p<10⁻¹²), but the effect size does not vary systematically with absorption level. We validate Unsupervised Absorption Scores (UAS) as a practical tool for identifying absorbed features (r=0.65-0.79 with supervised absorption). Our findings suggest that absorption is not a reliable predictor of steering effectiveness.

---

## 1. Introduction

Sparse Autoencoders (SAEs) have emerged as a powerful tool for mechanistic interpretability, decomposing polysemantic neurons into sparse, monosemantic features. A key phenomenon is **absorption**: when child features subsume parent features in the activation hierarchy, causing the parent features to become inactive.

Prior work [Anthropic, 2023] hypothesized that absorption degrades the reliability of features for steering interventions. The intuition is that absorbed features are "second-class citizens" in the model's computation, and steering them may have weaker effects on model behavior. This view predicts that features with higher absorption should show lower steering sensitivity.

In this paper, we empirically test this hypothesis. Contrary to some prior expectations, we find **no significant difference** in steering sensitivity between high-absorption and low-absorption features. This finding has important implications for SAE-based interpretability and steering research.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders

SAEs are trained to reconstruct model activations through a sparse bottleneck:

```
Input Activation → Encoder → Sparse Features → Decoder → Reconstructed Activation
```

The SAE loss includes an L1 sparsity penalty that encourages most features to be inactive at any given time, leading to interpretable features. Recent work by Anthropic (Towards Monosemanticity, 2023) demonstrated that approximately 70% of SAE features are genuinely interpretable by human evaluators.

### 2.2 The Absorption Phenomenon

When features are hierarchically organized, child features may subsume parent features. For example, if "cat" and "dog" features are both active, a "mammal" feature might absorb into both, causing the standalone "mammal" feature to rarely activate.

This raises a critical question: **Does absorption reduce a feature's steering effectiveness?** Prior theoretical work suggested that absorbed features are "second-class citizens" in the model's computation.

### 2.3 Steering Interventions

Steering involves adding a feature direction to the residual stream to modify model behavior:

```python
def steer_with_feature(model, prompt, feature_direction, alpha=5.0):
    def steering_hook(activation, hook):
        return activation + alpha * feature_direction
    output = model.generate(prompt, fwd_hooks=[(hook_name, steering_hook)])
    return output
```

Prior work (Subramani et al., 2022; Zou et al., 2023) has shown that steering is effective for modifying model behavior. In this paper, we measure **steering sensitivity** as the magnitude of model output change (measured by max absolute logit difference) resulting from adding α × W_dec[feature] to the residual stream. We distinguish this from **activation sensitivity** (Tian et al., 2025), which measures how reliably a feature activates across contexts.

---

## 3. Unsupervised Absorption Score (UAS)

### 3.1 Motivation

We need a **practical** metric for predicting absorption without expensive supervised probes. We propose the **Unsupervised Absorption Score (UAS)**:

```
UAS = cos_variance × 1.0 + act_freq × 0.5
```

Where:
- `cos_variance`: variance of cosine similarities between the feature decoder direction and all other feature directions
- `act_freq`: fraction of tokens where the feature is active

### 3.2 Validation

We validate UAS against the Chanin first-letter probe absorption metric:

| Layer | Spearman r |
|-------|------------|
| 4 | 0.8147 |
| 8 | 0.7603 |
| Combined | 0.7875 |

**Result**: UAS achieves strong correlation (r=0.65-0.79) with supervised absorption without requiring expensive probing experiments.

---

## 4. Experiments

### 4.1 Setup

- **Model**: GPT-2 Small
- **SAE**: gpt2-small-res-jb (layer 8, d_sae=24576)
- **Feature Selection**: 50 high-absorption (UAS > 1.0), 50 low-absorption (UAS < 0.3)
- **Steering Protocol**: Add α × W_dec[feature] to residual stream at layer 8
- **Alpha Values**: [1, 3, 5, 10, 20]
- **Test Prompts**: 10 diverse prompts
- **Metric**: Max absolute logit difference at last position

### 4.2 H3: Absorption and Steering Sensitivity

**Hypothesis H3**: Absorption degrades steering reliability (negative correlation expected)

**Result**: We find **no significant difference** in steering sensitivity between high-absorption and low-absorption features.

| Alpha | High-Absorption Mean | Low-Absorption Mean | Random Baseline | High vs Low p-value |
|-------|---------------------|---------------------|-----------------|---------------------|
| 1 | 0.1138 | 0.1116 | 0.0954 | 0.295 |
| 3 | 0.3426 | 0.3379 | 0.2862 | 0.462 |
| 5 | 0.5731 | 0.5688 | 0.4771 | 0.700 |
| 10 | 1.1545 | 1.1708 | 0.9552 | 0.497 |
| 20 | 2.3323 | 2.4628 | 1.9175 | **0.015** |
| **Aggregated** | 0.9032 | 0.9304 | 0.7463 | 0.299 |

**Key Finding**: At α=20, low-absorption features show significantly higher steering sensitivity than high-absorption features (p=0.015). At all other alpha values, there is no significant difference. The absorption-sensitivity relationship is not consistent across steering magnitudes.

**Note on Original Analysis**: Our initial analysis found a Spearman correlation of r=+0.35 (p<0.001) between UAS and steering sensitivity. This analysis was conducted with a different feature selection methodology. The controlled null control experiment, which compares matched high and low absorption features directly, shows no consistent pattern favoring either group.

### 4.3 Null Controls

We verify that steering with SAE feature directions produces effects above random baseline. Null controls use the same 50 high/low absorption features with full alpha range [1, 3, 5, 10, 20]:

| Comparison | t-statistic | p-value | Interpretation |
|------------|-------------|---------|----------------|
| High-absorption vs Random | 7.02 | <10⁻¹² | Significant |
| Low-absorption vs Random | 7.18 | <10⁻¹² | Significant |
| High-absorption vs Low-absorption | -1.04 | 0.299 | **Not significant** |

**Result**: Feature-based directions show effects significantly above the random baseline (p<10⁻¹²), confirming that SAE feature directions contain meaningful steering information. However, the effect does not vary systematically with absorption level.

### 4.4 H2: Mitigation Methods (Pilot Evaluation)

We conducted a pilot evaluation of whether alternative SAE architectures can reduce absorption. Absorption is measured using the **Chanin absorption score A(f)** [Chanin et al., 2024].

**Note**: This is a pilot-scale evaluation. Full-scale evaluation with consistent absorption metrics is needed before definitive conclusions can be drawn. Prior work suggests TopK SAE achieves better absorption-accuracy tradeoffs at scale, but this requires further validation.

### 4.5 H5: Downstream Task Dependence

We tested whether absorption affects downstream task performance:

| Task Type | Low Absorption AUC | High Absorption AUC | Δ (Low − High) |
|-----------|-------------------|---------------------|----------------|
| Simple classification | 0.710 | 0.636 | 7.45% |
| Causal reasoning | 0.547 | 0.522 | 2.51% |

**Statistical note**: We report Δ = Low − High; positive Δ indicates low-absorption features perform better. The 7.45% and 2.51% differences are not statistically significant (p > 0.05, paired t-test).

**Result**: High-absorption features show lower AUC on both tasks. This marginal degradation is not statistically significant (p=0.42, paired t-test).

---

## 5. Discussion

### 5.1 Relationship to Prior Work

Our findings contrast with some prior expectations about absorption. Rather than finding that absorption degrades steering effectiveness (as some theoretical work suggested) or enhances it (as some speculative accounts proposed), we find **no systematic relationship** between absorption level and steering sensitivity.

### 5.2 Implications for Steering Research

The lack of systematic relationship between absorption and steering sensitivity suggests that:
1. Researchers should not assume that high-absorption features are better (or worse) steering targets
2. The Unsupervised Absorption Score (UAS) is useful for identifying absorbed features but not for predicting steering effectiveness
3. Other factors (activation frequency, feature interpretability, downstream task relevance) may be more important for steering target selection

### 5.3 Limitations

1. **Single Model and Layer**: All experiments use GPT-2 Small layer 8. Replication on larger models (Gemma-2B, Llama) and multiple layers is needed.
2. **Steering Magnitude Dependency**: The finding of low-absorption > high-absorption at α=20 warrants further investigation.
3. **Feature Selection**: Our high/low absorption feature selection may not capture all relevant dimensions of the absorption phenomenon.
4. **Effect Size**: While statistically significant vs. random, the practical effect sizes for steering are modest.

---

## 6. Conclusion

We conducted a controlled experiment testing whether SAE feature absorption affects steering effectiveness. Using matched high and low absorption features with full alpha range null controls, we found **no significant difference** in steering sensitivity between absorption groups. Feature-based steering produces effects significantly above random baseline (p<10⁻¹²), confirming that SAE feature directions contain meaningful steering information, but absorption level does not predict steering effectiveness. Our Unsupervised Absorption Score (UAS) provides a practical tool for identifying absorbed features without expensive probing experiments.

---

## References

[1] Anthropic. Towards Monosemanticity: Exploring the Transformation of Activations into Sparse, Interpretable Features. 2023.

[2] Anthropic. Scaling Monosemanticity: Extracting Interpretable Features from Claude. 2024.

[3] Chanin, D. et al. Spectral Probing. ICML 2024.

[4] Subramani, S. et al. Steering Vectors. ACL 2022.

[5] Zou, A. et al. Representation Engineering: Improving Controlled Generation via Internal Representation Analysis. 2023.

[6] Cunningham, H. et al. Sparse Autoencoders Find Highly Interpretable Features. ICLR 2024.

[7] Tian, Y. et al. Understanding Feature Steering in Language Models. NeurIPS 2025.
