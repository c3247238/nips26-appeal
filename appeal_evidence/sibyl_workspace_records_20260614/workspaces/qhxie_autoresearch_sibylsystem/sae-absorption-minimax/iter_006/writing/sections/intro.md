# 1. Introduction

Sparse Autoencoders (SAEs) have emerged as a dominant tool for decomposing the polysemantic activations of large language models into sparse, human-interpretable features (Cunningham et al., 2023; Bricken et al., 2023). Two documented failure modes limit SAE reliability:

1. **Feature absorption** (Chanin et al., 2024): when child features subsume parent features in a semantic hierarchy, causing parent features to rarely activate
2. **Feature sensitivity** (Tian et al., 2025): when interpretable features fail to activate on semantically equivalent inputs

Recent work by Korznikov et al. (2026) reveals a troubling finding: random SAE baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). Understanding why requires characterizing SAE failure modes.

## 1.1 From Iteration 4: Absorption Does Not Predict Steering Sensitivity

Prior work hypothesized that absorbed features would show degraded steering effectiveness. A controlled experiment on GPT-2 Small layer 8 compared 50 high-absorption and 50 low-absorption features across steering magnitudes $\beta \in \{1, 3, 5, 10, 20\}$. The result: **no significant difference** in steering sensitivity between absorption groups (aggregated $p = 0.299$). Null controls confirmed that feature-based steering produces effects significantly above random ($p < 10^{-12}$). Absorption level is not a reliable predictor of steering effectiveness.

## 1.2 From Iteration 7: Mutual Coherence is Protective, Not Causal

A pilot study tested whether mutual coherence predicts absorption (H1). The finding ran counter to the hypothesis: **high mutual coherence is protective against absorption**, not causal of it. Features with high coherence showed 7.1% absorption rate versus 78.6% for low-coherence features (Spearman $r = -0.786$, $p < 0.0001$). This negative result suggests high-coherence features form robust clusters that resist absorption.

## 1.3 From Iteration 8 (Pilot): Compound Failure Hypothesis Weakened

This iteration proposed that absorbed + low-sensitivity features (doubly-compromised) would show near-random causal validity, and that the two failure modes would be independent ($r < 0.3$). Pilot results on 43 features from GPT-2 Small layer 8 falsified all three hypotheses:

| Hypothesis | Expected | Observed | Result |
|------------|----------|----------|--------|
| H5 (independence) | $r < 0.3$ | $r = 0.59$ | FALSIFIED |
| H6 (saturation) | norm ratio > 1.1 | ratio = 1.0 | FALSIFIED |
| H1-R (protective) | $r < -0.5$ | $r = +0.36$ | FALSIFIED |

Critically, Q4 (low absorption + high sensitivity) was empty: no features fell into the predicted best-case quadrant. The positive correlation between absorption and sensitivity ($r = 0.59$) suggests these failure modes may share a common cause rather than being independent.

## 1.4 Research Questions

Despite pilot failures, the Sanity Check crisis demands explanation. We frame three revised questions:

**RQ1**: Do absorbed features show different steering sensitivity than non-absorbed features? (From iter 4: no, $p = 0.299$)

**RQ2**: What is the relationship between absorption and sensitivity at the feature level? (From iter 8 pilot: positive correlation $r = 0.59$, not independent)

**RQ3**: Is the Sanity Check finding explained by absorption + sensitivity distribution? (From iter 8 pilot: Q4 empty, complicating the compound failure account)

## 1.5 Contributions

Our findings, while negative in specific hypotheses, paint a coherent picture:

1. **Absorption does not degrade steering**: absorbed features show equivalent steering sensitivity to non-absorbed features
2. **Absorption and sensitivity are positively correlated**: features that are absorbed tend to also have low sensitivity, suggesting a common underlying cause
3. **Mutual coherence is protective**: high-coherence features resist absorption, suggesting a potential mitigation strategy
4. **Q4 is empty**: the predicted best-case quadrant (low absorption + high sensitivity) contains no features, requiring theoretical revision

## 1.6 Paper Structure

Section 2 reviews related work on SAEs, absorption detection, and feature sensitivity. Section 3 describes the UAS metric and its validation against Chanin absorption scores. Section 4 presents experimental results. Section 5 discusses implications and limitations. Section 6 concludes.

\begin{figure}[htbp]
\centering
![Bar chart comparing steering effects across beta values for high-absorption, low-absorption, and random baseline features](figures/steering_by_absorption_bar.pdf)
\caption{Steering effects by absorption level across beta values. High-absorption and low-absorption features show equivalent steering sensitivity at all magnitudes.}
\end{figure}

<!-- FIGURES
- Figure 1: steering_by_absorption_bar.pdf — Bar chart of steering effects by absorption level
- None
-->