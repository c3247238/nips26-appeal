# 4. Experiments

## 4.1 Setup

All experiments use GPT-2 Small with the `gpt2-small-res-jb` SAE (layer 8, $d_{\text{sae}} = 24576$). We analyze features that pass activation thresholds for both the Chanin absorption protocol and Tian sensitivity protocol.

## 4.2 Pilot Results: All Hypotheses Falsified

### H5 (Independence): Falsified

**Hypothesis**: Absorption and sensitivity are weakly correlated (Spearman $r < 0.3$)

**Result**: We find $r = 0.59$ ($p = 3.15 \times 10^{-5}$), a **positive correlation** in the opposite direction. Features that are absorbed tend to also have low sensitivity.

This finding contradicts our hypothesis of independence and suggests absorption and sensitivity share a common underlying cause rather than being independent failure modes.

### H6 (Saturation): Falsified

**Hypothesis**: High-absorption features have decoder L2 norms 1.3-1.5x higher than low-absorption features

**Result**: Decoder L2 norm ratio (high-abs / low-abs) = 1.0. High-absorption and low-absorption features have **identical** decoder norms.

This finding contradicts the saturation hypothesis that was proposed to explain the beta=20 reversal observed in iter_004.

### H1-R (Protective Effect): Falsified

**Hypothesis**: Mutual coherence is protective against absorption ($r < -0.5$)

**Result**: Across layers 4, 8, and 10, mean Spearman $r = +0.356$ (positive correlation). The protective effect found in an earlier pilot ($r = -0.786$) was **not replicated**.

### Quadrant Classification: Q4 Empty

From 43 features classified into quadrants:
- Q1 (doubly-compromised): 15 features
- Q2 (absorbed + sensitive): 0 features
- Q3 (not absorbed + insensitive): 28 features
- Q4 (best-case): 0 features

**Critical finding**: Q4 (low absorption + high sensitivity) is empty. The predicted best-case quadrant contains no features, complicating the compound failure hypothesis.

## 4.3 Negative Results Are Actionable

Despite falsified hypotheses, our findings are meaningful:

1. **The positive correlation between absorption and sensitivity** ($r = 0.59$) is itself a novel finding. This suggests a common cause—perhaps both failure modes result from features being low-frequency or geometrically clustered in ways that make them both harder to detect and less reliable.

2. **The empty Q4 quadrant** suggests that features either become absorbed when they have high sensitivity, or that high-sensitivity features (which activate reliably) are systematically absorbed. This points to a trade-off rather than independence.

3. **The replication failure for the protective effect** ($r = -0.786$ in pilot, $r = +0.36$ in replication) indicates instability in the coherence-absorption relationship that warrants further investigation.

## 4.4 Steering by Absorption Level (From iter_004)

Prior experiments tested whether absorption level affects steering sensitivity using 50 high-absorption and 50 low-absorption features matched by decoder L2 norm.

\begin{table}[htbp]
\caption{Steering sensitivity by absorption level across steering magnitudes}
\label{tab:steering-by-absorption}
\centering
\begin{tabular}{ccccc}
\toprule
Beta & High-Absorption & Low-Absorption & Random & $p$-value \\
\midrule
1 & 0.1138 & 0.1116 & 0.0954 & 0.295 \\
3 & 0.3426 & 0.3379 & 0.2862 & 0.462 \\
5 & 0.5731 & 0.5688 & 0.4771 & 0.700 \\
10 & 1.1545 & 1.1708 & 0.9552 & 0.497 \\
20 & 2.3323 & 2.4628 & 1.9175 & \textbf{0.015} \\
\midrule
\textbf{Aggregated} & 0.9032 & 0.9304 & 0.7463 & 0.299 \\
\bottomrule
\end{tabular}
\end{table}

**Key finding**: At $\beta = 20$, low-absorption features show higher steering sensitivity than high-absorption ($p = 0.015$), but this does not survive Bonferroni correction. At all other beta values, no significant difference. Null controls confirm feature-based steering is above random baseline ($p < 10^{-12}$).

## 4.5 Correlation Analysis

We report the absorption-sensitivity correlation and its implications:

\begin{figure}[htbp]
\centering
![Scatter plot of absorption score vs sensitivity score with features colored by quadrant](figures/scatter_absorption_sensitivity.pdf)
\caption{Feature distribution in absorption-sensitivity space. Q1 (doubly-compromised) features cluster in the high-absorption/low-sensitivity region. Q4 (best-case) is empty. The positive correlation ($r = 0.59$) suggests shared underlying causes.}
\end{figure}

## 4.6 Summary of Findings

| Hypothesis | Expected | Observed | Result |
|------------|----------|----------|--------|
| H5 (independence) | $r < 0.3$ | $r = 0.59$ | FALSIFIED |
| H6 (saturation) | ratio > 1.1 | ratio = 1.0 | FALSIFIED |
| H1-R (protective) | $r < -0.5$ | $r = +0.36$ | FALSIFIED |
| Q4 non-empty | Q4 exists | Q4 = 0 | FALSIFIED |
| Absorption predicts steering | High-abs < Low-abs | No difference | NOT SIGNIFICANT |

<!-- FIGURES
- Figure 1: scatter_absorption_sensitivity.pdf — Scatter plot showing absorption vs sensitivity with quadrant coloring
- Table 1: tab:steering-by-absorption — Steering sensitivity by absorption level
- None
-->