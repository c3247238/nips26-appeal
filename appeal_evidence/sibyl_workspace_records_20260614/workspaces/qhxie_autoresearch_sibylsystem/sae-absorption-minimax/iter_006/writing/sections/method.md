# 3. Method

## 3.1 Unsupervised Absorption Score (UAS)

We propose the Unsupervised Absorption Score (UAS) as a practical metric for identifying absorbed features without expensive probing experiments. UAS is computed from feature geometry alone:

\begin{equation}
\text{UAS}(f) = \alpha \cdot \cosvar(f) + \beta \cdot \text{act\_freq}(f)
\end{equation}

Where:
- $\cosvar(f)$: variance of cosine similarities between feature $f$'s decoder direction and all other feature directions
- $\text{act\_freq}(f)$: fraction of tokens where feature $f$ is active

High $\cosvar(f)$ indicates the feature's decoder direction is an outlier in the feature geometry (suggesting absorption). High $\text{act\_freq}(f)$ indicates the feature activates frequently (inconsistent with absorbed parent features that should be rare).

## 3.2 Absorption Detection Protocol (Chanin et al., 2024)

For supervised absorption measurement, we implement the Chanin protocol:

1. **Task**: Classify tokens into A-M vs N-Z first-letter bins
2. **Positive set selection**: Features with mean activation $>\tau_{\text{fs}} = 0.03$
3. **Probe training**: Logistic regression on SAE feature activations to predict first-letter class
4. **Absorption score**:
   \begin{equation}
   A(f) = \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{1 - \text{acc}_{\text{sae}}}
   \end{equation}
   where $\text{acc}_{\text{resid}}$ is probe accuracy on residual stream (upper bound) and $\text{acc}_{\text{sae}}$ is probe accuracy on SAE features (lower bound)
5. **Classification threshold**: Feature is absorbed if $A(f) < \tau_{\text{pa}} = 0.4$

## 3.3 Sensitivity Measurement Protocol (Tian et al., 2025)

For sensitivity measurement, we implement the Tian protocol:

1. **Paraphrase generation**: Generate 50 paraphrase pairs for sentences where feature $f$ activates strongly
2. **Activation measurement**: Measure feature activation on original and paraphrase sentences
3. **AUC computation**: Compute area under the ROC curve across paraphrase pairs
4. **Classification threshold**: Feature is low-sensitivity if AUC $< 0.6$

## 3.4 Steering Protocol

For steering experiments:

1. **Steering direction**: Add $\beta \times W_{\text{dec}}[f]$ to residual stream at layer 8
2. **Beta values**: $\beta \in \{5, 10, 20\}$
3. **Test prompts**: 10 diverse prompts where feature $f$ activates strongly
4. **Effect measurement**: Max absolute logit difference at last position
5. **Baseline**: Random direction steering (matched L2 norm)

## 3.5 Quadrant Classification

We classify features into four quadrants based on absorption and sensitivity:

| Quadrant | Absorption | Sensitivity | Interpretation |
|----------|------------|-------------|----------------|
| Q1 | High (absorbed) | Low | Doubly-compromised |
| Q2 | High | High | Absorbed but sensitive |
| Q3 | Low | Low | Not absorbed but insensitive |
| Q4 | Low | High | Best-case |

Q1 features are predicted to show near-random steering validity (compound failure). Q4 features are predicted to show above-random steering (best-case scenario).

## 3.6 Experimental Setup

- **Model**: GPT-2 Small
- **SAE**: `gpt2-small-res-jb` (layer 8, $d_{\text{sae}} = 24576$)
- **Feature selection**: 43 features with sufficient activation for both protocols
- **Statistical tests**: Spearman correlation, Mann-Whitney U, bootstrap CI (n=1000)

\begin{table}[htbp]
\caption{Feature counts by quadrant from pilot classification}
\label{tab:quadrant-counts}
\centering
\begin{tabular}{lc}
\toprule
Quadrant & Features \\
\midrule
Q1 (doubly-compromised) & 15 \\
Q2 (absorbed + sensitive) & 0 \\
Q3 (not absorbed + insensitive) & 28 \\
Q4 (best-case) & 0 \\
\midrule
\textbf{Total} & 43 \\
\bottomrule
\end{tabular}
\end{table}

<!-- FIGURES
- Table 1: tab:quadrant-counts — Feature counts by quadrant
- None
-->