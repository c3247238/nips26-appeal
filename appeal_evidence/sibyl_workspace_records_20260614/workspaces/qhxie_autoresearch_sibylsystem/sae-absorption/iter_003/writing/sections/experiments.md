# 4. Experiments

## 4.1 Encoder Weight Norm Detection Performance

Table 1 reports EncNorm and EDA AUROC across Standard and TopK-32k SAEs at GPT-2 layer 6.

**Table 1: Detection Performance at GPT-2 Layer 6**

| Detector | Architecture | Hook | AUROC | 95% CI | $n_\text{pos}$ | Cohen's $d$ |
|----------|-------------|------|-------|--------|----------------|-------------|
| EncNorm | Standard/L1 | resid\_pre | **0.757** | [0.634, 0.864] | 18 | 0.971 |
| EDA | Standard/L1 | resid\_pre | 0.650 | [0.508, 0.786] | 18 | 0.532 |
| EncNorm | TopK-32k | resid\_post | **0.837** | [0.746, 0.911] | 77 | 1.235 |
| EDA | TopK-32k† | resid\_post | — | — | — | — |
| $O_\text{Jaccard}$ | Standard/L1 | resid\_pre | 0.721 | [0.576, 0.851] | 18 | — |

†EDA is ill-defined for TopK architectures (encoder-decoder cosine distance not interpretable without $\ell_2$-normalized decoder constraint under hard-$k$ training).

**EncNorm outperforms EDA at matched labels.** On Standard/L1 with gold IG labels ($n=18$), DeLong test: $p = 0.0012$, AUROC difference $= +0.107$ [CI: $+0.041, +0.173$]. Cohen's $d$ for group separation: $0.971$ vs.\ $0.532$ — nearly double the separation. Mean EncNorm for absorbed latents: $3.26$; for non-absorbed: $2.58$ (ratio $= 1.26$).

**Cross-architecture replication.** On TopK-32k (proxy labels), EncNorm achieves AUROC $= 0.837$ (Cohen's $d = 1.235$). This is the strongest detection result across all experiments. The AUROC cannot be directly compared to the Standard/L1 result due to the hook confound (resid\_pre vs.\ resid\_post; see Section 3.4), but confirms that EncNorm is applicable to TopK architectures where EDA is not.

**Layer analysis.** EncNorm ratio (absorbed/non-absorbed) peaks at layer 6 (ratio $= 1.267$) and decreases at deeper layers (L10: $0.933$), indicating the gradient competition mechanism is most pronounced at mid-layers. This matches the layer-dependent absorption prevalence observed by \citet{karvonen2025saebench}.

## 4.2 Co-occurrence Jaccard Signal

On 10,000 OpenWebText tokens, $O_\text{Jaccard}$ achieves AUROC $= 0.721$ (Table 1). More importantly, Precision@50 $= 0.100$ — the top 50 latents by $O_\text{Jaccard}$ contain $5$ of the 18 absorbed latents, a $13.9\times$ enrichment over random. AUPRC $= 0.075$ vs.\ $0.00073$ random baseline.

Spearman $\rho(\text{EncNorm}, O_\text{Jaccard}) = 0.044$ ($p = 0.67$, not significant), confirming the two signals are near-independent. A two-signal audit strategy — rank by EncNorm, then re-rank by $O_\text{Jaccard}$ — covers complementary failure modes: EncNorm identifies latents with inflated gradient competition; $O_\text{Jaccard}$ identifies latents with high co-occurrence overlap regardless of encoder geometry.

## 4.3 Amortization Gap Experiment: H2 Falsified

**Table 2: OMP Oracle vs.\ Feedforward Absorption Rates**

| Letter | FF AbsRate | OMP AbsRate | Ratio OMP/FF | Reduction |
|--------|-----------|-------------|--------------|-----------|
| a | 0.978 | 0.978 | 1.000 | 0.0% |
| e | 0.867 | 0.867 | 1.000 | 0.0% |
| s | 0.733 | 0.733 | 1.000 | 0.0% |
| **Mean** | **0.859** | **0.859** | **1.000** | **0.0%** |

OMP ($K=53$) achieves reconstruction MSE $= 0.219$ vs.\ feedforward $= 0.242$ — OMP is a strictly better encoder in reconstruction terms. Despite this, absorption rates are identical (ratio $= 1.000$ across all three letters, $p_\text{Fisher} > 0.99$).

The pre-committed falsification criterion was: OMP $\geq 80\%$ of feedforward absorption rate $\Rightarrow$ H2 falsified. The result ($100\%$ ratio) is unambiguous: **the amortization gap hypothesis is falsified**. The feedforward encoder is already near-optimal for absorbed features; improving it cannot reduce absorption. This directly supports the sparsity landscape account \citep{tang2025partial}: absorption is locked in by the training-time partial minimum, not by the encoder's inference-time approximation error.

**Control: Layer-6 OMP reconstruction quality.** OMP with $K=53$ achieves $R^2 = 0.87$ reconstruction quality on the held-out token set (vs.\ FF $R^2 = 0.84$), ruling out the possibility that OMP simply fails to encode the relevant signal.

## 4.4 Wider SAE Feature Recovery (F1)

To test whether dictionary width expansion remediates absorption, we compare decoder directions of the 18 absorbed Standard-24k latents against all 32,768 decoder columns of the TopK-32k SAE (cosine similarity threshold $= 0.80$).

**Table 3: Wider SAE Recovery of Absorbed Features**

| Metric | Value |
|--------|-------|
| $n$ absorbed (Standard-24k) | 18 |
| $n$ recovered in TopK-32k (cos\_sim $> 0.80$) | 12 (67%) |
| $n$ not recovered | 6 (33%) |
| Mean best cosine similarity | 0.791 |
| Median best cosine similarity | 0.815 |
| EncNorm (recovered, mean) | 3.289 |
| EncNorm (not recovered, mean) | 3.212 |

Two-thirds of absorbed features have geometric counterparts in the wider dictionary, suggesting that width expansion addresses absorption for features whose parent was simply absent from the narrower dictionary. However, one-third do not recover, indicating genuine semantic gaps that capacity increases alone cannot fill.

The nearly identical EncNorm for recovered vs.\ not-recovered features ($3.289$ vs.\ $3.212$, $t$-test $p = 0.73$) suggests encoder norm does not distinguish which absorbed features will benefit from width expansion — this distinction requires structural analysis of the absorbing child's decoder direction.

*Hook confound note*: The Standard-24k and TopK-32k SAEs differ in both dictionary size and hook point. Cosine similarity between decoder directions may reflect the different activation spaces rather than true feature recovery. This confound cannot be resolved without a matched-hook wider SAE.
