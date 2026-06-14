# 3. Methods

## 3.1 Encoder Weight Norm as Absorption Indicator

**Definition.** For SAE latent $j$ with encoder weight matrix row $\mathbf{w}_{e,j} \in \mathbb{R}^{d_\text{model}}$, define:
$$\text{EncNorm}(j) = \|\mathbf{w}_{e,j}\|_2$$

**Mechanistic motivation.** During SAE training, latent $j$'s encoder direction must overcome gradient competition from child features that fire on the same inputs. When child $c$ fires while parent $j$ should also fire, the sparsity penalty $\lambda \|\mathbf{z}\|_1$ creates a gradient that suppresses $z_j$. The encoder responds by inflating $\|\mathbf{w}_{e,j}\|_2$: a larger encoder weight increases the dot product $\mathbf{w}_{e,j}^\top \mathbf{x}$ for on-distribution inputs, partially counteracting the suppressive gradient from $c$. Absorbed features therefore tend to develop elevated encoder norms as a training-time signature of this gradient competition.

This mechanism predicts EncNorm$(j) > $ EncNorm for non-absorbed features, which we test empirically. Unlike EDA (which measures the angular divergence between encoder and decoder directions), EncNorm requires only the encoder weight matrix — it does not require the decoder to be unit-normed or the encoder-decoder pairing to be geometrically interpretable. This makes it applicable to both Standard/L1 and TopK architectures.

**Computation.** EncNorm for a full 65k-width SAE requires a single matrix row-norm computation: $O(d_\text{SAE} \cdot d_\text{model})$ = $O(65536 \times 768) \approx 50$M operations, completing in under 0.1 seconds on CPU.

## 3.2 Jaccard Co-occurrence Detector

**Definition.** For latent $j$ with activation set $A_j = \{t : z_j(t) > 0\}$ across a corpus of tokens $t = 1, \ldots, T$, the Jaccard co-occurrence score is:
$$O_\text{Jaccard}(j) = \max_{k: f_k > f_j} \frac{|A_j \cap A_k|}{|A_j \cup A_k|}$$
where $f_k = |A_k|/T$ is the activation frequency of latent $k$, and the max is over all latents with higher activation frequency than $j$.

**Motivation.** An absorbed latent $j$ should have high overlap with its absorbing child $c$: whenever $j$ should fire, $c$ fires instead, producing high $|A_j \cap A_c|$ relative to the union. The Jaccard score identifies candidate absorbers from activation statistics alone, without requiring probe directions.

**Complementarity.** Spearman $\rho(\text{EncNorm}, O_\text{Jaccard}) = 0.044$ (near-zero), confirming the two signals carry independent information. We use both in a two-signal audit: EncNorm for initial ranking, $O_\text{Jaccard}$ for candidates where EncNorm score is ambiguous.

## 3.3 Amortization Gap Controlled Experiment

**Hypothesis (H2) and pre-committed falsification criterion.** We test whether the feedforward encoder is suboptimal for absorbed features. If the amortization gap drives absorption, an oracle encoder should substantially reduce the absorption rate. Pre-committed criterion: if OMP achieves $\geq 80\%$ of feedforward absorption rate (i.e., $\leq 20\%$ reduction), H2 is falsified.

**Setup.** We use the GPT-2-small layer 6 Standard SAE (gpt2-small-res-jb, \texttt{blocks.6.hook\_resid\_pre}, $d_\text{SAE} = 24{,}576$) with a fixed, pre-trained decoder $W_d$. We vary only the encoding procedure:

- **Condition A (Feedforward):** $\mathbf{z} = \text{ReLU}(W_e \mathbf{x} + \mathbf{b}_e)$, mean $L_0 = 55.7$
- **Condition B (OMP oracle):** Orthogonal Matching Pursuit at $K = 53$ (matched to feedforward mean $L_0$). OMP iteratively selects the decoder column most correlated with the current residual, guaranteeing the optimal $K$-sparse approximation under the given dictionary.

OMP with the original SAE dictionary provides an upper bound on absorption reduction achievable by any improved encoder: if even OMP cannot reduce absorption, no inference-time encoder change will.

**Absorption measurement.** We use the Chanin et al.\ Integrated Gradients pipeline on letters $\{a, e, s\}$ from the GPT-2-small vocabulary ($\leq 30$ tokens per letter). A feature $j$ is counted as absorbed for letter $\ell$ if (a) its decoder direction aligns with the $\ell$-probe direction (cosine similarity $\geq 0.30$) and (b) $z_j = 0$ on $\ell$-positive tokens. The absorption rate ratio $= \text{AbsRate}_\text{OMP} / \text{AbsRate}_\text{FF}$ measures how much OMP reduces absorption relative to the feedforward baseline.

## 3.4 Experimental Setup

**Models.** We use GPT-2-small (117M parameters) with two SAEs from SAELens \citep{bloom2024saelens}:
- Standard/L1: \texttt{gpt2-small-res-jb}, layer 6, \texttt{blocks.6.hook\_resid\_pre}, $d_\text{SAE} = 24{,}576$
- TopK-32k: \texttt{gpt2-small-resid-post-v5-32k}, layer 6, \texttt{blocks.6.hook\_resid\_post}, $d_\text{SAE} = 32{,}768$, $k=32$

\textit{Note on hook confound:} The Standard SAE hooks into the residual stream \emph{before} the layer-6 attention sublayer (resid\_pre), while the TopK-32k SAE hooks \emph{after} (resid\_post). These are different activation spaces; AUROC values for the two architectures are not directly comparable, and the observed AUROC difference ($0.757$ vs.\ $0.837$) cannot be attributed to architecture alone.

**Labels.** Gold absorption labels for the Standard SAE are generated using the Chanin et al.\ \texttt{FeatureAbsorptionCalculator} (IG-based, exact): $n_\text{pos} = 18$, $n_\text{neg} = 24{,}558$ at layer 6. TopK-32k labels use decoder-alignment proxy (cosine similarity $\geq 0.30$ to letter probe): $n_\text{pos} = 77$, $n_\text{neg} = 32{,}691$.

**Metrics.** Primary: AUROC with 95\% bootstrap CI (10,000 resamples). Secondary: AUPRC, Precision@50, DeLong test for pairwise AUROC comparison, Cohen's $d$ for group separation.
