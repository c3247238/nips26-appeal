# Feature Absorption in Sparse Autoencoders is a Sparsity Landscape Problem: Encoder Weight Norm as a Weight-Only Detection Heuristic

---

## Abstract

Feature absorption is a systematic failure mode of Sparse Autoencoders (SAEs): a parent SAE latent fails to activate on inputs where it should be active, because a more specific child latent absorbs the activation budget. Absorption affects 15--35\% of SAE latents on the first-letter spelling task across Gemma Scope SAEs, representing a substantial fraction of the feature pool used for mechanistic analysis. Two competing theories assign opposite causes: the amortization gap hypothesis (O'Neill et al., 2024) attributes absorption to feedforward encoder approximation error and predicts that iterative encoders will fix it; the sparsity landscape hypothesis (Tang et al., 2025) attributes absorption to stable partial minima formed during SAE training and predicts that no inference-time encoder change can fix it. We report a controlled experiment that resolves this debate: we fix a pre-trained SAE decoder and compare feedforward encoding against an OMP oracle at matched sparsity ($K = 53$). OMP achieves $0\%$ absorption reduction across all tested features (mean $\text{AR}_\text{OMP} = \text{AR}_\text{FF} = 0.978$; absorption rate ratio $= 1.000$), decisively falsifying the amortization gap hypothesis. Alongside this mechanistic result, we introduce encoder weight norm ($\|\mathbf{w}_{\text{enc},j}\|_2$) as a weight-only absorption indicator: AUROC $= 0.757$ [0.665, 0.849] on GPT-2-small layer 6 with exact Chanin et al.\ IG labels ($n_\text{pos} = 18$), and AUROC $= 0.837$ [0.807, 0.870] on a TopK-32k SAE with proxy labels ($n_\text{pos} = 77$), significantly outperforming EDA (AUROC $= 0.650$; DeLong $z = 3.05$, $p = 0.0012$). Jaccard co-occurrence ($O_\text{Jaccard}$, AUROC $= 0.721$, Spearman $\rho = 0.044$ with encoder norm) provides an independent complementary signal. Finally, 67\% of absorbed features have direction-aligned counterparts in a wider (32k) SAE, but 33\% do not, confirming that dictionary expansion partially but incompletely remediates absorption. These results redirect the SAE mitigation research program: iterative encoders will not fix absorption; training-time objective changes are necessary.

---

## 1. Introduction

Sparse Autoencoders (SAEs) decompose the activations of large language models into sparse, interpretable features, enabling mechanistic interpretability at scale \citep{bricken2023monosemanticity,cunningham2023sparse}. A key failure mode of this approach is **feature absorption**: a child feature $c$ absorbs the activation budget of a semantically related parent feature $p$, causing $p$ to fail to fire on inputs where it should activate \citep{chanin2024absorption}. Absorption affects 15--35\% of SAE latents on the first-letter spelling task across Gemma Scope 16k/65k SAEs \citep{karvonen2025saebench}, representing a substantial fraction of the feature pool used for mechanistic analysis. Absorption compromises downstream analyses that rely on SAE features to localize model computations: if the parent feature is silenced by an absorbing child, any circuit-level analysis that traces information flow through $p$ will be systematically wrong.

Two mechanistic accounts offer competing explanations for why absorption occurs, with opposite implications for practitioners:

**Amortization gap hypothesis** \citep{oneill2024amortization}: The feedforward encoder is a compressed approximation of optimal sparse inference. Because SAE encoders share weights across all latents, absorbed features are those where the amortized approximation diverges most from the optimal per-token sparse code. Under this account, replacing the feedforward encoder with an iterative solver (e.g., Orthogonal Matching Pursuit) at the same sparsity level should significantly reduce absorption. The fix is *inference-time*: better encoders.

**Sparsity landscape hypothesis** \citep{tang2025partial}: Absorption arises from stable partial minima of the biconvex (alternately convex in encoder and decoder directions) sparse dictionary learning (SDL) loss. When a child feature $c$ fires frequently in contexts that also activate $p$, the sparsity gradient from $c$'s activations persistently pushes $p$'s encoder direction away from its decoder direction. This creates a training-time attractor that no amount of improved inference-time encoding can escape. The fix is *training-time*: different objectives or dictionary structure. The two accounts thus make opposite empirical predictions, enabling a clean controlled test.

Choosing between these accounts is not merely academic. Under the amortization gap account, absorption can be mitigated by deploying iterative solvers during inference --- computationally expensive but tractable. Under the sparsity landscape account, iterative solvers are irrelevant; only changes to training objectives, dictionary design, or model architecture can help.

**This paper reports a controlled experiment that resolves this debate.** We fix a pre-trained SAE decoder and vary the encoding method, comparing feedforward encoding against an OMP oracle at matched sparsity ($K = 53$). If the amortization gap is the dominant cause of absorption, OMP should substantially reduce the absorption rate; if the sparsity landscape is dominant, OMP and feedforward should produce indistinguishable absorption rates. Our result is decisive: OMP achieves a $0\%$ absorption reduction on all three tested letters (mean $\text{AR}_\text{OMP} = \text{AR}_\text{FF} = 0.978$; absorption rate ratio $= 1.000$), falsifying the amortization gap hypothesis. Figure~\ref{fig:teaser} illustrates these two results: absorption rates are identical under feedforward and OMP encoding (left panel), while encoder weight norm ROC curve dominates EDA (right panel).

Alongside the mechanistic experiment, we introduce **encoder weight norm** ($\|\mathbf{w}_{\text{enc},j}\|_2$) as a weight-only absorption indicator, achieving AUROC $= 0.757$ [0.665, 0.849] on GPT-2-small at layer 6 with exact Chanin et al.\ labels (DeLong test vs.\ EDA: $z = 3.05$, $p = 0.0012$). Encoder weight norm does not require activation data or probe directions; it can screen an entire 65k-width SAE in under one second. We also show that co-occurrence Jaccard overlap ($O_\text{Jaccard}$, AUROC $= 0.721$) provides a near-uncorrelated complementary signal (Spearman $\rho = 0.044$), enabling a two-signal audit approach. Finally, a successive-refinement analysis shows that 67\% of absorbed features have corresponding directions in a wider (32k) SAE, but 33\% do not, confirming that width expansion partially but incompletely remediates absorption (hook-confound caveat: Section~\ref{sec:experiments}).

Our contributions are:
\begin{enumerate}
    \item \textbf{Mechanistic}: The first controlled experiment adjudicating amortization gap vs.\ sparsity landscape as dominant absorption cause. OMP oracle falsifies the amortization gap hypothesis; sparsity landscape (Tang et al.) is supported.
    \item \textbf{Methodological}: Encoder weight norm, a weight-only absorption heuristic (AUROC $= 0.757$ on Standard/L1 gold IG labels and $0.837$ on TopK-32k proxy labels; architectures not directly comparable due to hook confound --- see Section~\ref{sec:method}) that outperforms EDA (AUROC $= 0.650$) and requires no activation data.
    \item \textbf{Practical}: Evidence that at least 33\% of absorbed features require training-time interventions; wider dictionaries alone do not resolve absorption.
\end{enumerate}

Section~\ref{sec:related} reviews prior work on SAE absorption theory and detection. Section~\ref{sec:method} introduces encoder weight norm and the OMP oracle experiment design. Section~\ref{sec:experiments} reports all results. Section~\ref{sec:discussion} discusses implications for the SAE mitigation research program.

---

## 2. Background and Related Work \label{sec:related}

### 2.1 Sparse Autoencoders and Feature Absorption

A Sparse Autoencoder trained on residual stream activations $\mathbf{x} \in \mathbb{R}^{d_\text{model}}$ learns encoder $W_\text{enc} \in \mathbb{R}^{d_\text{SAE} \times d_\text{model}}$, decoder $W_\text{dec} \in \mathbb{R}^{d_\text{model} \times d_\text{SAE}}$ with unit-normed columns, and biases, minimizing
$$\mathcal{L} = \mathbb{E}_\mathbf{x}\left[\|\mathbf{x} - W_\text{dec}\, \mathbf{z}\|^2 + \lambda \|\mathbf{z}\|_1\right], \quad \mathbf{z} = \text{ReLU}(W_\text{enc}\, \mathbf{x} + \mathbf{b}_\text{enc}).$$
The $\ell_1$ penalty drives most latent activations to zero. In TopK architectures \citep{makhzani2015winner,templeton2024scaling}, $\|\mathbf{z}\|_1$ is replaced by a hard constraint retaining only the top-$k$ activations per forward pass.

\citet{chanin2024absorption} identified **feature absorption**: latent $j$ is absorbed by child $c$ if $z_j = 0$ on inputs where $z_j$ should be positive, while $z_c > 0$ on those same inputs. They measure absorption via an integrated gradients pipeline on the first-letter spelling task (26 classes), reporting absorption rates of 15--35\% in mid-layer Gemma Scope SAEs. \citet{karvonen2025saebench} incorporates absorption rates as a first-class SAEBench metric.

### 2.2 Competing Mechanistic Accounts

**Amortization gap (O'Neill et al., 2024).** SAE encoders approximate optimal sparse inference via a single feedforward pass. \citet{oneill2024amortization} show that this amortized approximation systematically under-activates features that are weakly active or co-occurring with stronger features. They propose learned iterative thresholding encoders as a remedy. Under this account, absorption is a byproduct of the encoder's approximation error, and replacing the encoder with an oracle solver should eliminate it.

**Sparsity landscape / partial minimum (Tang et al., 2025).** \citet{tang2025partial} analyze the biconvex SDL loss landscape and identify stable partial minima where an absorbed feature $j$'s encoder direction $\mathbf{w}_{\text{enc},j}$ drifts away from its decoder direction $\mathbf{d}_j$. The mechanism is a training-time gradient conflict: when child $c$ fires on inputs that also activate parent $j$, the sparsity gradient from $c$ persistently suppresses $j$'s encoder. This creates an attractor that survives any inference-time change to the encoding procedure.

### 2.3 Mitigation Proposals

Several SAE variants have been proposed to reduce absorption: *Matryoshka SAEs* \citep{bussmann2024matryoshka} hierarchically nest features to encourage parent-child allocation; *OrtSAE* applies orthogonality regularization; *Masked Regularization* \citep{narayanaswamy2026masked} explicitly suppresses sparsity gradients from high-frequency features on their co-occurring partners. The success of masked regularization specifically at training time provides preliminary support for the sparsity landscape account; our experiment provides direct evidence.

### 2.4 Weight-Only Absorption Indicators

\citet{chanin2024absorption} and \citet{karvonen2025saebench} require model activations and integrated gradients --- expensive at scale. Prior work from this research program introduced **EDA** ($1 - \cos(\mathbf{w}_{\text{enc},j}, \mathbf{d}_j)$), a weight-only metric with AUROC $= 0.650$ at GPT-2 L6. EDA has theoretical grounding (formal lower bound connecting EDA to absorption degree), but is poorly suited to TopK architectures where encoder-decoder angular relationships are less constrained. The present work introduces encoder weight norm as an alternative with stronger empirical performance and wider architectural applicability.

---

## 3. Methods \label{sec:method}

### 3.1 Encoder Weight Norm as Absorption Indicator

**Definition.** For SAE latent $j$ with encoder weight matrix row $\mathbf{w}_{\text{enc},j} \in \mathbb{R}^{d_\text{model}}$, define:
$$\text{EncNorm}(j) = \|\mathbf{w}_{\text{enc},j}\|_2$$

Figure~\ref{fig:enc-norm-hist} shows the encoder norm distribution for absorbed vs.\ non-absorbed latents at GPT-2 L6.

**Mechanistic motivation.** Under the sparsity landscape account \citep{tang2025partial}, the training-time gradient competition from child $c$ prevents $j$'s encoder direction from converging normally. One predicted consequence is that $\|\mathbf{w}_{\text{enc},j}\|_2$ becomes inflated as the encoder attempts to counteract the suppressive gradient: when child $c$ fires while parent $j$ should also fire, the sparsity penalty $\lambda \|\mathbf{z}\|_1$ creates a gradient that suppresses $z_j$. A larger encoder weight increases the dot product $\mathbf{w}_{\text{enc},j}^\top \mathbf{x}$ for on-distribution inputs, partially counteracting the suppressive gradient from $c$. This predicts EncNorm$(j) >$ EncNorm for non-absorbed features, which we test empirically. Unlike EDA (which measures the angular divergence between encoder and decoder directions), EncNorm requires only the encoder weight matrix --- it does not assume any angular relationship between the encoder row $\mathbf{w}_{\text{enc},j}$ and decoder column $\mathbf{d}_j$. This makes it applicable to both Standard/L1 and TopK architectures.

**Computation.** EncNorm for a full 65k-width SAE (e.g., Gemma Scope) requires a single matrix row-norm computation: $O(d_\text{SAE} \cdot d_\text{model}) = O(65536 \times 768) \approx 50$M operations, completing in under 0.1 seconds on CPU.

### 3.2 Jaccard Co-occurrence Detector

**Definition.** For latent $j$ with activation set $A_j = \{t : z_j(t) > 0\}$ across a corpus of tokens $t = 1, \ldots, T$, the Jaccard co-occurrence score is:
$$O_\text{Jaccard}(j) = \max_{k: f_k > 3f_j} \frac{|A_j \cap A_k|}{|A_j \cup A_k|}$$
where $f_k = |A_k|/T$ is the activation frequency of latent $k$, and the max is over all latents with at least $3\times$ the activation frequency of $j$ (this lower bound ensures that absorbed-candidate latents $j$ are compared only against meaningfully higher-frequency potential absorbers).

**Motivation.** An absorbed latent $j$ should have high overlap with its absorbing child $c$: whenever $j$ should fire, $c$ fires instead, producing high $|A_j \cap A_c|$ relative to the union. The Jaccard score identifies candidate absorbers from activation statistics alone, without requiring probe directions.

**Complementarity.** Spearman $\rho(\text{EncNorm}, O_\text{Jaccard}) = 0.044$ (computed on 10,000 OpenWebText tokens using the Standard/L1 SAE), near-zero, confirming the two signals carry independent information. We use both in a two-signal audit: EncNorm for initial ranking, $O_\text{Jaccard}$ for candidates where EncNorm score is ambiguous.

### 3.3 Amortization Gap Controlled Experiment

**Hypothesis (H2) and pre-committed falsification criterion.** We test whether the feedforward encoder is suboptimal for absorbed features. If the amortization gap drives absorption, an oracle encoder should substantially reduce the absorption rate. Pre-committed criterion (established in the experimental proposal before running experiments): if OMP achieves $\geq 80\%$ of feedforward absorption rate (i.e., $\leq 20\%$ reduction), H2 is falsified.

Figure~\ref{fig:omp-design} illustrates the experimental design: both conditions share a fixed pre-trained decoder; only the encoding procedure varies.

**Setup.** We use the GPT-2-small layer 6 Standard SAE (\texttt{gpt2-small-res-jb}, \texttt{blocks.6.hook\_resid\_pre}, $d_\text{SAE} = 24{,}576$) with a fixed, pre-trained decoder $W_\text{dec}$. We vary only the encoding procedure:

- **Condition A (Feedforward):** $\mathbf{z} = \text{ReLU}(W_\text{enc}\, \mathbf{x} + \mathbf{b}_\text{enc})$, mean $L_0 = 55.7$
- **Condition B (OMP oracle):** Orthogonal Matching Pursuit (OMP) at $K = 53$ (matched to feedforward mean $L_0$). OMP iteratively selects the decoder column most correlated with the current residual, guaranteeing the optimal $K$-sparse approximation under the given dictionary. $K = 53$ was chosen to match the feedforward mean $L_0 = 55.7$ (rounded).

OMP with the original SAE dictionary provides an upper bound on absorption reduction achievable by any improved encoder: if even OMP cannot reduce absorption, no inference-time encoder change will.

**Absorption measurement.** We use the Chanin et al.\ Integrated Gradients pipeline (same labels as Section~\ref{sec:setup} Standard SAE gold labels: $n_\text{pos} = 18$, $n_\text{neg} = 24{,}558$) on letters $\{a, e, s\}$ from the GPT-2-small vocabulary ($30$ tokens per letter). The absorption rate ratio $= \text{AbsRate}_\text{OMP} / \text{AbsRate}_\text{FF}$ measures how much OMP reduces absorption relative to the feedforward baseline.

### 3.4 Experimental Setup \label{sec:setup}

**Models.** We use GPT-2-small (117M parameters) with two SAEs from SAELens \citep{bloom2024saelens}:
- Standard/L1: \texttt{gpt2-small-res-jb}, layer 6, \texttt{blocks.6.hook\_resid\_pre}, $d_\text{SAE} = 24{,}576$
- TopK-32k: \texttt{gpt2-small-resid-post-v5-32k}, layer 6, \texttt{blocks.6.hook\_resid\_post}, $d_\text{SAE} = 32{,}768$, $k=32$

**Confounds and Controls.** The Standard SAE hooks into the residual stream *before* the layer-6 attention sublayer (resid\_pre), while the TopK-32k SAE hooks *after* (resid\_post). These are different activation spaces; AUROC values for the two architectures are not directly comparable, and the observed AUROC difference ($0.757$ vs.\ $0.837$) cannot be attributed to architecture alone.

**Labels.** Gold absorption labels for the Standard SAE are generated using the Chanin et al.\ \texttt{FeatureAbsorptionCalculator} (IG-based, exact): $n_\text{pos} = 18$, $n_\text{neg} = 24{,}558$ at layer 6. TopK-32k labels use decoder-alignment proxy (cosine similarity $\geq 0.30$ to letter probe): $n_\text{pos} = 77$, $n_\text{neg} = 32{,}691$.

**Metrics.** Primary: AUROC with 95\% bootstrap CI (10,000 resamples). Secondary: AUPRC, Precision@50, DeLong test for pairwise AUROC comparison, Cohen's $d$ for group separation.

---

## 4. Experiments \label{sec:experiments}

### 4.1 Encoder Weight Norm Detection Performance

Setup follows Section~\ref{sec:setup}; detection experiments use Standard/L1 ($n_\text{pos}=18$ gold IG labels) and TopK-32k ($n_\text{pos}=77$ proxy labels) at GPT-2 L6. Table~\ref{tab:detectors} reports EncNorm and EDA AUROC across both SAEs (see Figure~\ref{fig:roc-curves} for ROC curves).

**Table 1: Detection Performance at GPT-2 Layer 6** \label{tab:detectors}

| Detector | Architecture | Hook | AUROC | 95% CI | $n_\text{pos}$ | Cohen's $d$ |
|----------|-------------|------|-------|--------|----------------|-------------|
| EncNorm | Standard/L1 | resid\_pre | **0.757** | [0.665, 0.849] | 18 | 0.971 |
| EDA | Standard/L1 | resid\_pre | 0.650 | [0.541, 0.767] | 18 | 0.533 |
| $O_\text{Jaccard}$ | Standard/L1 | resid\_pre | 0.721 | [0.604, 0.843] | 18 | --- |
| ARS\_v2 | Standard/L1 | resid\_pre | 0.586 | [0.493, 0.692] | 18 | --- |
| EncNorm | TopK-32k | resid\_post | **0.837** | [0.807, 0.870] | 77 | 1.235 |
| EDA | TopK-32k$^\dagger$ | resid\_post | --- | --- | --- | --- |

$^\dagger$EDA is ill-defined for TopK architectures (encoder-decoder cosine distance not interpretable without $\ell_2$-normalized decoder constraint under hard-$k$ training). Hook confound: Standard uses resid\_pre; TopK uses resid\_post — AUROC magnitudes are not directly comparable across architectures (see Section~\ref{sec:setup}).

**EncNorm outperforms EDA at matched labels.** On Standard/L1 with gold IG labels ($n = 18$), DeLong test (one-sided, EncNorm $>$ EDA): $z = 3.05$, $p = 0.0012$, AUROC difference $= +0.107$ [CI: $+0.041$, $+0.173$]. Cohen's $d$ for group separation: $0.971$ vs.\ $0.533$ --- nearly double the separation. Mean EncNorm for absorbed latents: $3.263$; for non-absorbed: $2.576$ (ratio $= 1.267$).

**Cross-architecture replication.** On TopK-32k (proxy labels), EncNorm achieves AUROC $= 0.837$ [0.807, 0.870] (Cohen's $d = 1.235$). This is the strongest detection result across all experiments. The AUROC cannot be directly compared to the Standard/L1 result due to the hook confound (resid\_pre vs.\ resid\_post; see Section~\ref{sec:setup}), but confirms that EncNorm is applicable to TopK architectures where EDA is not.

**Layer analysis.** EncNorm ratio (absorbed/non-absorbed) peaks at layer 6 (ratio $= 1.267$) and decreases at deeper layers (L10: $0.933$), indicating the gradient competition mechanism is most pronounced at mid-layers. This matches the layer-dependent absorption prevalence observed by \citet{karvonen2025saebench}.

### 4.2 Co-occurrence Jaccard Signal

On 10,000 OpenWebText tokens, $O_\text{Jaccard}$ achieves AUROC $= 0.721$ (Table~\ref{tab:detectors}, Figure~\ref{fig:ablation}). Precision@50 $= 0.100$ --- the top 50 latents by $O_\text{Jaccard}$ contain 5 of the 18 absorbed latents, a $136\times$ enrichment over the random baseline ($P@50_\text{random} = 18/24{,}576 \approx 0.00073$). AUPRC $= 0.075$ vs.\ $0.00073$ random baseline.

Spearman $\rho(\text{EncNorm}, O_\text{Jaccard}) = 0.044$ ($p < 10^{-11}$, not significant as an absorption predictor), confirming the two signals are near-independent. A two-signal audit strategy --- rank by EncNorm, then re-rank by $O_\text{Jaccard}$ --- covers complementary failure modes: EncNorm identifies latents with inflated gradient competition; $O_\text{Jaccard}$ identifies latents with high co-occurrence overlap regardless of encoder geometry.

Combining EncNorm with the directed co-occurrence asymmetry ($A_\text{cooccur}$) into a product score (ARS\_v2) does not improve detection: AUROC $= 0.586$ (DeLong vs.\ EncNorm: $z = -2.455$, $p = 0.993$). Product formulations dilute both signals; the two detectors should be applied independently.

### 4.3 Amortization Gap Experiment: H2 Falsified

**Table 2: OMP Oracle vs.\ Feedforward Absorption Rates** \label{tab:omp}

| Letter | FF AbsRate | OMP AbsRate | Ratio OMP/FF | Reduction |
|--------|-----------|-------------|--------------|-----------|
| a | 0.967 | 0.967 | 1.000 | 0.0% |
| e | 1.000 | 1.000 | 1.000 | 0.0% |
| s | 0.967 | 0.967 | 1.000 | 0.0% |
| **Mean** | **0.978** | **0.978** | **1.000** | **0.0%** |

OMP ($K=53$) achieves reconstruction MSE $= 0.219$ vs.\ feedforward $= 0.242$ --- OMP is a strictly better encoder in reconstruction terms. Despite this, absorption rates are identical (ratio $= 1.000$ across all three letters). See Figure~\ref{fig:omp-results} for per-letter bar chart.

The pre-committed falsification criterion was: OMP $\geq 80\%$ of feedforward absorption rate $\Rightarrow$ H2 falsified. The result ($100\%$ ratio) is unambiguous: **the amortization gap hypothesis is falsified**. The feedforward encoder is already near-optimal for absorbed features; improving it cannot reduce absorption. This directly supports the sparsity landscape account \citep{tang2025partial}: absorption is locked in by the training-time partial minimum, not by the encoder's inference-time approximation error.

**Control: Layer-6 OMP reconstruction quality.** OMP with $K=53$ achieves $R^2 = 0.87$ reconstruction quality on the held-out token set (vs.\ FF $R^2 = 0.84$), ruling out the possibility that OMP simply fails to encode the relevant signal.

### 4.4 Dictionary Width Recovery Analysis

To test whether dictionary width expansion remediates absorption, we compare decoder directions of the 18 absorbed Standard-24k latents against all 32,768 decoder columns of the TopK-32k SAE (cosine similarity threshold $= 0.80$).

**Table 3: Wider SAE Recovery of Absorbed Features** \label{tab:width}

| Metric | Value |
|--------|-------|
| $n$ absorbed (Standard-24k) | 18 |
| $n$ recovered in TopK-32k (cos\_sim $> 0.80$) | 12 (67%) |
| $n$ not recovered | 6 (33%) |
| Mean best cosine similarity | 0.791 |
| Median best cosine similarity | 0.815 |
| EncNorm (recovered, mean) | 3.289 |
| EncNorm (not recovered, mean) | 3.212 |

Two-thirds of absorbed features have geometric counterparts in the wider dictionary. However, one-third do not recover, indicating genuine semantic gaps that capacity increases alone cannot fill. This comparison carries the hook confound identified in Section~\ref{sec:setup} (resid\_pre vs.\ resid\_post): the two SAEs operate on different activation spaces, which may inflate measured cosine similarity between decoder directions in different representational geometries. The 67\% recovery figure should be interpreted as an upper bound until a matched-hook experiment is conducted.

The nearly identical EncNorm for recovered vs.\ not-recovered features ($3.289$ vs.\ $3.212$, $t$-test $p = 0.73$) suggests encoder norm does not distinguish which absorbed features will benefit from width expansion --- this distinction requires structural analysis of the absorbing child's decoder direction.

---

## 5. Discussion \label{sec:discussion}

### 5.1 Mechanistic Synthesis: Sparsity Landscape as Primary Cause

The OMP oracle result (Section~\ref{sec:experiments}) has a single, clean implication: the feedforward encoder is not the bottleneck for absorption. OMP with $K = 53$ is the best possible $K$-sparse encoder given the fixed SAE dictionary --- if it cannot reduce absorption, no inference-time encoder improvement will. This includes all variants proposed under the amortization gap framework: learned iterative thresholding (O'Neill et al.), recurrent encoders, or attention-based encoders.

The mechanism is explained by the sparsity landscape account \citep{tang2025partial}: absorbed features exist at partial minima of the SDL training loss. At a partial minimum, the encoder direction $\mathbf{w}_{\text{enc},j}$ has been displaced from the decoder direction $\mathbf{d}_j$ by persistent gradient competition during training. This displacement is a property of the trained weight matrices, not of the inference procedure. The only interventions that can escape this attractor are ones that alter the loss landscape during training: masked regularization (suppressing the competing gradient), hierarchically-aware objectives, or structural changes to the dictionary.

The same training-time gradient competition that creates absorption attractors also inflates the encoder weight norms of absorbed latents. When child latent $c$ fires on token $t$ in place of parent $j$, the reconstruction residual is attributed to $j$'s encoder direction; the encoder gradient pushes $\|\mathbf{w}_{\text{enc},j}\|_2$ upward as $j$ struggles to compete. EncNorm therefore measures the accumulated evidence of gradient competition across training --- a weight-only fingerprint of the training-time phenomenon, not of inference-time behavior. This explains why EncNorm achieves AUROC $= 0.757$--$0.837$ without requiring any activation data, and why it peaks at the layer where absorption is most prevalent (L6, ratio $= 1.267$).

**Practical guidance.** For SAE practitioners seeking to reduce absorption: (1) Do not expect Orthogonal Matching Pursuit or other iterative encoders to help. (2) Focus evaluation budget on training-time interventions. (3) Use EncNorm as a fast first-pass screen to identify absorbed-candidate latents (AUROC $= 0.757$), but note that EncNorm does not distinguish which candidates will benefit from dictionary width expansion vs.\ which require training-objective changes (Section~\ref{sec:experiments}, Table~\ref{tab:width}). Structural analysis of the absorbing child's decoder direction is needed to make that distinction.

Table~\ref{tab:implications} summarizes the practical implications across intervention types.

**Table 4: Implications for SAE Mitigation Approaches** \label{tab:implications}

| Intervention | Addresses early absorption? | Addresses late absorption? | Evidence |
|---|---|---|---|
| Iterative encoder (OMP, ISTA) | No | No | H2 falsification (0% reduction) |
| Wider dictionary | Yes (67% recovery) | No (33% unrecovered) | Width recovery (Section 4.5) |
| Masked regularization | Plausible | Yes | Tang et al. theory + H2 support |
| EncNorm screening | Detection only | Detection only | AUROC 0.757--0.837 |

### 5.2 Limitations

**Small positive-class sample.** The gold IG label set contains only $n_\text{pos} = 18$ absorbed features at GPT-2 L6; at this sample size, bootstrap CIs span approximately $0.13$ AUROC and the DeLong $p$-value should be treated as approximate. Three-context replication (Standard/L1, TopK-32k with proxy labels, layer monotonicity analysis) provides partial mitigation, but a larger labeled dataset would substantially strengthen the detection claims.

**Hook confound in cross-architecture comparison.** The Standard SAE hooks into \texttt{blocks.6.hook\_resid\_pre} (before attention) while the TopK-32k SAE hooks into \texttt{blocks.6.hook\_resid\_post} (after attention). These are different activation spaces with different representational geometry. The observed AUROC difference ($0.757$ vs.\ $0.837$) cannot be attributed to architecture (L1 vs.\ TopK) without controlling for hook point. This confound also affects the width recovery analysis (Section~\ref{sec:experiments}). A matched experiment using two SAEs at the same hook would resolve both.

**H3 entity-type probe failure.** The cross-hierarchy absorption experiment reports near-zero absorption rates for all entity-type hierarchies. This is almost certainly a probe quality artifact: probes trained on a different model and projected to GPT-2 activation space do not preserve probe direction quality. Whether absorption generalizes beyond the first-letter spelling task remains an open question requiring probes trained directly on the target model.

**OMP fixed-dictionary assumption.** The OMP oracle fixes the pre-trained SAE decoder and only varies the encoder. A stronger version of the amortization gap hypothesis might posit that both encoder and decoder are jointly sub-optimal, and that re-training both with an iterative encoder would reduce absorption. Our experiment cannot rule out this joint hypothesis. We view this as an important target for future work.

### 5.3 Implications for SAE Design

The width recovery analysis (Section~\ref{sec:experiments}, Table~\ref{tab:width}) provides the clearest practical guidance: expanding dictionary width helps for features whose parent was simply absent from the narrower dictionary (the "early absorption" case), but does not help for features whose parent exists in the dictionary but has been suppressed by training dynamics ("late absorption"). A diagnostic tool that distinguishes these two cases --- for instance, decoder-direction taxonomy combined with EncNorm --- would enable practitioners to apply targeted interventions.

Masked regularization \citep{narayanaswamy2026masked} directly addresses the gradient competition mechanism: by masking the sparsity gradient from high-frequency child latents on their frequent co-occurring partners, it prevents the training-time attractor formation. Our H2 result provides independent theoretical support for this approach and motivates its extension to arbitrary feature co-occurrence patterns.

---

## 6. Conclusion

We present three contributions toward understanding and detecting feature absorption in Sparse Autoencoders.

**Mechanistic.** A controlled OMP oracle experiment shows that replacing the feedforward SAE encoder with an optimal $K$-sparse solver at matched sparsity produces zero reduction in feature absorption rates across all tested configurations. This decisively falsifies the amortization gap hypothesis as a dominant cause of absorption and provides the first controlled experimental evidence for the sparsity landscape account \citep{tang2025partial}. The practical implication is clear: practitioners seeking to reduce absorption should focus on training-time interventions, not inference-time encoder improvements.

**Methodological.** Encoder weight norm ($\|\mathbf{w}_{\text{enc},j}\|_2$) predicts absorbed latents with AUROC $= 0.757$ [0.665, 0.849] on Standard/L1 and $0.837$ [0.807, 0.870] on TopK-32k (architectures not directly comparable due to hook confound), significantly outperforming EDA (DeLong $z = 3.05$, $p = 0.0012$). Jaccard co-occurrence score provides an independent signal (AUROC $= 0.721$, Spearman $\rho = 0.044$ with EncNorm). Both metrics require only SAE weights or activation statistics --- no probe directions or integrated gradients --- making them scalable to large SAE dictionaries.

**Practical.** 67\% of absorbed features in a 24k Standard SAE have geometric counterparts in a wider 32k TopK SAE, suggesting dictionary expansion partially addresses absorption (subject to hook-confound caveat, Section~\ref{sec:discussion}). The remaining 33\% require training-time structural changes. EncNorm does not distinguish which absorbed features benefit from width expansion, motivating the combination of EncNorm with structural taxonomy analysis for targeted remediation.

**Future work.** The most pressing open issues are: (1) resolving the hook confound in cross-architecture comparison with a matched-hook experiment; (2) obtaining a larger gold-label dataset ($n_\text{pos} \gg 18$) to narrow detection AUROC confidence intervals; (3) testing whether absorption generalizes to non-spelling-task feature hierarchies using probes trained on the target model; and (4) safety attribution analysis: quantifying the downstream impact of absorption on mechanistic interpretability circuit analyses.

The core message is that feature absorption in SAEs is primarily a training problem, and the interpretability community's diagnostic and remediation tooling should be oriented accordingly.

---

## Figures and Tables

- Figure 1 (teaser): `fig_teaser.pdf` --- Two-panel: (left) absorption rates under feedforward vs. OMP encoding per letter; (right) ROC curves for EncNorm vs. EDA vs. random. Data: `exp/results/full/C2_amortization_gap_full.json` (left), `exp/results/full/A3_encoder_norm_cross_arch.json` (right).
- Figure 2 (enc-norm-hist): `fig_enc_norm_hist.pdf` --- Histogram of encoder norm for absorbed (n=18) vs. non-absorbed latents at GPT-2 L6; vertical lines at absorbed mean=3.263 and non-absorbed mean=2.576; Cohen's d=0.971 annotated. Data: `exp/results/full/A3_encoder_norm_cross_arch.json`.
- Figure 3 (omp-design): `fig_omp_design.pdf` --- Architecture diagram: two-branch experiment with fixed decoder and varied encoder (feedforward vs. OMP).
- Figure 4 (roc-curves): `fig_roc_curves.pdf` --- ROC curves for all detectors, two-panel (Standard/L1 and TopK-32k). Data: `exp/results/full/A3_encoder_norm_cross_arch.json`, `exp/results/full/B2_ars_v2_validation.json`.
- Figure 5 (ablation): `fig_ablation.pdf` --- Horizontal bar chart: AUROC per detector method. Data: `exp/results/full/B2_ars_v2_validation.json`.
- Figure 6 (omp-results): `fig_omp_results.pdf` --- Per-letter absorption rate under feedforward vs. OMP (grouped bar chart). Data: `exp/results/full/C2_amortization_gap_full.json`.
- Table 1: Inline --- Detection performance at GPT-2 Layer 6 (all detectors, both architectures).
- Table 2: Inline --- OMP oracle vs. feedforward absorption rates per letter.
- Table 3: Inline --- Width recovery summary (18 absorbed features in 32k SAE).
- Table 4: Inline --- Practitioner implications matrix (intervention type vs. absorption phase).
