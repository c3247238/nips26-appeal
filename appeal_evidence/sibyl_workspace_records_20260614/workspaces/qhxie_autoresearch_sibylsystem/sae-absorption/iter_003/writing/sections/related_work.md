# 2. Background and Related Work

## 2.1 Sparse Autoencoders and Feature Absorption

A Sparse Autoencoder trained on residual stream activations $\mathbf{x} \in \mathbb{R}^{d_\text{model}}$ learns encoder $W_e \in \mathbb{R}^{d_\text{SAE} \times d_\text{model}}$, decoder $W_d \in \mathbb{R}^{d_\text{model} \times d_\text{SAE}}$ with unit-normed columns, and biases, minimizing
$$\mathcal{L} = \mathbb{E}_\mathbf{x}\left[\|\mathbf{x} - W_d \mathbf{z}\|^2 + \lambda \|\mathbf{z}\|_1\right], \quad \mathbf{z} = \text{ReLU}(W_e \mathbf{x} + \mathbf{b}_e).$$
The $\ell_1$ penalty drives most latent activations to zero. In TopK architectures \citep{makhzani2015winner,templeton2024scaling}, $\|\mathbf{z}\|_1$ is replaced by a hard constraint retaining only the top-$k$ activations per forward pass.

\citet{chanin2024absorption} identified \textbf{feature absorption}: latent $j$ is absorbed by child $c$ if $z_j = 0$ on inputs where $z_j$ should be positive, while $z_c > 0$ on those same inputs. They measure absorption via an integrated gradients pipeline on the first-letter spelling task (26 classes), reporting absorption rates of 15--35\% in mid-layer Gemma Scope SAEs. \citet{karvonen2025saebench} incorporates absorption rates as a first-class SAEBench metric, making it a standard benchmark for SAE quality.

## 2.2 Competing Mechanistic Accounts

\paragraph{Amortization gap (O'Neill et al., 2024).} SAE encoders approximate optimal sparse inference via a single feedforward pass. \citet{oneill2024amortization} show that this amortized approximation systematically under-activates features that are weakly active or co-occurring with stronger features. They propose \textit{learned iterative thresholding} encoders as a remedy. Under this account, absorption is a byproduct of the encoder's approximation error, and replacing the encoder with an oracle solver should eliminate it.

\paragraph{Sparsity landscape / partial minimum (Tang et al., 2025).} \citet{tang2025partial} analyze the biconvex SDL loss landscape and identify stable partial minima where an absorbed feature $j$'s encoder direction $\mathbf{w}_{e,j}$ drifts away from its decoder direction $\mathbf{d}_j$. The mechanism is a training-time gradient conflict: when child $c$ fires on inputs that also activate parent $j$, the sparsity gradient from $c$ persistently suppresses $j$'s encoder. This creates an attractor that survives any inference-time change to the encoding procedure.

## 2.3 Mitigation Proposals

Several SAE variants have been proposed to reduce absorption: \textit{Matryoshka SAEs} \citep{bussmann2024matryoshka} hierarchically nest features to encourage parent-child allocation; \textit{OrtSAE} applies orthogonality regularization; \textit{Masked Regularization} \citep{narayanaswamy2026masked} explicitly suppresses sparsity gradients from high-frequency features on their co-occurring partners. The success of masked regularization specifically at training time provides preliminary support for the sparsity landscape account; our experiment provides direct evidence.

## 2.4 Weight-Only Absorption Indicators

\citet{chanin2024absorption} and \citet{karvonen2025saebench} require model activations and integrated gradients — expensive at scale. Prior work from this research program (\citeauthor{eda2025}) introduced \textbf{EDA} ($1 - \cos(\mathbf{w}_{e,j}, \mathbf{d}_j)$), a weight-only metric with AUROC $= 0.650$ at GPT-2 L6 and $0.776$ at Gemma Scope L12-16k. EDA has theoretical grounding (formal lower bound connecting EDA to absorption degree), but is poorly suited to TopK architectures where encoder-decoder angular relationships are less constrained. The present work introduces encoder weight norm as an alternative with stronger empirical performance and wider architectural applicability.
