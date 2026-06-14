# Related Work

## Feature Absorption in Sparse Autoencoders

Chanin et al. \citeyear{chanin2024absorption} define feature absorption operationally: a
child latent fails to activate on inputs where its concept is present because a parent latent
at a higher level of the feature hierarchy has absorbed the activation signal. They validate
this phenomenon using the first-letter task---a controlled hierarchy where letter-class
features (parent) subsume individual token features (child)---and report absorption rates of
15--35\% across Gemma Scope, Llama 3.2, and Qwen2 SAEs spanning multiple widths and layers.
Their detection method, the FeatureAbsorptionCalculator, uses integrated-gradients ablation
over pre-specified linear probe directions and finds absorption in every tested SAE
architecture. Chanin et al. provide an informal argument that absorption is sparsity-efficient
(absorbing a parent-child pair saves one L0 per parent-only context) but do not formalize
this as a rate-distortion condition or derive a quantitative threshold.

SAEBench \cite{karvonen2025saebench} evaluates over 200 open-source SAEs across eight
architectures and includes absorption rate (via the Chanin et al. metric) as one of eight
metrics. A key negative finding: proxy metrics for SAE quality---cross-entropy loss and mean
sparsity---do not reliably predict absorption rate, meaning absorption cannot be inferred from
training curve diagnostics alone. SAEBench explicitly acknowledges that its absorption metric
relies on the first-letter hierarchy and may not generalize to domain-specific feature
absorption. Our work complements SAEBench: EDA provides a weight-only, probe-free pre-screen
that operates independently of activation data, and our phase-stability results show the
absorption task scores for SAEBench are unlikely to change dramatically across the
hyperparameter range SAEBench sweeps.

SynthSAEBench \cite{synthsaebench2026} constructs large-scale synthetic data with known
ground-truth feature hierarchies, including Zipfian firing distributions and realistic
correlation structure. The benchmark finds that SAEs substantially underperform direct probes
on feature recovery, confirming that absorption is not an artifact of the first-letter proxy
task. Our EDA detector operates as a weight-only pre-screen that does not require the ground
truth labels that synthetic benchmarks provide.

---

## Theoretical Frameworks for Absorption

Tang et al. \citeyear{tang2024unified} provide a unified theoretical framework casting sparse
dictionary learning (SAEs, transcoders, crosscoders) as piecewise biconvex optimization and
prove that absorption solutions are spurious minima of the SAE training loss. They show that
feature anchoring---pinning specific latents to ground-truth feature directions during
training---restores identifiability and eliminates absorption, but validate this only on
synthetic linear benchmarks. Our Proposition~1 refines the Tang et al. framework to a
concrete rate-distortion condition: for a specific parent-child pair, absorption is the loss-optimal
solution if and only if $\lambda > \sin^2(\theta_{p,c})$. Crucially, co-occurrence frequency
$p_\text{co}$ cancels from this threshold (Corollary~1), a prediction that the Tang et al.
framework does not make explicit.

Cui et al. \citeyear{cui2025limits} prove that SAEs generally fail to recover ground-truth
monosemantic features unless features are extremely sparse (density below a threshold that
scales with model width), establishing a fundamental limit on SAE identifiability. Their
reweighted SAE (WSAE) achieves better feature recovery under restrictive generative assumptions
but is not evaluated on absorption specifically. Chen et al. \citeyear{chen2025taming} propose
bias-adaptation training with theoretical recovery guarantees under a statistical generative
model, but their analysis does not model feature hierarchy and therefore does not address
absorption as a hierarchical failure mode. Our work operates in a complementary regime:
rather than proposing a new training procedure, we characterize the geometry of the absorbed
state in trained SAEs and derive a post-hoc weight-only diagnostic.

---

## Architectural Mitigations

Four architectural families target feature absorption directly. Matryoshka SAEs
\cite{bussmann2025matryoshka} train nested inner and outer codebooks simultaneously, with the
inner codebook learning general parent-level features. This hierarchical structure reduces the
gradient pressure that pulls child encoders toward parent directions during training; on
SAEBench, Matryoshka SAEs rank first on absorption, RAVEL, and sparse probing at the cost of a
small reconstruction penalty. OrtSAE \cite{korznikov2025ortsae} applies an orthogonality
penalty $\beta \sum_{i \neq j} (d_i \cdot d_j)^2$ to decoder columns, forcing decoder
directions apart and increasing $\theta_{p,c}$ for all feature pairs. Because our Proposition~1
requires $\sin^2(\theta_{p,c}) < \lambda$ for absorption to be loss-optimal, increasing
$\theta_{p,c}$ directly raises the threshold---OrtSAE reduces absorption by 65\% on the
first-letter task. ATM SAE \cite{li2025atm} introduces per-latent adaptive importance scoring
during training and achieves the lowest reported absorption score on Gemma 2 2B (0.0068 vs.
0.1402 for TopK and 0.0114 for JumpReLU), reducing the effective $\lambda$ for frequently
activated features. Masked regularization \cite{narayanaswamy2026masked} disrupts
co-occurrence patterns during training via token masking, improving out-of-distribution
robustness and reducing absorption. KronSAE \cite{kronsae2025} applies Kronecker product
factorization to SAE latents and reduces mean absorption fraction across all sparsity levels.

All architectural mitigations require retraining the SAE from scratch. Our detection
method---computing EDA and cross-directional cosines from existing weight matrices---is
post-hoc and applies to any deployed SAE without modification.

---

## Encoder-Decoder Alignment and the Amortization Gap

O'Neill et al. \citeyear{oneill2024amortization} characterize the amortization gap in SAEs:
the discrepancy between the optimal sparse code for a given input (achievable by iterative
pursuit) and the code produced by the encoder in a single forward pass. Their analysis shows
the encoder systematically underestimates feature activations due to the bias introduced by
L1 training, and that this gap is a distinct failure mode from absorption. The amortization
gap is a confounder for EDA: a feature with high EDA may reflect encoder drift from
absorption (as we predict via Proposition~2) or encoder-decoder misalignment from amortization
gap effects unrelated to absorption. We acknowledge this tension explicitly and note that EDA
achieves AUROC = 0.650 above permutation null despite this confounder, suggesting that
absorption-driven EDA is the dominant signal at GPT-2 Small layer 6.

Feature hedging \cite{chanin2025hedging} is a complementary failure mode: narrow SAEs
merge correlated features because they lack the capacity to represent each distinctly. Feature
hedging and feature absorption share a root cause---flat sparsity penalties under feature
correlation---but manifest in opposite capacity regimes: hedging in narrow SAEs
($d_\text{sae}$ too small), absorption when hierarchical features are present regardless of
width. Our phase-stability results (Section~\ref{sec:experiments}) are consistent with
this distinction: absorption rates remain uniformly high at 0.919--0.968 across widths from
12k to 98k, showing no mitigation from width alone.

---

## SAE Evaluation: Sensitivity and Feature Consistency

Tian et al. \citeyear{tian2025sensitivity} frame feature absorption as a special case of poor
feature sensitivity: a feature has low sensitivity if it fails to activate on texts similar
to its activating examples. Their sensitivity metric is computable from activation data but
requires curated test sets per feature. EDA is strictly more parsimonious: it requires only
the weight matrices, with no activation data or curation. The two metrics are complementary:
sensitivity measures the behavioral consequence of absorption; EDA measures the weight-level
geometric cause.

Song et al. \citeyear{song2025consistency} document that SAE features are inconsistent across
training runs (TopK SAEs achieve 0.80 pairwise match, the best among tested architectures),
raising the concern that absorption rates could vary with random seed. Our experiments use
canonical SAELens pre-trained SAEs at fixed releases, avoiding multi-seed variance; however,
the AJT polarity reversal we observe across training regimes (Section~\ref{sec:experiments})
shows that training procedure is a stronger modulator of EDA than layer or width.

---

## Cross-Directional Cosine Measures

No prior work uses $\cos(\hat{e}_p, d_c)$---the cosine between a parent feature's encoder
direction and a child feature's decoder direction---as a pairwise absorption signal. The
closest precedent is the Bricken et al. \citeyear{bricken2023monosemanticity} observation
that decoder-decoder cosine similarity clusters related features, but that metric operates at
the level of feature families rather than individual parent-child absorption pairs and does not
use encoder directions. Our finding that $\cos(\hat{e}_p, d_c)$ achieves AUROC = 0.730
(Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$)---higher than EDA's AUROC = 0.650 for the
same labels---is an empirical discovery not predicted by any prior theory and constitutes a
contribution independent of the rate-distortion framework.

---

**Summary.** Table~\ref{tab:related} positions the contributions of this paper against prior
work. Chanin et al. establish the measurement framework but require probes; Tang et al. prove
absorption is a spurious minimum but do not give a quantitative threshold; architectural
mitigations all require retraining; no prior method uses SAE weight matrices alone to detect
absorbed features at the individual-latent level.

| Work | Probe-free | Weight-only | Quantitative threshold | Post-hoc | Architecture-agnostic |
|---|---|---|---|---|---|
| Chanin et al. \citeyear{chanin2024absorption} | No | No | No | Yes | Yes |
| Tang et al. \citeyear{tang2024unified} | N/A | N/A | No | No (training) | Theoretical |
| Matryoshka SAE \cite{bussmann2025matryoshka} | No | No | No | No | No |
| OrtSAE \cite{korznikov2025ortsae} | No | No | No | No | No |
| ATM SAE \cite{li2025atm} | No | No | No | No | No |
| Tian et al. \citeyear{tian2025sensitivity} | No | No | No | Yes | Yes |
| **This work (EDA)** | **Yes** | **Yes** | **Yes ($\lambda > \sin^2(\theta_{p,c})$)** | **Yes** | **Yes** |

Building on the Tang et al. theoretical framework, we formalize the rate-distortion threshold,
derive the mechanistic conjecture connecting absorption to EDA, and validate both against exact
absorption labels from the FeatureAbsorptionCalculator.

<!-- FIGURES
- Table 1 (summary): inline — Positioning of this paper's contributions against prior work on probe-free detection, weight-only computation, quantitative threshold, post-hoc applicability, and architecture-agnosticism
- None (no data-driven figures assigned to this section in the Figure and Table Plan)
-->
