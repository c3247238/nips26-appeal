# 2 Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Neural networks encode more features than they have neurons by representing features as overlapping linear directions in activation space---a phenomenon called superposition (Elhage et al., 2022). Sparse autoencoders (SAEs) address this by projecting the residual stream activation $\mathbf{x} \in \mathbb{R}^d$ into an overcomplete basis of $m \gg d$ latent features $\mathbf{z} \in \mathbb{R}^m$ via a sparsity-inducing encoder, then reconstructing $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$. When each latent $z_i$ fires for a single coherent concept---monosemanticity---the sparse decomposition provides a human-readable vocabulary for model internals.

Early work demonstrated that SAEs recover interpretable features in small models: Cunningham et al. (2024) showed highly interpretable features in Pythia-70M/410M, and Bricken et al. (2023) decomposed a one-layer transformer's MLP activations into monosemantic directions. Rapid scaling followed. Anthropic applied SAEs to Claude 3 Sonnet and identified safety-relevant features including deception and sycophancy (Templeton et al., 2024). OpenAI trained a 16M-latent TopK SAE on GPT-4 with clean scaling laws (Gao et al., 2024). Google DeepMind released Gemma Scope, providing over 400 open JumpReLU SAEs across all layers of Gemma 2 models at widths from 16k to 1M (Lieberum et al., 2024). Anthropic's circuit tracing in Claude 3.5 Haiku demonstrated that reliable SAE features enable tracing of planning-ahead, hallucination, and jailbreak mechanisms through feature-level attribution graphs (Lindsey et al., 2025).

These successes established SAEs as the dominant tool for mechanistic interpretability. They also exposed systematic failure modes---feature absorption, hedging, inconsistency, and non-canonicality---that challenge whether SAEs reliably recover the features they promise.

## 2.2 Feature Absorption

Chanin et al. (2024) formalize feature absorption: a parent feature (e.g., "starts with S") fails to activate when a child feature (e.g., "the word Saturday") co-occurs, because the SAE achieves better sparsity by encoding the parent's information into the child's decoder direction $\mathbf{d}_c$. The mechanism is driven by decoder similarity: when $\cos(\mathbf{d}_c, \mathbf{w}_p)$ is high, the child's reconstruction already covers the parent direction, so encoding an additional parent latent wastes one unit of $L_0$ without improving reconstruction. Chanin et al. prove in a two-layer toy model that hierarchical feature structure is sufficient to produce absorption, and measure absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs.

Absorption is not a training error but a rational optimization outcome. For a parent-child pair $(p, c)$, an SAE with absorption encodes both concepts at $L_0$ cost $+1$ (child only), while absorption-free encoding requires $+2$ (child and parent). Any solution that fully eliminates absorption will occupy a worse position on the variance-explained vs. $L_0$ frontier (Chanin et al., 2024). Theoretical analysis reinforces this: Cui et al. (2025) show that SAEs generally fail to recover ground-truth features unless features are extremely sparse, and a unified theory of sparse dictionary learning casts absorption as a natural consequence of the piecewise biconvex optimization landscape shared by all SAE variants (Wright et al., 2025).

The canonical measurement procedure identifies false negatives (FN) via linear probes trained on raw activations, runs integrated-gradients attribution on FN tokens, and detects absorption when the highest-attribution latent has cosine similarity exceeding a threshold $\tau_{\cos}$ with the probe direction. SAEBench (Karvonen et al., 2025) adopted this procedure as one of its eight standardized evaluation metrics and found absorption present in every architecture tested---TopK, JumpReLU, BatchTopK, and Matryoshka. Tian et al. (2025) frame absorption as a special case of poor feature sensitivity: features that appear monosemantic from their activation examples may nonetheless fail to activate on semantically similar inputs.

A critical empirical limitation unifies all these measurements: every absorption rate in the literature is computed on a single task---first-letter spelling classification---using 26 letter classes with near-perfect probe accuracy ($F_1 = 1.0$) and 100% parent-child co-occurrence by construction. Whether these rates generalize to knowledge hierarchies, where class counts range from 6 to 80, class balance varies from moderate to extreme, and probe quality is imperfect ($F_1 = 0.73$--$0.87$), is unknown.

## 2.3 Adjacent Failure Modes

Absorption is one of several documented SAE failure modes, each with distinct causes and manifestations.

**Feature hedging** (Chanin et al., 2025) occurs when insufficient dictionary width forces correlated features to merge into a single latent. Hedging and absorption both manifest as features failing to fire where expected, but their root causes differ: width or sparsity limitations for hedging versus hierarchical feature structure for absorption. Matryoshka SAEs trade one for the other---their nested prefix loss reduces absorption to ${\sim}0.03$ (vs. BatchTopK's ${\sim}0.29$ on SAEBench) but introduces hedging at inner levels (Chanin et al., 2025). Incorrect $L_0$, which is common in practice, triggers hedging and feature mixing even in well-architectured SAEs: Chanin and Garriga-Alonso (2025) show that most open-source SAEs have $L_0$ that is too low, causing systematically wrong features.

**Feature inconsistency** (Song et al., 2025) describes the phenomenon where independent SAE training runs converge to different feature sets. TopK SAEs achieve only 0.80 consistency across runs as measured by pairwise mean cosine correlation.

**Feature non-canonicality** (Leask et al., 2025) challenges the assumption that SAE latents are atomic units of analysis. Meta-SAEs decompose individual latents into sub-features, and larger dictionaries discover qualitatively new latents missed by smaller ones---suggesting SAE features are neither complete nor atomic.

**Multi-dimensional features** (Engels et al., 2024) present a distinct challenge: irreducible circular representations for concepts like days of the week or months cannot be captured by one-dimensional SAE directions, potentially producing absorption-like pathologies when the model forces multi-dimensional structure into scalar latents.

## 2.4 SAE Architectures and Absorption Mitigation

Several architectures reduce absorption, though none eliminate it. JumpReLU SAEs (Rajamanoharan et al., 2024a) use a learnable threshold activation and form the backbone of Gemma Scope; SAEBench reports they exhibit the worst absorption among tested architectures at low $L_0$, possibly because longer training deepens the absorption optimum. BatchTopK SAEs (Rajamanoharan et al., 2024b) relax the TopK constraint to the batch level, enabling variable sparsity per sample; they show absorption comparable to standard TopK because the reconstruction incentive for absorption persists regardless of the sparsity mechanism. Matryoshka SAEs (Bussmann et al., 2025) train nested dictionaries of increasing size simultaneously, creating a natural feature hierarchy where smaller dictionaries learn general concepts; they achieve the lowest SAEBench absorption (${\sim}0.03$). OrtSAE (Korznikov et al., 2025) enforces orthogonality via pairwise cosine similarity penalties on decoder directions, reducing absorption by ${\sim}70\%$ vs. BatchTopK and achieving mean cosine similarity 2.7$\times$ lower. Adaptive Temporal Masking (Li et al., 2025) dynamically adjusts feature selection via per-latent importance scoring and reports the lowest published SAEBench absorption metric values (0.0068 vs. TopK 0.1402 on Gemma 2 2B). Masked regularization (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training via token masking, reducing absorption across architectures.

A limitation unites all these evaluations: every architecture comparison uses the first-letter spelling task as its absorption benchmark. Whether architecture rankings transfer to semantically richer hierarchies---where class counts, balance, and probe quality differ---is untested. This paper provides the first such cross-domain comparison (Section 6), though with limited statistical power (12 observations across four architectures).

## 2.5 Entity-Attribute Evaluation with RAVEL

RAVEL (Resolved Attribute Value Estimation for Language models; Huang et al., 2024) provides a framework for evaluating how language models encode entity-attribute relationships. The dataset contains thousands of entities (cities, people, objects) with validated attribute labels (country, continent, language, birthplace) and prompt templates designed to elicit attribute information. Linear probes trained on model activations with RAVEL labels achieve high accuracy for most attributes, establishing that models encode these relationships as recoverable linear directions.

RAVEL's city subset provides natural feature hierarchies for absorption measurement. The city "Paris" is a child of the country "France," the continent "Europe," and the language "French." These three hierarchies vary along dimensions relevant to absorption:

- **Class count.** City-continent has 6 classes (comparable to a coarse grouping), city-language has ${\sim}20$, and city-country has ${\sim}80$ (comparable to the 26-class first-letter task in scale, though far more imbalanced).
- **Class balance.** City-continent is moderately balanced; city-country is highly skewed (USA: 176 entities vs. Chad: 5).
- **Probe quality.** RAVEL probes achieve $F_1 = 0.73$--$0.87$ at layer 24 on Gemma 2 2B, compared to $F_1 = 1.0$ for first-letter probes. This gap is itself a potential confound that we explicitly decompose via the probe degradation ablation (Section 4.6).
- **Co-occurrence structure.** Unlike first-letter spelling, where every word co-occurs with exactly one letter, city-attribute relationships embed in a multi-level graph (city $\to$ country $\to$ continent) with non-trivial co-occurrence patterns.

These properties make RAVEL a natural testbed for extending absorption measurement beyond the controlled but potentially unrepresentative first-letter task.

## 2.6 Benchmarks and the Evaluation Gap

SAEBench (Karvonen et al., 2025) standardized SAE evaluation with eight metrics across 200+ SAEs and seven architectures, revealing that proxy metrics (cross-entropy loss recovered, sparsity) do not reliably predict practical performance on downstream tasks. SynthSAEBench (2026) provides a controlled synthetic environment with known ground-truth feature hierarchies, where logistic probes achieve $F_1 = 0.974$ while SAEs substantially underperform. CE-Bench (Gulko et al., 2025) offers a lightweight contrastive benchmark correlating at $> 70\%$ Spearman $\rho$ with SAEBench.

Across all three benchmarks, absorption is measured exclusively on first-letter spelling---a proxy task with 26 classes, near-uniform distribution, and perfect probe accuracy. No benchmark includes cross-domain absorption measurement. No cross-layer analysis examines how absorption varies from early to late transformer layers; published rates aggregate over single layers or report layer-averaged scores. Google DeepMind's decision to deprioritize SAE research was driven partly by 10--40% performance degradation on safety-relevant downstream tasks using SAE-reconstructed activations (Smith et al., 2025), but the specific contribution of feature absorption to this degradation has not been isolated.

Three specific gaps motivate this work:

1. **Single-task evaluation monoculture.** Every absorption rate in the literature comes from first-letter spelling. Absorption rates on knowledge hierarchies---where safety-relevant features reside---are unknown.
2. **No probe quality confound control.** First-letter probes achieve $F_1 = 1.0$; RAVEL probes achieve $F_1 = 0.73$--$0.87$. Cross-domain comparison without controlling for this difference risks attributing probe artifacts to hierarchy effects.
3. **No causal mechanism validation beyond spelling.** Chanin et al. (2024) provide correlational evidence (decoder cosine similarity, integrated-gradients attribution) for competitive exclusion on first-letter data. Whether the same causal mechanism operates in knowledge hierarchies has not been tested with interventional methods.

This paper addresses all three gaps. We extend absorption measurement to four feature hierarchies spanning syntactic and semantic domains, introduce a probe degradation ablation to decompose probe quality confounds from genuine hierarchy effects, and validate the competitive exclusion mechanism via activation patching across all tested hierarchy types.

<!-- FIGURES
- None
-->
