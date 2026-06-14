# 2. Related Work

## 2.1 Feature Absorption in Sparse Autoencoders

Feature absorption was first analytically characterized by Chanin et al. (2024), who proved that sparsity loss incentivizes the suppression of parent features when child features are active. In a sparse autoencoder (SAE) trained with an $L_1$ penalty on latents, the encoder must choose which features to activate for a given input. When a child concept (e.g., "short") and its parent (e.g., "starts with S") co-occur, the encoder can satisfy the sparsity constraint by activating only the child, discarding the parent. Chanin et al. introduced a detection protocol based on ground-truth logistic probes and k-sparse probing, measuring the accuracy drop between residual-stream and SAE-latent classifications. The resulting absorption score, $A_{\text{full}}$, quantifies the fraction of parent-feature information lost in the SAE representation.

This work established absorption as a central pathology in SAEs, but its evaluation task---first-letter classification (e.g., "starts with S" vs. "short")---is narrow and artificial. Chanin et al. explicitly noted this limitation, calling for "finding examples of feature absorption unrelated to character identification" as future work. Our study directly addresses this gap.

## 2.2 SAEBench and Benchmark Standardization

SAEBench (Karvonen et al., 2025) standardized the Chanin absorption protocol, making it one of eight canonical SAE evaluations. The benchmark adapted the metric technically by replacing ablation-based criteria with probe-projection criteria, enabling evaluation at all layers rather than only early layers. However, the underlying evaluation task remains first-letter classification. SAEBench reports absorption scores across architectures, layers, and sparsity levels, providing the community with a shared reference point. Our work tests whether this shared reference point generalizes beyond its training domain.

## 2.3 Architecture Advances Targeting Absorption

The absorption metric has directly shaped SAE architecture development. Matryoshka SAEs (Bussmann et al., 2025) use nested latent spaces to reduce absorption by allowing features to exist at multiple scales. OrtSAE (Korznikov et al., 2025) orthogonalizes decoder directions to minimize interference between parent and child features. Hierarchical SAEs (Zhan et al., 2026) explicitly model parent-child relationships in the architecture itself. All three report absorption reductions as primary contributions, using the first-letter benchmark. If first-letter absorption does not predict semantic-hierarchy absorption, these architectures may have been optimized for the wrong target.

## 2.4 Construct Validity in Neural Network Evaluation

The question of whether a benchmark measures what it claims to measure---construct validity---has received sustained attention in ML evaluation. Rajpurkar et al. (2016) showed that SQuAD reading comprehension scores do not predict performance on adversarially modified passages. Bowman & Dahl (2021) argued that many NLP benchmarks conflate multiple constructs, making score improvements uninterpretable. In mechanistic interpretability specifically, RAVEL (Vig et al., 2024) evaluates factual disentanglement using country-continent hierarchies, but does not frame its task as an absorption metric. No prior work has systematically tested whether the dominant absorption metric generalizes from first-letter tasks to real semantic hierarchies.

## 2.5 Semantic Hierarchies in Interpretability Research

WordNet (Miller, 1995) has been used extensively in NLP and interpretability research as a source of structured semantic knowledge. In the context of SAEs, prior work has used WordNet to evaluate feature quality (Bricken et al., 2023) and to construct concept hierarchies for probing (Lieberum et al., 2023). However, these studies do not measure absorption---they evaluate whether SAE features align with human-annotated semantic categories. Our work is the first to adapt the SAEBench absorption formula to WordNet hierarchies, creating a direct bridge between the benchmark's measurement protocol and real semantic structure.

## 2.6 Positioning This Work

We situate our contribution at the intersection of three lines of research: (1) the theoretical characterization of absorption (Chanin et al., 2024), (2) the standardization of absorption measurement (SAEBench), and (3) the use of semantic hierarchies for evaluating interpretable representations. Our study is the first to test the construct validity of the dominant absorption metric on real semantic hierarchies, using a rigorous statistical protocol with bootstrap confidence intervals, paired t-tests, and a random-SAE control. The results have direct implications for benchmark design and architecture evaluation in the SAE community.

<!-- FIGURES
- None
-->
