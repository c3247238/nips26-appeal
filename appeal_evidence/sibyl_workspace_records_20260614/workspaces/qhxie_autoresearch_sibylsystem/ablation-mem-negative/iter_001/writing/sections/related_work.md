# Related Work

### 2.1 Sparse Autoencoder Architectures

Sparse Autoencoders project neural activations into an overcomplete sparse basis via a sparsity-inducing objective. The standard formulation uses an L1 penalty on latent activations [Makhzani & Frey, 2014]. Recent architectures have introduced alternative sparsity mechanisms: JumpReLU employs a gating mechanism that improves sparsity control [Rajamanoharan et al., 2024]; TopK enforces hard sparsity via top-k selection [Gao et al., 2024]; BatchTopK extends this to batch-level selection; and Matryoshka SAEs use nested dictionaries for hierarchical feature learning [Bussmann et al., 2025]. Large-scale pretrained SAE suites such as GemmaScope [Lieberum et al., 2024] have made cross-architecture comparison feasible.

### 2.2 Feature Absorption

Chanin et al. [2024] formally defined feature absorption as the suppression of child features by parent features under co-occurrence. Their detection protocol requires known parent-child hierarchies (e.g., first-letter spelling tasks), limiting generalization. Absorption is connected to the broader phenomenon of superposition [Elhage et al., 2022], where neural networks represent more features than neurons by using overlapping directions. Hierarchical SAEs (HSAE) [Chen et al., 2025] propose architectural mitigation via explicit hierarchy, but require retraining. Existing approaches to reduce absorption all require SAE retraining---unlike our DFDA method, which operates at inference time.

### 2.3 SAE Evaluation Benchmarks

SAEBench [Dunefsky et al., 2024] provides standardized evaluation metrics including reconstruction quality, sparsity, and feature interpretability. Sparse probing [Marks et al., 2024] measures whether SAE features can predict ground-truth concepts. Feature attribution and steering [Rimsky et al., 2024] assess causal manipulability. While SAEBench evaluates reconstruction and sparsity, it does not include absorption-specific metrics, motivating our CAAB protocol. To our knowledge, no prior work systematically compares absorption across architectures or measures its causal downstream impact.

### 2.4 Positioning

Our work fills the gap between architecture-specific absorption anecdotes and unified quantitative understanding. Unlike Chanin et al. [2024], who focus on detection within a single architecture with predefined hierarchies, we benchmark across architectures and measure causal downstream impact---while also introducing the first unsupervised detection and SAE-retraining-free mitigation methods.

---
