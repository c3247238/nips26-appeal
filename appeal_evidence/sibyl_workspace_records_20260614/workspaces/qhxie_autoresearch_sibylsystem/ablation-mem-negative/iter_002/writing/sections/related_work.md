# 2. Related Work

## 2.1 Sparse Autoencoder Architectures

Sparse Autoencoders project neural activations into an overcomplete sparse basis via a sparsity-inducing objective. The standard formulation uses an L1 penalty on latent activations [Makhzani \& Frey, 2014]. Recent architectures have introduced alternative sparsity mechanisms: JumpReLU employs a gating mechanism that improves sparsity control [Rajamanoharan et al., 2024]; TopK enforces hard sparsity via top-k selection [Gao et al., 2024]; BatchTopK extends this to batch-level selection; and Matryoshka SAEs use nested dictionaries for hierarchical feature learning [Bussmann et al., 2025]. Large-scale pretrained SAE suites such as GemmaScope [Lieberum et al., 2024] have made cross-architecture comparison feasible.

## 2.2 Feature Absorption

Chanin et al. [2024] formally defined feature absorption as the suppression of child features by parent features under co-occurrence. Their detection protocol requires known parent-child hierarchies (e.g., first-letter spelling tasks), limiting generalization. Absorption is connected to the broader phenomenon of superposition [Elhage et al., 2022], where neural networks represent more features than neurons by using overlapping directions.

Hierarchical SAEs (HSAE) [Chen et al., 2025] propose architectural mitigation via explicit hierarchy, but require retraining. Matryoshka SAEs [Bussmann et al., 2025] reduce absorption rates from 0.49 to 0.05 through nested dictionary structure---a preventive rather than detective approach. To our knowledge, no prior work has systematically tested whether unsupervised detection of absorption is possible.

## 2.3 Unsupervised Detection Attempts

Several works have proposed using feature co-occurrence patterns to identify related or absorbed features without supervision. Co-occurrence clustering groups features that activate on similar inputs, with the intuition that absorbed features should cluster together [Chen et al., 2025]. However, this conflates two distinct phenomena: (1) features that co-occur because they are semantically related (e.g., "cat" and "dog"), and (2) features that suppress each other (absorption). Our work is the first to empirically demonstrate that this conflation leads to near-random detection performance.

## 2.4 Positioning

Our work fills the gap between supervised absorption detection (Chanin et al.) and preventive architecture design (Matryoshka SAEs). We ask: if prevention is not an option, can we at least detect absorption without labels? Our negative answer provides a theoretical foundation for why the field should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption at training time.
