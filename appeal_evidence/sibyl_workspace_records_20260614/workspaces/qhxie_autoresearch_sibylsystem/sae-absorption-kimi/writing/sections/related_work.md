## 2. Related Work

**Feature Absorption.** Chanin et al. [2] first systematically identified absorption, showing that sparsity optimization actively encourages parent features to be subsumed by children. They proposed a detection metric based on logistic regression probes and validated it across hundreds of LLM SAEs. However, their evaluation focused on first-letter spelling tasks, leaving open whether findings generalize to semantic features.

**Mitigation Architectures.** Matryoshka SAE [3] learns features at multiple scales via nested dictionaries, reducing absorption from 0.49 to 0.05. However, inner levels act as narrow SAEs, exacerbating feature hedging [8]. OrtSAE [4] enforces chunk-wise decoder orthogonality, claiming 65% absorption reduction with 4-11% compute overhead. GBA [9] provides theoretical feature recovery guarantees under specific generative model assumptions.

**Evaluation Benchmarks.** SAEBench [10] standardizes absorption evaluation across 8 metrics, but its absorption test is computationally expensive (~26 min per SAE). CE-Bench [11] offers a deterministic LLM-free alternative using contrastive story pairs.

**Gap.** No prior work controls for L0 when comparing architectures. The implicit assumption is that architecture effects are independent of sparsity—a claim we test directly.
