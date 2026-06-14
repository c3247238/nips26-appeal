# 2. Related Work

## 2.1 Sparse Autoencoders for Interpretability

Sparse Autoencoders (SAEs) trained on language model residual streams decompose polysemantic activations into sparse, interpretable features (Cunningham et al., 2023; Bricken et al., 2023). The SAE training objective includes an L1 sparsity penalty that encourages most features to be inactive at any given time, leading to human-interpretable feature directions.

SAELens (Bloom et al., 2024) provides the standard library for SAE training and analysis. Pre-trained SAE releases on GPT-2 Small (`gpt2-small-res-jb`) and Gemma models enable reproducible research. Multiple SAE architectures exist: standard ReLU, TopK (Gao et al., 2024), JumpReLU (Rajamanoharan et al., 2024), and orthogonality-penalized variants (OrtSAE, Korznikov et al., 2025).

## 2.2 Feature Absorption

### The Phenomenon

Feature absorption occurs when child features in a semantic hierarchy subsume parent features. For example, if "cat" and "dog" features are both active, a "mammal" feature might absorb into both, causing standalone "mammal" activation to be rare. Chanin et al. (2024) first systematically studied this phenomenon.

### Detection Protocol

The Chanin protocol detects absorption using first-letter classification (tokens split into A-M vs N-Z). A feature is considered absorbed if:
- Feature-split latents fail classification (threshold $\tau_{\text{fs}} = 0.03$)
- A different latent with cosine similarity $\geq \tau_{\text{ps}} = 0.025$ explains $\geq \tau_{\text{pa}} = 0.4$ of probe projection

### Mitigation Approaches

Multiple architectures attempt to reduce absorption:
- **OrtSAE** (Korznikov et al., 2025): Orthogonality penalty reduces absorption by 65%
- **ATM** (Li & Ren, 2025): Temporal importance tracking reduces absorption by 40%
- **HSAE** (Luo et al., 2026): Hierarchical structure learning reduces absorption by design

## 2.3 Feature Sensitivity

Tian et al. (2025) introduced feature sensitivity as an evaluation dimension orthogonal to absorption. Sensitivity measures how reliably a feature activates across semantically equivalent inputs. Key findings:
- Many interpretable features have poor sensitivity (fail on paraphrases)
- Sensitivity declines with SAE width
- Human evaluation confirms generated texts genuinely resemble original activating examples

The Tian protocol measures sensitivity using paraphrase AUC: generate paraphrases of sentences where a feature activates strongly, then measure activation reliability across paraphrase pairs.

## 2.4 The Sanity Check Crisis

Korznikov et al. (2026) revealed a fundamental challenge: random baselines match trained SAEs on standard interpretability benchmarks:
- Interpretability scores: 0.87 vs 0.90 (random vs SAE)
- Sparse probing: 0.69 vs 0.72
- Causal editing: 0.73 vs 0.72

In synthetic experiments, SAEs recover only 9% of ground-truth features despite 71% explained variance. This suggests current SAEs may not reliably decompose model internals. Our work investigates whether absorption and sensitivity failures explain the Sanity Check finding.

## 2.5 Steering Interventions

Steering interventions add $\beta \times W_{\text{dec}}[f]$ to the residual stream to modify model behavior (Subramani et al., 2022; Zou et al., 2023). Steering effectiveness is measured by max absolute logit difference at the last position.

Prior work by Subramani et al. (2022) showed steering is effective for behavioral modification. Our work extends this by investigating whether absorption level affects steering reliability—a question with conflicting prior evidence.

## 2.6 Relationship to Our Work

Prior work has studied absorption and sensitivity in isolation. We provide the first joint analysis of these failure modes, examining their correlation and combined effect on steering effectiveness.

| Paper | Focus | Contribution | Gap |
|-------|-------|-------------|-----|
| Chanin et al. (2024) | Absorption | Detection protocol | Not connected to sensitivity |
| Tian et al. (2025) | Sensitivity | Measurement protocol | Not connected to absorption |
| Korznikov et al. (2026) | Sanity Check | Benchmark findings | No mechanistic explanation |
| This work | Joint analysis | First correlation study | Connect failure modes |

<!-- FIGURES
- None
-->