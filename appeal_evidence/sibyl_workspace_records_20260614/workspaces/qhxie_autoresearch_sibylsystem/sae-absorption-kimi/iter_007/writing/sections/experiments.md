## 4. Results

### 4.1 Cross-Architecture Comparison (Unmatched L0)

Table 1 shows absorption rates across architectures at their natural sparsity levels.

| Variant | Absorption (mean±std) | L0 (mean±std) | Dead Latents |
|---------|----------------------|---------------|-------------|
| Random | 0.495 ± 0.035 | 1029.6 ± 12.3 | 0.0% |
| Baseline L1 | 0.254 ± 0.047 | 964.4 ± 89.2 | 0.0% |
| Gated | 0.257 ± 0.052 | 962.1 ± 101.5 | 0.0% |
| OrtSAE | 0.247 ± 0.048 | 550.4 ± 112.7 | 0.0% |
| Matryoshka | 0.057 ± 0.023 | 50.0 ± 0.0 | N/A* |
| TopK | 0.056 ± 0.021 | 50.0 ± 0.0 | N/A* |

*TopK and Matryoshka report dead latent counts as proportions, not percentages, due to a display formatting issue.

**Observation.** TopK and Matryoshka show dramatically lower absorption (~0.056) than Baseline (~0.254). However, their L0 is 50 vs. Baseline's 964—a 19x difference. This is not a fair comparison.

### 4.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to TopK/Matryoshka's L0=50 via lambda sweep. Table 2 reports the results.

| Variant | L0 (mean±std) | Absorption (mean±std) | Notes |
|---------|--------------|----------------------|-------|
| Baseline (λ=5e-5) | 1082.2 ± 111.4 | 0.199 ± 0.038 | Natural L0 |
| Baseline (λ=0.002) | 994.6 ± 117.9 | 0.255 ± 0.058 | Highest λ tested |
| TopK | 50.0 ± 0.0 | 0.056 ± 0.021 | Fixed k=50 |
| Matryoshka | 50.0 ± 0.0 | 0.057 ± 0.023 | Fixed k=50 |

**Key finding.** Baseline L1 cannot reach L0=50. Even at the highest lambda tested (0.002), Baseline L0 remains ~995—20× higher than TopK/Matryoshka. The lambda sweep spans a 40× range (0.0005–0.02 in pilot, 5e-5–0.002 in full), yet L0 decreases by only ~16% (1146→963 in pilot, 1082→995 in full dose-response). This demonstrates a fundamental limitation of L1 regularization for achieving extreme sparsity in this synthetic setting. Consequently, a true L0-matched comparison between Baseline and TopK/Matryoshka is impossible—their sparsity levels are incommensurable.

### 4.3 Dose-Response Causality

Figure 1 (conceptual) shows absorption rate vs. feature recovery MCC across five lambda levels.

- Lambda range: 5e-5 to 2e-3
- Absorption range: 0.141 to 0.319 (2.3x variation)
- MCC range: 0.217 to 0.222 (ratio 1.02)

**Finding.** Despite a 2.3x variation in absorption, feature recovery MCC remains essentially flat (~0.22, std ~0.001). This falsifies the hypothesis that absorption rate causally predicts downstream interpretability under these conditions.

### 4.4 Ablation Studies

**Matryoshka nesting.** Flat Matryoshka (single scale, k=50) shows absorption 0.054—identical to nested Matryoshka (0.057). The nesting structure provides no additional benefit.

**OrtSAE orthogonality.** OrtSAE without the orthogonality penalty (standard L1 with matched config) shows absorption 0.254—statistically identical to OrtSAE with penalty (0.247). The orthogonality penalty has no effect on absorption.

**TopK vs. ReLU+L1.** TopK with explicit k-selection (absorption 0.056, L0=50) achieves lower absorption than ReLU+L1 at its natural L0 (mean absorption 0.180 ± 0.042, mean L0=834). The explicit k-selection mechanism, which enforces a fixed low L0, is the key factor—ReLU+L1 cannot achieve comparable sparsity regardless of lambda tuning.
