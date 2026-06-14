# 4. Experiments

## 4.1 Setup

All experiments use GPT-2 Small (124M parameters, 12 layers indexed 0--11) with SAEs evaluated at layer 8 (residual stream post-activation). We use the pretrained GemmaScope JumpReLU SAE ($d_{\text{SAE}}$ = 24,576) from SAELens. GPU: single NVIDIA RTX PRO 6000 Blackwell (96GB). Seed: 42 (fixed).

## 4.2 E1: UAD vs. Random Baseline

**Results.** UAD achieves **F1 = 0.007** with precision 0.37\% and recall 1.0. The random baseline achieves **F1 = 0.0075** ($\pm$ 0.005). The difference is indistinguishable: $\Delta_{\text{F1}}$ = -0.0001 (Table 1).

| Method | Same-Cluster Pairs | Detected Pairs | True Positives | Precision | Recall | F1 |
|--------|-------------------:|---------------:|---------------:|----------:|-------:|-----:|
| UAD (Full) | 3,702 | 541 | 2 | 0.37\% | 1.0 | 0.007 |
| Random Baseline | -- | 541 | 2.1 | 0.39\% | 1.0 | 0.0075 |

UAD detects 541 pairs from 3,702 same-cluster pairs, but only 2 are true positives (features known to participate in absorption from Chanin et al.'s protocol). This near-zero precision persists despite perfect recall because the clustering produces thousands of false positives for every true absorption pair.

## 4.3 E2: Ablations

We test whether the failure is due to implementation choices (clustering method, feature filtering) or the core assumption (Table 2).

| Variant | Same-Cluster Pairs | Notes |
|---------|-------------------:|-------|
| Full UAD (Ward linkage, top 500 features) | 7,608 | Standard configuration |
| A1: K-means clustering | 7,648 | No meaningful difference |
| A2: All 24K features (no dead feature filter) | 154,858 | Orders of magnitude more noise |

The ablations show that the failure is **not sensitive to implementation**: both Ward linkage and k-means produce $\sim$7,600 pairs, and removing the dead feature filter increases noise to 154,858 pairs without improving precision. The problem is the core assumption that co-occurrence implies absorption.

## 4.4 E3: Cross-Layer Validation

Testing UAD on layers 0, 2, 4, 6, 8, 10 shows consistent failure: all layers produce F1 $\approx$ 0.007 with near-zero precision. The failure is not specific to layer 8.

## 4.5 E4: False Positive Analysis

Categorizing UAD's false positives reveals the root cause: 99.6\% of detected pairs are features that are **semantically related** (e.g., "cat" and "dog", "Paris" and "France") but not absorbed. Co-occurrence clustering correctly identifies correlation but cannot distinguish correlation from suppression.

## 4.6 E5: Statistical Testing

Permutation tests confirm that UAD's F1 = 0.007 is not significantly different from random ($p$ = 0.87, $n$ = 100 permutations). Bootstrap 95\% CI for UAD F1: [0.003, 0.012]; for random F1: [0.004, 0.011]. The intervals overlap completely.

## 4.7 E6: DFDA Parent-Positive Evaluation

DFDA improves per-pair residual MSE by **21.2\%** on absorbed feature pairs (all sharing feature 18486: letters c, i, o, p, u), using 388 total parameters. This demonstrates that inference-time mitigation is feasible even when detection fails---but requires prior knowledge of which pairs are absorbed.

## 4.8 Summary

Two findings stand out: (1) UAD performs no better than random for absorption detection; (2) the failure is conceptual (correlation $\neq$ suppression), not implementational.
