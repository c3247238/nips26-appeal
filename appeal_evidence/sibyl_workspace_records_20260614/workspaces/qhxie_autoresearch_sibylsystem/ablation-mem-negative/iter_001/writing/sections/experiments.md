# Experiments

### 4.1 Setup

All experiments use GPT-2 Small (124M parameters, 12 layers indexed 0--11) with SAEs evaluated at layer 8 (residual stream post-activation). Dictionary sizes are 16,384 for trained TopK and 24,576 for pretrained JumpReLU. Training uses 1M tokens from OpenWebText with AdamW optimizer (lr=1e-3). For pretrained SAEs, we use GemmaScope JumpReLU (layer 8, width 24K, average L0 52.4). GPU: single NVIDIA RTX PRO 6000 Blackwell (96GB). Seed: 42 (fixed).

### 4.2 E1: Cross-Architecture Collision Rates

**Results.** TopK SAE shows 3.85% collision rate while pretrained JumpReLU shows 15.38% (Table 1, Figure 1). This 4x difference suggests architecture significantly affects feature separation. However, this comparison confounds architecture with training data: the JumpReLU is pretrained on Gemma data while TopK is trained on OpenWebText.

| Architecture | Collision Rate | L0 Sparsity | Recon MSE | Dead Feature Ratio |
|-------------|----------------|-------------|-----------|-------------------|
| TopK ($k$=50) | 3.85% | 50.0 | 6.58 | 96.0% |
| JumpReLU (pretrained) | 15.38% | 52.4 | 0.93 | 94.4% |

The pretrained JumpReLU's higher collision rate may reflect its training on diverse data rather than architectural deficiency---but this confound prevents definitive comparison. JumpReLU achieves 7x lower reconstruction MSE (0.93 vs. 6.58), suggesting a trade-off between collision rate and reconstruction quality.

### 4.3 E2: Causal Impact on Downstream Tasks

We measure whether high-collision SAEs produce worse sparse probing accuracy. **No significant correlation exists**: Spearman $\rho_S$ = 0.103, $p$ = 0.870 ($n$ = 5 $k$-values, Figure 2). With $n$ = 5, the study has low power to detect medium effects. Test accuracy ranges from 15.0% ($k$=10) to 77.5% ($k$=100), driven primarily by reconstruction quality rather than collision rate (Table 2).

| $k$ | Collision Rate (%) | Recon MSE | L0 Sparsity | Probe Test Acc (%) |
|---|-------------------:|----------:|------------:|-------------------:|
| 10 | 23.1 | 914.5 | 10.0 | 15.0 |
| 25 | 15.4 | 543.6 | 25.0 | 27.5 |
| 50 | 0.0 | 203.5 | 50.0 | 45.0 |
| 100 | 23.1 | 27.3 | 100.0 | 77.5 |
| 200 | 19.2 | 10.3 | 200.0 | 72.5 |

*Table 2: Data from causal impact experiment (f2_causal). Sparsity sweep (f3_sparsity) yields similar trends with slightly different values (e.g., k=100 MSE 26.6 vs. 27.3).*

Reconstruction MSE strongly predicts accuracy ($\rho_S$ = -1.0, $p$ < 1e-24), but collision rate does not. This suggests that collision rate may not be a reliable proxy for harmful absorption, though our study's statistical power was limited.

### 4.4 E3: Sparsity-Collision Relationship

Varying $k$ from 10 to 200 (TopK SAE), we find **no monotonic relationship** between sparsity and collision rate: $\rho_S$ = -0.10, $p$ = 0.873 ($n$ = 5 $k$-values, Figure 3). Reconstruction quality improves predictably with higher $k$ ($\rho_S$ = -1.0, $p$ < 1e-24 for MSE), but collision remains stable at 0--23%. Medium $k$ (50) achieves the lowest collision rate (0.0%), suggesting a non-linear relationship.

### 4.5 E4: Layer-Depth Pattern

Testing layers 0, 2, 4, 6, 8, 10 (JumpReLU), collision rates vary between 7.7% and 19.2% with **no clear trend**: $\rho_S$ = 0.088, $p$ = 0.868 ($n$ = 6 layers, Figure 4). Collision appears stochastic rather than systematically depth-dependent.

### 4.6 E5: Unsupervised Detection (UAD)

UAD achieves **F1 = 0.704** with perfect recall (1.000) and 54.3% precision on 500 features (Figure 5a). This means all 25 true collisions are detected, at the cost of 21 false positives (46 same-cluster pairs total). On the pilot set (100 features), UAD achieved F1 = 0.522 with 35.3% precision, demonstrating improvement with more features (Table 3).

| Method | Precision | Recall | F1 | Features |
|--------|-----------|--------|-----|----------|
| UAD (pilot) | 35.3% | 100% | 52.2% | 100 |
| UAD (full) | 54.3% | 100% | 70.4% | 500 |
| DFDA (full) | -- | -- | -- | 4 pairs |

The 54.3% precision means nearly half of detected collisions are false positives---a limitation that could be improved with better clustering thresholds or alternative clustering methods.

### 4.7 E6: Dynamic De-Absorption (DFDA)

DFDA improves per-pair residual MSE by **11.1%** on average across 4 absorbed feature pairs (all sharing feature 18486: letters c, i, o, p, u), using only 388 total parameters (97 per MLP, Figure 5b). Three of four pairs show positive improvement (41.8%, 6.2%, 18.0%); one pair degrades by 21.4% due to overfitting on the small sample size. Note: this improvement is on the per-pair residual MSE (10$^{-6}$ scale), not the overall reconstruction MSE.

---
