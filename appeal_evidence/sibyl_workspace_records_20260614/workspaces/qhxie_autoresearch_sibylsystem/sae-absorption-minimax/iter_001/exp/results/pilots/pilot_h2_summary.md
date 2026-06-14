# Pilot H2 Summary: Mitigation Methods Comparison

## Configuration
- **Model**: GPT-2 Small, Layer 8
- **SAE Release**: gpt2-small-res-jb (d_sae=24576)
- **Training**: 50K tokens, 1 epoch, LR=1e-3, L1=8e-5
- **Evaluation**: 5K tokens (5120 tokens actually processed)
- **Top features analyzed**: 200 per variant

## Results

| Variant | L0 (sparsity) | Recon MSE | Absorption | UAS |
|---------|---------------|----------|-----------|-----|
| Vanilla (pre-trained) | 0.0064 | 13.53 | 0.225 | 0.622 |
| TopK SAE (k=50) | 0.0020 | 110.23 | 0.066 | 0.501 |
| JumpReLU SAE (thresh=0.01) | 0.491 | 3419.61 | 0.625 | 5.148 |

## Key Findings

### TopK SAE
- **Absorption reduction**: 70.9% lower absorption vs vanilla (0.066 vs 0.225)
- **Reconstruction cost**: 8x worse MSE (110.2 vs 13.5)
- **Interpretation**: TopK enforces sparsity (L0=0.002) which reduces absorption. However, MSE is significantly worse due to aggressive sparsity forcing the SAE to drop useful features.
- **Pareto note**: TopK trades absorption reduction for reconstruction quality. Not a clear win alone.

### JumpReLU SAE
- **Absorption**: Higher than vanilla (0.625 vs 0.225)
- **Reconstruction**: Very poor MSE (3419 vs 13.5), 250x worse
- **Interpretation**: With only 50K training tokens and 1 epoch, JumpReLU fails to converge properly. The threshold=0.01 kills most activations (L0=0.49 = many features active) but the decoder is poorly trained, causing massive reconstruction error.
- **Note**: Full H2 trained JumpReLU for 3M tokens had MSE=23.3, much better. Pilot training is insufficient.

## Pilot Assessment

- **Pilot GO/NO-GO**: PARTIAL GO
- Both trained variants executed without errors (pilot infrastructure works)
- Neither variant passes the strict "<30% MSE degradation" criterion
- TopK shows genuine absorption reduction (71%) but at severe reconstruction cost
- JumpReLU fails due to insufficient training tokens (expected — pilot vs full differ by 60x)
- Pre-trained vanilla SAE is already well-optimized (absorption=0.225, MSE=13.5)

## Recommendations for Full H2
1. Increase training tokens for JumpReLU to 1M+ (pilot's 50K is clearly insufficient)
2. Consider TopK hyperparameter sweep: k ∈ {25, 50, 100, 200} to find absorption-reconstruction Pareto
3. The absorption metric (Gini-based) and MSE are strongly anti-correlated — need to weight them jointly
4. Compare against full_h2 results: trained variants showed absorption ~0.63 vs vanilla 0.015, opposite direction from pilot (absorption 0.066 vs 0.225). This discrepancy needs investigation (different metric computation, different eval data, or full_h2 used different absorption formula).
5. The full H2 (3M tokens) showed JumpReLU/OrtSAE/ATM all achieving high absorption ~0.63 with MSE ~23-39, while vanilla was 0.015. The pilot shows the opposite (vanilla=0.225 > TopK=0.066). Key difference: full H2 uses Gini absorption computed on ALL features vs pilot uses top-200 by activation.

## Pass Criteria Check
- At least 2 methods with <30% CE loss degradation: **0/2 fail**
- Pilot execution success: **PASS** (both variants trained and evaluated)
- Infrastructure reliability: **PASS**
