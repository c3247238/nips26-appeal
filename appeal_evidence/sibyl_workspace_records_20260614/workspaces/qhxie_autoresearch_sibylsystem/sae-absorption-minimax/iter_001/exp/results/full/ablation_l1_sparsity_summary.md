# Ablation: L1 Sparsity Sweep

## Configuration
- Model: GPT-2 Small, Layer 8
- d_sae: 24576
- Training tokens: 500,000 per SAE (pilot budget)
- Epochs: 2
- Batch size: 64x128

## Results

| L1 | L0 (norm) | Recon MSE | Absorption | Gini Abs | Chanin Abs | UAS | Dead% | Pareto |
|----|-----------|-----------|-----------|----------|------------|-----|-------|--------|
| 1e-05 | 0.5015 | 1388.17 | 0.6861 | 0.3962 | 0.9760 | 5.15 | 0.0% | Pareto |
| 5e-05 | 0.5011 | 1339.96 | 0.6919 | 0.4116 | 0.9722 | 5.27 | 0.0% | Pareto |
| 1e-04 | 0.5005 | 1552.76 | 0.6898 | 0.4062 | 0.9734 | 5.22 | 0.0% | Dominated |
| 5e-04 | 0.4980 | 1427.93 | 0.6899 | 0.4029 | 0.9769 | 5.30 | 0.0% | Dominated |
| 1e-03 | 0.4980 | 1463.73 | 0.6887 | 0.4041 | 0.9733 | 5.20 | 0.0% | Dominated |

**Reference**: Pre-trained vanilla SAE (L1~8e-5): absorption=0.015, MSE=1.48, L0=0.003

## Key Findings

1. **L1-Absorption ceiling**: All L1 values produce nearly identical absorption (~0.69).
   Absorption is NOT avoided at any L1 level in short training (500K tokens, 2 epochs).
2. **L1-Sparsity saturation**: All L1 values produce L0~0.50 (50% of features active).
   No meaningful sparsity differentiation in this range.
3. **Reconstruction quality**: Very high MSE (~1340-1550) vs pre-trained SAE (1.48).
   SAEs are severely undertrained at 500K tokens.
4. **Pareto frontier**: L1=1e-05 and L1=5e-05 are Pareto-optimal.
   L1=5e-05 has best reconstruction (MSE=1340).
5. **UAS**: ~5.2 across all L1 values, consistent but higher than pre-trained SAE (0.74).

## Interpretation

The ceiling effect (all absorption ~0.69) is due to:
- Short training (500K tokens, 2 epochs) means SAEs have not converged
- High L0 (~50%) indicates the SAE is not learning sparse representations
- Chanin absorption ~97% means nearly every feature fires at nearly every position

The L1 coefficient in range [1e-5, 1e-3] has **minimal effect** on absorption during early training.
With longer training (3M tokens, 3 epochs), the L1 coefficient should differentiate more strongly.

## Comparison with Full H2 Reference

| SAE Type | Absorption | MSE | L0 (norm) |
|----------|-----------|-----|-----------|
| Pre-trained (L1~8e-5) | 0.015 | 1.48 | 0.003 |
| This ablation (L1=5e-5, 500K) | 0.692 | 1340 | 0.501 |

The pre-trained SAE (trained on 3M tokens) has dramatically better reconstruction and lower absorption,
confirming that short training produces degenerate SAEs.

## Pilot Assessment

- **Trained**: 5/5 SAEs successfully
- **Pareto-optimal**: 2/5 (L1=1e-05, L1=5e-05)
- **Key limitation**: Short training produces ceiling effects
- **Recommendation**: Run full experiment with 3M tokens for meaningful differentiation
- **Conclusion**: Absorption appears unavoidable at any L1 during short training; full training may reveal sweet spot
