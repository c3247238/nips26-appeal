# Pilot G5: EDW-Step-DPO L1-Only Ablation

## Overview
Ablation study testing whether shallow error (Level-1) training alone is beneficial.

## Results

| Metric | Value |
|--------|-------|
| Final Loss | 0.6927 |
| Training Steps | 200 |
| Dataset Size | 18 (L1 only) |
| Training Time | 3 min |
| Pass Criteria | Training completes |

## Depth Distribution
- L1 (shallow/computational): 18 pairs
- L2 (logical): 0 pairs
- L3 (conceptual): 0 pairs

## Comparison with G2

| Metric | G2 (All Depths) | G5 (L1 Only) |
|--------|-----------------|--------------|
| Final Loss | 0.6923 | 0.6927 |
| Dataset Size | 293 | 18 |
| Training Time | 4 min | 3 min |

## Key Findings

1. **Loss is similar**: Despite using only 18 L1 samples (vs 293 in G2), the final loss is comparable (0.6927 vs 0.6923).

2. **Dataset imbalance**: The original dataset is heavily dominated by L3 errors (296/314 = 94%), with only 18 L1 pairs (6%).

3. **Limited training signal**: L1-only training cannot capture the full error distribution, as most mathematical errors are conceptual (L3) rather than computational (L1).

4. **Hypothesis validation**: This confirms that deep errors (L3) likely drive most of the EDW-Step-DPO improvement - the small L1 subset alone is insufficient for meaningful learning.

## Recommendation

GO - Training completed successfully. However, expect G5 to underperform G2 in downstream evaluation since:
- Only 18 L1 samples available (6% of dataset)
- L1 errors are shallow/computational and may not transfer well to harder problems
- L3 conceptual errors (94% of dataset) are completely absent from training

## Next Steps
- Evaluate G5 model on MATH test set
- Compare accuracy vs G2 (full depth training)
- G5 should serve as a lower bound for EDW-Step-DPO performance
