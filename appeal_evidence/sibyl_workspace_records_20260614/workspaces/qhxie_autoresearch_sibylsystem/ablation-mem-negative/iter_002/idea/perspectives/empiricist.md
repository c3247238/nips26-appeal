# Empiricist Perspective (Iteration 2)

## Data-Driven Concerns
Iteration 1's key finding (collision rates differ 4x by architecture) was based on:
- JumpReLU: pretrained on Gemma data, d_SAE=24K, dead features=94.4%
- TopK: trained on OpenWebText, d_SAE=16K, dead features=96.0%

Both had >94% dead features! The comparison was between two broken dictionaries.

## Required Experiments
1. Train TopK with AuxK (target: dead features <5%)
2. Re-measure collision rates with living dictionary
3. Compare against pretrained JumpReLU (also has 94% dead, but different cause)
4. If both still differ significantly, the architecture effect is real; if they converge, dead features explain the difference

## Statistical Power
With n=3 k-values and n=6 layers, power remains low. But if the effect is large (convergence from 15.4%→~5%), even n=3 can detect it.
