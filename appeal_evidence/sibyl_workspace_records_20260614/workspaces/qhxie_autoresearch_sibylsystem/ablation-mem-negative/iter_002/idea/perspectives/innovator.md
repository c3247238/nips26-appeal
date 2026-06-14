# Innovator Perspective (Iteration 2)

## Core Insight
Iteration 1 failed because 89-99% dead features in trained SAEs invalidated all collision/absorption measurements. The solution is not to abandon trained SAEs but to adopt modern training techniques that eliminate dead features.

## Proposed Direction: CAAB-v2 with Living Dictionaries
Replace iteration 1's naive TopK training with AuxK-enhanced training + soft-capping, achieving <5% dead features. Then re-run the full CAAB benchmark with valid dictionaries.

## Novel Elements
1. Systematic dead-feature ablation: compare collision rates across dead-feature ratios (95% vs 5%)
2. Living-dictionary CAAB: first absorption benchmark where all architectures have healthy dictionaries
3. Quantify how dead features confound absorption measurements
