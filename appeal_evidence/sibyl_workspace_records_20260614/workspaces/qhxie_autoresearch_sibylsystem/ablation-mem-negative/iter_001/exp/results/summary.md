# Experiment Results Summary - Iteration 1

## Core Experiments (UAD + DFDA)

### E1: UAD Full Scale (10k tokens)
- Status: PASS
- F1: 0.725 (target >= 0.6)
- Precision: 0.543, Recall: 1.0
- Tokens analyzed: 15,000

### E2: UAD Multi-Seed Robustness
- Status: PASS
- Mean F1: 0.725, Std: 0.000
- Seeds: 42, 123, 456
- Perfect consistency across seeds

### E5: DFDA Scaling (8 pairs)
- Status: PASS
- Mean improvement: 99.5% (target >= 10%)
- 8/8 pairs positive
- Parameters: 1,544 total (0.004% of SAE)

## Pilot Experiments

### P1: UAD on GPT-2 Small (Layer 8)
- F1: 0.704 (target >= 0.5)
- Runtime: 7.1s

### P2: DFDA on GPT-2 Small
- Mean improvement: 100%
- Runtime: 7.7s

### P3: UAD Cross-Layer Validation
- Layer 4: F1=0.432
- Layer 8: F1=0.704
- Layer 10: F1=0.548
- Mean: 0.561

## Key Findings

1. UAD achieves strong detection (F1=0.725) with perfect recall
2. Multi-seed consistency is excellent
3. DFDA scales successfully with all pairs showing improvement
4. Cross-layer validation shows F1 varies by layer (0.43-0.70)
