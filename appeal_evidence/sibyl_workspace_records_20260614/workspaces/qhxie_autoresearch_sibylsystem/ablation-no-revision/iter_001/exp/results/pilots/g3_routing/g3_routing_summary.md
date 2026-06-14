# G3 Routing Experiment Results

## Task: g3_routing
## Date: 2026-04-29

## H3 Evaluation

**Hypothesis**: Single-pass accuracy > 75% on low-Ea problems

**Status**: FAIL

**Key Metrics**:
- Optimal Ea threshold: 9.999996
- Low-Ea accuracy: 68.4% (below 75% threshold)
- High-Ea accuracy: 63.6%
- Delta: -4.8% (high-Ea actually performs better)

**Pass**: NO

## Key Findings

### Finding 1: Ea Threshold Does NOT Predict Single-Pass Accuracy
- Low-Ea problems (Ea <= 9.999996): 68.4% accuracy
- High-Ea problems (Ea > 9.999996): 63.6% accuracy
- **The consistency-derived Ea is NOT a good predictor of single-pass accuracy**

### Finding 2: Ea-Level Correlation ≠ Ea-Accuracy Correlation
- While Ea correlates with MATH level (Spearman = 0.578, H2 PASS)
- Ea does NOT predict single-pass accuracy (H3 FAIL)
- This suggests Ea measures problem difficulty, not solveability

### Finding 3: Overall Single-Pass Performance
- 30 problems evaluated
- Overall single-pass accuracy: 66.7%
- Matches g1_saturation_v2 baseline (66.7% at k=1)

## Accuracy by MATH Level

| Level | Count | Mean Ea | Accuracy |
|-------|-------|---------|----------|
| 1     | 2     | 9.465   | 50.0%    |
| 2     | 4     | 9.599   | 75.0%    |
| 3     | 7     | 9.542   | 71.4%    |
| 4     | 8     | 9.666   | 62.5%    |
| 5     | 9     | 9.941   | 66.7%    |

## Routing Analysis

**Hypothesis H3 falsified**: Ea threshold cannot reliably identify problems solvable with >75% single-pass accuracy.

**Implication**: Consistency-based Ea estimation is insufficient for routing decisions. The model may "know" the answer regardless of consistency, or consistency measures something other than solveability.

## Recommendations

1. **NO_GO for routing based on consistency-Ea**
2. Alternative Ea estimation needed (e.g., per-layer attention patterns, token uncertainty)
3. Consider different routing signal: model confidence, not consistency

## Pilot Summary Integration

This completes the Round 4 pilot experiments:

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| G1 (H1) | PASS | Arrhenius kinetics confirmed (R2=0.936) |
| G2 (H2) | PASS | Ea correlates with difficulty (rho=0.578) |
| G3 (H3) | FAIL | Ea does NOT predict single-pass accuracy |
