# Pilot Summary - Iteration 8

## Tasks Completed

### 1. pilot_classify_features (H4+H5 Pilot)
- **Status**: COMPLETED
- **Features analyzed**: 43
- **Quadrant counts**:
  - Q1 (high absorption, low sensitivity): 15 features
  - Q2 (high absorption, high sensitivity): 0 features
  - Q3 (low absorption, low sensitivity): 28 features
  - Q4 (low absorption, high sensitivity): 0 features

### 2. pilot_decoder_norms (H6 Pilot)
- **Status**: COMPLETED
- **L2 norm ratio**: 1.0 (high-abs / low-abs)
- **Pass criteria**: ratio > 1.1

### 3. replicate_coherence_protective (H1-R Backup)
- **Status**: COMPLETED
- **Layers tested**: 4, 8, 10
- **Mean Spearman r**: 0.356

## Hypothesis Results

| Hypothesis | Result | Details |
|------------|--------|---------|
| H4 (Q1 steering near random) | NOT TESTED | Q4 empty, cannot compare best-case vs worst-case |
| H5 (independence, r < 0.3) | **FALSIFIED** | r = 0.59, p = 3.15e-05 |
| H6 (norm ratio > 1.1) | **FALSIFIED** | ratio = 1.0 |
| H1-R (protective, r < -0.5) | **FALSIFIED** | r = +0.356, not negative |

## Key Findings

1. **H5 FALSIFIED**: Absorption and sensitivity are POSITIVELY correlated (r=0.59), not independent. Features that are absorbed tend to also have low sensitivity.

2. **H6 FALSIFIED**: High-absorption and low-absorption features have identical decoder L2 norms (ratio=1.0). The saturation hypothesis is not supported.

3. **H1-R FALSIFIED**: The protective effect was NOT replicated. Coherence and absorption show POSITIVE correlation (r=0.36), not negative (r=-0.786 as in earlier pilot).

4. **Q4 Empty**: No features fell into the "low absorption + high sensitivity" quadrant, which was predicted to be the "best-case" scenario.

## Recommendations

1. The sensitivity-absorption "compound failure" hypothesis is WEAKENED by these results
2. The positive correlation between absorption and sensitivity suggests they may share a common cause
3. The H1-R finding from earlier pilot (r=-0.786) was NOT replicated - this needs investigation
4. Consider revising the theoretical framework before proceeding to full experiments

## Next Steps

Given that all pilot hypotheses are falsified:
- **GO/NO-GO**: NO-GO for full experiments
- **Reason**: Key assumptions (independence, protective effect, decoder norm differences) are not supported by pilot data
- **Recommendation**: Return to literature review and theoretical development
