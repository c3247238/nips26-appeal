# Summary Table: All Hypotheses and Results

| Hypothesis | Statement | Layer 4 Result | Layer 8 Result | Overall |
|------------|-----------|----------------|----------------|---------|
| H1 | Higher absorption -> lower raw steering | r=0.008, p=0.970 | r=-0.301, p=0.136 | FALSIFIED |
| H1b | Higher absorption -> lower delta-corrected steering | r=0.245, p=0.227 | r=-0.431, p=0.028 *** | SUPPORTED (L8) |
| H2 | Higher absorption -> lower probing F1 | r=-0.003, p=0.987 | r=-0.107, p=0.604 | FALSIFIED |
| H3 | Degradation coefficient consistent across layers | CV=1.53 (H1) | CV=1.32 (H2) | FALSIFIED |
| H4 | Absorption affects efficiency (higher EC50) | t=-1.23, p=0.23 | t=0.79, p=0.43 | NOT SUPPORTED |
| H5 | Absorption affects recall, not precision | Precision std < Recall std | Precision std < Recall std | SUPPORTED |

## Key Findings

1. **H1 (Raw Steering):** No significant correlation between absorption and raw steering effectiveness.
2. **H1b (Delta-Corrected):** Significant negative correlation at Layer 8 (r=-0.431, p=0.028) after random baseline correction.
3. **H2 (Probing):** No significant correlation between absorption and sparse probing F1.
4. **H3 (Consistency):** Degradation coefficients have opposite signs across layers (CV > 1.0).
5. **H4 (EC50):** No significant difference in EC50 between high and low absorption features.
6. **H5 (Precision-Recall):** Precision is invariant across features; recall varies and drives F1 variance.
