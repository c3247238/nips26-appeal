# Figure Generation Summary

All figures generated for the paper:
**Feature Absorption as Optimal Compression: Evidence that SAEs Correctly Handle Hierarchical Features**

## Figure List

| Figure | Description | Key Finding |
|--------|-------------|-------------|
| Fig 1 | Absorption rate distribution across layers (L0, L4, L8, L10) | Mean 2.1-3.9%, max 24.2% (Feature U at L8) |
| Fig 2 | Steering success vs absorption rate (L4, L8) | No significant correlation (H1 falsified) |
| Fig 3 | Precision-recall decomposition (k=5, 20) | Precision invariant, recall varies (H5 supported) |
| Fig 4 | EC50 vs absorption rate | No significant correlation (H4 falsified) |
| Fig 5 | Inhibition graph precision@k | 0/520 correct, precision@20 = 0.0 (H6 falsified) |
| Fig 6 | Cross-layer degradation coefficient comparison | Opposite signs, CV > 1.0 (H3 falsified) |
| Fig 7 | H9: Co-occurrence (p_11) vs absorption | Tautological by construction (r = -1.0) |
| Fig 8 | H10: Trained vs Random SAE absorption | Random > Trained (0.278 vs 0.034), methodological finding |
| Fig 9 | Summary table of all hypotheses | 1 supported (H5), 5 falsified, 1 invalid, 1 methodological |
| Fig 10 | Rate-distortion schematic | Conceptual framework for absorption as optimal compression |
| Fig 11 | Dose-response curves by absorption level | No systematic difference across absorption levels |

## Statistical Summary

### Multiple Comparison Correction (12 tests)
- Bonferroni alpha = 0.0042
- **0 tests survive Bonferroni correction**
- **0 tests survive BH-FDR correction**

### Key Results
- H1 (Raw Steering, L8): r = -0.301, p = 0.136
- H1b (Delta-Corrected, L8): r = -0.431, p = 0.028 (uncorrected)
- H2 (Probing, L8): r = -0.107, p = 0.604
- H4 (EC50, L8): r = 0.180, p = 0.380
- H5 (Precision-Recall): Precision = 1.0 at k>=5 universally; recall varies 0.05-1.0
- H6 (Inhibition Graph): Precision@20 = 0.0, Fisher p = 1.000
- H9 (Co-occurrence): r = -1.0 (tautological)
- H10 (Random SAE): Random mean = 0.278, Trained mean = 0.034, paired t p = 4.55e-07

## Paper Framing

The paper's central contribution is **honest null-result reporting** with rigorous controls:
1. **H1-H4**: Absorption does not significantly degrade downstream tasks in GPT-2 Small SAEs
2. **H5**: The one robust finding - absorption affects recall (coverage) but not precision (selectivity)
3. **H6**: A falsified mechanistic hypothesis (decoder correlations do not predict absorption)
4. **H10**: A methodological finding (Chanin metric is not specific to learned structure)
5. **Rate-distortion framing**: Provides theoretical grounding for why absorption may be optimal
