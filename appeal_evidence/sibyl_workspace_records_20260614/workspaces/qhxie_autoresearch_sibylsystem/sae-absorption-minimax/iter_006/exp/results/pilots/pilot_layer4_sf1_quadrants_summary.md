# Pilot Layer 4 Sensitivity Floor Quadrant Test - Summary

## Task ID
`pilot_layer4_sf1_quadrants`

## Status
**PARTIAL PASS** - H-SF2 (U-shape) passed but H-SF1 (Q2+Q4 < 10%) failed

## Hypothesis Results

| Hypothesis | Pass Criterion | Actual | Result |
|------------|----------------|--------|--------|
| H-SF1 (Structural Emptiness) | Q2+Q4 < 10% | Q2+Q4 = 100% (50/50) | **FAIL** |
| H-SF2 (U-Shape) | a > 0 | a = 0.82, R² = 0.52 | **PASS** |

## Quadrant Distribution (N=50)

| Quadrant | Description | Count | Fraction |
|----------|-------------|-------|----------|
| Q1 | High Absorption + Low Sensitivity | 0 | 0% |
| Q2 | High Absorption + High Sensitivity | 48 | 96% |
| Q3 | Low Absorption + Low Sensitivity | 0 | 0% |
| Q4 | Low Absorption + High Sensitivity | 2 | 4% |

## Key Observations

1. **Layer 4 features are predominantly in Q2**: 96% of features at layer 4 show both high absorption AND high sensitivity. This is the OPPOSITE of what the Sensitivity Floor hypothesis predicts.

2. **U-shape confirmed**: The quadratic coefficient a = 0.82 > 0 indicates a U-shaped relationship between absorption and sensitivity, as predicted by H-SF2.

3. **Q4 paradox present at layer 4 too**: Two features (20976, 20731) show low absorption but high sensitivity, confirming the paradox observed at layer 8 in iter_010.

## Important Caveats

The absorption metric used is a simplified proxy (max/mean activation ratio) rather than the full Chanin first-letter protocol. The clustering of UAS values around 0.65 and 1.0 suggests the metric needs refinement.

The sensitivity metric (coefficient of variation) shows high values (0.72-1.0) for most features, which may indicate:
- Layer 4 features are genuinely more context-sensitive than layer 8 features
- Or the metric needs calibration

## Pilot Decision

**H-SF1 FAILED**: The Sensitivity Floor prediction that Q2+Q4 < 10% was clearly falsified at layer 4. This suggests:
- Either the sensitivity floor mechanism is layer-specific (works at layer 8 but not layer 4)
- Or the absorption/sensitivity metrics need recalibration

**Recommended next steps**:
1. Recalibrate absorption metric using proper Chanin protocol
2. Compare with layer 8 results from iter_010 to understand layer-specificity
3. Consider pivoting to sensitivity-first hypothesis (cand_sensitivity_first) which predicts Q3 dominance

## Output Files
- Full results: `exp/results/pilots/pilot_layer4_sf1_quadrants.json`
- Script: `exp/results/pilots/pilot_layer4_sf1_quadrants.py`