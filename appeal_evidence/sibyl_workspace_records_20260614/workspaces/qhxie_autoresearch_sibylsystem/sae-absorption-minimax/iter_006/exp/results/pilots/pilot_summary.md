# Iteration 10 Pilot Summary

## Overall Recommendation: REFINE

## Pilot Results

### 1. pilot_layer_uas_mapping: PASS
- **H-L1**: PASS (UAS std > 0.1 at layer 4)
- **H-L2**: PASS (layer 4 is best)
- Layer 4: UAS std = 0.4810 (BEST)
- Layer 6: UAS std = 0.4000
- Layer 8: UAS std = 0.0000 (SATURATED)
- Layer 10: UAS std = 0.4330

**Key Finding**: Absorption IS measurable at layer 4. Layer 8 is confirmed saturated. Use layer 4 for future experiments.

### 2. pilot_q4_activation_patching: PASS (for different reason than expected)
- Feature 10236: absorbed=True (baseline=1.00, patched=0.54, random=0.54)
- Feature 6768: absorbed=True (baseline=1.00, patched=0.64, random=0.64)

**Interpretation**: patched≈random means features ARE absorbed. However, the printed "100% recovery" was a bug - the correct interpretation is that zeroing these features drops performance to random baseline, confirming absorption.

**CRITICAL PARADOX**: Q4 features are absorbed but have HIGH sensitivity (0.91, 0.80). This CONTRADICTS Sensitivity Floor which predicts absorbed features should have LOW sensitivity.

### 3. pilot_sensitivity_steering_correlation: FAIL (protocol broken)
- H-SENS: FAIL (|r| > 0.4)
- All 7 features showed steering_effect = 0.0
- **Issue**: Steering protocol needs debugging - steering is not affecting predictions

### 4. pilot_decoder_overlap_alternative: FAIL
- H-DEC: FAIL (high-sens NOT lower than low-sens overlap)
- High-sens overlap: 0.4827
- Low-sens overlap: 0.4655
- **Opposite of prediction**: High-sensitivity features have HIGHER decoder overlap

## Critical Paradox

The Q4 features (10236, 6768) present a paradox:
1. Activation patching confirms they ARE absorbed (patched ≈ random)
2. Yet they have HIGH sensitivity scores (0.91 and 0.80)

This directly contradicts the Sensitivity Floor hypothesis which states:
- Absorbed features have low specificity (decoder overlap with neighbors)
- Low specificity → low sensitivity
- Therefore absorbed = low sensitivity

But Q4 features are absorbed AND high-sensitivity.

## Recommendations

1. **Use layer 4** for future Sensitivity Floor experiments (not layer 8)
2. **Debug steering protocol** before H-SENS can be evaluated
3. **Investigate Q4 mechanism**: Why are these features absorbed but sensitive?
4. **Consider revising Sensitivity Floor**: The absorbed→insensitive link may not be universal
