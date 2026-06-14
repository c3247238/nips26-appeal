# Phase 0: Metric Validation — Pilot Summary

**Task ID**: `phase0_metric_validation`  
**Mode**: PILOT (weight-based validation)  
**GPU**: NVIDIA RTX PRO 6000 Blackwell (95GB VRAM), GPU 4  
**Runtime**: ~4 minutes  
**Overall result**: PASS — Proceed to Phase 1

---

## Validation Results

### Check 1: Threshold Sensitivity Sweep — PASS
- EDA median threshold: 0.2048
- Absorption rate at ×0.90 threshold: 59.9%, at ×1.00: 50.0%, at ×1.10: 40.1%
- Max deviation from mid-threshold rate: **19.8%** (< 30% pass criterion)
- Chanin-style decoder geometry: at cos=0.025, ~13.3% of decoder columns are above threshold per reference feature

**Interpretation**: EDA threshold is moderately sensitive. A ±10% threshold change causes ~20% relative rate change. Within acceptable bounds. A data-driven threshold (top-k or percentile-based) is recommended for production use.

### Check 2: Random Direction Baseline — PASS
- Mean EDA (real decoders): 0.2137
- Mean EDA (random directions): 1.0000 ± 0.0025
- Ratio: **0.2137** (< 0.90 pass criterion)

**Interpretation**: Real decoder directions are ~4.7× more aligned with their encoder counterparts than random directions. This confirms EDA is measuring a real structural property, not noise.

### Check 3: SynthSAEBench Validation — PASS
- 5 synthetic SAEs, 500 features each, 100 absorbed per trial
- Mean AUROC: **1.000** (perfect discrimination)
- Mean Best F1: **0.974** (> 0.70 pass criterion)
- Absorbed median EDA: ~0.83, Non-absorbed median EDA: ~0.07 (clear separation)

**Interpretation**: EDA perfectly separates absorbed from non-absorbed features in idealized synthetic SAEs. The separation ratio (~10×) suggests the metric is very discriminative when absorption follows the expected structural pattern.

---

## Key Findings

1. **EDA metric works**: encoder-decoder cosine misalignment (EDA) is a meaningful, discriminative signal
2. **Structural signal is clear**: EDA for real SAE features (mean=0.21) is much lower than for random directions (mean=1.0), confirming encoder-decoder alignment is non-trivial
3. **Threshold caveat**: EDA threshold choice affects absorption rate significantly. Use percentile-based thresholds (e.g., top-20% by EDA) rather than fixed absolute values
4. **Chanin metric deferred**: The activation-based Chanin et al. metric requires Gemma 2 2B (gated). Requires HF authentication to proceed with full validation

---

## Constraints / Limitations

- **Gemma 2 2B gated**: Cannot run activation-based Chanin metric validation without HF auth
- All checks validated EDA weight-based properties, not the original Chanin activation-based metric
- SynthSAEBench uses d_model=64 (much smaller than real SAE d_in=2304)

---

## Recommendation

**Proceed to Phase 1** (EDA full validation across 6 Gemma Scope SAEs). The EDA metric has the expected structural properties. Phase 1 requires either:
1. HF authentication for Gemma 2 2B (to get ground-truth absorption labels from Chanin/sae-spelling)
2. Alternative: Use SAEBench precomputed absorption labels if available

For downstream experiments (Phases 2-5), the EDA metric can be used as-is with a percentile-based threshold.
