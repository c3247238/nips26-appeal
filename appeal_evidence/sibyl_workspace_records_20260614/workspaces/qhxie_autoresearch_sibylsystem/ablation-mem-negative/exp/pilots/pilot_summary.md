# Pilot Experiment Summary

## Overall Recommendation: GO

**Selected candidate**: cand_a (Unsupervised Feature Absorption Detection + Dynamic Feature De-Absorption)

**Confidence**: 0.78

---

## Pilot Results

### P1: UAD on GPT-2 Small (Layer 8)
- **Status**: PASS
- F1: 0.704 (target >= 0.5)
- Precision: 0.543, Recall: 1.0
- TP: 19/35 same-cluster pairs are true collisions
- Runtime: 7.1s

### P2: DFDA on GPT-2 Small
- **Status**: PASS
- Mean improvement: 100.0% (target > 5%)
- Positive pairs: 2/2
- Total params: 386 (0.001% of SAE)
- Runtime: 7.7s
- **Caveat**: 100% improvement is suspiciously high due to near-zero parent values in child-dominant positions

### P3: UAD Cross-Layer Validation (Layers 4, 8, 10)
- **Status**: PARTIAL_PASS
- Layer 4: F1=0.432, Layer 8: F1=0.704, Layer 10: F1=0.548
- Mean F1: 0.561, Std: 0.111
- Runtime: 17.5s
- **Caveat**: Layer 4 slightly below 0.45 threshold. Cross-model validation blocked.

---

## Risks

1. **Cross-model validation blocked**: Gemma-2B gated on HuggingFace (no HF token). Pythia-2.8B SAE not available.
2. **DFDA metric artifact**: 100% improvement reflects near-zero parent values, not true recovery.
3. **Limited pair coverage**: Only 2/4 selected pairs had sufficient training data.

---

## Recommendations

1. Proceed with cand_a as primary contribution
2. Use GPT-2 Small multi-layer + multi-seed as validation proxy
3. Add reconstruction MSE check to DFDA evaluation
4. Consider training custom SAE on GPT-2 Medium for true cross-scale validation
5. Implement proper Chanin protocol for true absorption ground truth in full experiments
