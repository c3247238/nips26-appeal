# Phase 1 EDA/D-EDA Full Validation Results

**Mode:** FULL
**Timestamp:** 2026-04-11T20:47:27.388920
**Bootstrap:** 10,000 resamples (95% CI)

## Summary Table

| Config | AUROC_EDA | 95% CI | AUROC_DEDA | Prec@50_EDA | Prec@50_DEDA | Pass | SAEBench_r |
|--------|-----------|--------|------------|-------------|--------------|------|------------|
| L5-16k | 0.6982 | [0.637,0.779] | 0.6020 | 0.0038 | 0.0030 | PASS | 0.9999 |
| L5-65k | 0.6174 | [0.532,0.725] | 0.5338 | 0.0008 | 0.0006 | FAIL | 0.9999 |
| L12-16k | 0.7765 | [0.700,0.863] | 0.5790 | 0.0035 | 0.0015 | PASS | 0.9998 |
| L12-65k | 0.4683 | [0.315,0.620] | 0.4994 | 0.0003 | 0.0003 | FAIL | 0.9999 |
| L19-16k | 0.4579 | [0.317,0.590] | 0.5892 | 0.0013 | 0.0020 | FAIL | 0.9997 |
| L19-65k | 0.5623 | [0.438,0.683] | 0.4707 | 0.0004 | 0.0003 | FAIL | 0.9998 |

## Results

- **Passed (AUROC >= 0.65):** 2/6
- **D-EDA improvement >= 10pp (Layer 12):** 0/2
- **GO/NO-GO:** NO_GO

## Notes

- 10,000 bootstrap resamples (FULL mode)
- Neuronpedia proxy labels (Gemma 2 2B gated; Chanin et al. exact labels require model access)
- EDA = 1 - encoder_decoder_cosine_sim, cross-validated vs SAEBench (r > 0.999)
- Layers 5, 12, 19 (SAEBench available); methodology specified 6, 12, 20
