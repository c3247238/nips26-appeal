# R4-C Llama Replication — Pilot Summary

**Task:** r4c_llama_replication  
**Mode:** PILOT  
**GO/NO-GO:** GO  
**Elapsed:** 16.9s  
**Timestamp:** 2026-04-13T13:08

## Summary

R4-C successfully computes weight-only EDA on 2 Llama-3.1-8B Scope SAE configurations (Layer 6 and Layer 12, 8x width = 32768 latents each). Llama-3.1-8B model activations are HF-gated, so absorption labels cannot be generated and AUROC cannot be computed directly. The task contributes EDA distribution statistics for a third model family.

## Results

### Llama Scope EDA Distributions

| Config | Layer | d_in | d_sae | EDA mean | EDA std | EDA p50 | EDA p90 |
|--------|-------|------|-------|----------|---------|---------|---------|
| Llama-L6-8x | 6 | 4096 | 32768 | 0.5182 | 0.1631 | 0.5721 | 0.6848 |
| Llama-L12-8x | 12 | 4096 | 32768 | 0.4909 | 0.1630 | 0.5316 | 0.6635 |

### Cross-Model EDA Comparison

| Model | Config | Layer | d_model | AUROC_EDA | Label type |
|-------|--------|-------|---------|-----------|------------|
| Gemma 2B | L5-16k | 5 | 2304 | 0.698 | proxy |
| Gemma 2B | L12-16k | 12 | 2304 | 0.776 | proxy |
| Gemma 2B | L12-65k | 12 | 2304 | 0.468 | proxy |
| GPT-2 Small | L6 | 6 | 768 | 0.650 | direct |
| GPT-2 Small | L10 | 10 | 768 | 0.336 | direct |
| Llama-3.1-8B | L6-8x | 6 | 4096 | N/A (model gated) | — |
| Llama-3.1-8B | L12-8x | 12 | 4096 | N/A (model gated) | — |

### Internal Discriminability

EDA scores show strong separation between top and bottom quartile features:

| Config | Top-Q mean | Bottom-Q mean | Cohen's d |
|--------|-----------|---------------|-----------|
| Llama-L6-8x | 0.683 | 0.277 | 6.001 |
| Llama-L12-8x | 0.659 | 0.254 | 5.094 |

EDA = 1 - DecCos identity validated (Pearson r = -1.000 for both configs).

### EDA Mean Across Model Families

- Gemma 2B (Gemma Scope): EDA mean ≈ 0.19 (strongly aligned encoder-decoder)
- Llama-3.1-8B (Llama Scope): EDA mean ≈ 0.50 (intermediate alignment)
- GPT-2 Small (gpt2-small-res-jb): EDA mean ≈ 0.62 (weaker alignment)

This gradient suggests SAE training regimes differ in how much they enforce encoder-decoder alignment.

## Key Findings

1. **EDA computable weight-only across all tested architectures** — no model activations needed.
2. **Llama Scope SAEs have intermediate EDA** (~0.50 mean) between Gemma Scope (~0.19) and GPT-2 (~0.62). This is consistent with differences in SAE training objectives.
3. **Layer-wise decrease**: EDA mean drops slightly from L6 (0.518) to L12 (0.491) in Llama, suggesting increasing encoder-decoder alignment at deeper layers — consistent with Gemma findings.
4. **AUROC not computed** due to Llama model being HF-gated. This is documented as a limitation.

## Limitations

- Llama-3.1-8B model is HF-gated: cannot train first-letter probes or run FeatureAbsorptionCalculator.
- AUROC validation not possible for Llama configs.
- Cross-architecture label transfer (GPT-2→Llama) is invalid (d_in mismatch: 768 vs 4096).

## Pass Criteria

- [x] EDA computed for both Llama SAE configs (2/2)
- [x] EDA distribution statistics documented
- [x] Cross-model comparison table generated (7 entries, 3 model families)
- [x] Llama model gating documented as limitation
- [ ] First-letter probe accuracy >= 80% — N/A (model gated)
- [ ] AUROC computed for at least 1 Llama config — N/A (model gated)

**Overall: PILOT PASS** — sufficient for R4 final summary aggregation with appropriate framing.
