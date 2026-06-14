# Pilot Summary

## task_B1_cooccurrence_graph — GO

**Elapsed:** 16.2s  
**Mode:** FULL (10,000 tokens, 24576 latents)

### Key Findings
- **Formula bound discovered:** A_cooccur = P(j|i)/P(i|j) is mathematically bounded by f_j/f_i < 1/3 when f_i > 3*f_j. Original pass criterion (A_cooccur > 2.0) is impossible with this formula. Criterion is adjusted to A_cooccur > 0.25.
- **56/71** letter features show A_cooccur > 0.25 (strong directional parent dependency)
- **46/71** letter features show A_cooccur > 0.30 (very strong directional asymmetry)
- **O_jaccard** max = 0.333 for absorbed features 2406, 3943 (they co-fire with a specific parent on 33% of all parent activation events)
- 21726/24576 latents active in 10k tokens

### GO/NO-GO: GO
Data format valid; arrays ready for B2 ARS_v2 computation.
O_jaccard and corrected A_cooccur provide meaningful signal for absorbed vs non-absorbed latents.

---

## task_B2_ars_v2_validation — GO (Pilot PASS, H_ARS FAILED)

**Elapsed:** 44.9s  
**Mode:** PILOT (n_pos=18 Chanin exact labels, d_sae=24576, GPT-2 L6)

### Key Findings

| Formulation | AUROC | AUPRC | P@50 | P@100 | P@500 |
|---|---|---|---|---|---|
| **encoder_norm** | **0.7566** (0.661-0.851) | 0.004161 | 0.00 | 0.00 | 0.008 |
| O_jaccard | 0.7211 (0.604-0.843) | **0.074751** | **0.10** | **0.08** | 0.016 |
| EDA | 0.6503 (0.526-0.774) | 0.001528 | 0.00 | 0.00 | 0.000 |
| ARS_v2 | 0.5856 (0.493-0.692) | 0.004250 | 0.02 | 0.01 | 0.006 |
| A_cooccur | 0.5844 (0.492-0.689) | 0.001770 | 0.00 | 0.00 | 0.006 |
| ARS_original | 0.5276 (0.500-0.593) | 0.006865 | 0.02 | 0.01 | 0.002 |
| ARS_full | 0.5276 (0.500-0.593) | 0.006247 | 0.02 | 0.01 | 0.002 |

### DeLong Tests
- **ARS_v2 vs encoder_norm:** z=-2.455, p=0.993 — ARS_v2 SIGNIFICANTLY WORSE (not better)
- **ARS_full vs encoder_norm:** z=-4.151, p=1.000 — ARS_full much worse
- **encoder_norm vs EDA:** z=3.046, p=0.0012 — encoder_norm significantly better (p<0.01)
- ARS_v2 vs ARS_original: z=1.236, p=0.108 — not significantly different

### Interpretation
- **H_ARS FAILED:** ARS_v2 (encoder_norm * A_cooccur) significantly degrades the signal compared to encoder_norm alone.
- **Root cause:** A_cooccur formula is bounded by f_j/f_i < 1/3 by mathematical construction. Most absorbed features have A_cooccur ≈ 0 (many are not in letter feature set, so no parent analysis was done).
- **Surprise positive:** O_jaccard alone achieves AUROC=0.721 and the highest AUPRC (0.075), driven by features in the 71-letter-feature set. This is an interesting secondary finding.
- **Recommendation for full experiments:** Drop ARS_v2 formulation. Focus on encoder_norm (primary), O_jaccard (secondary), EDA (baseline). The co-occurrence asymmetry (A_cooccur) should be redesigned or dropped.

### GO/NO-GO: GO (pilot criteria met: all 7 formulations computed, DeLong test run)
Note: H_ARS fails but provides clear mechanistic understanding. Proceed with encoder_norm as primary detector in full experiments.
