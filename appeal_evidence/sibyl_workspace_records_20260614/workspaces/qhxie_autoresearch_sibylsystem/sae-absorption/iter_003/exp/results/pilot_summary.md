# Pilot Summary — Tasks A1, B1, C1 (Updated)

**Status: GO**
**Timestamp:** 2026-04-14T05:49:12
**Elapsed:** ~21 seconds
**GPU:** RTX PRO 6000 Blackwell (GPU 4)

## Task Executed

`task_A1_encoder_norm_replication` — Replicate encoder_norm AUROC=0.757 on GPT-2 L6 with exact Chanin IG labels (n_pos=18), extend to L10.

## Results Summary

### GPT-2 L6 (Primary — FeatureAbsorptionCalculator labels, n_pos=18)

| Detector | AUROC | 95% CI | AUPRC | P@50 | P@100 | P@500 |
|---|---|---|---|---|---|---|
| **encoder_norm** | **0.757** | [0.655, 0.837] | 0.0042 | 0.000 | 0.000 | 0.008 |
| EDA | 0.650 | [0.526, 0.774] | 0.0015 | 0.000 | 0.000 | 0.000 |
| activation_freq_inverted | 0.615 | [0.500, 0.743] | 0.0016 | 0.000 | 0.000 | 0.004 |
| decoder_norm | 0.515 | [0.393, 0.611] | 0.0007 | 0.000 | 0.000 | 0.000 |
| random | 0.496 | [0.363, 0.632] | 0.0009 | 0.000 | 0.000 | 0.000 |

- **DeLong test (encoder_norm > EDA):** z=3.04, p=0.0012 (one-sided) — statistically significant
- **Spearman(encoder_norm, EDA):** r=0.712, p≈0 — high but not redundant
- **Absorbed feature encoder_norm stats:** mean=3.26 vs. overall mean=2.58 (higher encoder norms in absorbed features)

### GPT-2 L10 (Extension — probe_decoder_alignment labels, n_pos=39)

| Detector | AUROC |
|---|---|
| **encoder_norm** | **0.645** |
| EDA | 0.344 |
| activation_freq_inverted | 0.453 |
| decoder_norm | 0.527 |

**Caveat:** L10 labels use proxy method (probe_decoder_alignment), not FeatureAbsorptionCalculator. The L10 AUROC comparison is less reliable than L6. Generate proper IG labels for L10 for definitive comparison.

## Pass Criteria Assessment

- [x] **L6 encoder_norm AUROC >= 0.70:** PASS (0.757)
- [x] **L10 AUROC measured:** PASS (0.645)
- **Overall: PASS**

## Key Findings

1. **H_ENC confirmed at pilot level:** encoder_norm replicates AUROC=0.757 from iter_002 to within rounding (our result: 0.7566, iter_002: 0.757). The finding is robust.

2. **DeLong test significant:** encoder_norm significantly outperforms EDA (p=0.0012), confirming it adds independent predictive value.

3. **Layer effect visible:** L6 AUROC=0.757 > L10 AUROC=0.645, consistent with absorption being more concentrated at earlier layers. However, L10 label quality is lower (proxy labels), so this comparison is confounded.

4. **Spearman r=0.712:** encoder_norm and EDA are highly correlated but not identical — they capture related but distinct geometric properties.

5. **AUPRC very low (0.004):** At 18/24576 = 0.07% prevalence, even AUROC=0.757 yields AUPRC=0.004. This is expected but important to report honestly alongside AUROC.

## Recommended Next Steps

- **Proceed to A2** (Theoretical grounding: Spearman by subtype, layer progression)
- **Proceed to A3** (Cross-architecture: Standard L1 vs. TopK)
- **Proceed to B2** (ARS_v2: does co-occurrence add signal above encoder_norm?)
- **Generate proper L10 IG labels** via sae_spelling for definitive cross-layer comparison
- **Flags for paper:** Report Precision@k = 0 at k=50/100 (not suppressed); cite AUPRC as primary practical metric; note that AUROC=0.757 remains the key detector for absorbed feature identification

---

## Task C1: OMP Pilot — Amortization Gap

**Status: PASS (with caveats) | Proceed to C2: NO**
**Elapsed:** ~23 seconds

### Pass Criteria Assessment

- [x] **OMP reconstruction error < feedforward:** PASS (OMP=0.219 < FF=0.242)
- [x] **Absorption direction reported on 3 letters:** PASS
- **Overall: PASS**

### Results

| Metric | Feedforward | OMP (K=56) | 2-pass (invalid) |
|---|---|---|---|
| Mean L0 | 55.7 | 34.9 | 3635 (buggy) |
| Mean recon error | 0.242 | 0.219 | 6.36 (buggy) |
| Absorption rate (a) | 0.950 | 1.000 | 0.950 |
| Absorption rate (e) | 1.000 | 1.000 | 1.000 |
| Absorption rate (s) | 1.000 | 1.000 | 1.000 |
| Overall AR | 0.981 | 1.000 | 0.981 |
| OMP reduction ratio | — | -0.019 | — |

### Key Findings

1. **OMP implementation valid:** Reconstruction error check passes (OMP < feedforward).

2. **Absorption measurement at ceiling:** All conditions show 98-100% absorption rate on letter-containing tokens. The proxy metric (checking if any of 18 absorbed features fires on a letter token) is too coarse — absorbed features rarely fire on letter tokens regardless of encoding method.

3. **OMP does NOT reduce absorption (marginally +1.9pp):** This is consistent with sparsity landscape dominance over amortization gap, but the measurement is too noisy to draw conclusions. The difference is well below the 5pp significance threshold.

4. **2-pass encoding bug:** 2-pass residual encoding has b_dec double-subtraction issue (residual signal encoded after b_dec already applied), causing L0=3635 and recon error=6.36. 2-pass results are invalid.

5. **DO NOT proceed to C2:** The absorption proxy metric cannot distinguish between encoding conditions. Full Chanin IG pipeline (letter probes + IG attribution) would be required for a meaningful experiment.

### H_OMP Assessment

**INCONCLUSIVE.** The pilot confirms OMP is technically feasible, but the measurement approach (checking if absorbed features fire on any letter token) lacks the sensitivity needed to detect amortization gap effects. The near-100% absorption rate suggests absorbed features almost never fire on letter-start tokens regardless of encoding method — the effect we're trying to measure is obscured by floor/ceiling effects in the proxy metric.
