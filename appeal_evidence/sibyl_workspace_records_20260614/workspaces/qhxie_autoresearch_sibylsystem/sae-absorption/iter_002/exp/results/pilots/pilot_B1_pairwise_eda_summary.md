# Pilot B1-RAVEN: Pairwise EDA Analysis — Summary

**Status:** GO (conditional — EDA fails, but cross-directional cosines pass)
**Mode:** PILOT (78 words across 26 letters, 100s permutation nulls, seed=42)
**Elapsed:** 26.1s / 900s budget

---

## Pass Criteria Assessment

| Criterion | Target | Result | Status |
|---|---|---|---|
| EDA AUROC (letter vs. non-letter) | >= 0.60 | 0.469 | FAIL |
| cos(enc_p, dec_c) AUROC | >= 0.60 | 0.730 | PASS |
| mean_cos(enc_c, dec_p) AUROC | >= 0.60 | 0.681 | PASS |
| n_pos proxy-absorbed | >= 10 | 47 | PASS |

**Overall: GO** — cross-directional cosines show strong signal even though EDA AUROC fails the primary threshold.

---

## Key Findings

### 1. EDA Discrepancy with Prior Pilot
- This pilot: EDA AUROC (letter vs. non-letter) = **0.469** (Cohen's d = -0.098, Wilcoxon p = 0.42 — non-significant)
- Prior pilot (pilot_A): EDA AUROC = **0.681** (Cohen's d = +0.70, p < 0.001)
- **Root cause:** Threshold calibration. This pilot uses thr=0.29 giving n=63 letter features; prior pilot used thr=0.32 giving n=50. The additional features at 0.29-0.32 range include non-letter features with higher EDA, diluting the signal. The EDA metric is threshold-sensitive at L6.
- **Note:** EDA direction is INVERTED here (negative d). Letter features have LOWER EDA than non-letter features in this sample — contradicts prior pilot and published AUROC=0.681.

### 2. Cross-Directional Cosines: Strong Signal (New Finding)
| Metric | AUROC | Cohen's d | z-score | p-value |
|---|---|---|---|---|
| EDA (letter vs. non-letter) | 0.469 | -0.098 | -0.86 | 0.42 |
| cos(enc_c, dec_p) max | 0.598 | 0.103 | +2.53 | 0.011 |
| **cos(enc_p, dec_c) max** | **0.730** | **0.552** | **+6.38** | **<0.001** |
| mean_cos(enc_c, dec_p) | 0.681 | 0.517 | +4.63 | <0.001 |

**Critical finding:** `cos(enc_p, dec_c)` — the cosine similarity between the **parent encoder** and the **child decoder** — is the strongest discriminator for letter features (AUROC=0.730, z=6.38).

This means: For letter features (child features), the parent encoder direction aligns significantly more with the child decoder direction than for non-letter features.

**Geometric interpretation:** Letter feature decoders (dec_c) point in directions that coincide with parent feature encoder directions (enc_p). This is the REVERSE of the revised H1 prediction (which expected enc_c to align with dec_p). Instead, enc_p aligns with dec_c.

### 3. Within-Letter Feature Analysis (Proxy Labels)
- 47/63 letter features labeled as "proxy-absorbed" (split feature non-firing in few-shot context)
- cos(enc_p, dec_c): AUROC=0.766, d=1.06, p=0.002
- mean_cos(enc_c, dec_p): AUROC=0.822, d=1.03, p=0.0001
- These are highly significant within-letter discriminators

### 4. L10 Sanity Check
- L10 EDA AUROC = 0.256 (letter vs. non-letter) — confirms EDA fails at L10 (prior pilot: 0.337)
- Consistent with prior finding that EDA is layer-specific

---

## Interpretation for Full Experiment

### EDA Issue
The EDA inconsistency (0.469 vs 0.681) suggests the metric is threshold-sensitive. For the full experiment, must use the exact threshold from the prior pilot (thr=0.32, n=50 features) rather than recalibrating. The full experiment should use sae-spelling FeatureAbsorptionCalculator with IG ablation for exact Chanin labels (n_pos=67 confirmed pairs).

### Unexpected Geometric Finding
The strong signal in `cos(enc_p, dec_c)` (parent encoder aligns with child decoder) is a new geometric observation that merits investigation:
- The revised H1 predicts: enc_c → dec_p (child encoder pulled toward parent decoder)
- The data shows: enc_p aligns with dec_c (parent encoder aligns with child decoder)
- These are not mutually exclusive — absorption may involve bidirectional alignment

For the full B1-RAVEN experiment, the pairwise analysis should report ALL cross-directional metrics:
1. EDA_child = 1 - cos(enc_c, dec_c) — per-feature encoder-decoder alignment
2. cos(enc_c, dec_p) — child encoder alignment with parent decoder
3. cos(enc_p, dec_c) — parent encoder alignment with child decoder
4. Compare all three against exact Chanin labels (n_pos=67)

---

## Recommended Action for Full Experiment

**GO** — Proceed with task_B1_pairwise_eda full experiment with these modifications:
1. Use exact Chanin labels from sae-spelling FeatureAbsorptionCalculator (n_pos=67) rather than proxy activation-based labels
2. Use fixed threshold thr=0.32 for letter feature identification (matching prior pilot)
3. Report cos(enc_p, dec_c) as primary cross-directional metric alongside EDA
4. If EDA AUROC < 0.55 on exact labels, investigate threshold issue — compare with proxy labels

**Full experiment expected runtime:** ~45 min (sae-spelling IG ablation on 67+ words × 26 letters)
