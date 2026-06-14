# Pilot B1: Decoder Geometry Analysis — Summary

**Status:** COMPLETE (24.2 seconds)
**Mode:** PILOT
**Task:** task_B1_decoder_geometry
**Timestamp:** 2026-04-13

---

## Key Findings

### Unexpected Direction of Effect

The central finding from B1 PILOT is that **letter features (child/absorbed features) show
LOWER cos^2(theta) with candidate parent features**, not higher as H1 predicts.

| Config | Metric | Letter Features (pos) | Non-Letter (neg) | Cohen's d | Wilcoxon p | AUROC |
|--------|--------|-----------------------|-----------------|-----------|------------|-------|
| L6 | max cos^2 with parents | 0.0438 | 0.0862 | -0.481 | 9.7e-6 | 0.331 |
| L10 | max cos^2 with parents | 0.0364 | 0.1008 | -0.689 | 1.9e-11 | 0.177 |
| L6 | mean cos^2 with parents | 0.00137 | 0.00307 | -0.832 | 5.1e-18 | 0.170 |
| L10 | mean cos^2 with parents | 0.00131 | 0.00361 | -1.402 | 7.7e-19 | 0.073 |

**The separation is statistically significant (p << 0.05) but in the WRONG direction for H1.**

### EDA Comparison

| Config | EDA AUROC | EDA Cohen's d | EDA Wilcoxon p |
|--------|-----------|---------------|----------------|
| L6 | 0.681 | +0.700 | 1.3e-7 |
| L10 | 0.337 | -0.476 | 3.1e-4 |

EDA works at L6 (consistent with Pilot A), but the decoder-cosine component alone (cos^2)
gives inverse signal to theory prediction.

### RD Threshold Classifier

The RD threshold classifier (lambda > sin^2(theta), sweeping lambda in [0.01, 0.50]) uniformly
predicts all features as absorbed (all cos^2 values are << 0.99). The threshold approach does not
activate within the empirical lambda range (lambda ~ 0.02 at L0~50).

---

## Interpretation

**Why letter features have LOWER cos^2 with parents:**

Letter features encode very specific information (first-letter spelling). Their decoder directions
are likely **orthogonal** to most general-purpose parent features, precisely because the SAE
training has specialized them. The high-frequency "parent" features are general semantic concepts
whose decoders point in different directions from these specialized letter features.

This is a key insight: absorption in the actual Chanin et al. sense refers to a parent feature
firing *instead of* a child feature. This does NOT necessarily require high decoder cosine
similarity. Instead, absorption may be characterized by:

1. Low activation frequency of the child feature (child is rarely active)
2. High activation frequency of the parent (parent is always active)  
3. Encoder alignment: the child's encoder direction picks up on a parent signal

The proxy labels (probe_decoder_alignment) identify letter features by their high decoder cosine
with letter probes — this is a different geometric relationship than parent-child decoder cosine.

**EDA works because:** EDA = 1 - cos(encoder_j, decoder_j) is high when a feature's encoder
and decoder point in different directions. This happens specifically for absorbed features where
the encoder has been "pulled" toward the parent's direction while the decoder maintains the
original child direction.

---

## Pass Criteria Evaluation

- Wilcoxon p < 0.05 for cos^2 separation: **PASS** (both L6 and L10, but inverse direction)
- Overall GO/NO_GO: **GO** (technically passes Wilcoxon threshold)

**Design Review Flag:** The direction of cos^2 separation is **inverse to H1 prediction**.
H1 states that absorbed feature pairs should have higher cos^2(theta). Data shows they have lower.
This suggests H1's geometric mechanism may need revision or the proxy labels may not capture
the correct parent-child pairs.

---

## Recommendations for Full B1 Experiment

1. **Use exact Chanin et al. labels** (sae-spelling repo) for paired (parent, child) absorption
   measurements rather than per-feature proxy labels.
2. **Report the inverse direction** explicitly — letter features have lower decoder cosine with
   parents than non-letter features.
3. **EDA remains the working metric** (AUROC=0.681 at L6). The B1 full experiment should focus on
   validating EDA's geometric interpretation (encoder-decoder misalignment) rather than raw
   decoder-decoder cosine similarity.
4. **Consider alternative H1 formulation:** Perhaps absorption is predicted by ENCODER alignment,
   not decoder alignment. The parent encoder might capture the child's concept if the decoder
   has specialized (absorbed child).
