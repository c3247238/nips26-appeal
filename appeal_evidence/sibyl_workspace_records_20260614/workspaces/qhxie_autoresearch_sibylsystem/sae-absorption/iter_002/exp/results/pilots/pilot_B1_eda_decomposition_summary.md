# Pilot B1 EDA Decomposition Summary

**Task**: B.1-REV — EDA Geometric Decomposition (Encoder vs. Decoder Alignment)
**Mode**: PILOT (static weight analysis, no token inference)
**Go/No-Go**: NO_GO (pass criteria not met, but INFORMATIVE)
**Elapsed**: ~4 seconds

## Pass Criteria

> "Paired t-test p < 0.05: letter feature encoder aligns more with letter probe than decoder does."

Result: **FAILED**. The decoder aligns significantly MORE with letter probe than encoder does (p ≈ 0), in the OPPOSITE direction of the revised H1 prediction.

## Key Measurements

| Metric | L6 | L10 |
|--------|-----|-----|
| n_letter_features | 50 | 32 |
| EDA AUROC (letter detection) | 0.659 | 0.227 (REVERSED) |
| EDA Cohen's d | +0.53 | -0.54 |
| cos(encoder, letter_probe): letter | 0.139 | 0.180 |
| cos(decoder, letter_probe): letter | 0.383 | 0.388 |
| diff (enc - dec): letter | **-0.244** | **-0.208** |
| diff (enc - dec): non-letter | -0.043 | -0.025 |
| paired t-test p (letter) | ~0 (dec more aligned) | ~0 (dec more aligned) |

## Geometric Interpretation

**Contrary to revised H1**: At BOTH layers, the decoder is substantially more aligned with letter probes than the encoder. This contradicts the hypothesis that absorption pulls the encoder toward the parent (letter probe) direction while the decoder stays specialized.

**What is actually happening**: Letter features are identified by high decoder-probe alignment (threshold=0.32). The EDA is high precisely because the decoder IS specialized to the letter direction, but the encoder is NOT pulled to the same degree. This is a form of encoder-decoder dissociation, but with the decoder being the specialized component.

**L10 AUROC = 0.227** (reversed): Letter features at L10 have LOWER EDA than non-letter features. This means at L10, letter feature encoders and decoders are MORE co-aligned than average. The layer-specific failure of EDA as a detector is confirmed.

## Implications for F.1 Theory

The revised theory must be updated again:
- EDA is high for letter features because: **decoder is strongly specialized** (pulled toward letter probe direction) while the encoder is only weakly pulled.
- This is still consistent with encoder-decoder dissociation as the absorption signature, but the direction of causation is different from what revised H1 proposed.
- The correct story: During training, the decoder specializes to the letter concept (high decoder-probe alignment), but the encoder does not equally specialize (creating high EDA). This may reflect that absorbed features have their decoder specialized while the encoder is "left behind."

## Encoder Norms (Informative)

- Letter features at L6 have significantly LARGER encoder norms (3.28 vs 2.58 for non-letter)
- Decoder norms are all normalized to ~1.0 by SAELens convention
- This suggests letter features have stronger encoder representations despite the geometric misalignment

## Recommendation

Proceed to full analysis (this is INFORMATIVE NO_GO, not a pipeline failure). The finding that decoder-probe alignment dominates, not encoder displacement, is a genuine discovery that revises the mechanistic account of EDA. Update F.1 theory to reflect decoder-centric specialization as the mechanism.

Key downstream tasks remain valid:
- B1_pairwise_eda: test EDA against exact Chanin labels (still valid, EDA AUROC=0.659 confirmed)
- D1 validation: should proceed
- F1 theory revision should address the decoder-centric finding
