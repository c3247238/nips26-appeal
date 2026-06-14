# pilot_h1_h4 Results

## Overall: FAIL

| Criterion | Threshold | Achieved | Pass |
|-----------|-----------|----------|------|
| H4 Spearman r | > 0.3 | 0.7875 | PASS |
| H1 signal (layer variation) | layer_8 > layer_4 | 0.0527 > 0.0684 | FAIL |
| Reconstruction MSE | < 10.0 | 0.6547225117683411, 1.2337689399719238 | PASS |

## Layer-wise Results

### Layer 4
- N features: 100
- Mean UAS: 0.2775
- Mean Chanin absorption: 0.0684
- Spearman r: 0.8147 (p=6.34e-25)
- Reconstruction MSE: 0.6547

### Layer 8
- N features: 100
- Mean UAS: 0.1933
- Mean Chanin absorption: 0.0527
- Spearman r: 0.7603 (p=4.52e-20)
- Reconstruction MSE: 1.2338

## Conclusion

- **H4**: UAS correlates with supervised absorption (Spearman r=0.7875)
- **H1 (preliminary)**: Layer 8 does NOT show higher absorption than layer 4
- **Pilot**: CAUTION - review results before proceeding

Generated: 2026-04-26T18:01:17.007285
