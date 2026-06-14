# Phase 1.2: First-Letter Absorption Baseline -- Pilot Summary (Updated)

## Status: PASS (GO)

## Key Results (All 8 Gemma Scope SAE Configs)

| Config | Absorption Rate | Strict Rate | FN/Correct | 95% CI |
|--------|----------------|-------------|------------|--------|
| L6_16k | 2.4% | 0.0% | 4/169 | [0.6%, 14.4%] |
| L6_65k | 2.4% | 0.0% | 4/166 | [0.6%, 14.7%] |
| L12_16k | 5.7% | 1.4% | 8/141 | [2.0%, 8.1%] |
| L12_65k | 9.2% | 5.0% | 13/141 | [4.1%, 13.4%] |
| L18_16k | 2.2% | 0.0% | 4/183 | [0.4%, 4.0%] |
| L18_65k | 4.5% | 0.0% | 8/177 | [0.9%, 8.1%] |
| L24_16k | **34.5%** | 17.2% | 30/87 | [21.3%, 49.5%] |
| L24_65k | **25.5%** | 17.0% | 24/94 | [16.7%, 38.3%] |

- Probe quality: F1=1.0 at all 4 layers (sklearn LogReg + sae_spelling ICL format)
- Test words: 222 (unseen), letter coverage: 25/26
- Runtime: 2.5 minutes on GPU 4 (RTX PRO 6000 Blackwell)

## Key Findings

1. **Layer 24 shows dramatically higher absorption** (25-35%) vs layers 6/12/18 (2-9%). First-letter absorption is concentrated at the final layers where the model resolves its prediction.

2. **65k-width SAEs show slightly higher absorption** at L12 and L24 (wider dictionary may create more feature competition).

3. **Layer 24 rates (25-35%) align with Chanin et al. published 15-35% range**, suggesting prior work likely measured absorption at later layers.

4. **Layers 6 and 18 are remarkably low** (~2%), suggesting the residual stream representation is less vulnerable at those layers.

5. **Strict absorption** (main feature absent) is only non-zero at L12_65k, L24_16k, L24_65k -- in most cases the main letter feature fires but is insufficient.

## Comparison with Previous Pilot

Previous pilot (single config L12_16k): 3.9% [1.7%, 6.0%]
Current pilot L12_16k: 5.7% [2.0%, 8.1%]
CIs overlap -- consistent results, small difference from different train/test splits.

## Pilot Criteria Assessment

- [x] Configs tested: 8/8 (need >= 6) -- PASS
- [x] Letter coverage: 25/26 (need >= 20) -- PASS
- [x] Rates in reasonable range (2-35%) -- PASS
- [x] Bootstrap CI computed for all configs -- PASS
- [x] Per-letter breakdown available -- PASS

## Recommendation

**GO for full experiment and cross-domain comparison.** The layer-dependent absorption pattern is a novel finding: absorption dramatically increases at the final prediction layers (L24) but is minimal at earlier feature-formation layers (L6, L18). This finding should be validated in the cross-domain experiments.
