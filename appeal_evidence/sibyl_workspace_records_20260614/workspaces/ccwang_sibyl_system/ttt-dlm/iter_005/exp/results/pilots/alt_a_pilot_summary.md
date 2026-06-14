# Pilot Summary: alt_a_pilot — Information-Gain Adaptive Unmasking

## GO/NO-GO: GO (with caveats)

## Results

| Method | GSM8K-100 Acc | Time/sample | Overhead |
|--------|--------------|-------------|----------|
| Vanilla (baseline) | 34.0% | 5.18s | 1.00x |
| Info-Gain Greedy | 16.0% | 4.67s | 0.90x |
| Info-Gain Soft (tau=1.0) | **39.0%** | 4.68s | 0.90x |

## Key Findings

1. **Info-Gain Soft passes the +1% threshold** (+5pp over vanilla), qualifying as GO
2. **Info-Gain Greedy catastrophically fails** (-18pp) — deterministic easy-first ordering resolves trivial tokens first, producing incoherent reasoning
3. **No computational overhead** — both IG methods are actually slightly faster (0.90x) because efficient unmasking leads to fewer remaining masks in later steps
4. **Statistical significance is marginal** — bootstrap p=0.182 (not <0.10 at pilot level)
5. **Text diversity is maintained** — Distinct-2: vanilla=0.840, IG-soft=0.892

## Caveats

- The +5pp improvement is not statistically significant at n=100 (p=0.182)
- Agreement analysis shows IG-soft gains 9 new correct but loses 27 vanilla was getting right — different solution distributions
- Needs replication on full GSM8K (n=1319) and additional benchmarks (ARC-C, HumanEval)
- The soft variant's tau=1.0 is untuned — there may be better temperature settings

## Entropy Analysis

- Entropy increases monotonically during denoising (~2.0 at step 0 to ~4.0 at step 112)
- Greedy selects near-zero entropy positions until step ~64, wasting early context-building on trivial tokens
- The entropy distribution is bimodal: many near-zero (function words) and many near-5.0 (content tokens)

## Recommendation

Proceed with Info-Gain Soft as a viable Alternative A track. Prioritize:
1. Full-scale GSM8K evaluation (n=1319) to confirm statistical significance
2. ARC-C and HumanEval evaluation to check generalization
3. Tau sweep (0.5, 1.0, 2.0, 5.0) to optimize the entropy temperature
4. Combination with A-CFG (from iter 4 pilot showing +12.5pp on n=16)
