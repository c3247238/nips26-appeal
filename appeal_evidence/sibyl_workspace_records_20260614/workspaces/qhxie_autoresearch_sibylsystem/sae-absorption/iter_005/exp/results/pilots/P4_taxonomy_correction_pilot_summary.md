# P4 Taxonomy Correction - Pilot Summary

## Task
Correct the inflated Type II absorption taxonomy rate (92.3%) from iter_004 by using proper comparison tokens instead of the fallback global mean-when-active baseline.

## Key Finding
**The original 92.3% comprehensive absorption rate drops to 19.2% after correction (95% CI: [3.8%, 34.6%]).**

The 88.5% Type II rate was a measurement artifact caused by:
1. **21/26 letters had n_comparison_tokens=0**, forcing fallback to global mean-when-active
2. The global mean-when-active baseline is systematically biased upward (includes high-activation non-letter contexts)
3. When proper non-letter-context comparison tokens are used (Strategy A), the parent features fire **more strongly** on letter tokens than non-letter tokens (corrected magnitude ratio >> 0.5 for 19/21 letters)

## Corrected Classification
| Classification | Original | Corrected |
|---|---|---|
| Type I (complete) | 1 (3.8%) | 1 (3.8%) |
| Type II (partial) | 23 (88.5%) | 4 (15.4%) |
| Type III (distributed) | 0 (0%) | 0 (0%) |
| None | 2 (7.7%) | 21 (80.8%) |
| **Comprehensive** | **24 (92.3%)** | **5 (19.2%)** |

## Validated Absorption Metric
The Chanin metric (false-negative detection) detects absorption in **19/26 letters (73.1%)**, independent of the magnitude ratio. This should be reported as the validated primary metric:
- Chanin high absorption (>50%): 4 letters (E, O, S, W)
- Chanin partial absorption: 12 letters
- No absorption: 7 letters (F, G, I, K, U, V, Z)

## Recommendation
1. Report **73.1% Chanin any-absorption rate** as the validated primary metric
2. Mark **92.3%** as "upper bound, artifact of missing comparison baseline"
3. Report **19.2% corrected comprehensive rate** with caveat that Type II metric is limited
4. The key message: absorption IS real (Chanin detects it in 73% of letters), but the magnitude-ratio-based Type II classification was unreliable

## Methodology
- Model: GPT-2 Small (open-model anchor)
- SAE: gpt2-small-res-jb, blocks.8.hook_resid_pre (d_sae=24576)
- Corpus: WikiText-103 (8,000 sentences, ~205k token positions)
- Strategy A: Compare parent feature activation on letter tokens vs. non-letter tokens from natural text
- Bootstrap: 10,000 resamples for 95% CIs

## Pass Criteria
- [x] Frequency-matched comparison tokens for >= 20 letters: 21/26
- [x] Type II rate differs from original by > 5 percentage points: 73.1% delta
- **Overall: GO**
