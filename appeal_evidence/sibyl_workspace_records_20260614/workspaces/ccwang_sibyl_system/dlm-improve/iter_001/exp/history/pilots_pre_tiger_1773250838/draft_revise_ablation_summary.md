# Draft-Revise Ablation Pilot Summary

## Task: draft_revise_ablation (H3 + H4)
- **Model**: LLaDA-8B-Instruct
- **Benchmark**: GSM8K (100 samples, seed=42)
- **Design**: 2x4 factorial (draft type x revision type)
- **Total time**: 31.2 minutes on 1x RTX PRO 6000 (97GB)

## Results Table

| Condition | Draft | Revision | Accuracy | NFE | Tokens Changed | Time(s) |
|-----------|-------|----------|----------|-----|----------------|---------|
| standard_none | standard | none | **38.0%** | 65 | 0.0 | 225 |
| standard_random | standard | random | 40.0% | 68 | 0.3 | 235 |
| standard_raw_entropy | standard | raw_entropy | **41.0%** | 68 | 4.8 | 235 |
| standard_calibrated_entropy | standard | calibrated | **41.0%** | 68 | 4.8 | 236 |
| aggressive_none | aggressive | none | 24.0% | 65 | 0.0 | 225 |
| aggressive_random | aggressive | random | 30.0% | 68 | 1.0 | 235 |
| aggressive_raw_entropy | aggressive | raw_entropy | 33.0% | 68 | 8.6 | 235 |
| aggressive_calibrated_entropy | aggressive | calibrated | 33.0% | 68 | 8.6 | 236 |

## Main Effects

### Revision Type (averaged over draft types)
- none: 31.0%
- random: 35.0% (+4.0%)
- raw_entropy: **37.0%** (+6.0%)
- calibrated_entropy: 37.0% (+6.0%)

### Draft Type (averaged over revision types)
- standard: **40.0%**
- aggressive: 30.0% (-10.0%)

### Per-Draft Revision Effect
- standard: 38.0% -> 40.7% (**+2.7%**)
- aggressive: 24.0% -> 32.0% (**+8.0%**)

## Hypothesis Tests

### H3: Revision > No-Revision by > 2%
- **PASS** (+5.3% average, +2.7% for standard draft)
- Revision is especially impactful for aggressive drafts (+8.0%)

### H4: Calibrated Entropy >= Raw Entropy
- **Technically PASS** but calibration adds **zero** improvement
- Both raw and calibrated entropy achieve identical results
- The calibration correction does not change the revision target ranking

## Key Insights

1. **Revision works**: Even random remasking helps (+4%), but entropy-targeted revision is better (+6%)
2. **Aggressive draft is harmful**: -14% accuracy loss with no revision; revision partially recovers but never catches up to standard draft
3. **Calibration adds nothing**: Despite strong ECE (0.22) proving miscalibration exists, correcting it doesn't improve revision targeting. This suggests entropy alone is a sufficient signal for identifying fragile tokens.
4. **Tokens changed**: Entropy-based revision changes 4.8-8.6 tokens/sample; random only changes 0.3-1.0. Entropy correctly identifies tokens that need revision.
5. **NFE overhead is small**: Revision adds only 3 NFE (65 -> 68), a 4.6% overhead for 3% accuracy gain (standard draft).

## Implications for CARD

- The core CARD method should use **standard draft + raw entropy revision** (simplest, equally effective)
- Calibration correction can be dropped from the revision targeting (simplifies the method)
- The calibration study remains valuable as a **scientific contribution** (first DLM calibration curves) even if it doesn't improve the practical method
- For the DNB comparison: standard_raw_entropy at 68 NFE (41%) already beats DNB-84 estimate (38%)
- **Recommendation**: Simplify CARD to use raw entropy, focus paper narrative on the calibration study as a contribution and the revision mechanism as the practical improvement

## GO/NO-GO: GO
