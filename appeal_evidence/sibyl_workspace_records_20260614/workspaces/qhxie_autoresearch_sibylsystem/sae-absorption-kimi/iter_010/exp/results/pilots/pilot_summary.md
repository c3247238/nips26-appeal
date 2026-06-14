# Pilot Experiment Summary (Iteration 10)

**Decision: GO_WITH_CAVEATS**

## Results

| Variant | Absorption Rate | MCC | L0 | Dead Latents % | Time (s) |
|---------|----------------|-----|-----|----------------|----------|
| baseline | 0.2258 | 0.2161 | 1047.5 | 0.0% | 19.1 |
| topk | 0.0328 | 0.2146 | 50.0 | 83.2% | 25.3 |
| multiscale | 0.0416 | 0.2193 | 50.0 | 58.4% | 27.2 |
| random | 0.4519 | 0.2210 | 1043.6 | 0.0% | 14.1 |

## Sanity Checks

- absorption_discriminates: PASS
- topk_low_absorption: PASS
- convergence_ok: PASS
- mcc_variation: FAIL

## Warnings

- TopK dead_latents_pct = 83.2% (known issue from prior iterations)
- Matryoshka dead_latents_pct = 58.4% (known issue)
- MCC variation is small (~0.006) across variants - metric may be insensitive
- Only 1 seed for pilot - full experiment needs 5 seeds for statistical power

## Recommendations

- Proceed to full experiment with 5 seeds
- Report dead_latents_pct prominently in results
- Consider alternative downstream metrics beyond MCC
- Add L0-matched baseline comparison
