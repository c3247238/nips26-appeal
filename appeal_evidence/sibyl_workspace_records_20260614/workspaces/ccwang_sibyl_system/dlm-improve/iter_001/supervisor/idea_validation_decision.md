# Idea Validation Decision

- Decision: `ADVANCE`
- Selected candidate: `cand_diag`
- Confidence: `0.88`

## Why

- `cand_diag` is the only candidate whose core claims are already supported by completed pilot evidence across all four planned slices.
- Honest compute accounting changed the GSM8K story in a meaningful way: `CORE-proxy-64` moved behind revision methods under actual compute, with a `7.81%` compute-gap mismatch.
- The observer-vs-controller framing is no longer speculative. Calibration had strong diagnostic value with zero deployed control gain, and entropy also showed a clear gap between observation quality and control benefit.
- The second reasoning benchmark was informative rather than flat: `MATH500` reordered to `Standard-64 > CORE-proxy-64 > TIGER > Entropy`, which reinforces the task-dependent revision story.
- Code-boundary evidence stays appendix-only: gated TIGER cut syntax failures, but it still underperformed `Standard` on `pass@1` and worsened runtime failures.
- `cand_minimal` is still a live backup, but it lacks equally strong pilot evidence today. `cand_factorization` remains more speculative and should only be revisited if full diagnostics stall.

## Next Actions

- Advance `cand_diag` into the next full experiment block.
- Scale matched-compute comparisons on the shortlisted reasoning benchmarks with actual NFE, latency, TPS, and batch size reported together.
- Add benefit-bucket and failure-taxonomy analyses to sharpen the mechanism story.
- Keep HumanEval gating as boundary evidence only; do not reopen `cand_tiger` as the main method narrative.
- Preserve `cand_minimal` and `cand_factorization` as backup pivots, not active front-runners.
