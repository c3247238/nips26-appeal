# Idea Validation Decision

- Decision: `PIVOT`
- Selected candidate: `cand_diag`
- Confidence: `0.93`

## Why

- GSM8K shortlist failed the core TIGER method gate: `TIGER = 0.39`, `Entropy = 0.39`.
- TIGER beat `Prophet` and `DNB`, but remained materially below `CORE-proxy = 0.46`.
- Gating helped code syntax failures, but did not restore a method-forward win and slightly reduced reasoning accuracy.

## Next Actions

- Reframe the contribution as a compute-normalized diagnostic benchmark study.
- Mark `math500_transfer` as skipped by pivot, not as a missing run.
- Keep code gating as boundary/appendix evidence only.
- Let the next iteration, if approved, start from `cand_diag`.
