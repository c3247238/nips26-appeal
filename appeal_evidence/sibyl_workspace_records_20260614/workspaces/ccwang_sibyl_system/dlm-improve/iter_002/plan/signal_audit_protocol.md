# Signal Audit Protocol

## Scope

This note defines the protocol objects used by the iter_002 diagnostic paper so that observer quality, controller behavior, and runtime execution are not conflated.

Primary source artifacts:

- `exp/results/canonical_asset_manifest.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/observer_controller_protocol.json`
- `exp/results/diag_signal_gap_audit.json`
- `exp/results/runtime_probe_iter2.json`

## Core Objects

### `d(s)`: observer diagnostic signal

`d(s)` is a sample-level observer score that estimates whether draft `s` is revision-worthy before any controller intervention.

Operational source:

- main artifact: `exp/results/diag_signal_gap_audit.json`
- proxy families currently allowed: entropy, instability, and draft-side uncertainty proxies

Interpretation:

- `d(s)` measures how suspicious or revision-worthy a draft looks
- `d(s)` does not measure realized gain
- `d(s)` is therefore a diagnostic object, not a success metric

### `g(s)`: realized controller gain

`g(s)` is a sample-level realized outcome delta after a concrete controller policy and runtime path are applied to sample `s`.

Operational source:

- main artifact: `exp/results/benefit_bucket_audit_pilot.json`
- bucket realization: `fixed`, `harmed`, `no_effect`

Interpretation:

- `g(s)` is defined on realized outcomes, not on draft-side suspicion alone
- `g(s)` depends jointly on controller policy, compute budget, and runtime path
- aggregate gain is a summary over many sample-level `g(s)` values

## Runtime Contract

All signal comparisons in this iteration are conditioned on an explicit runtime contract rather than nominal method names alone.

Runtime source:

- `exp/results/runtime_probe_iter2.json`

Recommended path from the probe:

- attention backend: `eager`
- compile: `True`
- safe batch size: `57`

Runtime-fairness reporting fields:

- `nominal_nfe`
- `actual_nfe`
- `latency_sec`
- `tokens_per_sec`
- `batch_size`
- `attention_backend`
- `compile_enabled`
- `rank_shift`

These fields are material because an observer/controller claim is only trustworthy when the realized execution path is auditable.

## Claim Mapping

### Claim C1: compute order can change under realized execution

Source:

- `exp/results/diag_compute_curve_gsm8k.json`

Required fields:

- `pairwise_reorders`
- `nominal_nfe`
- `actual_nfe`
- `latency_sec`
- `rank_shift`

### Claim C2: observer quality is not controller gain

Source:

- `exp/results/diag_signal_gap_audit.json`
- `exp/results/benefit_bucket_audit_pilot.json`

Required fields:

- observer-side signal metrics for `d(s)`
- bucket outcomes for `g(s)`
- shared runtime references from `exp/results/runtime_fairness_matrix.json`

### Claim C3/C4: reasoning and code are boundary slices, not universal laws

Sources:

- `exp/results/diag_math500_shortlist.json`
- `exp/results/diag_humaneval_guard_boundary.json`

Required rule:

- boundary slices can position the result
- boundary slices cannot be upgraded into universal cross-task claims

## Non-Equivalence Caveats

1. A stronger `d(s)` does not imply a stronger `g(s)`.
2. Controller comparisons must be interpreted under matched realized compute, not just matched nominal names.
3. A bucket decomposition explains aggregate gain structure, but does not by itself validate generalization beyond the audited slice.
4. Boundary evidence from reasoning and code tasks is supporting evidence only.

## Reporting Rule

Whenever the paper makes a revision-related headline claim, it should report:

1. which artifact defines `d(s)`
2. which artifact defines `g(s)`
3. which runtime-fairness fields make the comparison credible
4. which claim is headline evidence versus boundary evidence

This is the minimum protocol needed to keep the paper diagnostic-first and honest-compute-first.
