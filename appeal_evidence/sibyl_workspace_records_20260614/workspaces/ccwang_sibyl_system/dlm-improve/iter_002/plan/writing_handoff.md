# Iteration 2 Writing Handoff

## Final positioning

The paper should now be written as a **compute-normalized diagnostic / protocol paper**.

Do not revert to:

- a new hero controller story
- calibration-aware method branding
- a generic observer/controller optimization paper

## New evidence added in iter_002

### Runtime closure

- `exp/results/runtime_probe_iter2.json`
- `exp/results/runtime_fairness_matrix.json`

Usable statement:

- the current host supports `eager + torch.compile=True`
- safe batch size for the main GSM8K slice is `57`
- runtime assumptions are explicit rather than hidden in prose

### Bucket closure

- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/benefit_bucket_examples_pilot.json`

Usable statement:

- the headline GSM8K gain is decomposable into sample-level outcomes
- `fixed = 7`, `harmed = 4`, `no_effect = 89`
- the aggregate `+3pp` gain matches `fixed - harmed`

### Protocol closure

- `exp/results/canonical_asset_manifest.json`
- `exp/results/observer_controller_protocol.json`
- `plan/signal_audit_protocol.md`

Usable statement:

- observer signal, controller realization, and runtime execution are separate protocol objects
- claim-to-asset lineage is explicit and appendix-ready

### Seed closure

- `exp/results/seed_sensitivity_spotcheck.json`

Usable statement:

- only claim **sign consistency**
- seed deltas are `[+0.03, +0.01, +0.01]`
- wording should be: the headline direction remained the same across a minimal three-seed spot-check

## What to say in the paper

### Safe headline claims

1. Aggregate revision gain is not enough; bucket-level outcomes reveal the real effect structure.
2. Honest-compute conclusions are only credible when runtime-lineage and claim-to-asset mappings are auditable.

### Safe framing sentence

Observer quality does not automatically translate into controller gain, so observer diagnostics, controller outcomes, and runtime execution should be reported separately.

## What not to say

- Do not claim full robustness from the seed spot-check.
- Do not present the optional minimal controller probe as missing evidence; it was intentionally closed as `NO_GO`.
- Do not imply the reasoning-side result generalizes cleanly to code or all tasks.

## Optional probe disposition

The final optional task was intentionally closed via:

- `exp/results/min_controller_decoupling_probe.json`

Use this as an explicit explanation for why the paper stops at bucket + protocol + seed closure instead of opening a new controller line.
