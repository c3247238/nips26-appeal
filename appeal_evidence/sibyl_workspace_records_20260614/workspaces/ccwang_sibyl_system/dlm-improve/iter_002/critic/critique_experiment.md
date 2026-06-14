# Experiment Critique

## Evidence Basis

`exp/results/summary.md` is missing. This critique therefore uses the most representative available artifacts:

- `exp/results/pilot_summary.json`
- `exp/results/pilot_summary.md`
- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/benefit_bucket_examples_pilot.json`
- `exp/results/runtime_probe_iter2.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/seed_sensitivity_spotcheck.json`
- `exp/results/observer_controller_protocol.json`
- `exp/results/canonical_asset_manifest.json`
- `exp/results/min_controller_decoupling_probe.json`
- `exp/results/experiment_cycle_closeout.json`

## Main Verdict

The experiment package is good enough to support a narrow audited-slice result. It is not good enough to support a broader scientific thesis about observer-controller separation or a clean, end-to-end runtime-fairness story without qualification.

## Major Experimental Issues

### 1. The audited slice is too easy to mistake for full-benchmark evidence

The bucket audit has `num_samples = 100` and `coverage = 1.0` (`exp/results/benefit_bucket_audit_pilot.json:6-17`). The runtime probe shows the GSM8K dataset contains 1319 examples (`exp/results/runtime_probe_iter2.json:58-66`). So `coverage = 100%` means full coverage of a 100-example slice, not the benchmark.

That distinction must be made impossible to miss. Right now the manuscript's wording can still be read as benchmark-level closure.

### 2. The seed evidence weakens the headline more than the paper admits

The seed spot-check is honestly labeled as sign consistency, which is good. But the actual values matter:

- seed 40: `+0.03`
- seed 41: `+0.01`
- seed 42: `+0.01`

from `exp/results/seed_sensitivity_spotcheck.json:16-77`.

So the mean gain is only `+0.0167`, and the reference seed is `+0.01`. That is not a failure, but it does mean the paper should not keep speaking as if `+3pp` is the single natural headline.

### 3. The runtime-fairness artifact is internally inconsistent

The workspace's current audited runtime contract is:

- `eager|compile=True`
- safe batch size `57`

from `exp/results/runtime_probe_iter2.json:68-96`.

The bucket audit and seed spot-check both align with that contract (`benefit_bucket_audit_pilot.json:32-60`; `seed_sensitivity_spotcheck.json:8-15`). However, `runtime_fairness_matrix.json` still contains old headline rows with unmatched settings:

- `Standard-64`: batch `115`
- `Entropy-Revise-64+3`: batch `32`, `compile_enabled = false`

from `runtime_fairness_matrix.json:7-85`.

That means the runtime bundle is currently demonstrating why fairness matters, but it is not yet a clean matched-fairness table for the headline pair.

### 4. The observer-controller claim is not experimentally closed

The protocol bundle says observer signal and realized gain should be reported separately (`observer_controller_protocol.json:4-37`). That is reasonable. But the actual decoupling probe was skipped (`min_controller_decoupling_probe.json`), so the workspace does not contain a matched experiment that shows observer quality failing to become controller gain.

Worse, the bucket audit itself suggests the signal story is not straightforward. If you compute bucket-level means from `per_sample`, the average entropy signal is:

- `fixed`: `0.1312`
- `harmed`: `0.1285`
- `no_effect`: `0.2515`

That does not look like a clean recoverability selector.

### 5. The examples imply shallow repair, but the paper does not analyze it

The per-sample evidence is useful, and some examples are revealing:

- fixed cases such as indices 24, 31, and 33 are mostly short completion/final-answer repairs;
- harmed cases such as 48, 55, and 62 look like truncation or answer corruption;
- some no-effect wrong cases keep large entropy values without becoming recoverable.

This is exactly the kind of evidence that could have made the paper more interesting. Instead, the current writeup mostly stops at bucket counts.

## What the Experiment Package Does Well

- It closes the bucket-audit gap with real per-sample evidence.
- It adds a real seed quality gate instead of hiding behind a single seed.
- It explicitly records runtime-probe facts rather than burying them in prose.
- It makes the decision not to expand the controller family explicit, which is a disciplined experimental choice.

## Highest-Value Experimental Fix

If no new experiments will be run, the paper should explicitly present the result as:

"a 100-example GSM8K audited slice where the gain is small, sign-consistent across three seeds, and only interpretable under an explicit runtime contract."

That claim is honest and defensible. The broader one is not yet closed.
