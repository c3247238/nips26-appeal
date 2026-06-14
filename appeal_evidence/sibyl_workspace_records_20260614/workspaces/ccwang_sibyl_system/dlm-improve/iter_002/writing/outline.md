# Iteration 2 Paper Outline

## Working Title

Honest Compute for Diffusion Language Models: Bucket-Level Outcomes and Runtime-Lineage Audits

## Abstract

### Claim 1

Aggregate revision gains are not self-explanatory; bucket-level outcomes reveal whether improvements come from fixing wrong drafts, preserving correct drafts, or merely leaving errors unchanged.

### Claim 2

These conclusions are only credible when realized compute, runtime path, and claim-to-asset lineage are explicit and auditable.

### Explicit downscope

- The seed result is a minimal sign-consistency check, not a full robustness claim.
- The paper is not proposing a new controller family.

## 1. Introduction

### Motivation

- Diffusion LM comparisons are vulnerable to nominal-vs-realized compute mismatch.
- Observer-side diagnostics and controller-side gains are often conflated.

### Framing sentence

Observer quality does not automatically translate into controller gain, so observer diagnostics, controller outcomes, and runtime execution should be reported as separate protocol objects.

### Contributions

1. A bucket-level audit of the GSM8K headline pair showing `fixed / harmed / no_effect`.
2. A runtime-lineage protocol bundle that makes honest-compute claims auditable.
3. A minimal seed sign-consistency check for the headline pair.

## 2. Protocol Perspective

### 2.1 Observer / controller / runtime split

Use:

- `exp/results/observer_controller_protocol.json`
- `plan/signal_audit_protocol.md`

### 2.2 Honest-compute reporting contract

Use:

- `exp/results/runtime_probe_iter2.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/canonical_asset_manifest.json`

Key point:

- method names are insufficient; report `actual_nfe`, latency, batch size, backend, and compile status

## 3. Experimental Setup

### Main slice

- benchmark: GSM8K
- headline pair: `Standard-64` vs `Entropy-Revise-64+3`
- runtime path: `eager|compile=True`
- safe batch size: `57`

### Boundary slices

- reasoning boundary: `diag_math500_shortlist.json`
- code boundary: `diag_humaneval_guard_boundary.json`

State clearly that these are positioning evidence only.

## 4. Results

### 4.1 Runtime fairness first

Lead with:

- `exp/results/diag_compute_curve_gsm8k.json`
- `exp/results/runtime_fairness_matrix.json`

Main message:

- nominal compute labels and realized compute order can diverge

### 4.2 Bucket-level audit

Lead with:

- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/benefit_bucket_examples_pilot.json`

Main numbers:

- coverage `100%`
- `fixed = 7`
- `harmed = 4`
- `no_effect = 89`
- net gain `+3pp`

### 4.3 Minimal seed check

Lead with:

- `exp/results/seed_sensitivity_spotcheck.json`

Main message:

- all three seeds preserve the same sign: `entropy_better`
- deltas: `[+0.03, +0.01, +0.01]`

## 5. Discussion

### 5.1 What the paper now supports

- bucket-level outcomes explain the aggregate gain structure
- runtime-lineage makes the comparison auditable
- observer/controller non-equivalence is a reporting requirement, not a new method claim

### 5.2 What the paper does not support

- no full robustness claim
- no universal cross-task gain claim
- no new controller contribution

### 5.3 Optional probe disposition

Use:

- `exp/results/min_controller_decoupling_probe.json`

State that the optional minimal controller probe was intentionally closed as `NO_GO` to avoid story drift and fresh confounds.

## 6. Appendix

### Appendix A

Claim-to-asset lineage table from:

- `exp/results/canonical_asset_manifest.json`

### Appendix B

Runtime fairness table from:

- `exp/results/runtime_fairness_matrix.json`

### Appendix C

Observer/controller protocol definitions from:

- `exp/results/observer_controller_protocol.json`
- `plan/signal_audit_protocol.md`
