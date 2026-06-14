# 4. Experiments

We present the evidence in the same order as the paper's claims: runtime closure, bucket-level gain structure, minimal seed stability, and optional probe disposition. Boundary evidence from earlier reasoning and code slices is used only for positioning and is not promoted to a new headline.

## 4.1 Setup

The main audited slice is GSM8K with the headline pair `Standard-64` versus `Entropy-Revise-64+3`. All new assets in `iter_002` are tied to concrete JSON artifacts. The runtime probe records the feasible path on the current host, the bucket audit explains the net accuracy delta, the protocol bundle maps claims to assets, and the seed spot-check constrains the strength of the headline wording.

The runtime probe yields two operational facts that matter for all later interpretation. The recommended path on the present host is `eager|compile=True`, and the safe batch size for the audited GSM8K slice is `57`. Flash attention is not available in the current environment. These facts are not implementation trivia: they define the execution envelope under which the audited results should be read.

## 4.2 Runtime fairness must be explicit before the gain is interpretable

The runtime-lineage bundle turns the comparison into an auditable object. Rather than reporting only a nominal step label, we map each main claim to an artifact and to the execution fields required to interpret it. The key takeaway is simple: a revision claim is only meaningful if the reader can inspect the realized path used to obtain it.

For the current host, the important runtime statement is not that every method can be compared under an idealized setting, but that the actual audited path is explicit. `eager|compile=True` is the feasible path, the safe batch size is `57`, and the protocol files make it possible to trace a sentence in the paper back to a concrete runtime artifact. This is the minimum condition for what we call an honest-compute comparison.

## 4.3 Bucket-level outcomes explain the aggregate gain

The main empirical result is the benefit-bucket audit of the GSM8K headline pair. The audit covers the entire reviewed slice (`coverage = 100%`) and reports:

- `standard_accuracy = 0.34`,
- `entropy_accuracy = 0.37`,
- `accuracy_delta = +0.03`,
- `fixed = 7`,
- `harmed = 4`,
- `no_effect = 89`.

This decomposition matters because the aggregate gain is small enough that it can easily be over-interpreted. The bucket view shows that the gain is neither a broad rewrite of behavior nor a pure preservation story. Instead, the revised policy helps on a narrow subset, hurts on a smaller subset, and leaves the majority unchanged. That is exactly why the paper treats bucket-level outcomes as the main evidence layer.

Representative examples are stored separately so that the bucket labels are not just counts. The purpose of those examples is diagnostic rather than anecdotal: they reveal what kinds of corrections and regressions are actually responsible for the net gain.

## 4.4 Minimal seed closure supports sign consistency, not full robustness

The minimal seed spot-check evaluates the same headline pair across seeds `40`, `41`, and `42`. The resulting deltas are:

- seed `40`: `+0.03`,
- seed `41`: `+0.01`,
- seed `42`: `+0.01`.

All three runs preserve the same direction, `entropy_better`, so the headline result is sign-consistent under this minimal check. We explicitly stop there. The experiment does not justify a full robustness claim, a variance estimate for all downstream analyses, or any general statement about stability across tasks. Its role is to prevent the paper from depending on a single lucky seed.

## 4.5 Optional controller expansion was intentionally closed

The optional minimal controller probe was not executed as a new experimental branch. Instead, it was closed through an explicit `NO_GO` artifact. This is an important positive decision rather than missing work. By the time runtime, bucket, and seed evidence were all in place, an additional controller variant would have changed the story more than it would have improved the evidence. The paper therefore stops at the current closure and treats observer-controller separation as a reporting and interpretation constraint, not as an invitation to introduce a new controller family.

## 4.6 What the evidence supports

Taken together, the experiments support three safe claims. First, the GSM8K headline gain is explainable at the bucket level rather than only as an average. Second, runtime-lineage and claim-to-asset mappings are necessary to make that gain auditable. Third, the direction of the gain survives a minimal three-seed spot-check. What the evidence does not support is a universal cross-task improvement claim, a new controller contribution, or full multi-seed robustness.

<!-- FIGURES
- Table 1: runtime fairness summary from runtime_probe_iter2.json and runtime_fairness_matrix.json.
- Figure 3: bucket-level fixed/harmed/no_effect decomposition from benefit_bucket_audit_pilot.json.
- Table 2: seed sign-consistency summary from seed_sensitivity_spotcheck.json.
-->
