# 3. Protocol Perspective

This section defines the paper's protocol objects. The goal is not to introduce a new model or controller, but to specify how evidence should be separated and reported for a training-free DLM revision study.

## 3.1 Observer, controller, and runtime are different objects

We distinguish three layers.

1. **Observer layer.** An observer assigns a signal to a draft state, such as entropy or another uncertainty-derived score. Its role is descriptive: it indicates where the draft may be fragile or incorrect.
2. **Controller layer.** A controller converts some observer-side information into an intervention policy, for example by remasking a subset of tokens and running additional denoising steps.
3. **Runtime layer.** The runtime layer records how the policy is actually executed on hardware and software: backend, compilation path, batch size, latency, and realized forward evaluations.

This separation is the central protocol move of the paper. A strong observer does not automatically imply a strong controller, and neither one implies that the realized compute path is fair. We therefore report the three layers through separate artifacts rather than collapsing them into one benchmark table.

## 3.2 Headline pair and bucket-level outcome decomposition

Our main comparison is the GSM8K headline pair `Standard-64` versus `Entropy-Revise-64+3`. For each sample, the comparison is placed into one of three mutually exclusive outcome buckets:

- `fixed`: the revised policy corrects an answer that the standard policy got wrong,
- `harmed`: the revised policy changes a correct standard answer into an incorrect one,
- `no_effect`: both policies end with the same correctness status.

This decomposition is intentionally minimal. It does not claim a complete failure taxonomy. Instead, it provides the smallest structure needed to explain a nonzero aggregate delta at the sample level. In `iter_002`, the headline gain is explained by `7 fixed`, `4 harmed`, and `89 no_effect` samples over the audited slice.

## 3.3 Honest-compute reporting contract

A revision comparison is only credible when nominal method names are accompanied by execution metadata. Our reporting contract therefore requires the following fields to be explicit whenever a result is used in the paper:

- nominal and realized denoising steps,
- backend and compilation status,
- safe batch size on the actual host,
- latency or throughput,
- claim-to-asset lineage from the prose statement to a concrete artifact.

The role of this contract is not to define a universal benchmark standard. Its role is narrower: it prevents a paper from implying a cleaner comparison than the underlying execution path actually supports.

## 3.4 Minimal robustness closure

The present paper uses a minimal three-seed spot-check as a quality gate rather than a full robustness claim. The criterion is directional stability of the headline delta. If the sign flips under the minimal seed set, the headline gain should be treated as fragile. If the sign is preserved, the result may be reported as sign-consistent, but not as fully robust.

This design is deliberately conservative. It formalizes what the current evidence supports while preventing the paper from over-claiming statistical stability.

<!-- FIGURES
- Figure 2: observer-controller-runtime split diagram based on observer_controller_protocol.json and signal_audit_protocol.md.
-->
