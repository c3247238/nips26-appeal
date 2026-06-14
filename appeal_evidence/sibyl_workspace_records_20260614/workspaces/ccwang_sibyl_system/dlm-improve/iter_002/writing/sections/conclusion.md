# 6. Conclusion

This paper reframes training-free DLM revision as a compute-normalized diagnostic problem rather than a search for a new hero controller. The main headline pair on GSM8K improves by `+3` percentage points, but the contribution of that number lies in how it is explained and constrained: the gain decomposes into `7 fixed`, `4 harmed`, and `89 no_effect` examples; the runtime path is made explicit through a protocol bundle; and a minimal three-seed spot-check shows directional consistency without licensing a stronger robustness claim.

Under this framing, the durable contribution is not a new intervention family. It is a reporting contract for when a revision gain should be believed. Aggregate deltas need bucket-level explanation. Runtime claims need lineage and fairness metadata. Minimal robustness checks should constrain the language of the conclusion instead of being retrofitted after the fact.

The paper is therefore intentionally narrower than a benchmark or controller paper, but also more auditable. Future work can expand the robustness budget, enlarge the task coverage, and deepen the sample-level taxonomy. What the present work offers is the smaller but more reliable step: a protocol for interpreting revision gains in diffusion language models without hiding their structure or their execution conditions.

<!-- FIGURES
- None
-->
