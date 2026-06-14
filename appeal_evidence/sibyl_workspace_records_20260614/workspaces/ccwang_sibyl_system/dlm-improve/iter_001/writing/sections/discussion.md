# 5. Discussion

The evidence in this paper narrows the contribution to a more defensible shape. The strongest result is not a new controller. It is a diagnostic account of how revision claims should be evaluated in diffusion LMs.

## 5.1 Honest compute is a scientific claim, not a reporting nicety

The GSM8K comparison shows why nominal labels are not enough. A reader who only sees “64-step” or “67-step” method names can easily infer a cleaner comparison than the runtime evidence supports. Once actual NFE, latency, throughput, batch size, backend, and compile status are reported together, the Pareto story changes. This does not mean previous results are invalid; it means their interpretation is often too coarse. Future DLM revision papers should therefore treat runtime-fairness metadata as part of the main result table rather than as appendix residue.

## 5.2 Observer quality and controller quality must be separated

The signal audit suggests a useful reframing for the literature. Calibration, entropy, and instability are often described in a single language of “confidence signals,” but our results show that they answer different questions. Calibration is highly informative as an observer and still unconvincing as a controller. Entropy remains the most practical signal but is not transformative. Instability, at least in the tested form, does not justify a method-forward story. The practical consequence is simple: papers should report whether a signal predicts error well **before** claiming that it supports a strong intervention rule.

## 5.3 Task structure governs revision safety

The strongest negative result is not merely that code performance dropped. It is that the failure pattern is structured. Gating repaired syntax but not execution, which suggests that revision behaves differently when correctness depends on global structural constraints. Even within reasoning, MATH500 blocks a simple transfer story from GSM8K. Revision is therefore better understood as a task-dependent intervention with a recoverability boundary than as a generic improvement primitive.

## 5.4 Limits

Several limitations remain important.

- All main comparisons are single-seed.
- The detailed audits rely on `100`-example slices for GSM8K and MATH500 and `50` examples for HumanEval.
- We do not yet have a completed benefit-bucket audit separating draft-correct-then-harmed, draft-wrong-then-fixed, and no-effect cases.
- The shortlist includes a `CORE-proxy` rather than a full contemporary implementation of every concurrent revision method.

These limitations matter because they block stronger claims. In particular, the paper should not present itself as a benchmark standard-setter or imply a universal controller ranking.

## 5.5 Most valuable next steps

The next experiments should deepen the current diagnostic story rather than reopen a search for a new hero controller. The most valuable package is:

1. a benefit-bucket audit that turns aggregate comparisons into mechanism evidence;
2. a minimal seed-sensitivity spot-check on the most important pairwise comparisons;
3. appendix-grade runtime fairness and asset-lineage tables.

This ordering follows directly from the current evidence. The paper already has a stable diagnostic backbone; what it needs next is more uncertainty accounting and finer-grained failure decomposition.

<!-- FIGURES
- None
-->
