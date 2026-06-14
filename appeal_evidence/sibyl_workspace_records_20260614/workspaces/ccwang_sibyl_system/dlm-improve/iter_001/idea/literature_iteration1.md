# Iteration 1 Literature Memo

## Goal

This literature pass is not a general DLM survey. It is scoped to the concrete failure modes from iteration 0:

- single-seed and single-benchmark evidence
- calibration leakage / in-sample calibration
- nominal NFE not matching real compute
- entropy-only revision failing badly on code
- missing strong concurrent baselines
- slow evaluation pipeline preventing the right experiments

The purpose of this memo is to turn those lessons into hard constraints for the next planning and experiment stages.

## Lessons We Must Carry Forward

From `iter_001/reflection/reflection.md`, `iter_001/codex/result_debate_review.md`, and `iter_001/idea/result_debate/*.md`, the non-negotiable lessons are:

1. The paper cannot keep the old "Calibration-Aware" framing. Calibration correction did not help.
2. A single GSM8K win is not enough for a broad method claim.
3. Held-out calibration is mandatory if calibration remains part of the story.
4. Actual compute must be reported, not just nominal step budgets.
5. Code failure is mechanistic, not cosmetic. We either fix it or narrow scope honestly.
6. Batch-size and throughput optimization are not optional; otherwise we will keep underpowering the science.

## Most Relevant 2025-2026 DLM Inference Papers

### 1. CORE: Context-Robust Remasking for Diffusion Language Models

- Source: arXiv:2602.04096
- Type: training-free inference-time revision
- Core idea: do not trust static confidence; stress-test tokens under masked-context perturbations and revise the most context-brittle ones
- Why it matters to us:
  - This is the strongest direct competitor to entropy-based revision.
  - It directly attacks our biggest weakness: static confidence/entropy can miss tokens that are wrong but locally confident.
  - It reports the largest gains on structure-sensitive code generation, exactly where CARD failed.
- Key implication:
  - Any next-iteration method paper must compare against CORE or explicitly explain why it is out of scope.
  - If we stay with revision, "context robustness" is a stronger story than "calibration correction."

### 2. Prophet: Diffusion Language Models Know the Answer Before Decoding

- Source: arXiv:2508.19982
- Type: training-free early-stop / early-commit decoding
- Core idea: many DLM answers stabilize well before full decoding; use top-2 confidence gap to stop early
- Why it matters to us:
  - This is the cleanest speed baseline in the current space.
  - It keeps code quality roughly intact, unlike our entropy revision.
  - It reframes DLM acceleration as an optimal stopping problem rather than a revision problem.
- Key implication:
  - For speed claims, Prophet is a must-compare baseline.
  - If our method adds revision overhead, we must show why it beats a simpler "stop early" strategy.

### 3. STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking

- Source: arXiv:2601.04205
- Type: training-free dynamic remasking / acceleration
- Core idea: use temporal variance and spatial deviance of token confidence instead of a fixed threshold
- Why it matters to us:
  - It shows the field is already moving beyond static confidence thresholds.
  - It supports the critique that one-shot entropy ranking is too myopic.
  - It includes both GSM8K and code benchmarks, with strong speedups and near-maintained quality.
- Key implication:
  - If we use uncertainty at all, we should treat it as a dynamic trajectory signal, not just a scalar ranking.
  - Static entropy-top-k is no longer an adequate SOTA target by itself.

### 4. RDD: Reversible Diffusion Decoding for Diffusion Language Models

- Source: arXiv:2602.00150
- Type: training-free rollback / reversible decoding
- Core idea: detect stagnation from irreversible block commitments, roll back, and selectively remask
- Why it matters to us:
  - It offers a mechanistic explanation for why early commitments damage later decoding.
  - It reports across math and code, not just a single reasoning benchmark.
  - It strengthens the narrative that local early mistakes are a core DLM inference bottleneck.
- Key implication:
  - Our next-iteration framing should engage with irreversible commitment and recovery, not only calibration.
  - RDD is another strong baseline if we pitch revision/recovery.

### 5. DCD: Deferred Commitment Decoding for Diffusion Language Models with Confidence-Aware Sliding Windows

- Source: arXiv:2601.02076
- Type: training-free decode scheduling / commitment control
- Core idea: defer commitment through confidence-aware sliding windows
- Why it matters to us:
  - It sits in the same family of "avoid premature commitment" methods.
  - It is a direct contemporaneous comparator for acceleration-quality tradeoffs.
- Key implication:
  - Our paper cannot act as if the main comparison set is only standard decoding + DNB.

### 6. Saber: Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs

- Source: arXiv:2510.18165
- Type: training-free code-focused sampling
- Core idea: adaptive acceleration plus backtracking-enhanced remasking for code generation
- Why it matters to us:
  - Saber is highly relevant if code remains in scope.
  - It directly contradicts any claim that revision-like methods are inherently bad for code.
  - The right conclusion is not "revision hurts code" but "our entropy-only revision hurts code."
- Key implication:
  - If we keep HumanEval/MBPP in the headline story, Saber belongs in the baseline set.
  - If we cannot compare or improve on code, narrow scope to reasoning.

### 7. Parallelism and Generation Order in Masked Diffusion Language Models

- Source: arXiv:2601.15593
- Type: diagnostic / empirical study
- Core idea: MDLMs still lag because parallel probabilistic modeling weakens inter-token dependencies; generate-then-edit is supported as a promising direction
- Why it matters to us:
  - This supports the idea that task sensitivity is structural, not incidental.
  - It gives literature support for why code should be harder than math under local revision.
- Key implication:
  - Our code failure can be framed as a dependency issue, but only if we either fix it or explicitly scope it out.

### 8. Soft-Masked Diffusion Language Models

- Source: arXiv:2510.17206
- Type: training-based model modification
- Core idea: propagate partial information at masked positions instead of pure binary masking
- Why it matters to us:
  - Not a training-free baseline, but it shows that coding improvements may require richer structural signals than static uncertainty.
- Key implication:
  - Useful as a discussion point if we pivot from "final method paper now" to "diagnostic plus future directions."

## What The Literature Now Says About Our Position

### The old CARD story is too weak

The field has already moved toward:

- early stopping / deferred commitment
- dynamic remasking using token trajectories
- rollback / reversibility
- context-robust revision

Against that backdrop, "raw entropy top-k revision helps GSM8K" is too narrow as a standalone story.

### The diagnostic contribution is still salvageable

Our strongest defensible contribution remains:

- a DLM-specific diagnostic story about process-induced miscalibration / error accumulation
- when local uncertainty signals help
- when they fail
- why calibration correction itself is not the useful ingredient

This is consistent with both the internal debate and the recent literature: the strongest current papers are mechanism-driven, not just score-driven.

### Code is the main scope decision

The new literature weakens any attempt to bury HumanEval:

- CORE claims strong gains on structure-sensitive code tasks
- Saber explicitly targets code
- Prophet preserves code quality via safer early-stop behavior

So our next iteration must choose one of two honest paths:

1. Keep code in scope and add a real fix or strong comparator.
2. Drop universal claims and position the method as reasoning-focused.

There is no credible middle ground anymore.

## Protocol Rules For Iteration 1

These rules are now mandatory for planning and experiments.

1. Fit any calibrator on a held-out calibration split only.
   No reuse of the test subset used for headline evaluation.

2. Report actual compute, not just budgeted compute.
   At minimum: actual NFE, throughput, latency, and batching configuration.

3. Multi-seed the main claim.
   Minimum acceptable: 3 seeds on the primary positive benchmark, plus paired significance.

4. Add at least one stronger concurrent baseline beyond DNB.
   Priority order: Prophet, CORE, then STaRR or RDD depending implementation cost.

5. If code remains in scope, compare against a code-aware baseline.
   Priority: Saber or CORE. If we cannot do that, narrow scope to reasoning.

6. Do not make design decisions from pilot-only ablations if full-scale runs are affordable.
   This especially applies to remasking fraction, revision steps, and task gating.

7. Audit answer extraction and generation length.
   No more mixing pilot/full settings without explicit explanation.

8. Maximize batch size before any new experiment campaign.
   The literature now contains several stronger methods; wasting GPU on batch_size=1 is no longer scientifically defensible.

## Recommended Reframing Options

### Option A: Diagnostic / benchmark paper

Best if we want the most robust paper with current assets.

- Reframe around: when revision helps, when it hurts, and why calibration correction is not enough
- Add stronger baselines and proper protocol
- Emphasize compute-normalized evaluation and failure analysis

This path is the safest fit with the current evidence.

### Option B: Method pivot to context-aware or syntax-aware revision

Best if we want to stay method-forward.

- Replace entropy-only targeting with context-brittleness or structure-aware gating
- Focus on fixing code failure and strengthening benchmark coverage
- Still keep held-out calibration and compute accounting clean

This path is higher upside but requires new engineering and stronger experiments.

## Immediate Planning Consequences

For the next planning stage, the action order should be:

1. Throughput optimization first: batch search, flash attention, compile, multi-GPU split if available.
2. Rebuild evaluation protocol: held-out calibration split, actual compute accounting, paired significance.
3. Decide scope: reasoning-only vs reasoning+code.
4. Add strong baselines:
   - speed: Prophet
   - revision: CORE
   - optional extra: STaRR or RDD
5. Add a second meaningful reasoning benchmark:
   - highest priority: MATH500
   - acceptable alternatives: ARC-C or GPQA if implementation is cheaper
6. Rename the method / paper away from "Calibration-Aware."

## Source Shortlist

- CORE: `https://arxiv.org/abs/2602.04096`
- RDD: `https://arxiv.org/abs/2602.00150`
- STaRR: `https://arxiv.org/abs/2601.04205`
- Prophet: `https://arxiv.org/abs/2508.19982`
- DCD: `https://arxiv.org/abs/2601.02076`
- Saber: `https://arxiv.org/abs/2510.18165`
- Parallelism and Generation Order in MDLMs: `https://arxiv.org/abs/2601.15593`
- Soft-Masked Diffusion Language Models: `https://arxiv.org/abs/2510.17206`
