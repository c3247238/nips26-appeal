# When Revision Helps Diffusion LMs, When It Hurts, and Why Honest Compute Matters

## Abstract

Training-free revision methods for diffusion language models are often compared by nominal step counts and justified by uncertainty signals that are never cleanly separated into diagnostic and control roles. We revisit this literature as a compute-normalized diagnostic study rather than a controller paper. Across a focused shortlist of six training-free methods, we show three things. First, honest compute accounting changes key pairwise comparisons and Pareto conclusions on GSM8K: nominal labels hide meaningful differences in actual NFE, latency, throughput, batch size, and compile status. Second, strong diagnostic signals do not reliably become strong controllers under the tested policies: calibration is the strongest observer in our audit, entropy is the most practical signal, and instability does not justify a method-forward win. Third, revision response is task-dependent: the GSM8K story does not transfer cleanly to MATH500, and code results show that shallow syntax repair does not recover execution success. The resulting contribution is a diagnostic study, evaluation protocol, and failure taxonomy for revision in diffusion language models.

## 1. Introduction

Training-free inference has become the most active improvement path for diffusion language models (DLMs). As masked diffusion systems such as LLaDA and Dream matured into credible open-weight models, the frontier shifted away from proving that diffusion language modeling is possible and toward deciding how to decode these models efficiently and reliably. That shift produced a growing ecosystem of stopping policies, remasking rules, and fragile-token heuristics. Yet the literature still tends to conflate three different questions: how much compute a method really uses, whether a signal is a good observer of error, and whether that same signal remains useful once converted into a controller.

This paper argues that these questions should be separated. We do not present another hero controller. Instead, across a focused six-method shortlist and three benchmark roles, we show that a credible study of revision in DLMs must report honest compute, audit observer-controller mismatch, and test whether revision survives a change in task structure. Figure 1 previews the three corresponding takeaways: a compute-order shift on GSM8K, a diagnostic-versus-control gap for the main signals, and a code-boundary failure pattern on HumanEval.

The empirical motivation is straightforward. On GSM8K, `CORE-proxy-64` reaches the best raw shortlist accuracy (`0.46`), but it also runs with `actual_nfe=69`, `latency=482.95s`, `batch_size=1`, and `compile=false`, whereas `Entropy-Revise-64+3` reaches `0.39` with `actual_nfe=68`, `latency=210.67s`, `batch_size=32`, and `compile=false`. Nominal labels therefore hide a meaningful shift in Pareto interpretation. On the signal side, calibration is the strongest observer in the audit (`d(s)=0.6225`) but offers no deployed control gain, while entropy is useful yet only marginally better than random revision in the lightweight screen. On the task side, GSM8K supports a superficially positive revision story, but MATH500 and HumanEval block a simple generalization: `Standard-64` leads the MATH500 shortlist at `0.23`, and `Gated TIGER` lowers syntax failure on HumanEval while still trailing `Standard` on `pass@1` (`0.04` vs. `0.06`) and worsening runtime failure.

We therefore position the paper as a compute-normalized diagnostic study with three contributions:

1. We introduce an honest-compute comparison protocol for training-free DLM revision methods, reporting actual NFE, latency, throughput, batch size, backend, and compile status alongside task metrics.
2. We show that stronger observation does not reliably become stronger control under the tested interventions, motivating an explicit observer/controller split.
3. We provide task-dependent boundary evidence across GSM8K, MATH500, and HumanEval, arguing that revision behavior depends on recoverability and structural constraint.

Figure 1 is the paper's teaser because the paper itself is diagnostic: the left panel previews the compute-order shift, the middle panel previews the observer-controller gap, and the right panel previews the code-boundary failure.

> **Figure 1.** Composite teaser summarizing the paper's three claims. Left: headline step counts are not enough for GSM8K comparison. Middle: observer quality and controller gain split apart in the signal audit. Right: syntax repair does not restore execution success at the HumanEval boundary.

## 2. Related Work

The relevant literature spans three threads. The first is the DLM architecture line itself, from early discrete diffusion systems to scaled masked diffusion models such as LLaDA and Dream. The second is training-free acceleration and selective computation, including stopping, skipping, caching, and parallel unmasking strategies. The third is revision and remasking, where methods revisit positions deemed fragile, uncertain, or unstable after a draft pass.

Our paper differs from all three lines in emphasis. It does not introduce a new diffusion backbone. It does not present a new acceleration primitive. It does not claim a new best revision controller. Instead, it asks what conclusions remain once these methods are compared under a shared runtime-fairness protocol and once their signals are interpreted separately as observers and controllers.

This framing also changes how calibration enters the story. Prior work often uses confidence or entropy as operational inputs to a policy. Here, calibration is primarily a measurement object. Its value is diagnostic: it helps explain why strong uncertainty signals do not automatically translate into strong intervention rules under iterative denoising.

## 3. Experimental Protocol

### 3.1 Method family and benchmark roles

We study six representative methods:

- `Standard-64`
- `DNB-84`
- `Prophet-64`
- `CORE-proxy-64`
- `Entropy-Revise-64+3`
- `TIGER-Instability-64+3`

We deliberately assign different roles to the benchmarks rather than treating them as interchangeable leaderboard rows:

- GSM8K is the headline reasoning benchmark for honest-compute comparison.
- MATH500 is the transfer check on a second reasoning regime.
- HumanEval is the structural boundary test.

### 3.2 Honest-compute variables

For every run we track nominal NFE, actual NFE, latency, throughput, batch size, backend, compile status, and peak VRAM when available. This protocol is necessary because methods with similar names operate under materially different runtime conditions. A comparison that ignores these variables risks overstating both fairness and efficiency.

### 3.3 Observer-controller split

We use the language of observers and controllers throughout the paper. An **observer** is a signal that tracks likely error or revision opportunity. A **controller** is a policy that acts on that signal. For each signal $s$, we report a diagnostic score $d(s)$ and a control-effectiveness score $g(s)$. Calibration is treated only as an observer in this paper. Entropy-guided and instability-guided revision instantiate controller families.

Figure 2 summarizes the full protocol flow. The current workspace contains a detailed flow description, but the rendered figure will be finalized during LaTeX integration.

[TODO: Figure 2 rendered from `fig_protocol_flow_desc.md` during LaTeX integration.]

> **Figure 2.** Protocol flow diagram for the honest-compute shortlist, the observer-controller audit, and the HumanEval boundary test.

## 4. Results

### 4.1 Honest compute changes key comparisons on GSM8K

Table 1 gives the main GSM8K comparison under honest compute.

| Method | Accuracy | Nominal NFE | Actual NFE | Latency (s) | Tokens/s | Batch | Backend | Compile |
|--------|----------|-------------|------------|-------------|----------|-------|---------|---------|
| Standard-64 | 0.36 | 64 | 64.00 | 157.04 | 163.01 | 115 | eager | yes |
| DNB-84 | 0.36 | 84 | 83.00 | 180.61 | 141.75 | 115 | eager | yes |
| Prophet-64 | 0.34 | 64 | 63.93 | 147.13 | 173.99 | 57 | eager | yes |
| CORE-proxy-64 | **0.46** | 64 | 69.00 | 482.95 | 53.01 | 1 | eager | no |
| Entropy-Revise-64+3 | 0.39 | 67 | 68.00 | 210.67 | 121.51 | 32 | eager | no |
| TIGER-Instability-64+3 | 0.39 | 67 | 69.00 | 213.81 | 119.73 | 32 | eager | no |

**Table 1.** GSM8K matched-compute shortlist.

The main diagnostic point is not that `CORE-proxy-64` stops being strong. It remains the best raw-accuracy point in the shortlist. The point is narrower and more important: once actual compute is counted, one explicit reorder appears between `CORE-proxy-64` and `Entropy-Revise-64+3`, and the Pareto story becomes much less clean than the nominal labels suggest. Figure 3 visualizes this result directly by plotting accuracy against latency and annotating actual NFE.

> **Figure 3.** Pareto view under honest compute. `CORE-proxy-64` remains best on raw GSM8K accuracy but has far higher latency than the revision baselines, and the actual-compute ordering differs from the nominal one.

### 4.2 Good observers do not reliably become good controllers

The signal audit separates diagnostic quality from intervention gain.

| Signal | Diagnostic score | Control effectiveness | Gap |
|--------|------------------|-----------------------|-----|
| Calibration | 0.6225 | 0.0000 | 0.6225 |
| Entropy | 0.4140 | 0.0000 | 0.4140 |
| Instability | 0.0555 | 0.0100 | 0.0455 |

**Table 2.** Diagnostic-control gap under the tested policies.

Calibration is the cleanest example of mismatch: it is the strongest observer and not a useful controller in the current shortlist. Entropy remains diagnostically meaningful, but the lightweight screen shows little gain over random revision. Instability motivates TIGER, yet its observer evidence is weak and its controller does not beat the entropy baseline on GSM8K. Figure 4 makes this observer-controller separation visually explicit.

> **Figure 4.** Observer quality versus controller gain for calibration, entropy, and instability. The best observer is not the best controller under the tested policies.

### 4.3 Revision response is task-dependent

The GSM8K story does not transfer cleanly. Table 3 compares the reasoning shortlist across GSM8K and MATH500.

| Method | GSM8K | MATH500 |
|--------|-------|---------|
| Standard-64 | 0.36 | **0.23** |
| CORE-proxy-64 | **0.46** | 0.21 |
| Entropy-Revise-64+3 | 0.39 | 0.19 |
| TIGER-Instability-64+3 | 0.39 | 0.20 |

**Table 3.** Ordering across two reasoning datasets.

Given the `n=100` slice, we treat MATH500 as boundary evidence rather than a definitive leaderboard. Even so, it is sufficient to block a simple generalization from “revision helps on one reasoning benchmark” to “revision is a reasoning-wide improvement.”

HumanEval sharpens the structural interpretation. Gating repairs syntax but not executable success.

| Method | Pass@1 | Syntax failure | Runtime failure |
|--------|--------|----------------|-----------------|
| Standard | **0.06** | 0.46 | 0.48 |
| Entropy/TIGER Ungated Revision | 0.02 | 0.48 | 0.50 |
| Gated TIGER | 0.04 | **0.28** | 0.68 |

**Table 4.** HumanEval boundary evidence.

The gate lowers syntax failure by `0.20` relative to ungated revision, but runtime failure worsens by `0.18`, and `pass@1` remains below the standard baseline. Figure 5 visualizes that shallow-fix / deep-failure pattern.

> **Figure 5.** Code boundary failure breakdown across Standard, ungated revision, and Gated TIGER. Syntax improves, but execution success does not recover.

## 5. Discussion

The paper's contribution is strongest when read as a diagnostic study. Honest compute is not a reporting nicety; it is part of the scientific claim. Signals are not interchangeable because their observer and controller roles differ. And revision cannot be discussed independently of task structure.

Three cautionary points follow directly from the evidence.

First, the honest-compute result should be stated modestly: actual compute changes key comparisons and Pareto conclusions. It does not support a universal ranking rewrite. Second, the observer-controller mismatch should be scoped to the tested policies. Third, the cross-benchmark results remain slice-based (`n=100` for GSM8K and MATH500 diagnostics, `n=50` for HumanEval boundary evidence), so the task-dependence story is strong enough to guide framing but not yet strong enough to justify benchmark-standard language.

HumanEval belongs in the paper not as a second success domain but as structural stress-test evidence. That framing matters. The result is more interesting than “the method failed on code”; it shows that local repair can improve legality without repairing executable semantics when correctness depends on globally constrained structure.

The highest-value next steps are therefore diagnostic rather than algorithmic:

1. a benefit-bucket audit separating harmed, fixed, and no-effect cases;
2. a minimal seed-sensitivity spot-check on the core pairwise comparisons;
3. appendix-grade runtime fairness and asset-lineage tables.

## 6. Conclusion

We revisited training-free revision in diffusion language models as a diagnostic study rather than a controller paper. Across a focused shortlist, we showed that honest compute changes key comparisons on GSM8K, that strong diagnostic signals do not reliably become strong controllers under the tested policies, and that revision response is task-dependent across reasoning and code.

The resulting recommendation for the field is practical. Future DLM revision papers should report runtime-fairness metadata, separate observer from controller claims, and treat code as a structural stress test rather than as automatic evidence of generality. The current paper does not settle the whole ranking question, but it does provide a more credible protocol for asking it.

## Figures and Tables

- Figure 1: `fig1_teaser.pdf` — Teaser overview of honest compute, observer-controller mismatch, and code-boundary failure.
- Figure 2: `fig_protocol_flow_desc.md` — Protocol-flow specification for the integrated LaTeX figure.
- Figure 3: `fig2_honest_compute.pdf` — Honest-compute Pareto view on GSM8K.
- Figure 4: `fig3_signal_gap.pdf` — Observer quality versus controller gain.
- Figure 5: `fig4_code_boundary.pdf` — HumanEval boundary breakdown.
- Table 1: inline — GSM8K matched-compute shortlist.
- Table 2: inline — Diagnostic-control gap.
- Table 3: inline — GSM8K versus MATH500 ordering.
- Table 4: inline — HumanEval boundary evidence.
