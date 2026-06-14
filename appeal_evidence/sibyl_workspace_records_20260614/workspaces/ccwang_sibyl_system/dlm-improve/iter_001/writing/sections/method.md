# 3. Experimental Protocol and Diagnostic Definitions

This section defines the study design rather than a new decoding algorithm. Our goal is to evaluate training-free revision methods under a protocol that can support defensible claims about compute, signal quality, and task dependence.

## 3.1 Method family under study

We evaluate six representative methods:

- `Standard-64`: standard cosine denoising for 64 steps.
- `DNB-84`: additional standard denoising under a compute-matched budget.
- `Prophet-64`: a lightweight stopping-style baseline.
- `CORE-proxy-64`: a probe-based fragile-token baseline.
- `Entropy-Revise-64+3`: simple entropy-guided selective revision.
- `TIGER-Instability-64+3`: instability-guided selective revision.

The study uses this family to ask comparative questions. We do not treat any one member, including TIGER, as the paper's hero method.

## 3.2 Honest-compute protocol

The honest-compute protocol records both nominal and actual cost. For every run we log:

- nominal NFE implied by the method label;
- actual NFE measured at execution time;
- end-to-end latency;
- throughput in tokens per second;
- batch size;
- backend;
- compile status;
- peak VRAM when available.

We define the compute gap as the mismatch between nominal and actual NFE. This protocol matters because several methods incur hidden overhead or operate under very different runtime settings. On GSM8K, for example, `Standard-64` runs with `batch_size=115` and `compile_enabled=true`, whereas `CORE-proxy-64` runs with `batch_size=1` and `compile_enabled=false`. Any comparison that ignores those conditions risks overstating the fairness of the headline result.

The protocol is summarized in Figure X, which lays out the sequence from shortlist selection to honest-compute logging, observer-controller auditing, and task-dependent validation.

## 3.3 Observer and controller definitions

We distinguish observers from controllers.

- An **observer** is a signal that predicts error-prone or revision-sensitive positions.
- A **controller** is a policy that uses a signal to change the denoising trajectory.

For each signal $s$, we report a diagnostic score $d(s)$ and a control effectiveness score $g(s)$. Here, $d(s)$ measures how strongly the signal tracks error or revision opportunity in the diagnostic audit, while $g(s)$ measures the downstream gain obtained when that signal is used inside a tested intervention policy. In this paper, calibration is treated only as an observer, while entropy-guided revision and instability-guided revision instantiate controller families. The point of the factorization is to make mismatch measurable: a signal may be useful for diagnosis without yielding a correspondingly strong intervention policy.

## 3.4 Benchmarks and their roles

We assign different roles to the three benchmarks rather than treating them as interchangeable rows in a leaderboard.

- **GSM8K** is the headline reasoning benchmark for honest-compute comparison.
- **MATH500** is the transfer check on a second reasoning regime.
- **HumanEval** is a boundary benchmark for structured-output fragility.

This role assignment is deliberate. The paper's core argument is not “one method wins everywhere,” but “what survives once we compare methods honestly and test them against different structural demands.”

## 3.5 Failure taxonomy

For code, aggregate `pass@1` is too coarse. We therefore track a minimal failure taxonomy with at least two modes:

- syntax failure;
- runtime failure.

This minimal taxonomy lets us distinguish shallow repair from functional recovery. In the HumanEval boundary study, gating reduces syntax failure sharply but does not improve runtime success, revealing that local repair and executable correctness are different targets.

## 3.6 Scope and non-claims

The protocol supports three classes of claim:

1. whether honest compute changes key comparisons;
2. whether a signal is a better observer than controller under the tested policies;
3. whether revision behavior transfers across task structure.

It does **not** support broader claims such as a universal ranking rewrite, a universally best controller, or a benchmark-standard verdict for the whole field. Those stronger claims would require broader seeds, more datasets, and a completed benefit-bucket audit.

<!-- FIGURES
- Figure X: fig_protocol_flow_desc.md — Flow description of the honest-compute study design and observer-controller audit.
-->
