# 4. Experiments

We report results in the same order as the paper's claims: honest compute on GSM8K, observer-controller mismatch, and task-dependent revision behavior across MATH500 and HumanEval.

## 4.1 Setup

Our primary evidence comes from the diagnostic assets produced in the current workspace. The GSM8K honest-compute comparison and the signal audit each use `100` examples for detailed runtime analysis. The MATH500 shortlist also uses `100` examples. The HumanEval boundary study uses `50` examples. Although these slices are sufficient for a diagnostic paper, they remain a limitation for any stronger benchmarking claim, and all cross-benchmark conclusions should be read with that slice size in mind.

Across all reported runs, runtime settings vary materially. `Standard-64`, `DNB-84`, and `Prophet-64` on GSM8K run with compilation enabled, while `CORE-proxy-64`, `Entropy-Revise-64+3`, and `TIGER-Instability-64+3` do not. Batch size ranges from `115` for standard denoising to `1` for `CORE-proxy-64`. These differences are not implementation trivia; they are part of the result and motivate the honest-compute protocol.

## 4.2 Honest compute changes key comparisons on GSM8K

Table 1 reports the core GSM8K comparison. Two points matter most. First, nominal labels and actual compute do not align cleanly. Second, those mismatches change the comparison story.

| Method | Accuracy | Nominal NFE | Actual NFE | Latency (s) | Tokens/s | Batch | Backend | Compile |
|--------|----------|-------------|------------|-------------|----------|-------|---------|---------|
| Standard-64 | 0.36 | 64 | 64.00 | 157.04 | 163.01 | 115 | eager | yes |
| DNB-84 | 0.36 | 84 | 83.00 | 180.61 | 141.75 | 115 | eager | yes |
| Prophet-64 | 0.34 | 64 | 63.93 | 147.13 | 173.99 | 57 | eager | yes |
| CORE-proxy-64 | **0.46** | 64 | 69.00 | 482.95 | 53.01 | 1 | eager | no |
| Entropy-Revise-64+3 | 0.39 | 67 | 68.00 | 210.67 | 121.51 | 32 | eager | no |
| TIGER-Instability-64+3 | 0.39 | 67 | 69.00 | 213.81 | 119.73 | 32 | eager | no |

**Table 1.** Matched-compute comparison on GSM8K. Bold marks the best accuracy, not a full Pareto winner.

The most concrete reorder appears between `CORE-proxy-64` and `Entropy-Revise-64+3`. Nominally, `CORE-proxy-64` sits ahead of entropy revision in the compute ordering; once actual NFE is counted, it falls behind. That shift is only one reorder, but it is exactly the kind of change that affects pairwise interpretation and Pareto conclusions. Figure 2 visualizes this point by plotting quality against latency rather than relying on method names alone.

The section also clarifies what we do **not** claim. `CORE-proxy-64` still has the best raw GSM8K accuracy, and nothing in Table 1 invalidates that. The diagnostic point is narrower: once actual compute and runtime asymmetries are included, the apparent advantage of the method family changes shape. A paper that reported only nominal step counts would miss that shift.

## 4.3 Good observers do not reliably become good controllers

The signal audit in Figure 3 compares calibration, entropy, and instability as observers and controllers. The table below summarizes the headline quantities, while the accompanying text explains the off-table screening evidence that motivates the gap interpretation.

| Signal | Diagnostic score | Control effectiveness | Gap |
|--------|------------------|-----------------------|-----|
| Calibration | 0.6225 | 0.0000 | 0.6225 |
| Entropy | 0.4140 | 0.0000 | 0.4140 |
| Instability | 0.0555 | 0.0100 | 0.0455 |

**Table 2.** Diagnostic-control gap under the tested policies.

Calibration is the clearest example of mismatch. It has the strongest diagnostic score in the audit, yet it is not a deployed control policy in the shortlist and therefore contributes no direct control gain. Entropy remains diagnostically meaningful, but in the signal screen it does not outperform random revision: both reach `0.37` accuracy. Instability fares worse on the observer side and only marginally better on the controller side.

The shortlist comparison reaches the same conclusion from a second angle. On GSM8K, `Entropy-Revise-64+3` and `TIGER-Instability-64+3` tie at `0.39`. A more elaborate instability-themed controller therefore does not beat the simpler entropy baseline. Under the tested policies, the main lesson is not “instability wins,” but “controller sophistication does not rescue weak observer-controller alignment.”

## 4.4 Revision response is task-dependent

We next ask whether the GSM8K revision story transfers. It does not.

| Method | GSM8K | MATH500 |
|--------|-------|---------|
| Standard-64 | 0.36 | **0.23** |
| CORE-proxy-64 | **0.46** | 0.21 |
| Entropy-Revise-64+3 | 0.39 | 0.19 |
| TIGER-Instability-64+3 | 0.39 | 0.20 |

**Table 3.** Ordering across two reasoning datasets. Bold marks the best result in each column.

GSM8K suggests that revision methods are competitive with the standard baseline. MATH500 does not. On that dataset, `Standard-64` leads the shortlist and both revision methods fall below `0.21`. Given the `n=100` slice, we interpret this as boundary evidence rather than a final leaderboard claim, but it is still strong enough to block a simple generalization from “revision helps on one reasoning benchmark” to “revision is a reasoning-wide improvement.”

HumanEval sharpens the boundary interpretation. Figure 4 and Table 4 show that gating improves syntax but not executable success.

| Method | Pass@1 | Syntax failure | Runtime failure |
|--------|--------|----------------|-----------------|
| Standard | **0.06** | 0.46 | 0.48 |
| Entropy/TIGER Ungated Revision | 0.02 | 0.48 | 0.50 |
| Gated TIGER | 0.04 | **0.28** | 0.68 |

**Table 4.** HumanEval boundary study.

The gate lowers syntax failure by `0.20` relative to ungated revision, but runtime failure worsens by `0.18`, and `pass@1` remains below the standard baseline. This is exactly the pattern we would expect if local repair mainly fixes shallow legality while leaving deeper semantic and execution constraints unresolved. That makes HumanEval useful not as a second headline success domain, but as boundary evidence that task structure governs whether revision is safe.

## 4.5 What the experimental evidence supports

Taken together, the experiments support three claims. Honest compute changes key comparisons on GSM8K. Strong observers do not reliably become strong controllers under the tested interventions. Revision response is task-dependent both across reasoning datasets and at the code boundary. What the experiments do **not** yet support is a field-wide ranking rewrite or a claim that any one controller dominates across settings.

<!-- FIGURES
- Figure 2: gen_fig2_honest_compute.py, fig2_honest_compute.pdf — GSM8K honest-compute scatter with actual NFE annotations and Pareto interpretation.
- Figure 3: gen_fig3_signal_gap.py, fig3_signal_gap.pdf — Observer quality versus controller gain for calibration, entropy, and instability.
- Figure 4: gen_fig4_code_boundary.py, fig4_code_boundary.pdf — HumanEval boundary breakdown across standard, ungated revision, and gated TIGER.
- Table 1: inline — GSM8K matched-compute comparison.
- Table 2: inline — Diagnostic-control gap summary.
- Table 3: inline — GSM8K versus MATH500 ordering.
- Table 4: inline — HumanEval boundary study.
-->
