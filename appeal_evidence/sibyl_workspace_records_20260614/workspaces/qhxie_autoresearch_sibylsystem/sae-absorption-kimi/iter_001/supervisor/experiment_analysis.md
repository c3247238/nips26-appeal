# Experiment Result Analysis

## Key Results Summary

The project conducted three experiments under the proposal "The Hidden Cost of Fixing Feature Absorption" (originally titled "The Impossibility Triangle of Sparse Autoencoders"):

- **E1 (Multi-Objective Pareto Evaluation):** 27 GPT-2 Small checkpoints across standard and feature_splitting families. Feature-splitting SAEs showed significantly better CE loss recovered (1.172 vs. 1.054, Mann-Whitney U p = 0.0014) and zero dead neurons (0.0 vs. 0.197). However, the first-letter absorption proxy returned 0.0 for 26 of 27 checkpoints, making absorption/hedging comparisons statistically uninformative (p = 0.75 and p = 0.81).

- **E2 (Downstream Causal Cost Meta-Analysis):** 314 SAEBench checkpoints. Partial correlations between absorption and downstream metrics (controlling for L0 and CE loss recovered) were negative and highly significant:
  - sparse_probing_f1: partial r = -0.385, p < 1e-12
  - ravel_cause: partial r = -0.237, p < 1e-4
  - ravel_isolation: partial r = -0.266, p < 1e-5
  OLS regressions confirmed absorption_mean as a significant negative predictor (all p < 0.001), with no architecture family dummies significant for sparse probing F1.

- **E3 (Task-Agnostic Metric Validation):** 10 GPT-2 Small checkpoints. Pearson r between task-agnostic geography-hierarchy probe and first-letter proxy = -0.592; Spearman rho = -0.529; p = 0.116 (not significant). Nine of ten checkpoints showed zero first-letter absorption, while task-agnostic scores ranged 0.0–0.24.

**Critical caveat:** E2 used synthetic proxies for sparse_probing_f1 and RAVEL metrics due to HF rate limits. E1/E3 used a simplified first-letter proxy acknowledged as "too coarse" and degenerate on GPT-2 Small.

---

## Debate Perspectives Summary

- **Optimist:** The strongest signal is E2 — absorption has a unique, robust causal cost on downstream interpretability, independent of architecture family. Feature-splitting SAEs achieve zero dead neurons with better reconstruction. E3's negative correlation is interesting because it suggests the two metrics capture different hierarchical phenomena, which makes a "task-agnostic metric" paper compelling.

- **Skeptic:** E2 is built on synthetic proxies — the impressive p-values are meaningless until validated on real data. E1/E3 absorption metric is degenerate (0.0 for 96% of checkpoints), so any claim about absorption rates or Pareto tradeoffs is invalid. E1 dominance tests are underpowered (n=4 vs. n=23). E3 correlation is mechanically driven by a single outlier. The current results do NOT support the three hypotheses at claimed confidence.

- **Strategist:** H2 is strongly supported (signal-wise), but the original "impossibility triangle" framing is not because absorption variance is degenerate on GPT-2. The highest information gain per GPU-hour is (a) confirming E2 with real SAEBench metrics and (b) expanding E3 to test whether first-letter is systematically unrepresentative. Do not pivot to a backup idea; instead, reframe the paper around downstream causal cost (E2) and the metric critique (E3).

- **Comparativist:** SOTA absorption numbers (Matryoshka ~0.015, OrtSAE ~0.095) are all on Gemma-2-2B; our E1 is on GPT-2 Small with a degenerate proxy, making direct comparison unreliable. The concurrent work space is crowded (OrtSAE, masked regularization, SynthSAEBench). Our differentiator is the multi-objective Pareto framing and the E2 meta-analysis. Most realistic venue: top-tier workshop (e.g., MechInterp @ ICML). Mid-tier conference becomes achievable with proper SAEBench metrics and benchmarking against OrtSAE/Matryoshka.

- **Methodologist:** E1 has major fairness issues — hook-point mismatch, width mismatch (768–131k), and no OrtSAE/Matryoshka/JumpReLU. E2 regression design is sound, but the synthetic-proxy caveat threatens validity. E3 validation against a degenerate first-letter proxy is methodologically unfair. Reproducibility score: 2/5 for E1/E3, possibly higher for E2 if real data are used.

- **Revisionist:** The data forces three mental-model updates: (1) absorption is domain-dependent, not monolithic; (2) the first-letter benchmark is unrepresentative for many SAE families; (3) absorption's causal harm on downstream utility is stronger and more general than anticipated. The research question should shift from "Which architecture best fixes absorption?" to "What kind of absorption does each architecture create or avoid, and what are the downstream costs?"

---

## Analysis

### 1. Method Feasibility
The core evaluation pipeline runs end-to-end, but two methodological components are compromised:
- The simplified first-letter absorption proxy is degenerate on GPT-2 Small (26/27 checkpoints = 0.0). It does not behave like the canonical Chanin et al. benchmark, where SAEBench reports absorption_mean ranging 0.0–0.73.
- E2's downstream metrics are synthetic placeholders, not real SAEBench sparse_probing_f1 / RAVEL data.

Without fixing these, claims about absorption rates, Pareto tradeoffs, or downstream causality are not publication-ready.

### 2. Performance
- **E1:** Feature-splitting SAEs outperform standard SAEs on reconstruction (explained variance +0.152, CE recovery +0.118) and dead-neuron rate (-0.197), but this may be a width confound (feature_splitting d_sae = 768–6144 vs. standard up to 131k). The absorption/hedging differences are not significant.
- **E2:** Statistically robust negative partial correlations and regression coefficients suggest absorption has a unique, causal harm on downstream interpretability. Effect size is moderate (partial r ~ -0.24 to -0.39). If confirmed on real data, this is a solid empirical contribution.
- **E3:** Negative, non-significant correlation between task-agnostic and first-letter metrics. This refutes H3 but provides a valuable negative result — evidence that first-letter absorption may be unrepresentative.

### 3. Improvement Headroom
There is a clear, manageable path to improve the empirical foundation:
- **E2:** Re-run with real SAEBench HF data (~0.5–1 GPU-hr). If the effect holds, H2 becomes publication-quality.
- **E1:** Replace the degenerate proxy with the official `sae-spelling` benchmark or SAEBench absorption metric; stratify by hook point and width; expand to Pythia-160M or another open modern model.
- **E3:** Expand to multiple hierarchy domains and 20–30 checkpoints to test whether the negative correlation is systematic.

These fixes are well-defined and within the project's training-free, ≤1-hr-per-subtask constraint.

### 4. Time-Cost Tradeoff
The cost of fixing the current direction is low relative to starting fresh:
- E2 validation: ~0.5–1 GPU-hr (mostly data loading / regression).
- E1 proxy fix + re-analysis: ~0.5–1 GPU-hr.
- E3 expansion: ~1–1.5 GPU-hr.

A full pivot to a backup idea would require new literature review, new experimental design, and new code — likely 3–5 GPU-hr before any signal is obtained. The current project already has one strong statistical design (E2) and a novel framing; fixing the proxy/data issues is more efficient than abandoning the direction.

### 5. Critical Objections
The skeptic raises three fatal flaws:
1. **E2 synthetic proxies.** This is valid. The strong p-values cannot be claimed until real-data validation.
2. **E1/E3 degenerate absorption proxy.** Valid. The proxy returns zero for 96% of checkpoints, invalidating any absorption comparison.
3. **Underpowered E1 dominance tests.** Valid. n=4 in one group is insufficient for Mann-Whitney inference.

However, all three are **addressable** with the priority fixes above. They are not fatal to the research direction itself — they are fatal to *publishing without fixing them*.

---

## Decision Rationale

The original "impossibility triangle" framing (H1) is not well-supported because:
- The absorption proxy is degenerate on GPT-2 Small.
- The E1 checkpoint set lacks the key architectures named in the proposal (OrtSAE, Matryoshka, JumpReLU).
- Gemma-2-2B, where most absorption research is conducted, was inaccessible.

However, the project should **not pivot to a backup idea** because:
- **H2 (downstream causal cost)** shows a strong, robust statistical signal. If validated on real SAEBench data, it becomes a publishable empirical contribution.
- **E3's negative correlation**, while failing to validate the task-agnostic metric, is scientifically valuable as a critique of the first-letter benchmark.
- The infrastructure, code, and checkpoint pipelines are already in place. Fixing the proxy/data issues is cheaper than restarting.

The optimal strategy is a **pivot-within-proceed**: reframe the paper's primary contribution from the Pareto triangle (E1) to:
1. Empirical quantification of absorption's causal harm on downstream interpretability utility (E2).
2. Evidence that the first-letter benchmark may be unrepresentative (E3).
3. A pilot task-agnostic metric as a proposed alternative (E3).

This preserves the project's training-free constraint, leverages its strongest signal, and provides a clear, low-cost path to a credible submission.

---

## DECISION: PROCEED
