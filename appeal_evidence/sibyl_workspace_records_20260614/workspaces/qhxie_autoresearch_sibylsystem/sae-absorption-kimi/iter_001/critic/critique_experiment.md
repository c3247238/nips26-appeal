# Experiment Critique

## Experiment 1: Multi-Objective Pareto Evaluation

### Critical Flaw: Degenerate Absorption Proxy
The first-letter absorption metric returns exactly 0.0 on 26 of 27 checkpoints. This is not a finding that "absorption is low"—it is a finding that the *simplified proxy* lacks dynamic range. When a metric is degenerate across 96% of the sample, any statistical test on that metric is uninformative. The Mann-Whitney U test (U=48, p=0.75) does not mean Standard and feature_splitting have equivalent absorption; it means the proxy cannot resolve differences between them.

**The paper must not present this as evidence of "no significant advantage on absorption."**

### Family Collapse Obscures Real Patterns
The summary table compares only two aggregated families: "Standard" (n=23) and "feature_splitting" (n=4). But the 23 "Standard" checkpoints include resid_pre, resid_post, resid_mid, mlp_out, attn_out, and hook_z variants with very different behavior. For example, hook_z-kk-l8 has absorption=0.345, while attn-out SAEs show low hedging (0.28-0.32) and low CE recovery. Collapsing these into "Standard" masks potentially meaningful architectural variation.

### Hedging Proxy Is Too Coarse
The hedging score uses only 5 antonym pairs (good/bad, hot/cold, big/small, happy/sad, fast/slow). This is a toy proxy, not a robust metric. The pilot_summary.md explicitly calls it "coarse" and rates metric quality as "NO-GO for publication-ready numbers."

### Dead-Neuron Estimates Are Unreliable
Dead-neuron detection was run on only 2,048 tokens. The pilot summary notes that dead-neuron rates are "very high (33-88%)" and "likely because the pilot used only ~2k tokens." These numbers should not be presented as reliable.

### Positive Note
The pipeline itself works end-to-end and all 27 checkpoints loaded successfully. The *infrastructure* is sound; the *metrics* are not.

---

## Experiment 2: Downstream Causal Cost Meta-Analysis

### Major Update: H2 Is Falsified by New Full Results
Since the last critic round, E2 full GPT-2 and E2 full Pythia have been completed with the **official sae-spelling metric**. The results directly falsify H2's prediction that the first-letter benchmark would be degenerate on GPT-2 Small Standard/TopK:

- **GPT-2 Small Standard**: absorption_full ranges from 0.413 to 0.673
- **GPT-2 Small TopK**: absorption_full ranges from 0.036 to 0.411
- **Pythia-160M**: absorption_full ranges from 0.007 (GatedSAE trainer_0) to 0.579 (TopK trainer_0)

This is robust, meaningful variance. The paper must **stop claiming the first-letter benchmark is degenerate** and instead report H2 as falsified. The actual contribution of E2 is characterizing family-level and trainer-level variance in official absorption, not proving degeneracy.

### L0-Absorption Tradeoff Is Strong and Unexplored
The Pythia results reveal a striking pattern: low-L0 trainers (trainer_0, L0 ~20-50) show high absorption (0.20-0.58), while high-L0 trainers (trainer_3, L0 ~90-180) show low absorption (0.02-0.18). This classic sparsity-fidelity tradeoff may confound the H1 partial-correlation analysis if the relationship between L0 and absorption is non-linear. The current OLS model may not adequately capture this.

### absorption_full vs absorption_fraction Ambiguity
The E2 GPT-2 results show large divergences between absorption_full and absorption_fraction (e.g., Standard L4: 0.673 vs 0.388). The paper does not define these terms or explain which one supports which claim. This is a reproducibility gap.

### Strengths
This is the strongest experiment in the paper. It uses 314 SAEBench checkpoints, precomputed metrics, and appropriate statistical methods (partial correlation, OLS with cluster-robust SEs). The negative association between absorption and downstream utility is consistent across sparse probing F1, RAVEL Cause, and RAVEL Isolation.

### Weaknesses
1. **Observational design with causal framing.** The paper uses "causal cost" and "unique causal effect" in section titles and the abstract, but this is a meta-analysis of existing checkpoints. Unobserved confounders are not controlled.

2. **The absorption metric is still first-letter spelling.** While E2 benefits from a real, validated absorption measure, it is still the first-letter spelling metric. If E3 casts doubt on generalizability, then the E2 results are also potentially limited to that specific construct.

---

## Experiment 3: Task-Agnostic Absorption Metric Pilot

### Critical Reassessment: The Proxy, Not the Construct, May Be the Problem
The E3 pilot showed a negative correlation between the task-agnostic metric and the first-letter benchmark (r = -0.59). However, this comparison used the **simplified proxy**, which we now know is degenerate. Since the official sae-spelling metric (E2 full) shows robust variance, the E3 negative correlation may reflect **proxy invalidity** rather than a fundamental construct-validity problem.

**Recommendation:** Re-run E3 by correlating the task-agnostic metric against the *official* sae-spelling metric on the same 10 checkpoints. If the correlation improves, the problem was the proxy. If it remains negative, then there is a genuine construct-validity issue.

### Single-Outlier Dependence
Removing the TopK_Attn outlier (alpha_FL=0.654, alpha_TA=0.0) collapses the correlation to approximately zero. With N=10, the entire analysis is driven by one checkpoint and one domain.

### Sample Size
10 checkpoints and one hierarchy domain is far below the 20-50 checkpoint, multi-domain scale proposed in the methodology. The pilot was structurally underpowered.

---

## Experiment 4: Multi-Objective Pareto (Pythia-160M)

### Severe Underpowering
E4 full evaluates only 14 checkpoints across 7 families, with exactly 2 trainers per family, all within a single stratum (resid_post_layer_8, width=16384). The analysis runs 84 pairwise Mann-Whitney U tests and finds zero significant results. With n=1-2 per family in most comparisons, this is expected due to underpowering, not strong evidence of equivalence.

**The paper must not present this as conclusive evidence of "no dominance."**

### What E4 Actually Shows
The per-family Pareto points reveal interesting patterns:
- GatedSAE trainer_0: lowest absorption (0.007), highest EV (0.994), but highest dead neurons (0.509)
- TopK trainer_3: highest absorption (0.076), lowest dead neurons (0.165)
- BatchTopK trainer_0: high absorption (0.035) but also high sparse probing accuracy (0.939)

These are suggestive tradeoffs, but with n=2 per family, no statistical inference is justified.

---

## Cross-Cutting Concern: Failed Gemma Experiment

The planned `e1_full_gemma` task failed because Gemma-2-2B is a gated HuggingFace repo. This means the controlled Pareto evaluation (E1) and the metric pilot (E3) are entirely limited to GPT-2 Small—a 117M parameter model from 2019. The abstract's framing that the analysis "spans 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints" is technically true, but it obscures the fact that the *controlled* experiments are on a tiny, old model.

## Verdict

- **E1:** Weak. Degenerate proxy, coarse hedging metric, unreliable dead-neuron estimates, and family collapse.
- **E2:** Strong methodologically, but causal framing overreaches. **H2 is falsified by the new full results**—the official metric is not degenerate.
- **E3:** Weak and potentially misleading because it compared against the degenerate proxy. Needs re-run with official metric.
- **E4:** Underpowered. Report as exploratory pilot, not conclusive evidence.
