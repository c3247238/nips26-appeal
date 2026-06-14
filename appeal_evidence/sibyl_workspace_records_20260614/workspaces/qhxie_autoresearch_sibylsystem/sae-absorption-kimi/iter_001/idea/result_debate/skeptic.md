# Skeptic Analysis: Result Debate

## Overview

This analysis critically examines the three pilot experiments conducted for the proposal "The Impossibility Triangle of Sparse Autoencoders." The experiments are:

- **E1 (Multi-Objective Pareto Evaluation):** `e1_full_gpt2` — 27 GPT-2 Small checkpoints, 10 metrics, Pareto analysis.
- **E2 (Downstream Causal Cost Meta-Analysis):** `e2_meta` — 314 SAEBench checkpoints, partial correlation and OLS regression.
- **E3 (Task-Agnostic Metric Validation):** `e3_validation` — 10 GPT-2 Small checkpoints, correlation between task-agnostic and first-letter absorption.

---

## 1. Statistical Risk Inventory

### Risk 1: Extreme small-sample bias in E1 dominance tests
**Specific concern:** The Mann-Whitney U test comparing `standard` (n=23) vs. `feature_splitting` (n=4) on absorption yields U=48.0, p=0.754. With one group having only 4 observations, the test has virtually no power to detect differences. The reported p-value is meaningless for inference.
**Why unreliable:** A two-sample nonparametric test with n=4 in one cell cannot reliably distinguish distributions. The "no significant difference" conclusion is indistinguishable from "insufficient data."

### Risk 2: The E2 meta-analysis relies on synthetic downstream proxies
**Specific concern:** The `e2_meta` summary explicitly states: "Sparse probing and RAVEL metrics were unavailable from HF due to rate limits. Synthetic proxies were generated for pipeline validation in this pilot."
**Why unreliable:** The partial correlations (absorption vs. sparse_probing_f1: partial r=-0.385, p<1e-12) and regression coefficients are computed on *synthetic* data, not real SAEBench measurements. These numbers have no external validity.

### Risk 3: E3 correlation is driven by a single outlier and is not significant
**Specific concern:** Pearson r=-0.592, Spearman rho=-0.529, p=0.116 (n=10). The TopK_Attn checkpoint (`blocks.8.hook_attn_out`) has first-letter absorption=0.654 while all other 9 checkpoints have first-letter absorption=0.0.
**Why unreliable:** With 9 of 10 points sharing the same x-value (0.0), the correlation is mechanically determined by the 10th point. Removing the outlier collapses the correlation to undefined (zero variance on first-letter). The p=0.116 already fails standard significance thresholds.

---

## 2. Alternative Explanations

### For E1's "feature_splitting has higher CE loss recovered" claim
- **Alternative:** Feature-splitting SAEs have much smaller dictionary widths (768–6144) compared to standard SAEs (up to 131k). Smaller dictionaries may produce different CE-recovery dynamics due to lower reconstruction dimensionality, not because feature splitting is architecturally superior. The width confound is unaddressed.

### For E2's "absorption negatively predicts downstream utility" claim
- **Alternative 1 (synthetic data artifact):** The synthetic proxies may have been generated with an implicit negative correlation to absorption by the data-generation heuristic. Without the real SAEBench metrics, we cannot rule out algorithmic confounding.
- **Alternative 2 (omitted variable bias):** The regressions control for L0 and CE loss recovered, but not for training steps, random seed, or layer depth. SAEBench checkpoints include training-curve snapshots (step_0, step_2441, etc.) that are mechanically correlated with both absorption and performance. Step count is a massive confounder.

### For E3's "negative correlation between task-agnostic and first-letter absorption" claim
- **Alternative (metric degeneracy):** The "first-letter" proxy used in E1 and E3 is a coarse simplification of the Chanin et al. spelling benchmark, not the real metric. Nine checkpoints scoring exactly 0.0 suggests floor effects or implementation bugs rather than genuine zero absorption. The negative correlation may reflect that the simplified proxy is broken, not that the constructs diverge.

---

## 3. Proxy Metric Audit

| Claimed construct | What was actually measured | Gap / concern |
|---|---|---|
| Absorption (E1, E3) | Simplified first-letter proxy: k-sparse probe on a tiny set of parent-child token pairs | This is **not** the Chanin et al. benchmark. The proxy shows 0.0 for 26 of 27 checkpoints in E1 and 9 of 10 in E3, suggesting it is insensitive or miscalibrated. |
| Hedging (E1) | Correlated antonym pair overlap (mean top-k overlap) | Only 25 pairs evaluated. The metric conflates "same top feature" with genuine hedging. No validation against Chanin et al.'s hedging definition. |
| Downstream utility (E2) | Synthetic proxies for sparse_probing_f1, RAVEL Cause/Isolation | **Not real metrics.** The pilot note admits these are generated stand-ins. The paper cannot claim causal downstream cost without real data. |
| Task-agnostic absorption (E3) | Geography hierarchy probe (continent-country) with logistic regression and integrated-gradients ablation | Only **one** hierarchy domain (geography) and a small hand-built probe set. "Task-agnostic" is overstated. |
| Pareto front (E1) | Normalized metrics on GPT-2 Small only, with 27 checkpoints across 2 architecture families | The original proposal promised evaluation across Gemma-2-2B, Pythia-160M, and multiple architectures (OrtSAE, Matryoshka, JumpReLU, etc.). The actual run is a **tiny subset** on a single small model. |

---

## 4. Severity Classification

### Fatal flaws (invalidate main claims — must be fixed)

1. **E2 uses synthetic proxies for all downstream metrics.**
   - The regression tables and partial correlations are computed on fake data. Any claim about "absorption's causal cost on downstream utility" is unsupported until real SAEBench sparse_probing_f1 and RAVEL metrics are obtained.

2. **E1 and E3 absorption metric is a degenerate simplified proxy, not the validated benchmark.**
   - A metric that returns 0.0 for 96% of checkpoints (26/27 in E1) is not measuring absorption — it is measuring implementation limitations. Claims about absorption rates, Pareto tradeoffs, or architecture comparisons on this metric are invalid.

### Serious concerns (weaken claims — should address next iteration)

3. **E1 dominance tests are underpowered and architecturally narrow.**
   - Only 2 architecture families, one model (GPT-2 Small), and tiny group sizes. The Mann-Whitney test with n=4 is uninformative. The paper cannot generalize to "absorption-mitigation methods" from this sample.

4. **E3 correlation is mechanically determined by a single outlier and not significant.**
   - p=0.116, n=10, 9 points at x=0. The claim that "task-agnostic and first-letter metrics are negatively correlated" is statistically fragile and likely an artifact of proxy degeneracy.

5. **E1 feature_splitting advantage on CE loss recovered may be a width confound.**
   - Dictionary widths differ by 20x across groups. No width-controlled comparison exists.

### Minor caveats (worth noting but do not change conclusion)

6. **Dead-neuron rates in E1 pilot were unstable at 2k tokens.** The full E1 run used 2048 tokens; dead-neuron estimates may still be noisy, though less so than the pilot.
7. **Gemma-2-2B was inaccessible due to gating.** This is a resource limitation, not a methodological flaw, but it means the study cannot speak to the model family where most absorption research is conducted.

---

## 5. Concrete Remediation

### For Fatal Flaw 1 (E2 synthetic data)
**Required experiment:** Re-run `e2_meta` with real SAEBench HF data.
- **Dataset:** `adamkarvonen/sae_bench_results_0125` (or latest release).
- **Metrics to extract:** `absorption_mean`, `l0`, `ce_loss_recovered`, `explained_variance`, `sparse_probing_f1`, `ravel_cause`, `ravel_isolation`.
- **Expected outcome:** If the negative partial correlations hold on real data, H2 is supported. If they attenuate or flip, the causal-cost claim must be downgraded or abandoned.
- **Control addition:** Include `training_step` as a covariate to address the checkpoint-snapshot confound.

### For Fatal Flaw 2 (degenerate absorption proxy)
**Required experiment:** Replace the simplified first-letter proxy with the actual `sae-spelling` benchmark or SAEBench's absorption metric implementation.
- **Dataset:** Use the official Chanin et al. spelling-task dataset (first-letter parent-child pairs).
- **Checkpoints:** Re-evaluate all E1 checkpoints with the real metric.
- **Expected outcome:** The real metric should show variance across checkpoints (as it does in SAEBench, where absorption_mean ranges 0.0–0.73). If it still returns near-zero for most checkpoints, that indicates the GPT-2 SAEs genuinely have low first-letter absorption — a valid negative result, but very different from "the proxy is broken."

### For Serious Concern 3 (underpowered E1)
**Required experiment:** Expand E1 to Pythia-160M and include at least 4 architecture families with n>=8 per family.
- **Checkpoints:** Source SAEBench releases for Standard, TopK, JumpReLU, and Matryoshka on Pythia-160M (layer 8).
- **Analysis:** Re-run Mann-Whitney U or Kruskal-Wallis with adequate power.
- **Expected outcome:** If no architecture dominates after power adjustment, H1 is tentatively supported. If one architecture dominates, the "impossibility triangle" framing must be softened.

### For Serious Concern 4 (E3 outlier-driven correlation)
**Required experiment:** Re-run E3 with the real first-letter benchmark and at least 30 checkpoints spanning multiple layers and families.
- **Checkpoints:** GPT-2 Small (all available layers) + Pythia-160M (layer 8) if accessible.
- **Analysis:** Report correlation with and without the attention-output outlier; compute bootstrap CIs.
- **Expected outcome:** If the correlation remains weak or negative with the real metric and larger n, then H3 is falsified and the paper should pivot to analyzing *why* first-letter absorption does not generalize.

---

## 6. Bottom-Line Verdict

**The current experimental results do NOT support the paper's three hypotheses at the claimed level of confidence.**

- **H1 (tradeoffs):** E1 is too narrow (one small model, 27 checkpoints, 2 families, broken absorption proxy) to support any claim about multi-objective Pareto dominance.
- **H2 (downstream causality):** E2 is built entirely on synthetic proxies. The impressive p-values are meaningless until validated on real data.
- **H3 (task-agnostic metric):** E3 shows a non-significant, outlier-driven negative correlation using a degenerate proxy. This does not validate the task-agnostic metric; if anything, it validates that the simplified proxy is broken.

**Recommendation:** Before writing any paper section based on these experiments, the project must (1) obtain real SAEBench downstream metrics, (2) replace the absorption proxy with the validated benchmark, and (3) expand the checkpoint sample to at least two model families with adequate per-group sizes.
