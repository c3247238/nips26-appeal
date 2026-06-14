# Supervisor Review: The Impossibility Triangle of Sparse Autoencoders

**Overall Score: 5.5 / 10**  
**Verdict: CONTINUE**

---

## Executive Summary

The paper makes a genuinely novel conceptual reframing---shifting SAE research from "fixing absorption" to "navigating tradeoffs"---but its empirical foundation is undermined by a degenerate absorption proxy that returns zero on 96% of E1 checkpoints and 90% of E3 checkpoints. The negative correlation between the task-agnostic and first-letter metrics (r = -0.59) is more alarming than acknowledged, suggesting the two measures capture different constructs rather than a shared "absorption" phenomenon. E2 provides the strongest evidence with a large N=314 meta-analysis, but the core Pareto evaluation (E1) and metric validation (E3) rest on methodologically weak ground. The paper currently scores below the acceptance threshold and requires substantial revision before it can be considered a strong research contribution.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty & Significance** | 7 | The multi-objective Pareto reframing is novel and well-motivated by the literature. No prior work systematically evaluates the absorption-hedging-reconstruction tradeoff triangle in a training-free setting. However, the empirical basis is narrower than the framing suggests. |
| **Technical Soundness** | 5 | E2 is methodologically sound (partial correlations, cluster-robust SEs, proper controls). But E1 and E3 rely on a simplified absorption proxy that the authors' own pilot rated "NO-GO for publication-ready numbers." The causal language in E2 overreaches the observational design. |
| **Experimental Rigor** | 4 | The first-letter proxy is degenerate (0.0 on 26/27 checkpoints), making it impossible to discriminate architectures on absorption. The hedging proxy uses only 5 antonym pairs. E3 has N=10 and a negative correlation that undermines construct validity. Only E2 has adequate scale and rigor. |
| **Reproducibility** | 6 | The training-free design and use of public checkpoints are strengths. However, the simplified proxies are not the same as the full SAEBench pipelines, so exact reproduction requires reimplementing the custom proxies. The Gemma-2-2B experiment failed due to gated access, limiting model coverage. |

---

## Critical Issues

### 1. Degenerate Absorption Proxy (Critical)

**Evidence:** The raw E1 log shows `absorption_score: 0.0` on 26 of 27 GPT-2 Small checkpoints. The only non-zero checkpoint is `hook-z-kk-l8` (alpha = 0.345). In E3, 9 of 10 checkpoints also show zero first-letter absorption.

**Why it matters:** A metric with near-zero variance cannot support claims about architectural tradeoffs or Pareto dominance. The Mann-Whitney U test (U=48, p=0.754) is a statistical artifact of this degeneracy, not evidence of equivalent absorption rates.

**Fix:** Integrate the full `sae-spelling` / SAEBench absorption pipeline, or explicitly reframe E1 as a pipeline-validation study. Do not present the "no significant difference in absorption" conclusion as a primary empirical finding.

### 2. Construct Validity Crisis in E3 (Critical)

**Evidence:** The task-agnostic metric is *negatively* correlated with the first-letter benchmark (Pearson r = -0.592, Spearman rho = -0.529, p = 0.12, N=10). The single outlier (`gpt2-small-attn-out-v5-32k`, alpha_FL = 0.654, alpha_TA = 0.0) drives much of this relationship. Removing it collapses the correlation to approximately zero.

**Why it matters:** This is not merely a "weak negative result"---it is evidence that the two metrics may measure different phenomena. If geography-hierarchy absorption and first-letter absorption are uncorrelated or negatively related, the construct "absorption" lacks convergent validity across domains.

**Fix:** Elevate this to a serious construct-validity discussion. Either (a) provide a theoretical argument for why the two metrics should be negatively related, or (b) explicitly conclude that "absorption" as currently operationalized may not generalize across semantic domains. Do not bury this under "larger validation needed."

### 3. Conflated Architecture Comparison in E1 (Major)

**Evidence:** The E1 summary table compares only "Standard" (n=23) vs. "feature_splitting" (n=4). But the "Standard" bucket includes TopK, TopK_MLP, TopK_Attn, and multiple hook-point variants (resid_pre, resid_post, resid_mid, mlp_out, attn_out, hook_z). These are meaningfully different architectures with very different hedging scores (e.g., attn_out: 0.323, resid_pre: 0.985).

**Fix:** Disaggregate the comparison table to show all distinct families/hook points. Report pairwise Mann-Whitney tests for each family against a clear baseline (e.g., resid_pre Standard), not just a collapsed Standard vs. feature_splitting comparison.

### 4. Internal Contradiction on Metric Quality (Major)

**Evidence:** `pilot_summary.json` explicitly states the simplified absorption/hedging proxies are "too crude" and dead-neuron estimates at 2k tokens are "unreliable." It rates metric quality as "NO-GO for publication-ready numbers." Yet the main paper presents these exact numbers as primary evidence.

**Fix:** Add an explicit caveat in the paper that E1 and E3 use simplified proxies. Better yet, replace the crude hedging proxy with a larger correlated-pair set and increase the activation corpus for dead-neuron detection to at least 50k tokens.

### 5. Causal Overclaim in E2 (Major)

**Evidence:** The abstract and section titles use phrases like "downstream causal cost," "unique causal effect," and "causal cost meta-analysis." E2 is an observational meta-analysis on existing benchmark data. While the paper adds caveats, the causal framing is misleading.

**Fix:** Replace "causal" with "predictive" or "associational" in the abstract and section titles. Keep the careful caveats, but do not lead with causal claims that the design cannot support.

### 6. Misleading Model Coverage Claim (Major)

**Evidence:** The abstract states the analysis "spans 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints" and mentions "Gemma-2-2B and Pythia-160M." However, the controlled Pareto evaluation (E1) is GPT-2 Small only. Gemma-2-2B appears only in the SAEBench meta-analysis (E2), and the planned `e1_full_gemma` experiment failed due to gated HuggingFace access.

**Fix:** Clarify in the abstract that E1 (the controlled Pareto evaluation) is limited to GPT-2 Small, and that Gemma-2-2B coverage comes only from the precomputed SAEBench meta-analysis (E2).

---

## Minor Issues

- **Figure 4** is introduced for the first time in the Discussion (Section 7.1) with no forward reference in any prior section. Add a forward reference at the end of E2 or move the figure introduction into E2.
- **Tables 1-3** are inline markdown tables without LaTeX labels or captions. Convert them to proper `table` environments with `\label{}` and `\caption{}`.
- **Terminology inconsistencies:** "feature-splitting" vs. "feature_splitting", "CE recovered" vs. "CE loss recovered" vs. `$\text{CE}_{\text{recovered}}$`. Do a global alignment pass.
- **Novelty claim scope:** Temper to "first systematic multi-objective evaluation on open-model checkpoints" rather than implying comprehensive coverage of all architectures.
- **Replication protocol:** E3 lacks a pre-registered protocol for scaling to 20-50 checkpoints and multiple domains. Add one.

---

## Risks

1. **If the absorption proxy cannot be fixed, the core Pareto-evaluation claim (H1) may be unrecoverable.**
2. **The negative correlation in E3 could be interpreted by reviewers as evidence that the authors do not understand what absorption is, undermining the entire framing.**
3. **A skeptical reviewer may argue that E2 is merely a correlational observation on existing benchmark data, not a novel contribution.**

---

## Evidence Gaps

- No validation that the simplified first-letter proxy agrees with the full SAEBench absorption metric on the same checkpoints.
- No robustness check showing that E2's partial correlations hold under outlier removal or non-linear functional forms.
- No cross-model validation of the task-agnostic metric (Gemma-2-2B was inaccessible).
- No statistical power analysis for E3; N=10 is almost certainly underpowered to detect a moderate correlation.

---

## What Would Raise the Score

**To raise by 1 point:**
1. Replace the degenerate absorption proxy with the full SAEBench/sae-spelling pipeline, or demonstrate that the simplified proxy correlates strongly with the canonical metric on a validation set.
2. Properly address the negative E3 correlation as a construct-validity problem rather than a weak negative result.
3. Disaggregate E1's architecture comparison and report all families separately.

**To raise by 2 points:**
- Additionally, run the task-agnostic metric on at least 30 checkpoints across 2+ hierarchy domains and show convergent or constructively divergent validity.
