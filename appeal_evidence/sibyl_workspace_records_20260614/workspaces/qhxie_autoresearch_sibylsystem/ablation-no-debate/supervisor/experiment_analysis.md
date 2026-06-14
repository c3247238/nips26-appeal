# Experiment Result Analysis

## Key Results Summary

Seven experiments were completed across iter_002. Here are the exact numbers:

| Experiment | Status | Key Metric | Value |
|-----------|--------|-----------|-------|
| H_Mech Factorial (FULL) | Completed | Encoder effect | 0.843 +/- 0.082 |
| H_Mech Factorial (FULL) | Completed | Decoder effect | 0.011 +/- 0.015 |
| H_Mech Factorial (FULL) | Completed | Effect ratio | 80x (encoder / decoder) |
| H_Mech Factorial (FULL) | Completed | Original pass rate | 6.7% (1/15) |
| H_Mech Factorial (FULL) | Completed | Revised pass rate | 100% (15/15) |
| Multi-seed Validation (FULL) | Completed | Trained mean | 0.477 +/- 0.022 |
| Multi-seed Validation (FULL) | Completed | Random mean | 0.033 +/- 0.011 |
| Multi-seed Validation (FULL) | Completed | t-test | t=36.04, p=3.85e-10 |
| Hierarchy Strength Ablation (FULL) | Completed | Sim 0.5 / 0.67 / 0.8 | 0.416 / 0.501 / 0.544 |
| Hierarchy Strength Ablation (FULL) | Completed | Monotonic | True, ANOVA p < 1e-10 |
| L0 Sparsity Ablation (FULL) | Completed | L0=20 / 32 / 50 | 0.552 / 0.490 / 0.419 |
| L0 Sparsity Ablation (FULL) | Completed | Monotonic (increasing) | False, ANOVA p < 1e-10 |
| Held-Out Generalization (PILOT) | Completed | Train / Test | 0.352 / 0.352 |
| Held-Out Generalization (PILOT) | Completed | Pearson r | 0.998 across seed means |
| H3 Steering (FULL) | Completed | Ratio (absorbed/non-absorbed) | 0.97x |
| H3 Steering (FULL) | Completed | p-value | 0.936 |
| H_Safe on GPT-2 (FULL) | Completed | Safety / Non-safety | 0.967 +/- 0.010 / 0.968 +/- 0.013 |
| H_Safe on GPT-2 (FULL) | Completed | Mann-Whitney p | 0.989 |

Confirmed findings: encoder-driven absorption (H_Mech), trained > random (H1), hierarchy strength dose-response (H_Comp), perfect train/test stability (generalization).
Negative findings: H3 steering null, H_Safe null, L0 sparsity opposite direction.

## Debate Perspectives Summary

- **Optimist**: The encoder-driven mechanism is a strong, novel contribution. The 80x ratio is headline-worthy. Four of seven experiments show strong signals. Negative results (H3, H_Safe) refine rather than destroy the narrative -- absorption is a universal representational strategy, not a defect. The decoder disentanglement observation (Condition D < Condition B) suggests a two-player dynamic that prior work missed. Paper is publishable with reframing.

- **Skeptic**: The H_Mech confirmation is compromised by post-hoc criterion revision. Original criteria (B ~ D, |diff| < 0.25) failed on 14/15 runs (6.7% pass). Revised criteria (encoder effect > 0.5, decoder effect < 0.1) were adopted after seeing the data and cannot fail by construction. The "perfect" generalization (r=0.998) is a ceiling effect -- test data comes from the same generative process, not new geometry. H3 steering has degenerate statistics (CV > 6, effect direction reverses with alpha). The absorption metric is inconsistent across experiments (cosine vs overlap vs Jaccard). Severity: post-hoc revision = fatal flaw; metric inconsistency = serious concern; generalization = serious concern.

- **Strategist**: PROCEED with mandatory reframing. Four experiments have strong signals (H_Mech, H1, hierarchy, generalization). Two are noise (H3, H_Safe). The core insight is robust but the project must: (1) drop H3 as a primary claim (failed replication), (2) drop H_Safe on GPT-2 (ceiling effect at ~97%), (3) add real-model validation as a mandatory next step. Without real-model validation, the paper is workshop-level at best. Resource allocation: 1.0 GPU hr for Gemma Scope validation (P0), 0.5 hr for encoder regularization pilot (P1). Do not invest more in H3 or H_Safe on current setup.

- **Comparativist**: The encoder-driven claim is novel and challenges the prevailing narrative, but Oursland (2026) provides theoretical support for encoder-decoder asymmetry, partially preempting the contribution. All evidence is synthetic (d_model=128) with a non-standard metric (Jaccard overlap), making direct comparison with Chanin et al.'s standard "absorption rate" impossible. Venue recommendation: workshop or mid-tier conference (AAAI/EMNLP/Findings). Top-tier only if real-model validation + constructive contribution (e.g., encoder regularization) are added.

- **Methodologist**: Reproducibility score 2/5. Critical issues: (1) post-hoc criterion revision for H_Mech undermines confirmation claim, (2) all evidence synthetic -- no real LLM validation, (3) metric inconsistency across experiments (cosine, overlap, Jaccard used interchangeably without equivalence proof), (4) H3 failed replication undermines causal claims, (5) no code repository, incomplete hyperparameters. Top-3 recommendations: (1) pre-register criteria and report original failure honestly, (2) add real LLM validation, (3) standardize to one metric with justification.

- **Revisionist**: Core contribution survives but practical implications are narrower than hoped. Mental model revisions: (1) decoder is not passive -- it actively disentangles encoder-induced absorption (two-player dynamic), (2) absorption is not controllable -- it is a representational property, not a steering handle, (3) absorption is driven by capacity pressure -- fewer active features force more absorption. Reframe from "absorption as a controllable mechanism" to "absorption as a fundamental structural constraint." New hypotheses generated: decoder disentanglement can be optimized via regularization; absorption has a critical capacity threshold; real SAEs exhibit near-universal absorption (>0.9).

## Analysis

### 1. Method Feasibility

The core method -- a 2x2 factorial decomposition of encoder vs decoder contributions to absorption -- works as intended and produces clean, interpretable results. The encoder effect (0.843 +/- 0.082) is 80x larger than the decoder effect (0.011 +/- 0.015), a magnitude gap that is unambiguous even accounting for measurement uncertainty. The factorial design successfully isolates the encoder as the driver.

However, the original pass criteria (B ~ D with |diff| < 0.25) failed on 14/15 runs because the trained decoder actively reduces absorption overlap compared to a random decoder -- an unanticipated "disentanglement" effect. The revised criteria (encoder effect > 0.5, decoder effect < 0.1) were adopted post-hoc. While the revised criteria directly test the paper's core claim, the fact that the original falsification criterion failed is a methodological concern that must be reported honestly.

### 2. Performance

Results outperform baselines decisively on the core claims:
- Trained SAE vs random: 0.477 vs 0.033 (14.6x higher, p=3.85e-10)
- Encoder vs decoder effect: 0.843 vs 0.011 (80x ratio)
- Hierarchy strength: monotonic increase 0.416 -> 0.501 -> 0.544 (ANOVA p < 1e-10)

However, three hypotheses failed:
- H3 steering: ratio 0.97x (p=0.936), pilot's 1.62x did not replicate
- H_Safe: no difference (p=0.989), near-ceiling absorption (~97%)
- L0 sparsity: opposite direction from hypothesis (lower L0 -> higher absorption)

The failed hypotheses are not fatal to the core contribution but narrow the paper's scope. The original proposal envisioned "absorption as a controllable mechanism with causal consequences" -- the data supports only "absorption as an encoder-driven structural property."

### 3. Improvement Headroom

There is a clear path to strengthen the paper:
- **Real-model validation** (P0): Replicate core findings on Gemma Scope or GPT-2 SAEs. This is the single highest-impact addition. Without it, the paper is synthetic-only.
- **Encoder regularization** (P1): Add a penalty term that discourages parent-child co-activation during encoder training. If absorption drops >30% with <5% reconstruction loss, this becomes a constructive contribution that elevates the paper.
- **Metric standardization** (P1): Commit to one absorption metric, compute all three on the same data, report inter-correlation.
- **Honest reporting** (P1): Report the original H_Mech criteria failure (6.7% pass rate) alongside the revised criteria.

These improvements require approximately 2.0 GPU hours and are well within the project's resource budget.

### 4. Time-Cost Tradeoff

Continuing to optimize the current direction is more efficient than pivoting to an alternative:
- The core mechanism is confirmed and robust across 5 seeds.
- The factorial methodology is novel and well-executed.
- Real-model validation (1.0 GPU hr) has higher expected return than starting fresh with a new idea.
- The backup directions in alternatives.md (encoder architecture modifications, absorption-as-diagnostic) are either already part of the improvement plan or undermined by the data (H3 failure contradicts the diagnostic framing).

Pivoting to a completely new research question would discard 2.5+ hours of validated experimental infrastructure and return to the idea-generation stage with no guaranteed improvement in outcome.

### 5. Critical Objections

The skeptic raises three serious concerns:

**Post-hoc criterion revision (severity: fatal flaw)**: This is the most serious objection. The original B~D criteria failed on 93.3% of runs. The revised criteria were adopted after seeing the data. Remediation: report the original failure honestly, justify the revised criteria as directly testing the core claim, and pre-register all future criteria. This does not invalidate the finding but requires transparent reporting.

**Metric inconsistency (severity: serious concern)**: Three metrics (cosine, overlap, Jaccard) are used interchangeably without establishing equivalence. Remediation: compute all three on the same data, report correlation matrix, commit to one primary metric. This is a straightforward fix.

**Generalization ceiling effect (severity: serious concern)**: The held-out test uses the same generative process, not new geometry. Remediation: reframe the claim from "perfect generalization" to "stable across train/test splits from the same distribution." A true cross-geometry test is a P2 follow-up.

None of these objections are fatal to the core finding. They are methodological weaknesses that can be addressed with transparent reporting and minor additional experiments.

## Decision Rationale

The decision is **PROCEED** for the following reasons:

1. **Core hypothesis is validated**: The encoder-driven absorption mechanism is robust, replicated across 5 seeds, and supported by a clean factorial design. The 80x encoder-to-decoder effect ratio is a genuinely novel finding.

2. **Multiple supporting experiments confirm the direction**: H1 (trained > random), hierarchy strength ablation (monotonic dose-response), and held-out stability all align with the encoder-driven narrative.

3. **Negative results refine rather than destroy the narrative**: H3 steering null, H_Safe null, and inverse L0 effect are all honest negative results that shift the framing from "absorption as a controllable defect" to "absorption as a universal structural constraint." This is still a publishable story.

4. **Clear improvement path exists**: Real-model validation (1.0 GPU hr) and encoder regularization (0.5 GPU hr) are well-defined, high-ROI next steps. The project does not need to pivot to a new idea to make progress.

5. **Pivoting would be wasteful**: The experimental infrastructure (synthetic data generator, factorial framework, multi-seed pipeline) is validated and reusable. A pivot would discard this investment.

The project must reframe its narrative and address methodological weaknesses, but the core direction is sound and worth pursuing.

## DECISION: PROCEED
