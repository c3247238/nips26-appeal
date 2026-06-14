# Experiment Result Analysis

## Key Results Summary

The experiment evaluated the construct validity of the SAEBench first-letter absorption metric by testing its correlation with semantic-hierarchy absorption across 8 SAE architectures on Pythia-160M.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| H1: First-letter vs semantic-hierarchy correlation (Pearson r) | **0.463** | Below the 0.6 success threshold |
| Bootstrap 95% CI for H1 | **[-0.389, 0.981]** | Catastrophically wide; inconclusive |
| H2: Hierarchy specificity (paired t-test) | **t = -4.748, p = 0.0032** | Significant but **reversed**: semantic (0.235) < non-hierarchy (0.331) |
| H3: Robustness across tau_fs | r = 0.468, 0.463, 0.471 | Highly stable but of an imprecise estimate |
| Random SAE semantic absorption | **0.352** | Identical to Standard SAE (0.352); metric not specific to learned structure |
| Probe AUROC (all hierarchies, all SAEs) | **1.0** | Perfect probes suggest trivial task or ceiling effect |
| GPT-2 replication (2 SAEs) | 0.0, 0.003 | Near-zero absorption; starkly different from Pythia-160M |

## Debate Perspectives Summary

- **Optimist:** The architecture ranking is preserved across tasks (PAnneal < Matryoshka < Gated < JumpRelu < TopK < BatchTopK), indicating ordinal construct validity. The tau_fs robustness is remarkably stable. The GPT-2 near-zero result and Random-SAE floor effect are surprising but publishable boundary conditions. The story should be reframed as a *construct-validity stress-test*.

- **Skeptic:** The experiment is uninterpretable due to two fatal flaws: (1) perfect probe AUROCs (1.0) suggest a degenerate probe, invalidating the semantic-hierarchy measurement; (2) the Random-SAE control fails the pre-registered expectation (0.352 vs near-zero), proving the metric is not specific to learned structure. The wide CI on n=7 SAEs makes the primary correlation pure noise. Before claiming "lack of construct validity," the authors must show the semantic task measures absorption at all.

- **Strategist:** Signal strength is mixed: H1 is weak/inconclusive, H2 is strong but opposite-direction, H3 is strong but vacuous. The dominant strategy is to **PROCEED** with two high-ROI follow-ups: (1) a weak co-occurrence non-hierarchy control to explain the reversed H2, and (2) a full GPT-2 cohort replication to validate the model-family effect. A pivot would discard a compelling, partially-complete story. The paper should be reframed as "A Construct-Validity Stress-Test of the SAEBench Absorption Metric."

- **Comparativist:** This is a methodological diagnostic paper, not a SOTA-beating result. The novelty is genuine but narrow: it is the first systematic construct-validity test of the dominant SAEBench absorption metric. Venue recommendation: ICLR/NeurIPS MI Workshop as-is; EMNLP Findings / AAAI with a Gemma-2-2B replication. The narrative space is becoming more crowded (2025–2026 papers questioning SAEBench validity), so the window may close in 6–12 months.

- **Methodologist:** The experiment is methodologically sound in design but suffers from three critical issues: (1) small-sample correlation inference with an uninformative CI, (2) a ceiling-effect probe artifact (AUROC = 1.0), and (3) an unvalidated GPT-2 replication anomaly. Reproducibility score: 3/5. Top recommendations: diagnose the GPT-2 pipeline, replace trivial probes with harder hierarchies, and add a natural-frequency baseline ablation.

- **Revisionist:** The core mental model must be revised. First-letter absorption is **not** a general measure of "feature absorption"; it is a task-specific metric measuring how well an SAE avoids splitting *character-level* features. The Random-SAE result and reversed hierarchy specificity indicate the metric is confounded by base-model representational geometry and general correlation structure. The research question should shift from "Does it generalize?" to "What specific properties does it actually capture, and why do they fail to generalize?"

## Analysis

### 1. Method Feasibility
The core method *technically works*: the custom pipeline successfully computed absorption scores across 8 SAEs, 10 hierarchies, and a non-hierarchy control. The tau_fs robustness and ranking preservation demonstrate that the measurement is stable and internally consistent. However, the method's *validity* is in serious doubt. The Random-SAE control matching the Standard SAE (0.352 vs 0.352) is a sharp signal that the semantic-hierarchy task may not measure "absorption" as a SAE-specific pathology at all. The perfect probe AUROCs (1.0) further suggest the task is trivially easy, potentially creating a mechanical floor effect where any SAE—or random directions—achieves similar scores.

### 2. Performance
The results do **not** outperform baselines in the traditional sense. The pre-registered success criterion for H1 (r > 0.6) was not met. H2 was statistically significant but in the opposite direction. The "performance" of this study must be judged on its critical findings: the Random-SAE control failure, the reversed hierarchy specificity, and the GPT-2 model-family divergence. These are negative/diagnostic results, not empirical improvements. They are valuable as methodological contributions but do not demonstrate that a new method works better than existing ones.

### 3. Improvement Headroom
There is a **clear and manageable path to improvement** within the current direction:
- **Weak co-occurrence control** (~0.3 GPU-hr): Directly tests the leading explanation for the reversed H2.
- **Full GPT-2 cohort** (~0.8 GPU-hr): Validates or falsifies the dramatic model-family effect.
- **OrtSAE + HSAE expansion** (~0.3 GPU-hr): Tightens the H1 correlation CI from n=7 to n=9.
- **Harder hierarchies** (~0.3 GPU-hr): Addresses the ceiling-effect concern by dropping AUROCs below 1.0.

Cumulatively, this is ~1.4 GPU-hours—well within the project's iterative budget. These follow-ups directly address the skeptic's fatal flaws and the methodologist's concerns. The headroom is not speculative; it is a concrete list of experiments with falsifiable outcomes.

### 4. Time-Cost Tradeoff
Continuing to optimize is **more efficient than pivoting**. The current dataset already contains several publication-quality surprises (Random-SAE floor effect, reversed H2, GPT-2 divergence). A pivot to an alternative idea (e.g., FastProbe-Absorb, theory bounds) would discard this sunk investment and require building a new empirical foundation from scratch. The strategist's opportunity-cost analysis shows the highest information-gain-per-GPU-hour experiments are follow-ups within the current framework. The reframed narrative—"A Construct-Validity Stress-Test"—is arguably more compelling than the original confirmatory hypothesis.

### 5. Critical Objections
The skeptic raises two "fatal flaws": perfect probe AUROCs and the failed Random-SAE control.
- **Perfect AUROCs (1.0):** This is addressable. The follow-up experiment with harder/deeper hierarchies (3-level, rare concepts, or more abstract distinctions) is designed to drop AUROCs into the 0.7–0.95 range. If the story remains similar with non-trivial probes, the ceiling-effect objection is defanged.
- **Random-SAE control:** This is the sharper objection. If a second random baseline (random encoder + decoder) also scores ~0.35, it confirms the metric is not SAE-specific on semantic tasks. However, this does not invalidate the paper—it reframes it. The paper's contribution becomes: "We show that the SAEBench absorption metric, when applied to semantic hierarchies, measures base-model representational geometry rather than SAE architecture quality." This is still a publishable, field-relevant finding.

The skeptic's concerns are **serious but addressable**, not fatal. They do not demand abandonment; they demand a diagnostic reframing and 1–2 follow-up experiments.

## Decision Rationale

The project should **PROCEED** for the following reasons:

1. **At least one robust, surprising signal exists.** H2 (reversed hierarchy specificity) is statistically strong (p = 0.0032) and unexpected. The Random-SAE result and GPT-2 divergence are sharp, visually compelling findings that challenge benchmark assumptions.

2. **The narrative can be productively reframed.** The original confirmatory story ("construct validity confirmed") is unsupported. The reframed story ("A Construct-Validity Stress-Test of the SAEBench Absorption Metric") is more interesting: it validates ordinal utility (ranking preservation) while identifying critical boundary conditions (model-dependence, non-hierarchy specificity, random-SAE floor).

3. **Clear path to publication-quality results.** Two follow-up experiments (~1.1 GPU-hr) directly address the strongest objections. The weak co-occurrence control tests the reversed H2 explanation. The full GPT-2 cohort validates the model-family effect. These are not open-ended fishing expeditions; they have pre-specified success criteria.

4. **Pivoting is more costly than continuing.** A pivot discards a partially complete, empirically grounded story with several surprising results. The alternatives (FastProbe-Absorb, theory bounds, validity-aware analysis) are either untested or better suited as sequels.

5. **The venue target is appropriate.** A workshop paper (ICLR/NeurIPS MI Workshop) or a mid-tier conference submission (EMNLP Findings, AAAI) is a realistic goal for a methodological diagnostic paper with negative results. The current direction does not need to reach a top-tier main conference to be a successful research output.

## DECISION: PROCEED
