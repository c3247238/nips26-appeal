# Working Title

Entropy-Routed Compute for Training-Free Discrete Diffusion Language Models: A Bounded Full-Scale Study with Sham Controls

## Core Thesis

We do not claim a benchmark-level breakthrough. We claim that, under a unified runtime contract, entropy is better interpreted as a routing/stopping signal than as a semantic controller, and that this interpretation survives a matched fixed-frontier sham on full-scale GSM8K.

## Section Outline

### 1. Introduction

- Motivate the problem: training-free intervention for discrete diffusion language models has entered a small-gain regime where attribution matters as much as raw score.
- State the key failure of the old framing: entropy can be informative without being a validated semantic controller.
- Present the bounded contribution: a full-scale GSM8K study showing that entropy-routed compute (`cand_espd`) yields a modest but credible gain over shared controls and a fixed-frontier sham.

**Key evidence**

- `cand_espd` accuracy `0.4041` vs `RAND-84` `0.3980`, `CARD-84` `0.3965`, `ESPD-FixedFrontier` `0.3988`.
- `cand_espd` speed at equal-quality band `124.42 tok/s` vs fixed-frontier `105.73 tok/s`.

**Transition**

Move from the problem of over-claiming weak gains to the need for a more disciplined framing of uncertainty-guided test-time compute.

### 2. Background and Related Work

- Cover discrete diffusion language models and inference-time engineering for dLLMs.
- Position our work against test-time scaling/search/guided-decoding lines such as Prism, SOAR, CD4LM, MetaState, and sampler-centric evaluation.
- Clarify the distinction between observer-side uncertainty signals and controller-side intervention rules, linking to calibration/failure-prediction caution.

**Key argument**

Our contribution is not "another stronger dLLM method"; it is a bounded result on attribution, routing, and control design under honest runtime accounting.

**Transition**

Having established that the field now cares about both test-time scaling and attribution, introduce the concrete hypothesis we test.

### 3. Problem Reframing

- Describe the iteration-4 pivot from object-first/controller-first narratives to an evidence-first framing.
- State the central update: entropy appears more useful for deciding where to spend compute than for deciding what semantic revision to make.
- Explain why a sham-controlled comparison is required for small-gain claims.

**Key evidence**

- Shared-control result: `CARD-84` does not cleanly beat `RAND-84`.
- Result-debate synthesis: `H2` supported modestly; `H1`, `H3`, and `H5` remain untested or inconclusive.

**Transition**

This reframing motivates the method family and control design used in the full-scale study.

### 4. Method

- Define `cand_espd` as an entropy-routed compute allocation method with active-frontier selection and early stopping.
- Define `ESPD-FixedFrontier` as the matched sham: same frontier ratio family and stopping template, but no entropy-based frontier placement.
- Separate observer, router, and controller roles explicitly to avoid overstating what the method does.

**Key evidence**

- `frontier_ratio = 0.1211`
- `avg_nfe = 67.93`
- `stopped_step_histogram = {1: 702, 2: 4, 3: 613}`

**Transition**

After defining the method, describe the evaluation contract needed to keep runtime and quality comparisons honest.

### 5. Experimental Setup

- Specify GSM8K full-scale evaluation with `n=1319`.
- Describe shared controls (`CARD-84`, `RAND-84`) and the fixed-frontier sham.
- State the runtime contract used in the final comparison: eager backend, compile disabled, flash attention disabled, separate logging of auxiliary overhead.

**Key evidence**

- Candidate effective batch `54`; fixed-frontier `52`; shared controls `57`.
- Candidate peak VRAM `15289 MB`; fixed-frontier `41973 MB`; shared controls `40910 MB`.

**Transition**

With the setup fixed, present the main bounded result before moving into interpretation and limitations.

### 6. Main Results

- Present the main result table with quality and equal-quality speed.
- Show that `cand_espd` improves over shared controls only modestly, but improves over the sham in both quality and speed.
- Emphasize that the most important evidence is structural, not magnitude alone.

**Key evidence**

- `cand_espd`: accuracy `0.4041`, equal-quality speed `124.42 tok/s`
- `ESPD-FixedFrontier`: accuracy `0.3988`, equal-quality speed `105.73 tok/s`
- `CARD-84`: accuracy `0.3965`
- `RAND-84`: accuracy `0.3980`
- `speed_gain_vs_fixed_frontier = 18.69 tok/s`

**Transition**

Because the gains are small, the paper must immediately move into paired comparisons, runtime accounting, and claim boundaries rather than celebratory summary.

### 7. Analysis

- Analyze paired repair/harm decomposition to show where the modest gains come from.
- Interpret the stopping histogram as evidence that the method is allocating compute non-uniformly rather than simply running fewer steps everywhere.
- Discuss why the fixed-frontier sham matters more than the absolute speed ranking against shared controls.

**Key evidence**

- vs `RAND-84`: `fixed=73`, `harmed=65`, `net=+8`
- vs `CARD-84`: `fixed=52`, `harmed=42`, `net=+10`
- vs fixed-frontier: `fixed=57`, `harmed=62`, `net=-5` from the sham's perspective

**Transition**

The analysis supports a routing interpretation, but it also makes the remaining weaknesses impossible to ignore.

### 8. Limitations and Claim Boundary

- State explicitly that this is not a benchmark-level SOTA claim.
- State explicitly that the current evidence does not validate the object-level mainline (`cand_bsr`) or a general semantic-controller theory.
- Enumerate the remaining weaknesses: small margins, missing multi-seed stability, partially unmatched sham, and missing cross-task validation.

**Key argument**

The value of the paper depends on saying exactly what the evidence supports and no more.

**Transition**

Once the claim boundary is clear, the final section can discuss what future work would most efficiently raise the confidence level.

### 9. Discussion and Next Steps

- Argue that the correct next step is a narrowed continuation of the entropy-routed compute line, not a reset.
- Prioritize one external validation, one runtime-lineage artifact, and one routing/stopping split ablation.
- Reposition `cand_bsr` as a challenger that must earn its way back into the mainline through evidence.

**Closing message**

Small-gain dLLM research should be evaluated as much by attribution discipline as by raw benchmark movement.

## Transition Logic Summary

- Introduction -> Related Work: from the general risk of over-claiming to the concrete literature that makes attribution salient.
- Related Work -> Problem Reframing: from what the field is doing to what this paper is actually testing.
- Problem Reframing -> Method: from the conceptual distinction between observer and router to a concrete method/sham pair.
- Method -> Setup: from algorithm description to the contract needed for honest comparison.
- Setup -> Results: from evaluation design to bounded empirical findings.
- Results -> Analysis: from headline table to structural interpretation.
- Analysis -> Limitations: from what survived scrutiny to what still does not.
- Limitations -> Discussion: from claim boundary to efficient next actions.

## Figure & Table Plan

### Figure 1: Entropy-Routed Compute vs Fixed-Frontier Sham (Section: Method)

- **Purpose**: Show the conceptual difference between entropy-based frontier placement and a fixed-frontier control.
- **Type**: flow_chart
- **Content**: Shared 64-step draft -> frontier scoring -> active frontier selection -> stopping check -> selective revision; parallel branch for fixed-frontier sham with the same ratio but no entropy routing.
- **Key takeaway**: The tested claim is about dynamic frontier placement, not merely keeping a fixed amount of compute.
- **Generation**: manual_diagram
- **Data source**: `exp/results/espd_gsm8k_full_v1.json`, `exp/results/espd_fixed_frontier_gsm8k_full_v1.json`

### Table 1: Main Full-Scale GSM8K Results Under a Unified Runtime Contract (Section: Main Results)

- **Purpose**: Present the main quality and speed numbers for candidate, sham, and shared controls in one place.
- **Type**: comparison_table
- **Content**: accuracy, correct count, equal-quality speed, avg NFE, frontier ratio, effective batch size.
- **Key takeaway**: `cand_espd` is modestly better in quality than shared controls and clearly better than the sham in the candidate-vs-sham comparison.
- **Generation**: data_table
- **Data source**: `exp/results/espd_fullscale_bundle_v1.json`, `exp/results/gsm8k_controls_full_v1.json`

### Figure 2: Quality-Speed Positioning with Shared Controls and Sham (Section: Main Results)

- **Purpose**: Visually communicate the bounded gain and prevent readers from over-reading the result.
- **Type**: scatter
- **Content**: x-axis = equal-quality speed, y-axis = quality at equal compute; one point each for `cand_espd`, `ESPD-FixedFrontier`, `CARD-84`, `RAND-84`; annotate the candidate-sham gap.
- **Key takeaway**: The candidate is not the absolute fastest point, but it dominates the matched sham and slightly improves quality over shared controls.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/espd_fullscale_bundle_v1.json`

### Figure 3: Stopping Behavior of the Entropy-Routed Candidate (Section: Analysis)

- **Purpose**: Show that the method is allocating compute non-uniformly rather than behaving like a trivial fixed-step heuristic.
- **Type**: bar_chart
- **Content**: histogram of stopped steps (`1`, `2`, `3`) for `cand_espd`, optionally compared with the sham.
- **Key takeaway**: The method exhibits sharply bimodal stopping behavior that supports the routing interpretation, while also motivating further ablation.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/espd_gsm8k_full_v1.json`, `exp/results/espd_fixed_frontier_gsm8k_full_v1.json`

### Table 2: Paired Repair/Harm Decomposition (Section: Analysis)

- **Purpose**: Prevent the paper from leaning only on aggregate accuracy.
- **Type**: ablation_table
- **Content**: fixed, harmed, unchanged_correct, unchanged_wrong, net repaired for candidate vs `RAND-84`, candidate vs `CARD-84`, and sham vs candidate.
- **Key takeaway**: The gains are small but not purely aggregate noise; they arise from a modest positive repair-harm balance.
- **Generation**: data_table
- **Data source**: `exp/results/espd_fullscale_bundle_v1.json`, `exp/results/espd_gsm8k_full_v1.json`, `exp/results/espd_fixed_frontier_gsm8k_full_v1.json`

### Figure 4: Runtime Lineage Audit (Section: Limitations)

- **Purpose**: Make runtime caveats reviewer-visible instead of burying them in prose.
- **Type**: bar_chart
- **Content**: batch size, peak VRAM, auxiliary overhead, extra forward passes for candidate, sham, and shared controls.
- **Key takeaway**: The claim is bounded because the runtime path is not fully matched or optimized; the paper is honest about that.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/espd_fullscale_bundle_v1.json`, `exp/results/espd_gsm8k_full_v1.json`, `exp/results/espd_fixed_frontier_gsm8k_full_v1.json`, `exp/results/gsm8k_controls_full_v1.json`

## Visual Storytelling Notes

- Mention Figure 1 before introducing the sham to ensure the reader understands exactly what is controlled.
- Introduce Table 1 first in Experiments, then immediately interpret it with Figure 2 to avoid over-reading raw numbers.
- Use Figure 3 and Table 2 together to move from aggregate scores to structural interpretation.
- Use Figure 4 in Limitations, not Appendix, because runtime caveats are part of the core contribution boundary.
