# Experiments Section Critique

**Section**: 4. Experiments
**Reviewer**: Section Critic Agent
**Date**: 2026-03-10

---

## Overall Score: 6 / 10

The Experiments section provides a well-organized structure covering setup, main results, cross-benchmark evaluation, cross-model verification, and scaling curves. However, it suffers from several critical and major issues: incomplete results for the paper's headline method (DTA), reliance on N=16 pilot data for 3 out of 5 subsections, absence of promised statistical significance tests, and missing visual elements from the Figure & Table Plan. The section reads more like an interim progress report than a finished experimental evaluation.

---

## Critical Issues

### C1. Incomplete results for the headline method (Section 4.2, Table 1, lines 52-54)
**Severity**: CRITICAL

DTA and DTA+ReMDM — the paper's primary contributions — show "pending" in the central results table. A paper claiming to introduce DTA cannot publish without full-scale results for DTA itself. The section currently proves that DMI works, not DTA. This fundamentally undermines the paper's thesis.

**Suggestion**: Complete full-scale DTA and DTA+ReMDM experiments on Countdown-500 before submission. If results remain unavailable, restructure the narrative to position DMI as the primary contribution and DTA as a theoretically motivated direction with preliminary evidence.

### C2. SCP interim accuracy discrepancy (Section 4.2, line 51 vs. data source)
**Severity**: CRITICAL

Table 1 reports SCP interim accuracy as "~8.4%", but the data source (`interim_countdown_results.json`) shows `"interim_accuracy": "7.3-9.3%"`. The section cherry-picks the midpoint without reporting the range. Furthermore, SCP is marked with dagger for "results still in progress," yet the prose on line 62 presents it with apparent confidence ("approximately 150/500 samples per seed suggest accuracy around 8.4%"). The text should report the observed range (7.3-9.3%) and more prominently caveat the interim nature.

**Suggestion**: Replace "~8.4%" with the range "7.3-9.3%" or the range midpoint with explicit uncertainty. Add a sentence noting that the final SCP accuracy may shift as the remaining ~350 samples per seed complete.

### C3. Statistical significance tests promised but never delivered (Section 4.1 vs. 4.2, line 31)
**Severity**: CRITICAL

Section 4.1 promises "McNemar's test with Bootstrap 95% confidence intervals and Bonferroni correction for multiple comparisons." However, Section 4.2 presents no p-values, no confidence intervals, and no formal significance tests. The claim that "Neither result [ReMDM-conf, RCR] is statistically significant" (line 60) is asserted without evidence. Similarly, DMI's improvement is described as a "98% relative improvement" without a significance test.

**Suggestion**: Run McNemar's test on paired per-sample outcomes (vanilla vs. DMI, vanilla vs. ReMDM-conf, vanilla vs. RCR) and report p-values with Bonferroni correction. Add Bootstrap 95% CIs for the mean accuracy differences. If these tests have not been performed, remove the promise from Section 4.1.

---

## Major Issues

### M1. Pilot-to-full-scale inconsistency in Table 1 timing data (lines 43-54)
**Severity**: MAJOR

Table 1 lists Time/sample values (e.g., DTA: 23.0s, SCP: 45.9s), but these come from the pilot summary (`task_5a_summary.md`, N=16) rather than the full-scale experiments. Full-scale runs may have different per-sample timing due to GPU memory pressure, batching effects, etc. The table mixes full-scale accuracy data (for vanilla, ReMDM-conf, RCR, DMI) with pilot-scale timing data without disclosure.

**Suggestion**: Measure and report timing from full-scale runs. If pilot timing must be used, add a footnote: "Timing measured on pilot-scale (N=16) runs; full-scale timing may differ."

### M2. Cross-benchmark results (Section 4.3) lack RCR and DMI (lines 66-93)
**Severity**: MAJOR

Table 2 omits RCR and DMI for GSM8K and MBPP, even though these methods are tested on Countdown at full scale and are key contributions. The absence makes the cross-benchmark comparison incomplete — the reader cannot assess whether DMI's strong Countdown performance transfers to other tasks.

**Suggestion**: Run DMI and RCR pilots on GSM8K and MBPP, or explicitly explain their omission (e.g., "DMI requires task-specific tuning not yet completed for GSM8K/MBPP").

### M3. Pilot data presented without adequate statistical framing (Sections 4.3, 4.4, 4.5)
**Severity**: MAJOR

Three of five subsections rely on N=16 pilot data. While the caveat note on line 93 is helpful, it is buried at the end of Section 4.3 and not repeated in Sections 4.4 or 4.5. The reader may forget that Table 3 and Table 4 are also N=16. At N=16, a single-problem difference is 6.25%, making most comparisons noise.

**Suggestion**: (1) Add a visual marker (e.g., shaded background or "Pilot" label) to Tables 2-4. (2) Repeat the pilot caveat prominently in Sections 4.4 and 4.5 introductions. (3) Consider whether N=16 results add enough value to include, or whether they should be moved to an appendix with a one-sentence summary in the main text.

### M4. Missing Figure 5 — Scaling curves (Section 4.5)
**Severity**: MAJOR

The Figure & Table Plan specifies Figure 5 as an inference-time scaling curve (line plot with dual y-axis). The section contains only Table 4. The HTML comment on line 143 acknowledges this gap ("pilot data is insufficient for meaningful visualization"), but scaling curves are standard for inference-time scaling papers and would greatly aid readability even with pilot data (with appropriate error bars/caveats).

**Suggestion**: Generate Figure 5 from the pilot data with a "preliminary" label and wide confidence bands or individual data-point markers showing the N=16 noise. Alternatively, if full-scale scaling data is forthcoming, leave a placeholder with "[Figure 5: pending full-scale data]" in the main text rather than hiding the gap in a comment.

### M5. Table 1 formatting inconsistency — bold on wrong method (line 50)
**Severity**: MAJOR

The bolding in Table 1 marks DMI as the best result, which is correct for completed methods. However, the table header says "Bold indicates best result" without qualifying "among completed methods." If DTA or DTA+ReMDM ultimately outperform DMI, the bolding will need to change. More importantly, DMI's per-seed values (7.8, 9.6, 10.6) are bolded, but these should not be individually bolded since they are not the highest per-seed values in an absolute sense — they are just the DMI values.

**Suggestion**: Bold only the Mean column value for the best method. Remove bold from individual seed columns unless doing a per-seed comparison. Add qualifier: "Bold indicates best result among methods with completed full-scale evaluation."

---

## Minor Issues

### m1. Overhead multiplier inconsistency (Table header vs. prose)
**Severity**: MINOR

The setup table (lines 17-25) lists SCP overhead as "~7x" but DMI as "~1.2x". In the prose (line 62), SCP is described as "~7x the computational overhead of vanilla (45.9s vs 3.7s per sample)." But 45.9/3.7 = 12.4x, not 7x. The "~7x" likely refers to forward-pass FLOP overhead rather than wall-clock time.

**Suggestion**: Clarify whether "overhead" refers to wall-clock time or FLOPs. If wall-clock, SCP is ~12x, not ~7x. If FLOPs, state this explicitly.

### m2. "98% relative improvement" framing (line 58)
**Severity**: MINOR

"98% relative improvement" (from 4.7% to 9.3%) sounds impressive but is misleading when absolute accuracy is under 10%. A reader might infer a large practical gain. The absolute improvement is 4.6 percentage points on a task where even the best method solves fewer than 1 in 10 problems.

**Suggestion**: Lead with the absolute improvement (+4.6pp) and report relative improvement parenthetically, or omit the relative figure.

### m3. Missing RCR from cross-benchmark and cross-model tables
**Severity**: MINOR

RCR appears in Table 1 but vanishes from Tables 2-4. This inconsistency makes it harder to track method performance across sections.

**Suggestion**: Either include RCR in all tables or note its omission explicitly.

### m4. Hardware description lacks batch size and parallelization details (line 33)
**Severity**: MINOR

The hardware section mentions 4x GPUs but does not specify batch size, whether methods are parallelized across GPUs, or whether timing is measured on a single GPU. This matters for interpreting the reported Time/sample values.

**Suggestion**: Add: "All timing measurements are single-GPU, single-sample. Methods were parallelized across 4 GPUs for throughput but timing reflects single-GPU latency."

### m5. Table 4 — DTA at T=128 shows 0.0% (line 131)
**Severity**: MINOR

DTA accuracy is 0.0% at T=128 (the default configuration), but 6.2% at both T=64 and T=256. This is the same configuration as the main comparison (Table 1, where DTA is "pending" at full scale). Verified against source data (`task_6a_scaling_curve.json`: T=128 DTA correct=0). This 0.0% result, while within noise at N=16, is worth flagging for the reader since T=128 is the default.

**Suggestion**: Add a brief note: "DTA's 0.0% at T=128 reflects the high variance of N=16 evaluation rather than a genuine failure mode, as the default DTA configuration shows positive results at other step counts."

### m6. "origin" sampling algorithm not defined (line 29)
**Severity**: MINOR

The generation configuration mentions "the Dream-native `origin` sampling algorithm" without citation or definition. A reader unfamiliar with Dream's codebase will not understand what this means.

**Suggestion**: Add a brief parenthetical explanation or cite the Dream paper with the relevant section.

---

## Visual Communication Review

### Missing visual elements from Figure & Table Plan:
1. **Figure 5 (Scaling curves)**: Not generated. The outline specifies a dual-y-axis line plot. Only a raw table is present. (See M4 above.)

### Present visual elements:
1. **Table 1**: Present and well-formatted. Grouping by method category is helpful.
2. **Table 2**: Present. Adequate for pilot data.
3. **Table 3**: Present. Side-by-side Dream vs. LLaDA comparison is clear.
4. **Table 4**: Present. Would benefit from Figure 5 as companion visualization.

### Suggestions for visual improvement:
- Add a **bar chart** comparing method accuracy on Countdown-500 (full-scale) for the completed methods (Vanilla, ReMDM-conf, RCR, DMI). This would be more impactful than the raw table for the key finding.
- Consider a **radar/spider chart** or **grouped bar chart** for the cross-benchmark comparison (Table 2) to highlight the task-dependent effectiveness pattern visually.
- Tables are referenced before they appear (good). Captions could be more descriptive — e.g., Table 1's caption should mention the key finding (DMI dominance) rather than just describing the table contents.

---

## Summary

| Category | Count |
|----------|-------|
| Critical | 3 |
| Major | 5 |
| Minor | 6 |

The section's structure and writing quality are good, but the incomplete experimental results (DTA/SCP pending) and reliance on underpowered pilots severely weaken the empirical contribution. The most urgent priorities are: (1) completing full-scale DTA results, (2) running the promised statistical significance tests, and (3) generating Figure 5. Once full-scale data is available, the section has the potential to score 8-9/10.
