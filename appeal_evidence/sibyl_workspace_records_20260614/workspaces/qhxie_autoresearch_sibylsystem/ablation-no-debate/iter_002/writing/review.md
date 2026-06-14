# Writing Quality Review

## Summary

This paper presents a controlled factorial decomposition of feature absorption in sparse autoencoders (SAEs), identifying the encoder as the dominant driver through a 2x2 design crossing trained/random encoder with trained/random decoder. The paper is well-organized and clearly written, with a logical flow from problem statement through methodology, experiments, and discussion. The central finding -- encoder effect 80x larger than decoder effect -- is clearly supported by the factorial data. However, the paper contains an internal inconsistency in the H3 steering results (sensitivity ratio and p-value differ between abstract/Section 4.3 and the experiment summary data), a figure reference mismatch at Section 3.4, and relies on three different absorption metrics without establishing their equivalence, creating confusion when comparing results across experiments.

## Detailed Assessment

### Structural Coherence: 8/10

The paper has a clear, logical structure: Introduction establishes the problem and research questions; Background covers related work; Methodology details the factorial design and metrics; Experiments present results grouped by hypothesis; Discussion interprets findings; Conclusion summarizes. The abstract accurately represents the paper's content and results.

The argument structure is coherent: problem (absorption threatens SAE reliability) → approach (2x2 factorial decomposition with synthetic ground-truth) → evidence (encoder effect 0.843 vs decoder effect 0.011) → implications (encoder regularization, not decoder modification). Transitions between sections are generally smooth.

Two structural issues reduce the score: (1) Section 3.4 references "Figure 7" for the 2x2 factorial design, but the image displayed is `figure_7_heldout_generalization.pdf` showing train vs. test scatter -- the factorial architecture diagram appears to be missing or mislabeled; (2) the outline plans 8 figures (including a dedicated architecture diagram for the factorial design), but the paper presents only 7, with the held-out generalization figure apparently substituting for the factorial diagram at the methodology reference point.

### Notation & Terminology Consistency: 7/10

Most terminology is consistent with glossary.md:
- "absorption rate" used as decimal (0.50), not percentage -- correct per glossary
- "L0" used correctly (not "L0 norm" or "sparsity level")
- "safety-critical" hyphenated correctly throughout
- "multi-child" hyphenated correctly
- "TopK activation" (not "Top-k sparsity") -- consistent
- "Jaccard overlap" (not "Jaccard similarity") -- consistent
- "Mann-Whitney U test" (not "Wilcoxon") -- consistent

However, the notation document uses `cos(u, v)` for cosine similarity while the paper text uses "cosine similarity" in prose. This is acceptable but could be tightened.

The notation document defines symbols for the 2x2 factorial conditions (A, B, C, D) and the decomposition formula on page 4, which is referenced correctly in Section 4.1.

The paper is largely consistent internally, but the **three different absorption metrics** (Jaccard overlap, overlap fraction, cosine-based proportional absorption) create confusion when comparing across experiments. The paper acknowledges this in Section 3.3 ("direct comparison across experiments should be interpreted with this in mind"), but this admission also serves as a warning flag: readers cannot directly compare, for example, the factorial decomposition absorption values (Jaccard-based) with the safety analysis absorption values (cosine-based). This is a Major issue for a paper whose central claim relies on comparing effect sizes across experiments.

### Claim-Evidence Integrity: 6/10

**CRITICAL INCONSISTENCY -- H3 Steering Results:**

The abstract (line 7) states: "steering interventions do not differentially affect absorbed features (sensitivity ratio=0.776, p=0.273)"

Section 4.3 (line 222) states: "The primary condition (parent input, parent-direction steer) yields a ratio of 0.776 +/- 0.066 (p = 0.273)"

But `exp/results/full/experiment_summary.json` records for h3_steering:
```
"ratio": 0.9744269212513145,
"p_value": 0.9355699146104653
```

These numbers are irreconcilable. Either the paper reports incorrect values for the H3 steering experiment, or the experiment_summary.json reflects a different experimental run. A ratio of 0.776 vs 0.974 and p=0.273 vs p=0.936 are substantively different interpretations -- one suggests a non-significant trend toward lower sensitivity for absorbed features (ratio < 1), the other is a near-perfect null result (ratio ≈ 1). **This must be resolved before publication.**

**Other claim-evidence checks (verified against experiment_summary.json):**

| Claim | Paper Value | JSON Value | Status |
|-------|-------------|------------|--------|
| Encoder effect (H_Mech) | 0.843 ± 0.082 | 0.843 | MATCH |
| Decoder effect (H_Mech) | 0.011 ± 0.015 | 0.011 | MATCH |
| Trained SAE absorption (H1) | 0.477 ± 0.022 | 0.477 | MATCH |
| Random baseline (H1) | 0.033 ± 0.011 | 0.033 | MATCH |
| Safety absorption | 0.967 ± 0.010 | 0.967 | MATCH |
| Non-safety absorption | 0.968 ± 0.013 | 0.968 | MATCH |
| H3 steering ratio | 0.776 (abstract) | 0.974 | MISMATCH |
| H3 steering p-value | 0.273 (abstract) | 0.936 | MISMATCH |
| L0=20 absorption | 0.552 | 0.552 | MATCH |
| L0=50 absorption | 0.419 | 0.419 | MATCH |
| Hierarchy cos=0.5 | 0.416 | 0.416 | MATCH |
| Hierarchy cos=0.8 | 0.544 | 0.544 | MATCH |

**H3 steering is the only irreconcilable mismatch.** All other numerical claims match the experiment_summary.json source data.

For the multi-seed claim "t=36.04, p=3.85e-10" -- this matches the JSON's p=3.85e-10 exactly, so it is verified.

The claim "Cohen's d > 10" for multi-seed effect size is stated but not numerically verified in the JSON. This is an estimation rather than a direct report and may be acceptable as a descriptive characterization.

### Visual Communication: 6/10

The paper has the required visual elements: 1 method diagram (the 2x2 factorial design, though the reference is problematic -- see above), 1 results table (Table 1 with all hypotheses), and analysis figures (Figures 2-6 for multi-seed, steering, safety, hierarchy, L0).

**Critical figure reference issue at Section 3.4 (line 135):**
The text states "Figure 7 illustrates the [2x2 factorial] design" and displays an image tag showing `figure_7_heldout_generalization.pdf` -- a train vs. test scatter plot. This means the 2x2 factorial architecture diagram is either missing from the figure set or mislabeled. The outline specifies Figure 7 should be an architecture diagram of the factorial design; the paper uses a held-out generalization scatter plot at that reference instead. Reviewers will be confused when they look for the factorial diagram and find a scatter plot.

All figures are referenced before they appear in the text -- this is done correctly throughout.

Table 1 (inline summary) is well-constructed and presents all key results in one place, which reviewers will appreciate.

Captions are self-explanatory: Figure 1 includes error bar description, condition labels, and key takeaway; Table 2 includes statistical test results inline.

The text-heavy Discussion section (Section 5) could benefit from a summary figure or table showing the key decomposition visually, since the factorial architecture diagram is problematic.

**Positive**: The paper explicitly flags metric inconsistency for visual elements (Section 3.3 note), which is honest but also signals that figures using different metrics cannot be directly compared.

### Writing Quality: 8/10

The writing is clear and direct. Sentences are mostly short and specific. The paper leads with concrete results ("encoder effect is 0.843") before explaining implications.

**No banned patterns detected.** The paper avoids:
- Generic openings ("In recent years...") -- opens directly with "Feature absorption..."
- Hollow self-praise ("To the best of our knowledge...") -- absent
- Filler transitions ("Moreover", "Furthermore") -- minimal use
- Hype words ("groundbreaking", "game-changing") -- absent
- Vague claims without evidence -- all major claims have specific numbers

**Specific unclear or problematic sentences:**

1. "This is a genuine negative result: absorbed features do not respond differently to steering than non-absorbed features." (Section 4.3, line 222) -- The problem here is that this claim is based on the numbers in the paper (ratio 0.776, p=0.273), which contradict the experiment_summary.json (ratio 0.974, p=0.936). If the JSON is correct, the claim that the result is "genuine" may still hold (both are null results), but the specific characterization is inconsistent.

2. "The decoder disentanglement interpretation is post-hoc and requires future validation with a dedicated statistical test." (Section 5.1, line 280) -- This is honest and well-stated.

3. Section 3.3 note on metric consistency (lines 128-129): "Three different absorption metrics appear in this paper's experiments... direct comparison across experiments should be interpreted with this in mind." -- This is appropriately flagged but the acknowledgment does not resolve the underlying problem that the paper's central narrative requires comparing effect sizes across experiments using different metrics.

**Positive**: The paper's treatment of negative results is exemplary. The H3 and H_Safe negative results are presented honestly, with appropriate statistical detail, and their implications are discussed constructively ("positive implication for SAE-based safety analysis").

## Issues for the Editor

1. **Critical** **[H3 Steering Numbers Inconsistency]**: Abstract and Section 4.3 report sensitivity ratio=0.776, p=0.273; experiment_summary.json reports ratio=0.974, p=0.936. These are irreconcilable without access to the underlying steering experimental runs. Determine which numbers are correct, update accordingly, and verify all H3 steering figures (Figure 3) reflect the correct data. A reviewer will inevitably try to replicate the steering experiment and find this inconsistency.

2. **Critical** **[Section 3.4 Figure Reference Mismatch]**: Text at line 135 says "Figure 7 illustrates the [2x2 factorial] design" but displays `figure_7_heldout_generalization.pdf` (a train vs. test scatter plot). The 2x2 factorial architecture diagram is missing or mislabeled. Either: (a) add the correct factorial architecture diagram and renumber figures, or (b) update the text reference to point to the correct figure number for the factorial results (Figure 1 in the current numbering).

3. **Major** **[Metric Incomparability Across Experiments]**: The paper uses Jaccard overlap, overlap fraction, and cosine-based proportional absorption across different experiments. Section 3.3 acknowledges this but does not resolve it. The central narrative requires comparing effect sizes across experiments (e.g., "encoder effect is 80x larger than decoder effect"), but these are computed on different metrics. Consider: (a) adding a conversion experiment establishing formal equivalence between metrics, or (b) restructuring the narrative to avoid cross-experiment comparisons on different metrics.

4. **Minor** **[Section 3.3 Note Undermines Trust]**: The explicit note that "direct comparison across experiments should be interpreted with this in mind" is methodologically honest but signals to reviewers that the paper's own numbers may not be comparable. Consider removing this note and instead ensuring all experiments use a single consistent metric, or restructuring the paper so that each experiment's claims only reference that experiment's internal comparisons.

5. **Minor** **[Cohen's d > 10 Not Verified]**: Section 4.2 states "Effect size is Cohen's d > 10" but this exact value does not appear in any JSON result file. Either compute and report the exact value or change to "effect size is extremely large (Cohen's d >> 1)" for accuracy.

## What Works Well

1. **The factorial design is a genuine methodological contribution** (Section 3.4, 4.1): The 2x2 decomposition isolating encoder vs. decoder contributions is novel and well-executed. The validation criteria (encoder effect > 0.5, decoder effect < 0.1) directly test the paper's central claim. The honest acknowledgment that original criteria failed and were revised is commendable.

2. **Negative results are treated as first-class findings** (Sections 4.3, 4.4): The H3 steering and H_Safe safety experiments are presented with the same rigor and detail as confirmatory results. The discussion of their implications (Section 5.3, 5.4) is constructive rather than defensive, framing null results as positive findings for the field.

3. **Tables 1 and 2 are exemplary** (Section 4, line 171-200): The inline summary table and the factorial decomposition table present all key statistics clearly. Reviewers can verify claims without consulting external files. The statistical tests, effect sizes, and conclusions are all in one place.

SCORE: 7
