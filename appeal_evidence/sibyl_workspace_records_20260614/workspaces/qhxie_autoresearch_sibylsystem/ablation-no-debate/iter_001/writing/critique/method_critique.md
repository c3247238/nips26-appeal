# Critique: Methodology

## Summary Assessment
The Methodology section is technically sound and logically organized, correctly presenting the experimental design including synthetic hierarchy generation, SAE training configuration, baseline methods, multi-child proportional ablation methodology, steering intervention protocol, and statistical analysis framework. The main weaknesses are: (1) a critical inconsistency between alpha values in the method versus experiments section, (2) premature reporting of experimental results in the methodology section, and (3) the architectural diagram lacks proper figure numbering convention.

## Score: 7/10
**Justification**: The methodology is well-structured and technically accurate against the proposal. However, the alpha inconsistency between method (5 values) and experiments (3 values) is a critical issue that suggests either a design change was not propagated or results were computed with a different protocol than described. Reaching the next level requires resolving this inconsistency and ensuring methodology presents the design, not preliminary results.

## Critical Issues

### Issue 1: Alpha Value Mismatch Between Method and Experiments
- **Location**: Section 5.5 (Steering Protocol), line 104 vs. Experiments Section 6.4.1, line 115
- **Quote**: Method: "alpha in {0.05, 0.1, 0.15, 0.2, 0.25}" vs. Experiments: "alpha in {0.0, 0.1, 0.2}"
- **Problem**: The method specifies 5 alpha values (0.05, 0.1, 0.15, 0.2, 0.25), but experiments Section 6.4.1 reports only 3 values (0.0, 0.1, 0.2). This is a significant discrepancy that suggests either: (a) the method was updated but experiments were not re-run, or (b) the method was written from an earlier design iteration.
- **Fix**: Reconcile the alpha values. If the experiment used 3 values, update Section 5.5 to match. If the method's 5 values are correct, verify experiments used the same protocol.

### Issue 2: Experimental Results Reported in Methodology Section
- **Location**: Section 5.4 (Multi-Child Ablation), lines 72-73 and Section 5.4 (Proportional Variant), line 80
- **Quote**: "trained SAEs show abs_5 = 0.50 while random decoder shows abs_5 = 0.059" and "var(prop) = 0.115 versus 0.004 for random decoder"
- **Problem**: The methodology section should describe the experimental design, not report results. These specific numerical results (0.50, 0.059, 0.115, 0.004) belong in the Experiments section, not the Methodology. Reporting results in method creates confusion about what was pre-registered versus post-hoc.
- **Fix**: Rewrite lines 72-73 to describe the expected behavior without specific numbers: "When k=1, both trained and random SAEs saturate at abs_1 = 1.0. We predict trained SAEs will maintain measurable absorption while random decoder approaches zero." Similarly for line 80, remove specific variance values.

## Major Issues

### Issue 3: Missing Figure Number Convention
- **Location**: Line 120, "![Multi-Child Proportional Ablation Procedure](figures/fig5_method_architecture.pdf)"
- **Quote**: The markdown image syntax lacks the "Figure 5:" prefix used elsewhere in the paper
- **Problem**: According to the glossary, figures should follow "Figure X" format. The outline specifies "Figure 5: Architecture Diagram" but the actual markdown uses only alt-text, not a figure number prefix. This is inconsistent with experiments section (e.g., "Figure 1 presents...") and may confuse LaTeX conversion.
- **Fix**: Change to proper figure reference: "**Figure 5:** Multi-Child Proportional Ablation Procedure" with the image embedded.

### Issue 4: Sec 5.5 Steering Protocol Missing Key Detail
- **Location**: Section 5.5 (Steering Protocol), lines 100-104
- **Quote**: "We apply steering by adding scaled directions to activations" with alpha values listed
- **Problem**: The section describes the steering mechanism but does not specify: (a) which alpha value(s) produced the reported results, (b) whether alpha values were swept or a fixed value was used, or (c) how multiple alpha values were aggregated if all were tested.
- **Fix**: Add: "We sweep alpha across the range {0.05, 0.1, 0.15, 0.2, 0.25} and report the maximum sensitivity improvement observed across alpha values for each feature."

## Minor Issues

- **Line 31**: "We use the SAELens framework for implementation" — consider citing the specific SAELens paper or GitHub commit for reproducibility.
- **Line 88**: "In our pilot, this identified 7 absorbed features" — this pilot reference is appropriate but could be confused with pre-registered analysis. Consider changing to "In our pilot validation, this threshold identified..."
- **Line 118**: "Pass thresholds are pre-registered" — this statement belongs here but should also appear in the Experiments section as a reminder. Cross-reference: "See pre-registration details in Section 5.6."

## Visual Element Assessment
- [x] Figure 5 referenced (architecture diagram of multi-child ablation)
- [x] All visuals referenced before appearance (Figure 5 appears at end of Section 5.4)
- [ ] Figure 5 lacks proper "Figure X:" numbering convention
- [ ] Method section would benefit from a table summarizing the experimental design (similar to outline's Figure & Table Plan)

## What Works Well

1. **Logical organization**: The progression from synthetic hierarchy generation (5.1) through SAE training (5.2), baselines (5.3), measurement methodology (5.4), intervention (5.5), to statistical analysis (5.6) follows a clear experimental design logic that readers can easily follow.

2. **Technical precision in geometric constraints**: The table in lines 15-19 precisely specifies the hierarchy geometry (cosine similarities, orthogonality constraints) with mathematical rigor, enabling exact replication.

3. **Clear explanation of the saturation problem**: Section 5.4's explanation of why single-feature ablation saturates (lines 62-65) is well-written and sets up the key insight for multi-child ablation effectively. The intuition that "remaining children compensate" is communicated clearly.

4. **Baseline methodology justification**: Section 5.3's explanation of what each baseline tests (lines 52-58) is excellent. The mapping from baseline design to specific null hypotheses (a), (b), (c) makes the experimental logic transparent.

## Cross-Section Consistency Check

| Element | Method Section | Experiments Section | Status |
|---------|---------------|---------------------|--------|
| Absorption formula (abs_k) | Line 70 | Line 33 (identical) | Consistent |
| Proportional contribution formula | Line 78 | Not repeated | Consistent |
| SAE architecture parameters | Lines 35-44 | Lines 13 (identical) | Consistent |
| Random seeds | {42, 43, 44, 45, 46} | Same (line 13) | Consistent |
| Alpha values | {0.05, 0.1, 0.15, 0.2, 0.25} | {0.0, 0.1, 0.2} (line 115) | **INCONSISTENT** |
| Parent direction formula | Line 94 | Line 113 (identical) | Consistent |
| Absorbed feature threshold | 0.5 | 0.5 (line 111) | Consistent |
| Number of absorbed features | 7 absorbed, 1014 non-absorbed | Same (Table 4, line 127) | Consistent |
| Absorption rates (trained SAE) | 0.50 (line 72) | 0.5000 (Table 1) | Consistent |
| Random decoder absorption | 0.059 (line 72) | 0.0590 (Table 1) | Consistent |
