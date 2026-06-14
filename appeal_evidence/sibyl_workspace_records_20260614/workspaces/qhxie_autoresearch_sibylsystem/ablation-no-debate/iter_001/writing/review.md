# Writing Quality Review

## Summary

This paper investigates feature absorption in Sparse Autoencoders using multi-child proportional ablation, a new measurement methodology that resolves saturation problems in prior single-feature ablation approaches. On synthetic hierarchies with ground truth geometry, trained SAEs show absorption rates of 0.50 compared to 0.059 for random baselines (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine representational phenomenon. However, two key predictions are falsified: absorption does not inversely correlate with feature frequency (rho = +0.17, contradicting competitive exclusion theory), and steering interventions do not restore sensitivity for absorbed features (zero improvement). The authors conclude that absorption may be epistemic rather than causal, contributing to growing concerns about SAE reliability for safety-critical interpretability.

## Detailed Assessment

### Structural Coherence: 8/10

The paper has a clear logical structure: Introduction establishes the problem and measurement crisis, Background situates the work in related literature, Methodology describes the experimental design, Experiments present results for three hypotheses, and Discussion interprets findings and limitations. The argument flows from problem statement through measurement solution to experimental validation.

The abstract accurately represents the paper's content and results, correctly stating the key numbers (0.50 vs 0.059, Cohen's d = 8.94) and the main conclusions. Transitions between sections are generally smooth, with Section 4.2 leading naturally into 4.3 and 4.4.

**Minor issues**:
- Section 5.5 (Limitations) appears after Section 5.6 (Future Directions), which is unconventional. Typically limitations precede future directions.
- The paper would benefit from a clearer preview paragraph at the end of the Introduction, explicitly stating the three hypotheses and their fates.

### Notation & Terminology Consistency: 7/10

Most terminology is consistent with glossary.md. "Feature absorption" is correctly used throughout without incorrect variants. "Parent feature" and "child feature" are used consistently. "Multi-child proportional ablation" is correctly hyphenated.

**Issues identified**:

1. **Notation mismatch**: The paper uses `act(p | c_1, ..., c_k ablated)` in the main text (Section 3.4, line 144) but notation.md defines `abs_k` with formal notation. The mathematical notation could be cleaner.

2. **Missing notation for proportional variance**: The paper uses "var(prop)" and "proportional variance" but notation.md defines `var(prop)` as variance of proportional contributions. This is used in Table 3 caption and Section 4.2.3 but never formally defined in the Methodology section.

3. **Inconsistent steering alpha range**: The Methodology (Section 3.5) states alpha sweeps {0.0, 0.1, 0.2} but the notation.md specifies {0.05, 0.1, 0.15, 0.2, 0.25}. The implementation used {0.0, 0.1, 0.2} so this is a documentation inconsistency between paper and notation.md.

4. **Notation for "parent direction"**: The paper uses `parent_dir` (underscore) in equations but notation.md defines `parent_direction` (no underscore). This is a minor inconsistency.

### Claim-Evidence Integrity: 9/10

The paper demonstrates strong alignment between claims and source data from exp/results/summary.md.

**Verified claims**:

| Claim | Paper Value | Summary.md Value | Status |
|-------|-------------|------------------|--------|
| Trained SAE absorption rate | 0.5000 | 0.5000 | MATCH |
| Random Decoder absorption rate | 0.0590 | 0.0590 | MATCH |
| Cohen's d (Trained vs Random) | 8.94 | 8.94 | MATCH |
| p-value | 3.16e-133 | 3.16e-133 | MATCH |
| Spearman rho | +0.1711 | +0.1711 | MATCH |
| H1 result | SUPPORTED | PASSED | MATCH |
| H2 result | NOT SUPPORTED | FAILED | MATCH |
| H3 result | NOT SUPPORTED | FAILED | MATCH |
| Proportional variance (Trained) | 0.1154 | 0.1154 | MATCH |

**Minor issue**: The paper claims "only 7 absorbed features were tested" in H3 but exp/results/summary.md does not explicitly validate this count. The number is stated but the derivation (threshold of 0.5) is clear.

### Visual Communication: 6/10

The paper references figures in a somewhat confusing order:

**Figure ordering problems**:
1. Figure 5 (method architecture diagram) is referenced in Section 3.4 (line 160-162) BEFORE Figure 1 appears in Section 4.2.2 (line 243). While Figure 5 is methodological and precedes experimental results, this creates reader confusion about what Figure 1-4 will show.

2. Figure 3 (proportional variance) is referenced in Section 4.2.3 (line 249) but appears before the discussion of proportional variance results, which is appropriate.

3. Figure 2 (scatter plot for H2) is correctly placed before H2 results discussion.

**Missing elements**:
- Figure 1 is referenced in Section 4.2.2 ("Figure 1 provides visual comparison") but Table 1 and Table 2 are presented before the figure reference. The tables should be referenced before the figure to maintain logical flow.

**Positive aspects**:
- All five figures have clear captions explaining the key takeaway
- Figure captions reference the specific statistical findings
- Tables are well-structured with clear headers

### Writing Quality: 7/10

**Banned patterns found**:
1. "a growing body of work" (Section 1, line 374) - This is a filler phrase. Should be more specific: "Korznikov et al. (2026) showed that..."

2. "contribute to a growing body of work" (Section 5.4, line 372) - Same filler pattern.

**Unclear sentences**:

1. "This finding is statistically robust: both Spearman and Pearson correlations are positive and highly significant." (Section 4.3.2, line 287)
   - Problem: "highly significant" is vague. Should state the exact p-values.

2. "The proportional variance measures how asymmetrically children substitute for the parent" (Section 3.4, line 155)
   - Problem: This is circular. Higher variance means... higher variance. What does this mean practically?

3. "Single-child ablation, which ablates the top-k children simultaneously" (Section 5.1, line 350)
   - Problem: Typo/error. "single-child ablation" does NOT ablate top-k children. Should say "Multi-child ablation."

**Sentence-level issues**:
- The abstract (line 4-6) is very dense with multiple subordinate clauses. Consider splitting.
- Section 4.5.2 (line 328-330) contains two possible interpretations without clearly choosing or ranking them.

## Issues for the Editor

1. **[Critical] Figure order confusion**: Figure 5 (method diagram) is referenced before Figure 1 (main results). Consider reorganizing so methodology figures appear before experimental results OR add a forward reference paragraph at the end of Methodology: "We present four figures in the Experiments section: Figure 1 shows [X], Figure 2 shows [Y]..."

2. **[Major] Table before figure reference**: Table 1 is presented before "Figure 1 provides visual comparison" is mentioned. Reverse the order or add "Table 1 presents these results" before the table.

3. **[Major] Typo in Discussion**: Section 5.1 line 350 says "Single-child ablation, which ablates the top-k children simultaneously" - this describes MULTI-CHILD ablation, not single-child. Change "Single-child ablation" to "Multi-child ablation."

4. **[Minor] Missing notation for proportional variance**: Section 3.4 defines proportional contribution and proportional absorption but not "proportional variance." Add: "We define proportional variance as Var(prop_1, ..., prop_k), which measures the asymmetry of child contributions to parent reconstruction."

5. **[Minor] Clarify n=7 limitation**: H3 discussion mentions "only 7 absorbed features" but the implications for statistical power are understated. Add: "The small n=7 absorbed features provides insufficient statistical power to detect moderate effect sizes."

6. **[Minor] Remove banned patterns**: Replace "a growing body of work" with specific citations or stated facts.

7. **[Minor] Abstract density**: The abstract has long compound sentences. Consider breaking the final sentence into two shorter sentences.

## What Works Well

1. **Excellent claim-evidence alignment**: Every quantitative claim (H1, H2, H3 results, absorption rates, effect sizes) matches the source data in exp/results/summary.md exactly. This is a model of rigorous experimental reporting.

2. **Clear hypothesis structure**: The paper correctly states each hypothesis with its falsification criterion, then presents unambiguous results (SUPPORTED / NOT SUPPORTED). This pre-registration-style presentation strengthens credibility.

3. **Transparent negative result reporting**: The Discussion (Section 4.5) handles failed hypotheses with appropriate nuance, suggesting multiple interpretations without overclaiming. The paper correctly avoids calling results "failed experiments" and instead frames them as falsified predictions.

4. **Methodological clarity**: The saturation problem and its multi-child solution are explained clearly, with the key insight stated explicitly: "ablating one child allows remaining children to reconstruct the parent."

SCORE: 7
