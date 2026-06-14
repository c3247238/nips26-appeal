# Writing Quality Review

## Summary

This paper investigates whether feature absorption in sparse autoencoders (SAEs) degrades downstream interpretability tasks---specifically feature steering and sparse probing---using GPT-2 Small SAEs (gpt2-small-res-jb, 24,576 latents). The authors measure absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10, then test for correlations with steering success and probing F1. The results are mixed: raw steering and probing show no significant correlation with absorption, but delta-corrected steering (subtracting a random baseline) reveals a significant negative correlation at layer 8 (Pearson r = -0.431, p = 0.028; Spearman rho = -0.502, p = 0.009). The relationship is inconsistent across layers (H3 not supported). The paper argues that absorption's impact is subtle, layer-dependent, and only detectable after controlling for baseline steering effects.

---

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a standard IMRAD structure with clear logical flow. The abstract accurately represents the paper's content and results. The argument structure (problem -> approach -> evidence -> conclusion) is clear and well-motivated. Section 3 (Research Questions and Hypotheses) with formal falsification criteria adds rigor. The Results-Discussion separation is clean.

Remaining issues:

1. **The Conclusion's "Closing Thought" (Section 8.3) introduces a new argumentative frame.** The phrase "Null results are valuable, but so are carefully controlled positive findings" is elegant but appears nowhere else in the paper. This is acceptable as a closing reflection but slightly shifts the paper's framing at the last moment.

2. **Section 6.3 (Comparison with Pilot) is well-placed but could be shortened.** The pilot comparison adds methodological credibility but repeats numbers already in the Results. A single sentence summarizing the key insight (baseline subtraction was the critical addition) would suffice.

3. **The Discussion's "What Would Change Our Conclusion?" (Section 6.4) is excellent but isolated.** No other section uses this proactive falsifiability framing. Consider adding a brief forward-looking sentence to the Abstract.

### Notation & Terminology Consistency: 8/10

Significant improvement since the first review. The model size is now consistent ("124M parameters, 85M non-embedding"). The H3 notation is clean. The "not supported" terminology is consistent. The absorption classification thresholds are aligned across paper, notation.md, and glossary.md (HIGH >= 10%, MEDIUM 5--10%, LOW < 5%). The CV definition now uses the absolute-value denominator consistently.

Remaining issues:

1. **"res-jb" vs. "gpt2-small-res-jb" inconsistency persists in minor places.** The Abstract uses "res-jb" while Section 4.2 uses "gpt2-small-res-jb SAE release." The glossary defines "res-jb SAEs" as a family. Standardize: use "gpt2-small-res-jb" on first mention in each major section, then "res-jb" thereafter.

2. **"first-letter" vs. "first letter" hyphenation.** The glossary specifies "first-letter" (hyphenated when used as modifier). The paper is mostly consistent but has at least one instance of unhyphenated "first letter" (Section 4.3: "first letter features" should be "first-letter features").

3. **Table 1a caption uses "Verdict" while the text uses "Result" and "Not supported."** Standardize on "Result" or "Verdict" consistently.

### Claim-Evidence Integrity: 8/10

Reported numbers are consistent with source data (correlation_report_full.json, correlation_with_baseline.json). The correlation values match across Abstract, Results, and Conclusion. Table 1 and Table 1a values are accurate. The random baseline validation (Table 4) is properly cited with effect sizes.

Remaining issues:

1. **The DeepMind claim has been softened but still lacks citation.** Section 1.1 now states "These concerns have led some research groups to deprioritize SAE research after finding negative results on downstream tasks (Bricken et al., 2023; Lieberum et al., 2023)." However, neither Bricken et al. (2023) nor Lieberum et al. (2023) are in the reference list at the end of the paper. Either add full citations or remove the parenthetical references.

2. **"First systematic study" claim is now properly scoped.** The Abstract says "the first systematic study connecting feature absorption detection to downstream task performance for steering effectiveness and sparse probing accuracy." This is appropriately scoped to the specific tasks tested.

3. **Power analysis is precise.** Section 4.6 states "approximately 64% power" which matches the exact calculation for n=26, r=0.50, alpha=0.05 (two-tailed).

4. **The random feature baseline is now properly addressed.** Table 4 validates that feature-specific steering exceeds random baseline (p < 0.0001, Cohen's d > 1.1). The delta steering metric is well-justified.

### Visual Communication: 8/10

All 6 figures and 5 tables from the outline are present. Figure numbering is sequential with no gaps. All figures are referenced before they appear. Captions are self-explanatory. The color scheme is uniform.

Remaining issues:

1. **No confidence intervals on regression lines.** Figures 3, 4, and 5 show regression lines without confidence bands. For a mixed-result paper, visualizing the uncertainty would strengthen the message.

2. **Figure 6 (dose-response) shows category means but not individual trajectories.** The category means (HIGH/MEDIUM/LOW) may mask heterogeneous behavior within categories. Individual feature trajectories would reveal more.

3. **Table 1a is functional but visually dense.** The H3 consistency table packs 4 hypotheses x 5 columns into a single table. Consider splitting into two smaller tables (one for sign consistency, one for magnitude consistency) for readability.

### Writing Quality: 8/10

The writing is clear, direct, and largely free of banned patterns. No "In recent years," "Furthermore," or "It is worth noting that" survive. Hype words are absent. The "challenge the assumption" repetition has been addressed.

Minor issues:

1. **"The results are mixed" appears 3+ times** (Abstract, Introduction 1.3, Conclusion 8.1). Vary the phrasing: "Our findings are heterogeneous," "The pattern of results is inconsistent across hypotheses."

2. **"Practitioners should use delta-corrected steering metrics" (Conclusion 8.2) is strong but justified.** The evidence supports this recommendation. However, "should adopt" could be softened to "would benefit from adopting" to match the paper's otherwise cautious tone.

3. **"This task-specificity is itself an important finding" (Conclusion 8.2) is slightly self-congratulatory.** The evidence supports the claim, but the phrasing could be more neutral: "This task-specificity has direct implications for future SAE research."

---

## Issues for the Editor

1. **[Minor] Add full citations for Bricken et al. (2023) and Lieberum et al. (2023) to the reference list, or remove the parenthetical references in Section 1.1.** Currently these citations appear in the text but not in the reference list.

2. **[Minor] Fix "first letter features" -> "first-letter features" in Section 4.3.** The glossary specifies hyphenated form when used as a modifier.

3. **[Minor] Standardize "res-jb" vs. "gpt2-small-res-jb" in the Abstract.** The Abstract uses "res-jb" without the full prefix; consider "gpt2-small-res-jb (res-jb)" on first mention.

4. **[Minor] Vary "mixed results" phrasing.** The phrase appears in Abstract, Introduction, and Conclusion. Use synonyms to avoid repetition.

5. **[Minor] Consider adding confidence bands to Figures 3--5.** Not critical for the review, but would strengthen the visual argument.

---

## What Works Well

1. **The H1 vs. H1b contrast is the paper's strongest intellectual contribution.** The text clearly shows that the same absorption rates and steering data produce no correlation in raw form (H1) but a significant negative correlation after baseline subtraction (H1b). This is a genuinely important methodological insight for the field.

2. **Honest reporting of mixed results.** The paper does not oversell the single significant finding (H1b at layer 8) nor does it bury it. The Abstract, Results, and Conclusion all accurately report the pattern: null for raw steering and probing, significant only for delta-corrected steering at one layer.

3. **"What Would Change Our Conclusion?" (Section 6.4) remains rare and valuable.** The four conditions (larger models, semantic features, alternative metrics, different tasks) are specific, testable, and demonstrate scientific maturity.

4. **The random baseline validation (Table 4) is well-executed.** Feature-specific steering exceeds random by 132% (layer 4) and 126% (layer 8) with large effect sizes (Cohen's d > 1.1), validating that the decoder directions capture meaningful structure.

5. **The U-at-layer-8 counterexample is powerful and memorable.** "The most absorbed feature in our sample (U at layer 8, A(U) = 0.242) achieves 100% raw steering success at s = 50" immediately challenges the absorption-degradation intuition.

---

SCORE: 8
