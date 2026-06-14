# Writing Critique: Feature Absorption Paper

## Overall Assessment

The paper's writing is functional but has significant clarity issues that will draw reviewer criticism. The methodology presentation is generally clear, but the results and discussion sections contain inconsistencies and unexplained artifacts that undermine confidence.

## Abstract

**Issue**: Overloaded single sentence containing 5 hypotheses and results.

The abstract packs H1, H_Mech, H3, H_Safe, and H2 findings into one sentence. The final clause about safety analysis being "not systematically compromised" is buried.

**Recommendation**: Split into 2-3 sentences:
- Sentence 1: Multi-child proportional ablation methodology and H1/H_Mech findings
- Sentence 2: Safety-critical feature finding (H_Safe)
- Sentence 3: Negative result (H2)

**Issue**: Abstract claims "Cohen's d = 8.94, p < 10^-133" but the exact value in Table 2 is "3.16 × 10^-133".

**Recommendation**: Use consistent precision. Either round consistently in abstract ("p < 10^-133") OR use exact values throughout.

## Introduction

**Issue**: "Despite growing interest in absorption, the field faces a measurement crisis."

Generic opening. The "measurement crisis" framing is good but buried.

**Issue**: Paragraph 3 on absorption definition is well-written but long. Consider breaking at "This creates a fundamental reliability problem..."

## Methodology (Section 3)

**Issue**: Section 3.7 introduces "overlap method" without defining it.

The entire paper builds around multi-child proportional ablation, yet H_Safe suddenly uses a different method with no explanation. This is a major gap.

**Issue**: Figure 1 has only a caption, no prose walkthrough.

Readers need textual guidance through the methodology figure.

**Issue**: "act(p)" in equation for abs_k is undefined.

Should explicitly state this means parent activation magnitude.

## Results (Section 4)

**Critical Issue**: Table 6 contains degenerate percentage values.

"134,717,856%" and "1.52 billion%" will immediately draw reviewer objections. These are computation artifacts from dividing by near-zero baseline. Remove the percentage column.

**Critical Issue**: Table 1 shows Trained SAE with Std=0.0000 (exact).

This is suspicious across 5 seeds. Either state it's truly constant or verify the measurement.

**Critical Issue**: H_Mech Table 5 shows Condition C = 0.299 (Std=0.010), identical to Condition A = 0.299 (Std=0.010).

If trained decoder adds nothing beyond random decoder, then "decoder contributes 0.299" is contradicted. The paper's own decomposition table appears internally inconsistent.

**Critical Issue**: Shuffled features (0.487) and Permuted Encoder (0.484) nearly equal Trained SAE (0.500).

The paper acknowledges this but doesn't explain why. If broken feature identity produces same absorption as trained SAE, what is the method actually measuring?

## Discussion (Section 5)

**Issue**: Section 5.1 says "decoder geometry contributes 0.299" but Section 4.4 shows Condition C (decoder only) = 0.299, Condition A (random/random) = 0.299 - these are the same.

The decomposition logic needs fixing. If Condition C ≈ Condition A, decoder contributes ZERO to absorption, not 0.299.

**Issue**: Section 5.2 on competitive exclusion is well-written but the Lotka-Volterra reference needs brief explanation for non-ecology readers.

**Issue**: Conclusion (5.8) largely restates the abstract verbatim.

Some trimming needed.

## References

**Issue**: "Kalmykov & Kalmykov (2012)" cited for competitive exclusion in SAEs context, but appears to be a physics paper, not an SAE paper.

Verify citation attribution is correct.

## Specific Lines Requiring Revision

1. **Line 5 of Abstract**: "suggesting SAE-based safety analysis is not systematically compromised" - bury this deeper with clearer separation
2. **Section 3.7**: Define "overlap method" or align with multi-child methodology
3. **Section 4.5.2, Table 6**: Remove degenerate percentage column
4. **Section 4.4**: Reconcile Condition C = Condition A contradiction in decomposition
5. **Figure 1**: Add 2-3 sentence textual walkthrough

## What Works

1. Consistent hypothesis testing format is excellent
2. Negative results (H2, H_Safe) reported with appropriate rigor
3. Figure captions are self-explanatory
4. Background section effectively frames the problem