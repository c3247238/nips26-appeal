# Critique: Experiment Design and Execution

## Summary

The H3 experiment is methodologically sound for the matched design. However, the beta-conditional reversal (low-absorption > high-absorption at beta=20, p=0.015) is the most scientifically interesting finding and requires a mechanistic explanation that is currently absent. The paper lacks all four planned figures, making it visually inadequate for a top-tier submission.

## Critical Issues

### 1. Beta-Conditional Reversal: p=0.015 at Beta=20 Is Unexplained

Table 2 shows that at beta=20, LOW-absorption features have significantly HIGHER steering sensitivity than HIGH-absorption features (p=0.015). This is not explained by any stated mechanism. The paper notes it "warrants further investigation" but does not even state what it might mean.

**Potential explanations that should be discussed**:
- **Saturation hypothesis**: High-absorption features have higher decoder L2 norms (they activate rarely but fire strongly when they do). At high beta values, these high-norm directions may saturate the residual stream, causing diminishing returns. Low-norm directions (low-absorption features) scale more linearly.
- **Competition hypothesis**: At high steering magnitudes, multiple features compete for representational space. High-absorption features that are absorbed into many others may create interference patterns that limit steering effectiveness.
- **Methodology artifact**: The matched design matched on activation frequency and decoder L2 norm, but the beta-20 reversal might reflect imperfect matching on some unobserved confound.

**Recommendation**: Add a one-paragraph discussion of the saturation hypothesis as the most plausible explanation. "At high steering magnitudes, high-norm decoder directions (characteristic of absorbed features) may saturate the residual stream, leading to diminishing marginal returns. Low-norm directions scale more linearly, producing larger incremental effects at high beta values."

### 2. No Figures: Conference Submission Risk

The paper has zero figures despite planning 4. Tables 1-3 are well-formatted but cannot convey:
- The distribution of steering effects across individual features (a scatter plot would show whether the correlation is driven by outliers)
- The beta-conditional relationship visually (a grouped bar chart with error bars is essential)
- The UAS validation correlation (a scatter plot with marginal histograms would show the distribution)
- The H2 mitigation Pareto frontier

**Recommendation**: Generate all four figures before submission. At NeurIPS/ICLR, visual communication of results is not optional.

### 3. Saturation Confound in Beta-20 Finding

The beta-20 reversal may be a methodology artifact. High-absorption features are defined by UAS > 1.0, which includes high cos_variance and high act_freq. High-absorption features may also have higher decoder L2 norms by construction (features that activate rarely but contribute strongly to reconstruction). At high beta values, high-norm directions may saturate while low-norm directions remain in the linear regime.

**Recommendation**: Add a supplementary analysis checking whether decoder L2 norm correlates with absorption level in the selected features. If it does, the beta-20 effect may be confounded by norm differences, not absorption per se.

## Major Issues

### 4. UAS Baseline Comparison Missing

UAS is validated against Chanin supervised absorption (r=0.65-0.79) but not compared against trivial baselines. Is UAS better than activation frequency alone? Is it better than L2 decoder norm alone?

**Recommendation**: Report correlations: act_freq vs Chanin absorption (r=?), L2_decoder_norm vs Chanin absorption (r=?). Quantify UAS's improvement over these baselines.

### 5. Random Baseline Inconsistency Across Tables

Table 2 (main results) random baseline at beta=5: 0.4771
Table 3 (null controls) random baseline at beta=5: 0.6207

These should be identical if they use the same random directions and same prompts. If they differ, the comparison across tables is not apples-to-apples.

**Recommendation**: Verify the random baseline computation. If different random seeds were used, this must be stated. If they should be identical, recompute.

### 6. H2 Mitigation Results Are Negative and Should Be Stated Clearly

The paper says: "We conducted a pilot evaluation of whether alternative SAE architectures can reduce absorption." All variants (TopK, MultiScale, OrtSAE) show HIGHER absorption than vanilla SAE. This is a negative result.

**Current framing**: Vague, suggests further validation is needed.
**Honest framing**: "None of the evaluated alternative architectures achieved lower absorption than vanilla SAE in our pilot evaluation."

**Recommendation**: State H2 as a negative result: "In our pilot evaluation, none of the tested alternative SAE architectures (TopK, MultiScale, OrtSAE) achieved lower absorption than vanilla SAE on the Chanin first-letter probe metric. This suggests absorption resistance may require fundamentally different training objectives."

## What Works

1. **Matched feature selection**: Controlling for activation frequency and decoder L2 norm is methodologically correct.
2. **Full beta range null controls**: The null controls now use the full beta range matching the main experiment, addressing the iteration-1 flaw.
3. **Appropriate N**: 50 matched pairs is reasonable for this effect size.
4. **Honest negative result reporting**: H2 is correctly noted as pilot-scale and inconclusive.
