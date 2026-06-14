# Critique: Ideation and Research Question

## Summary

The research question (does absorption reduce steering effectiveness?) is well-motivated against prior work. The counter-intuitive finding potential is genuine. However, the ideation has not evolved across 4 iterations: the same critiques persist, the paper lacks a clear novelty story, and the beta-conditional reversal (low-absorption > high-absorption at beta=20, p=0.015) is the most interesting finding in the data but is hidden rather than highlighted.

## Strengths

1. **Well-motivated question**: The hypothesis directly tests a claim from prior work (Bricken et al., Chanin et al.) about absorption degrading steering reliability. This positions the paper as an empirical test of a specific prior claim.

2. **Methodological maturity**: The matched feature selection design (controlling for activation frequency and decoder L2 norm) is the right approach. This is a genuine methodological contribution over earlier versions.

3. **Beta-conditional finding**: At beta=20, low-absorption features show significantly higher steering sensitivity (p=0.015). This is the most interesting result in the data but the paper frames it as an anomaly rather than a finding.

## Critical Weaknesses

### 1. The Beta-Conditional Effect is the Real Finding and It Is Buried

The paper's title and abstract claim "no significant difference." But Table 2 shows p=0.015 at beta=20 with low-absorption features outperforming high-absorption features. This is:
- Statistically significant
- In the OPPOSITE direction of the original hypothesis (which predicted high-absorption < low-absorption, the opposite of what p=0.015 shows)
- An effect that only emerges at high steering magnitudes

This pattern — high absorption = lower steering at low beta, high absorption = lower steering at high beta, but low absorption = higher steering at beta=20 — suggests a non-monotonic, beta-conditional relationship. The paper's null framing misses this.

**Recommendation**: Reframe the paper around the beta-conditional effect. The contribution is NOT "absorption doesn't matter for steering." The contribution is "the absorption-steering relationship is beta-conditional and reverses at high steering magnitudes."

### 2. The Two-Analysis Contradiction Is Unresolved

The abstract mentions "our initial analysis found r=+0.35 (p<0.001)" but the matched design shows p=0.299. This is presented as a note ("Note on Original Analysis") rather than a structural explanation. The ideation has not resolved which analysis is authoritative and why they differ.

**Recommendation**: State explicitly: "Our initial analysis used unmatched feature selection, where high-absorption features systematically differed from low-absorption features on confound variables correlated with steering sensitivity. After controlling for activation frequency and decoder L2 norm through matched feature pairs, the correlation becomes non-significant. This suggests the original r=+0.35 result was confounded."

### 3. Iteration History: No Evolution

The ideation from iteration 1 to iteration 4 has not evolved. The reflection.md documents 4 iterations with identical critical issues. The ideation should have pivoted the paper to focus on the beta-conditional effect, which is genuinely novel and interesting, rather than continuing to polish a null result framing.

## Recommendations

1. **Pivot to the beta-conditional finding as the primary contribution**: This is the most scientifically interesting result and should drive the paper's narrative.

2. **Resolve the two-analysis story explicitly**: The confounded vs. controlled analysis framing is actually a strength — it shows methodological rigor in identifying and correcting for confound.

3. **State what prior work specifically predicted**: Add a paragraph stating exactly what Bricken et al. and Chanin et al. predicted about absorption and steerability, so the paper's contribution is clearly positioned as a replication or rebuttal.
