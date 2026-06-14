# Strategist Analysis: Publication Roadmap

## Overview
This analysis prioritizes which findings are publishable and recommends concrete next steps.

## Findings Assessment

### Tier 1: Publishable (Strong Evidence)

**H3 REVERSED Finding**
- Strengths: r=+0.35, p<0.001, clear directional hypothesis
- Framing: "Absorption as Steering Signature" rather than "Absorption Degrades Reliability"
- Position: Lead contribution—novel and counter-intuitive

### Tier 2: Supportive (Needs Strengthening)

**UAS Validation (implied by H3 results)**
- If UAS correlates with steering sensitivity, it's validated as a useful metric
- Currently lacks dedicated experiment (full_h4.json missing)
- Recommendation: Run full_h4 or frame as "UAS correlates with effect size"

### Tier 3: Problematic (Needs Resolution)

**Layer-wise Absorption (H1)**
- Status: SKIPPED—contradiction unresolved
- Run 1: L4 < L8 (+10.6%)
- Run 2: L4 > L8 (-22.9%)
- Recommendation: **Frame as limitation/future work**—don't claim definitive layer ordering

**pilot_h3_null Contradiction**
- Shows high ≈ low absorption (0.7485 vs 0.7543)
- Contradicts full_h3 spearman correlation
- Recommendation: **Investigate and explain** before submission

## Publication Strategy

### Option A: Focused Paper
- Center: H3 REVERSED
- Supporting: Theoretical discussion of entanglement hypothesis
- Limitations: Acknowledge H1 unresolved, single-model experiments
- Target: NeurIPS Workshop or ICLR

### Option B: Comprehensive Paper
- Center: H3 REVERSED + UAS validation
- Additional: OrtSAE/TopK mitigation results (H2)
- More experiments needed (replication, H1 resolution)
- Target: ICLR Main Track or NeurIPS

## Recommended Next Steps

1. **Immediate** (this iteration):
   - Resolve pilot_h3_null vs full_h3 contradiction
   - Write partial draft with current findings

2. **Short-term** (next iteration):
   - Replicate on GPT-J or Llama
   - Run full_h4 for UAS validation
   - Investigate H1 layer ordering

3. **Before submission**:
   - Minimum 2 model replications
   - Consistent effect direction across experiments

## Conclusion
H3 REVERSED is publishable as-is, but a focused workshop paper with honest limitations is better than overclaiming.
