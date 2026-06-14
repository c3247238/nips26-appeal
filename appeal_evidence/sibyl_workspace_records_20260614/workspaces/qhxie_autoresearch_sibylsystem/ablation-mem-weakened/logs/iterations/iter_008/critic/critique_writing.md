# Critique of Writing

## Abstract and Introduction

**Issue 1 (Critical):** The abstract presents multiple claims as established facts when they are actually null results or post-hoc interpretations:

- "absorption does not significantly degrade steering effectiveness or sparse probing accuracy after rigorous multiple comparison correction" — This is a null result, framed as a positive finding. The correct framing should use tentative language: "we found no significant evidence that absorption degrades..."
- "trained SAEs exhibit significantly LOWER absorption than random baselines (mean 0.034 vs 0.278, p < 0.001)" — This is ONE significant result out of 12 tests, and its interpretation is ambiguous (does it mean training reduces absorption, or does it mean the metric is flawed?).

**Issue 2 (Major):** The "optimal compression" framing in the abstract and introduction is presented as an established explanation when it is actually:
- Post-hoc theoretical framing
- Not directly tested (H6 is falsified)
- Used to explain why null results should be interpreted positively

**Issue 3 (Minor):** The introduction does not clearly state the paper's limitations upfront. Readers should know that:
- n=26 features provides ~20% power for medium effects
- Zero significant results after MCP correction
- The LCA connection is theoretically motivated but empirically unsupported

## Claims and Hedging

**Issue 4 (Major):** The paper claims "absorption is not a critical failure mode" but provides no evidence that would support this strong claim. The evidence is:
- Null results (no significant degradation found) — but null results ≠ evidence of no effect
- One suggestive uncorrected p-value at L8 — does not survive MCP
- Theoretical framing — not empirically tested

The claim should be softened to: "we found no significant evidence that absorption degrades steering or probing in our setup" or "absorption may be less problematic than previously thought."

**Issue 5 (Minor):** "Honest null-result reporting" is presented as a positive contribution. This framing is problematic because:
- Null results at low power are not inherently virtuous
- The field benefits more from well-powered studies that find effects than under-powered studies that don't
- The honest framing should acknowledge that the study may simply be underpowered

## Discussion Section

**Issue 6 (Major):** The discussion does not adequately address the probe quality confound. The paper notes that absorption rate correlates with probe F1 at rho=-0.67 but never controls for it in the CMI-absorption analysis.

**Issue 7 (Major):** Section discussing "Why Prior Work Found Null Results" (Section 5.2) should be renamed to "Why This Study Found Null Results" — the framing implies prior work also found null results, which is not clearly established.

**Issue 8 (Minor):** The practical implications (Section 5.3) recommend using the inhibition graph as a diagnostic, but H6 showed precision@20=0.0. This recommendation contradicts the empirical results.

## Overall Writing Quality

The writing is clear and well-structured. The logical flow from introduction to methods to results to discussion is appropriate. However, the framing throughout tends to put a positive spin on null results, which may not withstand reviewer scrutiny.

**Recommendation:** Use more tentative language throughout. Null results are valid scientific findings, but they should be reported with appropriate uncertainty, especially when statistical power is low.
