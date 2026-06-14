# Critique: Related Work

## Summary Assessment
The Related Work section provides comprehensive coverage of the relevant literature, properly situating the paper within SAE research, absorption characterization, architectural solutions, and evaluation benchmarks. The structure is logical and the writing is clear. The section's primary weakness is a tendency to describe rather than critically evaluate prior work, and the gap section at the end is too short to fully leverage the preceding analysis.

## Score: 7/10
**Justification**: Thorough literature review with good organization. Deducted points for: (1) insufficient critical evaluation of prior work (most paragraphs end with description rather than assessment), (2) the gap section (4.5) is abrupt and could synthesize the preceding analysis more explicitly, (3) some potentially relevant references appear missing.

## Critical Issues

### Issue 1: Insufficient Critical Evaluation of Architecture Solutions
- **Location**: Section 4.3, lines 18-22
- **Problem**: The paragraph on architectural solutions describes each method's claims but does not critically evaluate them. For example: Matryoshka SAE's "0.49 to 0.05" absorption reduction -- is this on the same benchmark? Under the same conditions? OrtSAE's "65% absorption reduction" -- reduction from what baseline? The numbers are presented without context that enables comparison.
- **Fix**: Add a critical sentence to each architecture description. For Matryoshka: note that the 0.05 residual absorption and 50% computational overhead should be compared against the paper's target use case. For OrtSAE: clarify whether the 65% figure is from the paper's own benchmarks or third-party evaluation. For AdaptiveK: note that training on 2,000x less data makes direct comparison to other methods' benchmarks problematic.

### Issue 2: Missing Discussion of Why Prior Work Did Not Attempt Decomposition
- **Location**: Section 4.5, lines 32-33
- **Problem**: The gap section states "no prior work empirically decomposes absorption" but does not explain *why* this decomposition was not attempted. Possible reasons: (1) the factorial design requires controlling for encoder and decoder independently, which standard SAE training does not support; (2) measuring absorption requires ground-truth hierarchy, which real LLM SAEs lack; (3) the field's focus on architectural solutions implied absorption was a joint problem. Explaining this would make the contribution feel less like an oversight and more like a deliberate methodological choice.
- **Fix**: Expand 4.5 to explain why the decomposition was technically challenging prior to this work (needs synthetic data with ground truth, needs independent parameter freezing, etc.).

### Issue 3: Missing Relevant References
- **Location**: Section 4.2 and 4.3
- **Problem**: Several potentially relevant papers are not cited. Specifically: (1) Engel et al. (2025) on feature geometry and absorption -- appears to be missing from the absorption literature review; (2) The recent "SAE Collapse" work by Liu et al. (2025) on training dynamics that may relate to when absorption emerges; (3) Nanda et al. (2025) on the relationship between SAE training steps and feature emergence, which may connect to when absorption becomes established.
- **Fix**: Search for and incorporate these references. If they are not relevant, add a brief note in 4.2 explaining why.

### Issue 4: Section 4.4 Evaluation Benchmarks Has No Critical Assessment
- **Location**: Section 4.4, lines 27-29
- **Problem**: The paragraph describes SAEBench and SynthSAEBench findings without critically evaluating the benchmarks themselves. Key question: do these benchmarks actually measure what they claim to measure? SynthSAEBench uses 16,000 features -- does absorption behavior differ at this scale compared to the 4,096 used in this paper?
- **Fix**: Add a critical paragraph assessing the benchmarks' strengths and limitations. Note any ways the benchmarks' settings differ from the paper's experimental conditions.

## Minor Issues

- **Citation consistency**: Some references use "2023/2025" for the same paper (Bricken et al. in 4.1). Determine the correct year and use consistently.
- **Feature hedging mention**: Line 19 mentions "feature hedging" as related to Matryoshka inner levels but does not cite Chanin & Garriga-Alonso (2025) in the text, only in the reference parenthetical. Add the in-text citation.
- **The Oursland (2026) citation**: The year 2026 appears for Oursland -- verify this is correct as it suggests a very recent or forthcoming paper.
