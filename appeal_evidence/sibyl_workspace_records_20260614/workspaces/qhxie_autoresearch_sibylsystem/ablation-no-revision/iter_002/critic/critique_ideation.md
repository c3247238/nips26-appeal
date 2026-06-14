# Critique: Ideation (proposal.md, hypotheses.md)

## Summary

The proposal presents five hypotheses about feature absorption in SAEs, but three are falsified, one is uninformative, and one remains untested. The fundamental issue is that H1's threshold (>20%) was wildly miscalibrated: the observed rate is 0.19% at layer 8 and 49.3% at layer 4. The research direction identified a legitimate gap but the hypotheses were insufficiently pre-calibrated against real SAE behavior.

## Critical Issues

### 1. H1 Threshold Was Not Pre-Calibrated Against Real SAEs

**Location**: proposal.md, H1

The hypothesis predicts ">20% of latents in mid-to-deep layers" would have A_f > 0.5. This threshold appears to have been chosen without first measuring what actual absorption rates look like in pretrained SAEs. The pilot revealed 0.19% at layer 8 — two orders of magnitude below the threshold. A properly calibrated hypothesis would have had a pre-registered threshold informed by pilot measurements.

**Fix**: Before finalizing hypotheses, run a small exploratory study (10 sequences, 2 layers) to establish baseline absorption rates. Use these to set falsification thresholds that are challenging but not trivially easy to falsify. The gap between 0.19% observed and >20% predicted suggests the hypothesis was speculative rather than empirically grounded.

### 2. H1 Hypothesis Is Internally Inconsistent

**Location**: proposal.md, H1

The H1 hypothesis states: ">20% of latents in mid-to-deep layers have A_f > 0.5" with falsification criterion "<10% prevalence across layers 4-10."

**Problem**: The falsification criterion tests across "layers 4-10" but layer 4 shows 49.3% (not <10%) and layer 8 shows 0.19% (falsified). The hypothesis cannot be falsified across all layers 4-10 when one of those layers contradicts it.

**Fix**: Restate H1 as layer-specific hypotheses (H1a: layer 4 > 20%, H1b: layer 8 > 20%) or specify that the falsification applies to layer 8 specifically. The current presentation misleads readers into thinking H1 is fully falsified when layer 4 actually confirms it.

### 3. H4 Experiment Design Is Fatally Flawed

**Location**: plan/methodology.md, H4 description

The H4 experiment selects bottom-10% and top-10% absorption latents, zeroes out the other 90%, and tests patching faithfulness. Both subsets yield 0.0.

**Root cause**: The faithfulness metric requires the SAE to produce a meaningful reconstruction. Zeroing out 90% of the dictionary means the SAE can only reconstruct 10% of the residual stream, so any patching produces near-random results. This is a methodological error, not an interesting finding about absorption.

The paper itself notes a task-aware redesign would be more appropriate but does not emphasize this as a critical design flaw.

**Fix**: Redesign H4 to compare FULL SAE representations at layers with different absorption profiles (e.g., layer 4 vs layer 8), not subsets of a single layer. Use the full dictionary at each layer to preserve reconstruction quality.

### 4. H2 Never Tested Despite Being Major Hypothesis

**Location**: proposal.md, H2; paper Section 5.5

H2 (token frequency correlation with absorption) was abandoned citing "early termination after H1/H3/H4 falsification." This is weak justification — failed experiments on different hypotheses don't make H2 untestable. The paper's own rationale ("insufficient absorption variance") is correct but buried.

**Fix**: Either run H2 or formally retire it with explicit justification. The correct rationale is: "With only 46 of 24,576 latents (0.19%) showing A_f > 0.5, there is insufficient variance in absorption scores to detect frequency-dependent variation. A redesigned H2 would require either a model with higher overall absorption rates (e.g., GemmaScope SAEs) or a continuous absorption metric capturing partial absorption below the binary threshold."

### 5. Absorption Metric Validated on Null Case Only

**Location**: plan/methodology.md, Section 3

The absorption metric (top-5 co-firers, >80% RVE) is validated against a random dictionary control (correctly yields 0.0) but there is no positive control showing the metric detects real absorption when present.

**Fix**: Add synthetic positive controls: inject known absorption patterns into a random dictionary and verify the metric recovers them. For example, create a dictionary where one latent's decoder vector is a linear combination of others, and verify A_f = 1.0 for the absorbed latent. This validates the metric before applying it to real SAEs.

## Major Issues

### 6. H3 Inverted-U Pattern Has No Mechanistic Theory

**Location**: proposal.md, H3

H3 hypothesizes a monotonic increase of absorption with sparsity. The pilot found an inverted-U (peaking at layer 4 with 49.3%, declining to 17.3% at layer 10). The proposal provides no theory for why this pattern would occur. Post-hoc speculation ("mid-layers process densest abstract and semantic content") is not a mechanistic account.

**Fix**: Either propose a mechanistic hypothesis for the inverted-U before running H3 (so it can be tested), or explicitly frame the inverted-U as an exploratory finding requiring future investigation. Do not present it as a positive contribution without theoretical backing.

### 7. No Contingency for "Absorption Is Rare" Scenario

**Problem**: The methodology assumes absorption is common enough to study at scale. H1's result (0.19% at layer 8) means the phenomenon is too rare for the designed experiments, and this contingency was not planned for.

**Fix**: Add contingency plans for each major hypothesis. For H1: "If absorption rate is <1%, switch to continuous absorption metric (mean RVE across all tokens) and redesign H2 around that metric." For H4: "If subsetting destroys signal, redesign to compare full SAE at layers with different absorption profiles."

### 8. H5 Is Insignificant But Marked "Not Falsified"

**Location**: proposal.md, H5; paper Section 5.4

H5 (larger dictionaries reduce absorption) is marked "not falsified" because the direction matches the hypothesis. But the practical significance is negligible: 97.75% of latents are not absorbed even at 2K dictionary size. The effect is real but explains <1% of variance.

**Fix**: Add an effect size threshold to the falsification criterion. A hypothesis that moves in the right direction but explains <1% of variance is not scientifically meaningful for practical guidance. Consider labeling H5 as "confirmed_direction_insufficient_magnitude."

### 9. No Decision Framework for Failed Pilots

**Location**: plan/methodology.md, task dependencies

The plan specifies task dependencies (h1_pilot gates h1_full, etc.) but does not specify what happens when a pilot fails. Early termination after H1/H3/H4 suggests no contingency plan existed. The decision to abandon H2 was made without a clear framework.

**Fix**: Add a decision tree to the plan: if pilot produces falsification vs uninformative vs unexpected result, what are the options? (redesign, terminate, continue to full experiment) Define thresholds for "no-go" vs "redesign vs proceed."

## Minor Issues

### 10. H5 Simulation May Not Reflect Real Training

**Location**: plan/methodology.md, H5

H5 uses cumulative subsampling of the 24K dictionary to simulate 2K and 8K dictionaries. This does not reflect how independently-trained SAEs of different sizes would behave.

**Fix**: Either train actual SAEs at different dictionary sizes (2K, 4K, 8K, 16K) with the same data and training setup, or explicitly note that subsampling is a simulation requiring empirical validation in the methodology.

### 11. Novelty Claim Is Unsubstantiated

**Location**: proposal.md, contribution claims

The proposal claims "first systematic quantification of feature absorption prevalence" but does not cite prior work on related phenomena (feature interference, dictionary learning quality, superposition metrics).

**Fix**: Add a prior work section distinguishing absorption from superposition, feature collision, and interference. Show how the proposed metric and results compare to existing measurements of similar concepts.

### 12. Five Hypotheses May Be Too Many for Single Pilot Phase

**Problem**: Testing 5 hypotheses in a single pilot with 100 sequences increases the risk that failures cascade (as seen with H2 abandonment after upstream issues).

**Fix**: Prioritize hypotheses. Run H1 and H3 pilots first (they share cached activations). Only proceed to H2/H4/H5 if H1 shows sufficient absorption variance. Sequential prioritization avoids investing in dependent experiments when upstream results are uncertain.

## Summary

The proposal identified a legitimate research gap (feature absorption in SAEs) but the hypotheses were insufficiently pre-calibrated:
- H1's >20% threshold was two orders of magnitude above observed rates at layer 8, but below observed rates at layer 4 — the hypothesis is internally inconsistent with results
- H4's design was fundamentally flawed (zeroing 90% of dictionary destroys signal regardless of absorption)
- H2 was abandoned without proper justification or contingency planning
- No positive control exists for the absorption metric
- No decision framework specifies what to do when pilots fail

Main improvements for future ideation:
1. Run exploratory pilots to calibrate thresholds before committing to falsification criteria
2. Design hypotheses with explicit dependencies and contingencies
3. Add synthetic positive controls to validate metrics before real experiments
4. Use task-aware designs for causal experiments (compare full representations, not subsets)