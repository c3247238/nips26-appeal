# Ideation Critique: Iteration 5

## Overall Assessment

The ideation for iteration 5 is the strongest in the project's history. The pivot from novel detection methods (EDA, EncNorm, ITAC -- which consumed iterations 1-3 with minimal publishable results) to empirical characterization and confound control addresses the correct blocking issues identified in iteration 4's reflection. The three-contribution structure (confound resolution, cross-domain, scaling surface) is logically coherent, with pre-registered hypotheses and explicit falsification criteria representing a step-change in scientific rigor. However, execution reveals that one of the three contributions (cross-domain) was doomed by model access constraints, and the surviving two contributions face novelty challenges.

---

## Strengths

### 1. Correct identification of blocking issues

The proposal correctly identifies the three weaknesses from iteration 4 as blocking: (a) the L0 confound, (b) single-task evaluation, (c) absence of joint scaling characterization. Addressing all three simultaneously is strategically sound -- even partial success on two of three produces a publishable paper.

### 2. Pre-registered falsification criteria

Explicit go/no-go thresholds (|partial_r| > 0.2 for H1, 3x shuffled baseline for H2, interaction p < 0.05 for H3) are a genuine improvement. The proposal specifies what "killed" looks like: H1 AND H2 simultaneously falsified triggers a pivot to Alternative 3 (honest audit framing). This is how science should work.

### 3. Backup alternatives are well-designed

Four backup alternatives (immunodominance, phase diagram, honest audit, dictionary coverage) cover the major failure modes. Alternative 3 (honest audit) is especially well-conceived: "feature absorption is less severe, less general, and less causal than believed" would be a valuable contribution if the data supported it.

### 4. Methodological novelty via epidemiological methods

Importing Baron-Kenny mediation and Rosenbaum sensitivity analysis to SAE evaluation is a genuine first. These are textbook methods in their home disciplines but have never been applied to SAE evaluation. The novelty is real, albeit in application rather than method development.

---

## Weaknesses

### 1. Cross-domain contribution was doomed by model access

The proposal planned Gemma 2 2B with Gemma Scope SAEs (16k-1M features). The execution used GPT-2 Small (124M) with a 24k SAE. This model substitution was not merely a quantitative downgrade -- it changed the experiment from "test absorption on knowledge hierarchies with SAEs that can represent knowledge features" to "test absorption on knowledge hierarchies with SAEs that cannot represent knowledge features." The risk assessment listed "Gemma 2B model access blocked" at 15% likelihood with "Fall back to GPT-2 Small" as mitigation. But the mitigation does not preserve the scientific value of the experiment. The proposal should have either (a) secured model access before committing to this contribution, or (b) designed the cross-domain experiment around GPT-2 Small from the start (e.g., using syntactic features like noun-verb agreement rather than knowledge hierarchies).

### 2. Novelty concerns for the surviving contributions

With Phase 2 producing a negative result, the paper rests on Phase 1 (confound resolution) and Phase 3 (scaling surface). Both have novelty challenges:

- **Phase 1** applies standard statistical methods to existing data. A reviewer could view this as "you ran a mediation analysis on 48 SAEs" -- technically correct but not a research contribution in statistics, and the within-width matching null undermines the causal narrative.
- **Phase 3** uses precomputed SAEBench data and fits a GAM. Chanin et al. Figures 9b/9c already show the qualitative pattern. The novelty is formalizing what the community can see informally.

Together, these may not clear the novelty bar for a top venue.

### 3. Insufficient consideration of confounding in causal design

The proposal's causal inference design (mediation analysis, Rosenbaum bounds) is sound for what it does, but it does not address the fundamental limitation: all data is cross-sectional from a single model family. The proposal mentions "interventional study" in Future Work but does not design one. Given that OrtSAE code is available and could be trained at matched width/L0, a direct comparison between standard SAEs and OrtSAE at the same hyperparameters would provide much stronger causal evidence than mediation analysis on observational data. This was feasible within the GPU budget and would have been a stronger Contribution 1.

### 4. Hypothesis strength calibration

- **H1** threshold (|r| > 0.2) is low. A partial correlation of 0.2 with p < 0.05 on n = 48 explains 4% of variance. A reviewer could reasonably argue this threshold is too lenient for claiming an "independent quality predictor."
- **H2** threshold (10% absorption, 3x shuffled) is reasonable but depends on the model having knowledge features in the SAE -- a condition that was not met for GPT-2 Small.
- **H3** threshold (interaction p < 0.05) is standard but the actual result (p = 3.1e-15) so far exceeds it that the pre-registration provides no meaningful constraint.

---

## Alternatives Assessment

### Alternative 3 (Honest Audit) should have been the framing

Given that: (a) Phase 2 is a conclusive negative, (b) Phase 1's causal claims are undermined by the within-width null, and (c) Phase 4 reveals a massive taxonomy artifact, the paper's strongest framing would be Alternative 3: "Feature absorption is real but more confined, more metric-dependent, and less causally understood than previously believed." This framing:
- Makes the taxonomy correction the headline (92.3% -> 19.2% is dramatic)
- Frames the Phase 2 negative as a domain limitation finding (absorption requires dedicated SAE features)
- Frames Phase 1 as "the causal chain survives L0 control statistically but not within-width, suggesting width is the deeper driver"
- Uses Phase 3 as the constructive contribution (practitioners can use the scaling surface)

The current framing tries to sell all four contributions as positive, which requires misframing Phase 2 and overclaiming Phase 1.

---

## Score

Ideation quality: **7/10**. The research questions are correct, the structure is logical, and the pre-registration is genuine. The execution was undermined by an external constraint (model access) that the ideation failed to adequately mitigate, and the surviving contributions face novelty challenges that the backups partially but not fully address.
