# Ideation Critique -- ComposeAccel iter_002

## Overall Assessment: 5.5/10

The research question ("how do training-free DLM acceleration methods compose?") is genuinely important and unanswered. The proposal correctly identifies a real gap: no published work systematically evaluates pairwise or higher-order composition of DLM inference acceleration methods. The four research questions (orthogonality, Pareto frontier, task dependence, AR comparison) are well-motivated by iter_001 pilot data.

However, the ideation suffers from three structural weaknesses that propagated through the entire pipeline.

---

## Strength: Honest Grounding in Pilot Data

The proposal is unusually transparent about iter_001 failures. Table in Section "Motivation" reports both positive results (M1+M3 at 2.25x/99.7%) and negative results (M2 accuracy collapse to 54.4%, IGSD at 36% retention). The "Evidence-Driven Revisions" section explicitly lists six findings from iter_001 and their implications. This is a model of how iterative research should build on prior evidence.

## Strength: Clear Scope Limitation

Section 1.3 (Scope) correctly frames ComposeAccel as "an analysis paper" and IGSD as "simple by design." This manages expectations appropriately.

---

## Weakness 1: IGSD Was Designed as a Novel Algorithm but Is Actually Naive Step Truncation

The proposal frames IGSD as a "novel, simple (50 lines of code) training-free step scheduler using inter-step logit KL divergence." Contribution 2 in the proposal lists IGSD as the second-most-important contribution after the composition study itself.

The iter_002 ablation (Section 4.4) confirms what iter_001 already suggested: at T_draft=32, tau=0.0 (no confidence partitioning) and tau=0.9 produce identical GSM8K accuracy (49.5%). IGSD's KL-motivated confidence gate is inert at the primary operating point.

**Root cause in ideation**: The proposal did not include a pre-registered null hypothesis for IGSD's confidence gate. There was no a priori plan to compare IGSD against naive step truncation (run T_draft steps, stop). This comparison should have been the FIRST experiment, not a secondary ablation. The fact that both iter_001 and iter_002 discover the same null result suggests the ideation phase failed to think adversarially about IGSD.

**Impact**: IGSD cannot be claimed as an algorithmic contribution. The paper must reframe it as "we tested whether KL-guided confidence partitioning improves upon naive truncation; it does not." This is a useful negative result but a dramatically weaker contribution than what was proposed.

## Weakness 2: The "Three Orthogonal Axes" Framing Collapses Under Evidence

The proposal organizes the acceleration landscape into three axes: KV cache approximation, adaptive step scheduling, and guided/speculative decoding. This is a clean taxonomy, and the factorial design (test all pairs) is sound.

However:
- **M1 (KV caching)** achieves 1.16x measured speedup -- effectively a no-op. The axis exists in theory but not in practice given the implementation.
- **M2 (step scheduling)** is excluded as a structural NO_GO. This was the most natural "step scheduling" method; IGSD was supposed to be a speculative method but now reduces to step scheduling.
- **IGSD** reduces to naive step truncation (see Weakness 1), making it an extremely simple instantiation of the step scheduling axis.

The actual axes tested are: (1) entropy monitoring with negligible speedup, (2) step truncation, and (3) AR guidance. Only (2) and (3) are genuine acceleration mechanisms. The "three orthogonal axes" framing promised in the proposal and maintained in the paper does not match the experimental reality.

**Impact**: The paper should reframe from "three axes" to "two-and-a-half" or honestly to "two: step reduction and AR guidance, with KV caching projected but not achieved." This is less elegant but more honest.

## Weakness 3: Hypotheses Were Not Pre-Registered with Falsification Criteria

The proposal lists four hypotheses (H1-H4) with expected outcomes, but the falsification criteria are loose:
- H1 ("composition ratio < 0.5 on reasoning"): No pre-specified confidence threshold or sample size justification.
- H2 ("quality-first composition dominates"): Falsified (M1+M3 interference), but the proposal did not anticipate this outcome.
- H3 ("IGSD composable, ratio > 0.8"): Partially supported but on pilot data only.
- H4 ("task-dependent recipes"): Supported directionally but MATH500 baseline is too weak for statistical claims.

H6 ("inverted-U KL profile") is explicitly falsified by the monotonically decreasing KL profile. This is a genuine scientific prediction that failed -- good science. But H5 ("KL sufficient for step skipping") was poorly specified: "at most 40 effective forward passes" at "iso-accuracy within 1%" is never achieved; the closest is T_draft=48 at 73.3% retention, far from iso-accuracy.

---

## Recommendations for Future Ideation

1. **Pre-register null hypotheses for novel algorithms**: Before proposing IGSD as a contribution, specify "IGSD vs. naive-T_draft truncation" as a mandatory first experiment. If the confidence gate adds nothing, IGSD is not a contribution.

2. **Stress-test the axis taxonomy**: Before committing to "three orthogonal axes," verify that each axis has a working implementation that produces measurable speedup. The d2Cache integration risk was identified but underestimated.

3. **Include an "anti-proposal" section**: For each hypothesis, explicitly write "this hypothesis is wrong if..." with specific numeric thresholds. This forces adversarial thinking at the ideation stage.

4. **Separate methodological from algorithmic contributions**: The Ortho metric and factorial design are methodological contributions. IGSD was supposed to be an algorithmic contribution. These should be evaluated independently at the ideation stage.
