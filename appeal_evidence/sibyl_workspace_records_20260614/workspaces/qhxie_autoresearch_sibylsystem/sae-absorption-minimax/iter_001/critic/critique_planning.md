# Critique: Planning and Methodology

## Overview

The planning documents (proposal.md, methodology.md, plan/methodology.md) are well-structured with clear hypotheses and a logical research arc. The decision to pivot from "absorption degrades reliability" to "absorption is a steering signature" based on empirical evidence is methodologically sound. However, the planning phase failed to catch several critical issues that led to the experimental execution problems documented in critique_experiment.md.

---

## Critical Issues

### 1. Absorption Metric Standardization Was Deferred Too Late

The methodology specifies multiple absorption measurement approaches:
- Chanin absorption score (first-letter probe method)
- Gini absorption (used in full_h1_gpt2)
- UAS (proposed unsupervised metric)

The proposal acknowledges that "absorption score" is ambiguous in Section 4.4 of the paper but does not resolve the ambiguity in planning. The pilot used first-letter probe absorption; the full H2 used Gini absorption. These differ by 15x for the same SAE.

**Planning should have established**: One canonical absorption metric, used in ALL experiments (pilot and full-scale), before running any experiments. The choice of metric should be justified (what does it measure? what range does it take?).

### 2. H5 Pass/Fail Criteria Were Not Pre-Registered

The proposal defines H5 as: "High-absorption features perform worse than low-absorption features on both task types."

The full experiment added pass criteria (>8% causal delta) that are not mentioned in the proposal. The proposal mentions "5% threshold" but this threshold does not appear in the experimental design. The full_h5_downstream_tasks.json shows the actual criteria were different.

**Planning should have pre-registered**: Exact pass/fail thresholds for each task type, with statistical justification (power analysis for minimum detectable effect size).

### 3. Pilot/Full Transition Criteria Were Not Defined

The proposal acknowledges pilot/full contradictions (H3: pilot showed negative correlation, full showed positive; H1: pilot showed contradictory layer orderings) but does not define what happens when pilot and full-scale contradict.

**Planning should have specified**:
- If pilot and full-scale contradict, which takes precedence?
- What analysis is needed to determine which result is reliable?
- Should writing proceed with pilot-only results or wait for full-scale?

The paper proceeds as if pilot and full-scale results are consistent despite clear contradictions in H2 and H3.

---

## Major Issues

### 4. Feature Selection Protocol Not Fully Specified

The methodology states high-absorption: "UAS > 1.0" and low-absorption: "UAS < 0.3." The full_h3 experiment used Gini absorption for feature selection (UAS ~0.57 for "high" features), not UAS > 1.0.

The feature selection protocol was not fully specified in planning. Key questions unanswered:
- Which metric (UAS or Gini absorption) determines high/low classification?
- What is the exact threshold for each?
- How are ties handled?
- What happens if not enough features meet the threshold?

### 5. Null Control Protocol Was Not Pre-Specified

The proposal mentions null controls as a follow-up experiment but does not define:
- What null conditions to test (random directions? shuffled directions?)
- Which alpha values to test
- How many null controls per alpha value
- What constitutes a passing null control (must match random baseline? must not differ between absorption bins?)

The null controls as executed (alpha=5 only) do not match the main experiment protocol (alpha range [1,3,5,10,20]). This mismatch should have been caught in planning.

### 6. H2 Mitigation Success Criteria Were Vague

H2 states: "OrtSAE and ATM reduce absorption by >40% relative to vanilla SAE, but at a significant reconstruction cost."

But:
- What is "significant" reconstruction cost? (No upper bound specified)
- What is the minimum absorption reduction? (>40%? Any reduction?)
- What if ALL variants INCREASE absorption? (This happened in full H2)

The success criteria were not specific enough to detect a contradictory result. The full H2 assessment code flags h2_pass=true when absorption increases, which is the wrong direction.

### 7. H1 Experimental Design Did Not Account for Instability

Two contradictory pilot runs for H1 should have triggered a redesign before running full-scale experiments. The methodology did not specify:
- What causes the pilot instability? (token sampling? SAE seed?)
- What controlled experiment would isolate the cause?
- What sample size is needed per layer to detect the expected pattern?
- How to handle contradictory runs (stop? aggregate? analyze separately?)

### 8. Gemma-2B Dependency Was Not Flagged as Risk

The methodology plans Gemma-2B experiments but does not flag HF authentication as a prerequisite risk. The full_h3_gemma was skipped due to HF authentication, leaving the paper single-model only.

**Planning should have included**: A checklist of prerequisites for Gemma-2B experiments, with explicit contingency plans if prerequisites were not met.

---

## Strengths of Planning

1. **Clear hypothesis structure**: H1-H5 are well-defined with specific predictions, making falsification possible (even if some criteria were not specific enough).

2. **Good pilot design**: The 10-15 minute pilot target is appropriate for rapid iteration.

3. **Risk identification**: The Risks and Mitigations section in methodology.md identifies several real risks (JumpReLU convergence, Gemma-2B availability).

4. **Alternative candidates**: The alternatives section shows good forward thinking about pivot options.

5. **Honest hypothesis status tracking**: The Evidence-Driven Revisions section in proposal.md is exemplary — it honestly documents H3 reversal, H1 instability, and H2 partial confirmation.

---

## Recommendations for Planning

1. **Pre-register one absorption metric**: Before any experiments, document the canonical absorption metric with justification.

2. **Pre-register pass/fail criteria**: For each hypothesis, specify exact thresholds and statistical tests before running experiments.

3. **Define pilot/full contradiction protocol**: What happens when pilot and full-scale contradict? Specify before running full-scale.

4. **Document feature selection protocol completely**: Which metric? Which thresholds? How are ties handled?

5. **Pre-specify null control protocol**: Match null controls exactly to main experiment protocol (same alpha values, same prompts).

6. **Add upper bounds to H2 success criteria**: "Significant reconstruction cost" needs a specific definition.

7. **Account for prerequisite risks**: Gemma-2B HF authentication should have been flagged and resolved before planning Gemma-2B experiments.
