# Ideation Critique: Construct Validity of SAEBench Absorption Metric

## Overall Assessment

The research idea is timely, well-motivated, and addresses a genuine gap in the SAE literature. The construct-validity framing is conceptually sound. However, the idea's execution is undermined by several design choices that were foreseeable during the ideation phase.

**Idea Quality: 7/10**
**Execution Quality: 5/10**

---

## Strengths of the Idea

### 1. Genuine Field-Wide Blind Spot

The proposal correctly identifies that no prior work has validated the SAEBench absorption metric on real semantic hierarchies. Chanin et al. (2024) explicitly called for this as future work. The novelty claim is accurate and well-supported by the literature review in the proposal.

### 2. Perfect Alignment with Training-Free Constraints

The idea requires only existing pretrained SAEs and the SAEBench codebase. No model training is needed. This makes it feasible within the 1-hour-per-experiment constraint.

### 3. Clear Falsification Criteria

The pre-registered hypotheses (H1: r > 0.6, H2: semantic > non-hierarchy, H3: robust across tau_fs) provide clear success/failure criteria. This is good scientific practice.

### 4. High Stakes Either Way

A positive result validates a cornerstone benchmark; a negative result reveals a methodological blind spot. Both outcomes are publishable and actionable.

---

## Weaknesses in the Idea Design

### 1. Underestimated Sample Size Requirements

The proposal planned for "6-8 SAEs" with the expectation that this would be sufficient for correlation analysis. With n=7 trained SAEs, the bootstrap CI spans 1.37 correlation units---making the primary test uninformative. The proposal should have recognized that n=7 is inadequate for reliable correlation inference and either:
- Planned for 15-20 SAEs (if feasible)
- Used a different statistical framework (e.g., Bayesian inference with informative priors)
- Framed the study as exploratory/pilot rather than confirmatory

### 2. Hierarchy Selection Lacks Theoretical Grounding

The WordNet hierarchies were selected based on convenience (single tokens, concrete concepts) rather than theoretical relevance to absorption. The proposal mentions "direct or near-direct hypernym relationship" but does not justify why these particular hierarchies should exhibit absorption.

Key questions unaddressed:
- Why should "building -> house" exhibit absorption while "doctor -> hospital" does not?
- What property of the parent-child relationship determines absorption likelihood?
- Are frequency-matched synthetic templates the right control, or do they introduce their own artifacts?

### 3. Random-SAE Control Design Was Under-Specified

The proposal states: "Compute absorption on an SAE with randomly permuted decoder directions. Should yield near-zero absorption on all tasks; if not, the metric is not specific to learned structure."

This design is problematic:
- If decoder directions are permuted, encoder outputs are unchanged, so absorption scores (which depend on encoder outputs) will be identical to the original SAE. The control is guaranteed to "fail" by construction.
- If encoder directions are permuted (as Section 3.1 claims), the control is more meaningful but the proposal does not explain why this would be the preferred randomization.
- The proposal's expectation ("should yield near-zero absorption") is inconsistent with either permutation strategy.

### 4. Control Condition is Structurally Mismatched

The non-hierarchy control uses binary word pairs (big-large, doctor-hospital), while the hierarchy condition uses multi-class parent-child relationships (building vs. {house, school, library}). This structural mismatch was foreseeable and undermines the hierarchy specificity test.

A better design would have used structurally equivalent controls from the start.

### 5. GPT-2 Replication Was Under-Powered

The proposal planned GPT-2 as a "replication control" but only tested 2 SAEs (Standard and TopK). With n=2, no correlation can be computed, and the near-zero scores suggest a ceiling effect rather than a meaningful replication.

---

## Missed Opportunities from Alternatives

### Alternative A (FastProbe-Absorb)

The front-runner idea could have incorporated the FastProbe-Absorb concept as a secondary analysis: instead of just testing construct validity, also test whether a lightweight probe-based screen correlates with the full absorption metric. This would have added practical utility.

### Alternative B (Theoretical Bound)

The theoretical bound idea could have provided a conceptual framework for WHY certain hierarchies should exhibit absorption. Without such a framework, the hierarchy selection feels arbitrary.

### Alternative C (Validity-Aware Analysis)

The validity-aware analysis idea directly anticipated the Random-SAE finding. Incorporating its random-baseline comparison would have strengthened the paper's claims about metric validity.

---

## What the Idea Phase Got Right

1. **Identified the right question:** Whether first-letter absorption generalizes to semantic hierarchies is genuinely important.
2. **Chose a feasible scope:** Training-free, existing SAEs, SAEBench codebase.
3. **Designed clear hypotheses:** Pre-registered criteria with falsification thresholds.
4. **Planned appropriate controls:** Random-SAE, non-hierarchy, GPT-2 replication.

## What the Idea Phase Got Wrong

1. **Underestimated sample size needs:** n=7 is inadequate for correlation analysis.
2. **Under-specified the Random-SAE control:** The design was internally inconsistent.
3. **Mismatched control conditions:** Binary vs. multi-class comparison is confounded.
4. **Overlooked metric design flaws:** Perfect probe accuracy collapsing the absorption formula was foreseeable.
5. **Under-powered replication:** GPT-2 with 2 SAEs provides no meaningful replication.

---

## Recommendations for Future Iterations

1. **Increase SAE sample size** to 15-20 architectures for adequate correlation power
2. **Redesign the Random-SAE control** to permute encoder directions and verify the implementation
3. **Use structurally equivalent controls** (binary hierarchy pairs vs. binary non-hierarchy pairs)
4. **Add a task-difficulty control** (same hierarchy, different probe complexities)
5. **Incorporate the FastProbe-Absorb screen** as a secondary validation
6. **Test with natural corpus sentences** instead of synthetic templates
