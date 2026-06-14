# Result Debate Verdict: Executive Summary

**Project**: SAE Feature Absorption Audit | **Iteration**: 9 | **Date**: 2026-04-15

---

## Result Quality Score: 7.0 / 10

A solid experimental foundation with two strong empirical pillars (metric audit + L0 transition), one metric-independent causal test (activation patching), and a notably honest portfolio of negative results (7+ of 10 hypotheses falsified). The collapsed CMI theoretical pillar and single-model scope prevent a higher score. After writing revision, the paper should reach 7.0-7.5 at NeurIPS/ICML review standard, with 8.0 upside if writing is excellent.

---

## Key Conclusion

**The Chanin absorption metric does not measure competitive exclusion on JumpReLU SAEs.** Three independent evidence streams converge:

1. **Control failure**: Shuffled-label controls exceed measured absorption in all 5 domains because the cosine threshold (>= 0.025) identifies 23% of decoder columns as candidates in R^2304, causing the metric to degenerate into a false-negative rate proxy.

2. **Causal test**: Activation patching yields 0/8 parent feature recovery for persistent false negatives across three methods, with 0/65 control recovery. (Caveat: JumpReLU hard threshold confound must be discussed.)

3. **Hedging decomposition**: Only 6.2% of false negatives show parent-specific recovery at high L0 (strict hedging), while 93.8% resolve through entirely different features -- the "98.6% hedging" headline was a construct validity failure from an overly loose definition.

The dominant false-negative mechanism is not competitive exclusion but **encoding absence under capacity constraint**: at low L0, the SAE encoder simply does not allocate capacity to surface-level attributes like first-letter identity. L0 is the primary control variable (42.85% at L0=22 -> 0.84% at L0=176, cross-layer CV<10%).

---

## Action Plan

### Decision: PROCEED to Gate 2 (Writing Revision)

All experiments are complete. No additional GPU time is needed. The sole bottleneck is integrating new results and correcting the narrative.

### 5 Critical Writing Revisions (P0)

| # | Revision | Rationale |
|---|----------|-----------|
| 1 | Rewrite Section 6: CMI from "diagnostic" to "exploratory negative result" | H4 falsified (L0=22: rho=+0.044, p=0.835) |
| 2 | Revise hedging narrative: 3-category decomposition (6.2% strict / 87.6% compensatory / ~0% persistent) | 92.3pp gap between loose and strict definitions |
| 3 | Add Section 4.3: Activation patching 0/8, with JumpReLU threshold caveat | First causal evidence against competitive exclusion |
| 4 | Revise title and abstract: remove rate-distortion, adopt audit framing | CMI pillar collapsed; title must reflect actual contributions |
| 5 | Eliminate all CMI overclaims (7+ locations) | Every CMI mention must reference L0=22 falsification |

### Estimated Timeline
- Writing revision: ~12 hours (1 week)
- Final validation: automated cross-check (validate_integration)
- arXiv preprint: immediately after revision
- Target venue: ICLR 2027 (primary), COLM 2027 (backup)

### Key Risk
Re-introducing CMI overclaims during writing revision. Enforce rule: every CMI mention must cite the L0=22 replication failure.

---

## Hypothesis Scorecard

| ID | Hypothesis | Verdict | Key Evidence |
|----|-----------|---------|--------------|
| H1 | Metric does not transfer to JumpReLU | **Confirmed** | 5-domain control failure, mechanistic diagnosis |
| H2 | >80% hierarchy-driven absorption | **Falsified** | 6.2% strict hedging, 0/8 patching recovery |
| H3 | L0 sparsity transition | **Confirmed** | 42.85%->0.84%, cross-layer CV<10% |
| H4 | CMI predicts absorption susceptibility | **Falsified** | L0=22 rho=+0.044; probe quality confound |
| H5 | Cross-domain absorption >10% in 2+ domains | **Falsified** | Universal control failure |
| H8 | Patching recovers 7+/9 parents | **Falsified** | 0/8 recovery, three methods |
| H9 | Strict hedging >80% | **Falsified** | 6.2% (CI: 4.4%-8.2%) |
| H10 | CMI at L0=22 rho<-0.3 | **Falsified** | rho=+0.044, p=0.835 |

**Summary**: 2 confirmed, 7+ falsified. The high falsification rate is itself a scientific contribution -- it reveals systematic bias in the existing understanding of SAE feature absorption.
