# Ideation Critique: Research Proposal Review

> Critic: sibyl-critic (heavy tier)
> Date: 2026-04-29
> Scope: proposal.md + alternatives.md + hypotheses.md + novelty_check.md

---

## Overall Ideation Assessment

**Strength**: The proposal honestly pivots from a falsified positive result (Iteration 2: "UAD works") to a negative result (Iteration 3: "UAD fails"). This intellectual honesty is the proposal's greatest strength.

**Weakness**: Several issues in the original ideation (positive-result framing) were carried over into the negative-result reframing without adequate reassessment.

---

## 1. Research Question Clarity

**Strengths:**
- RQ1, RQ2, RQ3 are well-structured and answerable
- The central question ("Does co-occurrence clustering actually detect feature absorption?") is crisp

**Issues:**

1. **RQ2 is circular**: "Is collision rate a robust proxy for true absorption rate?" --- But collision rate IS defined as the operationalization of true absorption rate. This is not a proxy validation question; it is a reliability/consistency question.

2. **RQ3 is speculative**: "What alternative approaches are theoretically better suited?" --- The proposal answers this with decoder weight similarity and causal intervention, but neither has been tested. Proposing untested alternatives as a "contribution" is weak.

**Fix:** Reframe RQ2 as "Does the collision rate operationalization reliably capture absorption patterns across diverse concept hierarchies?" Reframe RQ3 contribution as "directions for future work" rather than a contribution.

---

## 2. Hypothesis Quality

**Strengths:**
- Pre-registered falsification criteria (H1, H2, H3)
- Clear go/no-go gates
- Honest reporting of falsified hypotheses

**Issues:**

1. **H1's falsification criterion is too loose**: "If F1 > 0.01, H1 would be falsified" --- With 7 GT pairs, F1 = 0.01 would mean detecting ~0.07 true positives (less than 1). This criterion is essentially impossible to falsify. The threshold should have been set relative to a meaningful baseline.

2. **H2's falsification criterion is also loose**: "If any two absorption features co-occur on > 5% of tokens, H2 would be falsified" --- The paper only tested 4 features in one sequence. The criterion is not tested against.

3. **H-S2 (decoder weight similarity) is untested but presented as a hypothesis** --- A hypothesis without any evidence is just speculation.

**Fix:** Acknowledge that the falsification criteria were designed for the positive-result framing and are not well-calibrated for the negative-result outcome.

---

## 3. Novelty Assessment Problems

**Critical Issue**: The novelty check (`novelty_check.md`) was performed on the **original positive-result framing** (UAD as a new method). The current paper is about **why UAD fails**. The novelty of these two framings is completely different:

- **Original framing**: "First unsupervised absorption detection method" --- HIGH novelty
- **Current framing**: "Co-occurrence clustering cannot detect absorption" --- MEDIUM novelty (methodological observation)

The novelty check does NOT assess:
- Whether "UAD fails" is a novel finding
- Whether the token-level mutual exclusivity insight is novel
- Whether collision rate validation at n=56 is novel

**Fix:** Re-run novelty assessment for the actual paper's contributions. Search for prior work that:
1. Systematically evaluates co-occurrence clustering for absorption detection
2. Identifies token-level mutual exclusivity as a barrier
3. Validates collision rate as an absorption proxy at scale

---

## 4. Contribution Claims Analysis

### Contribution 1: "Empirical Falsification of Co-occurrence Clustering"

**Strength**: First systematic evaluation of UAD on pretrained SAEs.

**Weakness**: The evaluation is on ONE SAE with ONE hierarchy type and only 7 GT pairs. Calling this "empirical falsification" is strong language for limited evidence.

**Fix**: Frame as "empirical demonstration of failure" rather than "falsification." Add caveat about limited scope.

### Contribution 2: "Root Cause Identification"

**Strength**: The token-level mutual exclusivity explanation is elegant and theoretically grounded.

**Weakness**: Only demonstrated on number words in a synthetic sequence. The claim that this is "the fundamental reason" is an overgeneralization.

**Fix**: Qualify as "the fundamental reason for token-disjoint hierarchies" and acknowledge untested cases.

### Contribution 3: "Validation of Collision Rate as Proxy Metric"

**Strength**: Strong correlation (r=0.869) across 56 pairs and 2 hierarchy types.

**Weakness**: Collision rate IS the operational definition of absorption. This is not independent proxy validation.

**Fix**: Reframe as "collision rate reliably operationalizes absorption patterns across diverse hierarchies."

### Contribution 4: "Constructive Forward Look"

**Strength**: Proposes concrete next steps.

**Weakness**: The proposed alternatives (decoder weight similarity, causal intervention) are entirely untested. Proposing them as a "contribution" is weak---they are future work, not contributions.

**Fix**: Reframe Contribution 4 as "Directions for Future Work" rather than a contribution.

---

## 5. Alternatives Assessment

The `alternatives.md` file provides a well-structured pivot decision matrix. However:

1. **Alternative A (decoder weight similarity)** is ranked "Highest Priority" but has zero empirical support. The rationale is purely theoretical.

2. **Alternative C (collision rate standalone)** is ranked "Safe fallback" but is actually the strongest validated finding in the paper.

3. **The pivot path** recommends "write negative result paper now, then pilot Alternative A." This is reasonable but the paper should not present Alternative A as a validated contribution.

---

## 6. Evolution from Prior Iterations

The proposal's Section 2.1 "What Changed from Iteration 2" is excellent:
- Clearly documents falsified claims
- Shows honest revision in response to data
- Maintains intellectual integrity

This section should be preserved and potentially expanded in the paper's appendix.

---

## Summary: Ideation Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Honesty | 9/10 | Excellent pivot from falsified positive result |
| RQ Clarity | 7/10 | RQ2 is circular; RQ3 is speculative |
| Hypothesis Quality | 6/10 | Falsification criteria poorly calibrated |
| Novelty Assessment | 5/10 | Performed on wrong framing; needs re-assessment |
| Contribution Claims | 6/10 | Overstates scope and independence of findings |
| Alternatives | 7/10 | Good structure but untested proposals overvalued |
| **Overall** | **6.5/10** | Honest but needs reframing of several claims |
