# Critique: Ideation

**Reviewer:** sibyl-critic
**Date:** 2026-04-15
**Target:** `current/idea/proposal.md`, `current/idea/alternatives.md`, `current/idea/hypotheses.md`

---

## Overall Assessment: 6/10

The ideation demonstrates genuine intellectual honesty -- particularly in tracking hypothesis falsification across iterations (H2 falsified, H4 refuted, H5 not supported, CMI demoted). This is exemplary scientific practice. However, the proposal suffers from three structural problems that weaken it as the foundation for a high-impact paper.

---

## Strengths

### S1. Honest hypothesis tracking across iterations
The progression from "H2: first-letter is worst case" through "H2 REFUTED by pilot data" to "H2' REFUTED at L24: first-letter is actually highest" demonstrates exactly how evidence should update priors. This is the proposal's strongest quality and should be preserved in the paper narrative.

### S2. Well-defined falsification criteria
Each hypothesis has explicit falsification conditions (e.g., H1: "Rates within 5 pp across all types after controlling for L0 and width"). This is above-average for ML interpretability proposals.

### S3. Negative results elevated, not buried
GAS failure, CMI non-result, and Absorption Tax prediction failure are prominently featured, not hidden. This is consistent across all nine iterations and is the paper's most reviewer-friendly aspect.

---

## Weaknesses

### W1. CRITICAL -- The probe quality confound was known from the start but not resolved in the experimental design

The proposal acknowledges "Probe quality is the critical blocking dependency" (line 189) and sets a quality gate of F1 >= 0.90. The results show only 2/20 probes pass this gate. Despite knowing this was the binding constraint, the ideation phase did NOT include a degraded-probe ablation in the experimental plan. This is not a post-hoc discovery -- the probe quality problem was flagged in the proposal's own risk assessment ("RAVEL probes still below 0.90 at all layers: High severity, 30% likelihood").

The proposal's mitigation for this risk is "Test layers 18, 24. Relax to 0.85 with documented caveat. Fall back to GPT-2 Small." These are all attempts to find better probes rather than controlling for probe quality as a confound. The correct mitigation was to design an experiment that is robust TO probe quality variation (e.g., degraded-probe ablation, or probe-independent absorption measurement via activation patching across hierarchies).

### W2. MAJOR -- Three of seven hypotheses were already falsified or not supported BEFORE full-mode experiments

The proposal enters full mode with H2 refuted, H4 refuted, and H5 not supported. The remaining hypotheses are H1 (cross-domain variation), H3 (hedging decomposition), H6 (architecture generalization), and H7 (causal absorption). Of these:
- H6 turns out to be inconclusive (p=0.87, underpowered)
- H3 is partially supported but the cross-domain hedging comparison is invalid (N=3 for city-language)

This means the paper rests on two solid results: H1 (confounded by probe quality) and H7 (limited to first-letter at L12). The ideation should have pivoted more aggressively after the pilot results showed most hypotheses failing, rather than running "full mode" on a hypothesis set that was already largely exhausted.

### W3. MAJOR -- The 'Absorption Tax' theoretical framework persists despite total empirical failure

The Absorption Tax theorem (T(G) = sum p_c * R_pc) is promoted as a "Tertiary" contribution and receives a dedicated section in the experimental plan despite:
- Absorption-MSE correlation: rho=0.08 (pilot)
- R_pc prediction: rho=0.16 (pilot)
- Ranking prediction: rho=-0.20 (full)
- Pairwise concordance: 50% (chance)

The proposal correctly notes these as "NOT SUPPORTED" but still allocates experimental resources (Phase 3.1-3.2: 1.5 hours) to a framework that has been comprehensively falsified. The T(G) formulation's failure is not due to insufficient data -- the correlation is near zero. The ideation should have DROPPED the Absorption Tax from the experimental plan entirely after the pilot results.

### W4. MINOR -- Backup ideas are high-quality but none were activated

The alternatives (Backup A: controlled dictionary experiment, Backup B: ecological phase transition) are genuinely creative and potentially more impactful than the current paper's cross-domain extension. Backup A (separating encoder from dictionary using OMP encoding with the same decoder) directly tests the mechanism question that the current activation patching result only partially addresses. The ideation process generated strong backup ideas but was too committed to the current trajectory to activate them.

### W5. MINOR -- Venue targeting is optimistic given the confound

"Primary: NeurIPS 2026 main conference or ICLR 2027 main conference." Given that the primary contribution is confounded by probe quality, two of the four contributions are restricted to first-letter only, and all theoretical contributions failed, ICLR 2027 main conference is optimistic. A workshop paper or EMNLP (the listed fallback) is more realistic without the degraded-probe ablation.

---

## Ideation Quality Summary

| Dimension | Score | Comment |
|-----------|-------|---------|
| Originality of research questions | 7/10 | Cross-domain absorption extension is overdue but not conceptually deep |
| Hypothesis rigor | 8/10 | Explicit falsification criteria, honest tracking |
| Risk anticipation | 5/10 | Identified probe quality risk but did not design around it |
| Pivot readiness | 4/10 | Strong backup ideas never activated despite widespread hypothesis failure |
| Resource allocation | 5/10 | Spent time on falsified Absorption Tax; no degraded-probe ablation designed |
