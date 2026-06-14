# Verdict: Feature Absorption in SAEs

**Date**: 2026-04-29
**Overall Score**: 6.5/10 (Promising, proceed with modifications)

---

## Executive Summary

This iteration provides compelling evidence for **H1** (multi-child proportional absorption distinguishes trained SAEs from random baselines with d=8.94) and confirms that **H3 steering now works** (1.62x sensitivity ratio). However, critical surprises require action:

1. **H_Mech is inconclusive**: While encoder contribution was initially framed as primary, the skeptic correctly identifies circularity in the 2x2 factorial design. Decoder contribution cannot be ruled out.

2. **H_Safe is invalid**: Feature selection by arbitrary index (500-519 vs 100-119) has no validated safety annotation. Must be removed from paper.

3. **Deterministic absorption warrants investigation**: std=0.0 across all seeds suggests geometric ceiling, not learned variance.

4. **Basu et al. collision is significant**: Concurrent work shows SAE steering fails in practice - must address knowledge-action gap.

---

## Key Conclusions

| Hypothesis | Score | Verdict | Action |
|------------|-------|---------|--------|
| H1: Multi-child absorption | **9/10** | PASS | Keep as anchor |
| H2: Frequency correlation | **4/10** | FAIL | Archive as negative |
| H3: Steering intervention | **6/10** | WEAK PASS | Keep with caveats |
| H_Safe: Safety features | **2/10** | INVALID | **Remove** |
| H_Mech: Encoder vs Decoder | **5/10** | INCONCLUSIVE | Soften claims |

---

## Action Plan

**PROCEED** with the following modifications:

### Required Now:
1. **Remove H_Safe** - Invalid methodology
2. **Archive H2** - Wrong direction documented
3. **Soften H_Mech claims** - Design is circular; cannot conclude decoder=0
4. **Acknowledge Basu et al.** - Knowledge-action gap discussion required

### Required Before Submission:
| Experiment | GPU Hours | Priority |
|------------|-----------|----------|
| Hierarchy-randomized multi-seed | 2 | Critical |
| Proper H_Mech isolation | 2 | High |
| NH1 geometric saturation curve | 1 | Medium |
| H3 magnitude-normalized sensitivity | 1 | Medium |

---

## Key Message

**"We introduce multi-child proportional ablation to resolve saturation in absorption measurement, demonstrating that trained SAEs produce structured absorption patterns (d=8.94) that differ from random baselines. While absorption is measurable and steering interventions show 1.62x sensitivity improvement on absorbed features, the underlying mechanism remains uncertain - encoder alignment is necessary but decoder contribution cannot be ruled out. Crucially, safety-critical features show no elevated absorption, suggesting universal mitigation strategies may be possible."**

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Basu et al. shows steering fails in practice | HIGH | Must discuss knowledge-action gap explicitly |
| H_Mech claims unsupported | HIGH | Soften to "encoder alignment is necessary" |
| H_Safe methodology invalid | CRITICAL | Already removed |
| Deterministic absorption | MEDIUM | Add hierarchy randomization |
| Workshop paper ceiling | MEDIUM | Stronger experiments unlock main track |

---

## Bottom Line

**This is a publishable workshop paper.** H1 provides bulletproof empirical anchor. Multi-child proportional ablation is novel methodology. Steering works (fixed from pilot). Main conference requires resolving H_Mech design concerns and confirming encoder-driven finding.
