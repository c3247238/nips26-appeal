# Supervisor Review -- Iteration 9

**Score: 6.0 / 10** | Verdict: **continue** | Trend: **declining** (6.5 -> 6.0)

---

## Executive Summary

The paper has undergone a major strategic restructuring since iter_007/008: all Lyapunov/PMP-WD/certified-band theoretical apparatus has been removed, repositioning the paper as an empirical contribution. This is the correct strategic move -- the empirical story (Phi Framework + diagnostic metrics + Phi Invariance Conjecture) is cleaner and more defensible than the over-ambitious theory-heavy version.

However, the score **decreases** from 6.5 to 6.0 because the figure-table-text consistency problems are now WORSE than before. The paper's central quantitative claim ("0.25pp spread") is contradicted by its own Figure 3(a) ("0.49pp spread"), which uses different data (iter_006) than Table 2 (iter_003). PMP-WD appears in figures but not in any table or text. The half_lambda BEM bug (Figure 4 shows BEM~0.0, Table 6 says 0.500) has persisted for 3 iterations unfixed.

A paper that contradicts itself on its central numbers is not submittable to any venue.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|:-----:|---------------|
| Novelty | 7 | Phi Framework taxonomy is genuinely useful; no prior work provides a unified mathematical interface for WD methods. The "weight decay illusion" framing is compelling. However, the novelty is primarily organizational (taxonomy + notation) rather than algorithmic or theoretical. |
| Soundness | 5 | Three self-contradictions in figures vs tables vs text. Data provenance mismatch (0.33pp inter-iteration shift exceeds the claimed 0.25pp inter-method spread). These are not minor presentation issues -- they undermine the paper's central quantitative claims. |
| Experiments | 6 | 105 controlled experiments with good statistical methodology (paired t-tests, Bonferroni correction, TOST). But CIFAR-only scale is insufficient by 2026 standards. N=3 seeds is underpowered for a null-result paper. NoBN ablation data exists but is unused. |
| Reproducibility | 5.5 | Training details are adequate (Section 4). However, the data provenance mismatch means a reader cannot reproduce the paper's numbers -- they do not know which codebase version produced which table. CSI's smoothness constant L estimation is undocumented. |

---

## Critical Issues (would cause rejection)

### C1: Figure-Table Data Provenance Mismatch
**Table 2** reports constant AdamW CIFAR-10 = 90.13 +/- 0.31 (iter_003 data: seeds 90.48, 90.03, 89.89).
**Figure 3(a)** shows constant at ~89.80 (iter_006 data: seeds 89.72, 90.15, 89.54).

The 0.33pp shift between identical configurations in different iterations is **larger than the paper's headline finding of 0.25pp inter-method spread**. The central claim -- that WD methods are equivalent within a 0.25pp band -- is confounded by uncontrolled cross-iteration variance that exceeds the signal.

**Fix**: Rerun all 7 methods with iter_006 instrumented code (21 runs, ~6 GPU-hours) for provenance consistency, OR regenerate Figure 3(a) from iter_003 data.

### C2: Figure 4 BEM Contradiction
Figure 4 (fig3_bem_vs_accuracy.png) plots half_lambda at BEM approximately 0.0 on the CIFAR-10 panel. Table 6 reports BEM = 0.500. This is a direct visual contradiction between a figure and a table, flagged in iter_007 AND iter_008, still unfixed.

**Fix**: Regenerate Figure 4. 20 minutes.

### C3: Three Contradictory Spread Numbers
- Figure 3(a) annotation: "Spread: 0.49pp"
- Section 5.1 text: "0.25 percentage points"
- Table 2 data: max(90.13) - min(89.88) = 0.25pp

The 0.49pp in Figure 3(a) comes from iter_006 data with a different method set. The paper cannot disagree with itself on its central number.

**Fix**: Regenerate Figure 3 with consistent data, update annotation.

### C4: No ImageNet
CIFAR-10/100 with ResNet-20 (270K params) and VGG-16-BN (15M params) are toy-scale. No comparable paper at ICLR/NeurIPS 2025-2026 publishes WD results without ImageNet or LLM-scale validation.

**Fix**: ImageNet-100 subset, 3 methods x 3 seeds x ResNet-50, ~6-8 GPU-hours.

---

## Major Issues (significantly weaken the paper)

### M1: PMP-WD Ghost Method
Figure 3(a) shows PMP-WD but it appears in NO table and is not in the paper's "7 methods" list. Figure 3 panels show different method sets, making the spread comparison invalid.

### M2: Underpowered TOST
N=3 gives ~15-20% power at delta=0.5%. TOST confirms equivalence for only 6/12 comparisons at delta=1.0%.

### M3: Unused NoBN Data
Data exists from iter_005. Zero compute needed to test the BN mechanism hypothesis (Section 6.2).

### M4: CSI as Contribution
Arbitrary weights, no predictive value. Should be demoted from contribution to exploratory diagnostic.

### M5: Misleading Abstract
Implies full factorial design. Should explicitly state non-factorial structure.

### M6: Section 5.7 Negative Result
rho=-0.379, p=0.121 is not significant. Should be framed honestly or removed.

---

## What Would Raise the Score

| Action | Time | Score Impact | Priority |
|--------|------|:----------:|:--------:|
| Fix all figure-table-text consistency (C1-C3, M1) | ~4 hours, 0 GPU | 6.0 -> 6.5 | P0 |
| Consistency rerun: 7 methods, iter_006 code | ~6 GPU-hours | 6.5 -> 7.0 | P1 |
| ImageNet-100: 3 methods x 3 seeds | ~6-8 GPU-hours | 7.0 -> 7.5 | P1 |
| Integrate NoBN data | ~1 hour, 0 GPU | +0.25 | P1 |
| Add 2 seeds for TOST power | ~2 GPU-hours | +0.25 | P2 |
| Demote CSI, fix abstract | ~1 hour | +0.1 | P2 |

**Total to reach ~8.0**: ~16 GPU-hours + ~6 hours of figure/text work.

---

## Positive Aspects

1. **Strategic restructuring was correct**: Removing the Lyapunov/PMP-WD theory eliminates the theory-experiment mismatch (SGD theory, AdamW experiments) and removes multiple attack surfaces. The paper is now empirically honest.

2. **Statistical methodology is above community norm**: Paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, explicit power analysis. This is the paper's competitive advantage.

3. **Phi Framework taxonomy (Table 1)** provides genuine organizational value. No prior work provides a unified mathematical interface for WD methods with a pluggable software interface.

4. **"Weight decay illusion" framing** is compelling and well-supported by the AdamW vs SGD contrast.

5. **Section 4.3 Hyperparameter Fairness Protocol** is a model for how controlled WD comparisons should be done.

---

## Calibration Note

This score (6.0) reflects a paper with a good idea (Phi Framework + Phi Invariance Conjecture) undermined by execution problems (data inconsistency, scale limitation, low power). The novelty score (7) is stable across iterations because the framework genuinely fills a gap. The soundness score (5) is the bottleneck -- a paper that contradicts itself on its central quantitative claim cannot score above 6 overall regardless of other strengths. Fixing the figure-table consistency issues is the single highest-ROI action.
