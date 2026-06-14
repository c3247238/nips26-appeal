# Final Review (Round 2): "When Does Dynamic Weight Decay Help? A Unified Framework Analysis"

**Reviewer**: Sibyl Final Critic (NeurIPS/ICML standard)
**Date**: 2026-03-18
**Paper version**: Iteration 4 integrated paper, Revision 2 (74KB, ~530 lines)
**Basis**: integrated_paper.md (revised), pilot_summary.md, previous final_review.md
**Review round**: 2 (post-revision comparison)

---

## Overall Score: 6.5 / 10

**Verdict**: The revision has addressed many of the writing-level and framing issues identified in Round 1, producing a noticeably more honest and well-calibrated paper. The overclaiming problem (previously the most jarring weakness) is substantially resolved: the abstract, conclusion, and discussion now consistently hedge claims with appropriate qualifiers. However, the fundamental experimental gaps remain unchanged---no new compute has been added between rounds. The paper is now a well-written, honestly-scoped analysis that makes the best possible case from limited data, but it remains an incomplete empirical study by top-venue standards. The score improvement from 5.5 to 6.5 reflects genuine progress in calibration, framing, and transparency, not a change in the underlying evidence.

---

## Comparison with Round 1 Review

### Issues Resolved or Substantially Improved

| Round 1 ID | Issue | Status | Notes |
|-----------|-------|--------|-------|
| M1 | "Theorem 1" is a trivial algebraic identity | **FIXED** | Now labeled "Observation 1 (Dual Characterization)" with appropriate framing: "while immediate once stated, was not made explicit in either prior work." This is exactly the right calibration. |
| M4 | SGD/AdamW comparison confounded by different rho | **FIXED** | The paper now explicitly states the rho asymmetry (0.005 vs 0.5) in the abstract, Section 4.3, Section 6.2, and the conclusion. The 18.3x ratio is consistently presented as reflecting "the combined effect of optimizer mechanism and operating-point difference." This is thorough and honest. |
| M5 | Proposition 2 has acknowledged formal gap | **IMPROVED** | The formal gap is now explicitly flagged in the text: "extending Sun et al.'s fixed-lambda analysis to time-varying lambda_t involves a formal gap" and is marked as "an open theoretical question." Appropriate hedging. |
| M12 | Abstract/conclusion overclaim relative to evidence | **SUBSTANTIALLY FIXED** | The abstract now uses "no statistically significant differences were detected" rather than "statistically indistinguishable." The conclusion uses "conditional equivalence observation" and "consistent with conditional equivalence but does not constitute proof." The discussion opens with "tentative practical message" and "we detected no statistically significant benefit." Section 7.1 heuristic is labeled "preliminary, conjectured." This is a major improvement. |
| M13 | Regime boundary presented as more established than it is | **IMPROVED** | The conjecture is now more carefully framed with explicit falsification criteria (Section 7.2) and the conclusion acknowledges "currently supported at a single operating point." The three falsifiable predictions are well-structured. |
| m3 | SWD BEM "varies" unexplained | **FIXED** | Table 2 footnote now explains that SWD's BEM is time-varying due to gradient-norm sensitivity, with the mean BEM reported as approximately -0.3. |
| M11 | CIFAR-100 SGD no_wd has n=1 | **FIXED** | Now disclosed in Section 5.2 ("Data completeness note"), Section 6.2 ("not included in statistical comparisons"), and Limitation 8 in Section 7.3. |
| M3 | O(1/rho) scaling prediction dimensionally confused | **FIXED** | The problematic scaling derivation has been removed. The paper no longer claims a specific functional form for the ratio, instead presenting the 18.3x as an empirical observation with the rho confound clearly stated. |

### Issues Partially Addressed

| Round 1 ID | Issue | Status | Notes |
|-----------|-------|--------|-------|
| M2 | Lemmas 1-3 proofs missing (Appendix D) | **PARTIALLY ADDRESSED** | The paper now provides the intuition for why the bound becomes second-order in rho (Section 4.2, detailed paragraph after Lemma 3). This is substantially more content than before, including the key telescoping argument and the explanation of exponential damping. Appendix D is still "in preparation," but the main text now contains enough sketch to evaluate the argument's plausibility. The critical gap---formal verification that the Adam saturation condition holds globally rather than at 80%---is now explicitly flagged as a limitation. Remaining gap: the actual written appendix is still missing. |
| M9 | BN confound unresolved | **PARTIALLY ADDRESSED** | The paper now repeatedly and prominently acknowledges this confound (Sections 4.2, 6.4, 7.1, 7.2, 7.3). It is identified as "the highest-priority blocking experiment" in multiple locations. The competing mechanisms (AdamW l-infinity vs BN scale-invariance) are explicitly presented as alternatives. However, the confound is still unresolved empirically---no NoBN experiment has been run. The improvement is in transparency, not in evidence. |
| m1 | Adam saturation verified at 80%, not 100% | **PARTIALLY ADDRESSED** | Now explicitly flagged: "the gap between 80% and 100% is not formally addressed and represents a limitation of the current analysis" (Section 4.2). Good transparency but the gap remains. |
| m5 | No training curves or trajectory figures | **NOT ADDRESSED** | Still no figures in the integrated paper. The paper references "weight norm convergence" data in Section 6.4 but shows no trajectories. |
| m7 | No figures in integrated paper | **NOT ADDRESSED** | Same as above. For a paper with 87 experiments, the absence of any visualization is a significant presentation weakness. |

### Issues Unchanged (Require Compute)

| Round 1 ID | Issue | Status |
|-----------|-------|--------|
| M6 | Single architecture (ResNet-20 only) | Unchanged---requires VGG/ViT experiments |
| M7 | n=3 insufficient for equivalence claims | Unchanged---requires more seeds |
| M8 | Regime boundary untested (rho=0.5 only) | Unchanged---requires lambda sweep |
| M10 | No ImageNet or large-scale validation | Unchanged---requires ImageNet runs |

---

## Detailed Assessment by Dimension

### 1. Novelty and Originality: 6.5 / 10 (up from 6)

**Improvement**: The demotion of "Theorem 1" to "Observation 1" and the more careful positioning relative to prior work (especially the explicit comparison paragraphs in Section 2 for Xie & Li, D'Angelo, Wang & Aitchison, Chou) make the novelty claims more defensible. The paper no longer overstates its mathematical contributions.

**Remaining concerns**:
- The core empirical finding (AdamW insensitivity) remains strongly anticipated by existing work. A reviewer who has read Xie & Li (2024) and D'Angelo et al. (2024) would predict this outcome.
- The genuinely novel contribution---the controlled cross-optimizer comparison and the 18.3x ratio---is now better presented but is a single data point with known confounds.
- The Phi framework as a taxonomy is useful but incremental; it organizes existing methods rather than enabling new ones.

### 2. Technical Correctness: 6 / 10 (up from 5)

**Improvements**:
- Removal of the confused O(1/rho) scaling derivation eliminates a mathematical error.
- The "Observation 1" framing is honest about its algebraic nature.
- The proof sketch for the O(rho^2) bound (Section 4.2) is now present and the reasoning is sound in outline.
- The rho confound in the 18.3x ratio is thoroughly disclosed.

**Remaining concerns**:

**M2' (downgraded from Critical to High)**: Appendix D still does not exist. The inline proof sketch is helpful but insufficient for a venue that expects formal proofs. Lemma 3's claim that the damped sum is O(rho^2) is the nontrivial step; the sketch provides the right intuition (l-infinity bound on ||w_s|| converts each term to O(rho * eta), exponential damping gives geometric sum O(1/lambda), total is O(eta^2/lambda) = O(rho^2 * eta^2 / ... )), but the details need to be verified. The BEM-equivalent normalization step in particular is hand-waved.

**m2 (unchanged)**: CSI metric weights (1/3 each) remain unjustified. This is minor but should be acknowledged as an ad hoc choice.

**m4 (unchanged)**: Cosine schedule low variance is still informally noted. Not a major issue but a loose end.

**New concern (N1)**: The paper states "accuracy range < 0.3% on CIFAR-10" (Section 6.1, point 1) but the actual range is 90.13% - 89.88% = 0.25%. This is correct but the "< 0.3%" framing is awkward---it is exactly 0.25%. On CIFAR-100, the range is 0.76%, which is much larger and falls right at the MDE boundary. The paper correctly notes this but could be more explicit that the CIFAR-100 results are genuinely ambiguous rather than clearly supporting equivalence.

### 3. Experimental Sufficiency and Rigor: 4.5 / 10 (up from 4)

The 0.5-point improvement reflects the better experimental reporting (n=1 disclosure, pilot framing, data completeness notes) rather than new experiments.

**Unchanged critical gaps**:

**M6 (Critical)**: Single architecture. This remains the single biggest weakness. The VGG pilot (10 epochs, 1 seed) is now correctly labeled as "not intended as scientific validation," which is good framing but does not solve the problem. A reviewer will still reject on this basis.

**M7 (Critical)**: n=3. The paper now handles this much better linguistically ("no statistically significant differences were detected" vs. "statistically indistinguishable"), but the underlying problem remains. The CIFAR-100 spread of 0.76% at MDE = 0.77% means the data is genuinely ambiguous for CIFAR-100---the paper could be missing a real effect.

**M8 (Critical)**: Regime boundary at rho=0.5 only. The conjecture now has three explicit falsifiable predictions (Section 4.2), which is good scientific practice, but none of them are tested beyond the first. A "phase diagram" from one measurement remains unconvincing.

**M10 (High)**: No ImageNet. By 2026 standards, CIFAR-only results are insufficient for claims about "when dynamic weight decay helps" in general.

**Improved aspects**:
- The pilot summary (T0.1-T0.4, T1.1-T1.2) demonstrates good experimental hygiene: the BEM bug was found, diagnosed, and fixed with verification.
- The SGD baseline analysis (42 runs, proper statistical testing) is solid work.
- The framing of VGG pilot as infrastructure validation rather than evidence is honest.

### 4. Writing Quality: 7.5 / 10 (up from 7)

**Improvements**:
- The calibration between claims and evidence is now consistently maintained throughout the paper. The abstract, introduction, results, discussion, and conclusion all use appropriate hedging language. This is the single biggest improvement in the revision.
- The "Statistical honesty statement" in Section 6.2 remains a model of responsible reporting.
- Section 7.3 (Limitations) is now even more thorough, with 8 explicitly enumerated limitations including the BN confound, rho asymmetry, and data completeness.
- The related work comparisons (Section 2) are now structured as explicit "vs." paragraphs, making the positioning clearer.
- The falsification criteria (Section 7.2) are well-specified and scientifically rigorous.

**Remaining concerns**:

**m7/m5 (unchanged, now elevated to Medium)**: The complete absence of figures is a significant presentation weakness for a paper with 87 experiments. Bar charts of accuracy by method, weight norm convergence trajectories, a rho regime diagram, and BEM-vs-accuracy scatterplots would substantially improve readability and impact. The iter_003 figures exist but are not integrated.

**m6 (unchanged)**: Several references still lack complete publication details (arXiv IDs for Chou 2025, He 2025, Kosson 2023, Loshchilov 2023, Tian 2024). Note: Defazio (2025) now has an arXiv ID (2506.02285), which is an improvement.

**m8 (unchanged)**: At 74KB in markdown (~25+ pages in conference format), the paper is too long. Sections 4 and 7 could be condensed; some material (detailed proof sketches, extended discussion of prior work) should move to appendices.

### 5. Impact and Significance: 6 / 10 (unchanged)

The impact assessment is unchanged because the underlying evidence has not changed. The practical message is clear and useful, but:
- It confirms the status quo (use constant WD with AdamW) rather than changing practice.
- The scope (CIFAR, ResNet-20) limits generalizability.
- The "null result" framing requires the regime boundary validation to become a "conditional positive," and that validation is missing.

The Phi framework's impact as a taxonomy stands regardless of empirical outcomes, but a taxonomy without extensive empirical validation is a workshop-level contribution, not a main-track one.

---

## New Issues Identified in Revision 2

| ID | Issue | Section | Priority |
|----|-------|---------|----------|
| N1 | CIFAR-100 spread (0.76%) at MDE boundary deserves more prominent discussion | 6.1 | Medium |
| N2 | Proof sketch hand-waves the BEM normalization step in O(rho^2) derivation | 4.2 | Medium |
| N3 | The paper's length (74KB) has increased from the previous version (~62KB) without adding new experimental content | All | Medium |
| N4 | Future work section (8.2) lists 8 directions with time estimates, giving the impression that these are tractable but not yet done; this may frustrate reviewers who will ask "why not run them?" | 8.2 | Low |

---

## Summary of Remaining Issues

### Critical (must address before submission)

| ID | Issue | Required Action |
|----|-------|----------------|
| M6 | Single architecture | Run VGG-16-BN full (200 epochs, 3 seeds) + ideally ViT |
| M7 | n=3 insufficient | Increase to n>=5 for core AdamW comparisons |
| M8 | Regime boundary untested | Lambda sweep: rho in {0.05, 0.5, 5, 50} |
| M9 | BN confound unresolved | NoBN ablation (highest priority) |

### High Priority

| ID | Issue | Required Action |
|----|-------|----------------|
| M2' | Appendix D still missing | Write formal proofs for Lemmas 1-3 |
| M10 | No ImageNet validation | ImageNet pilot (ResNet-50, key methods) |
| m7/m5 | No figures | Integrate existing figures from iter_003/iter_004 |
| m8 | Paper too long | Trim to conference length, move material to appendices |

### Medium Priority

| ID | Issue | Required Action |
|----|-------|----------------|
| N1 | CIFAR-100 ambiguity understated | More prominent discussion of CIFAR-100 MDE boundary |
| N2 | Proof sketch hand-waves | Tighten BEM normalization step |
| N3 | Paper length increased | Condense sections 4, 7 |
| m6 | Incomplete references | Add arXiv IDs |
| m2 | CSI weights unjustified | Acknowledge as ad hoc |

---

## Venue Recommendation (Updated)

| Condition | Score | Venue |
|-----------|-------|-------|
| Current state (Revision 2, as-is) | 6.5 | Strong workshop paper (NeurIPS OPT, ICML Optimization) |
| + NoBN ablation + lambda sweep + n=5 + figures | 7.0-7.5 | TMLR / NeurIPS poster (competitive) |
| + VGG-16-BN full + matched-rho SGD + Appendix D | 7.5-8.0 | NeurIPS/ICML main track (solid) |
| + ImageNet + ViT + formal proofs | 8.0-8.5 | NeurIPS/ICML strong accept |

---

## What Changed Between Rounds: Summary

The revision represents a **writing and framing improvement, not an evidence improvement**. The paper now says what the data supports rather than what the authors hope is true. Specifically:

1. **Overclaiming resolved**: The most important change. Every major claim is now appropriately hedged. The paper reads as honest science rather than advocacy.

2. **Confounds disclosed**: The rho asymmetry (M4) and BN confound (M9) are now prominently discussed, not buried. The reader can assess the evidence with full information.

3. **Mathematical errors removed**: The O(1/rho) scaling confusion (M3) is gone; Theorem 1 is correctly relabeled (M1).

4. **Data gaps disclosed**: CIFAR-100 SGD n=1 (M11) is now reported. The VGG pilot is properly framed.

5. **No new evidence**: The fundamental experimental gaps (single architecture, n=3, single rho, no ImageNet, BN confound) remain. These require compute, not writing.

The score improvement from 5.5 to 6.5 is earned: the paper is now a competent, honest piece of work that makes the best possible case from limited data. The remaining 3.5-point gap to a strong accept is almost entirely about experimental coverage.

---

## Final Assessment

This is now a well-calibrated paper that knows what it has and what it lacks. The Phi Modulator Framework is a clean organizational contribution. The controlled comparison methodology is sound. The statistical reporting is exemplary in its honesty. The falsification criteria for the conjecture are well-specified.

The bottleneck is purely experimental. The four blocking experiments (NoBN, lambda sweep, n>=5, VGG full) would collectively require approximately 10-15 hours of compute on the available hardware (8x RTX PRO 6000). Running these would likely move the paper from workshop-quality to main-track competitive. The figure integration (zero compute) would further improve presentation.

The paper's greatest strength is also its greatest vulnerability: it proposes a regime theory from a single operating point. This is either visionary (if the regime boundaries are confirmed) or premature (if they are not). The revision has made the "premature" case less damaging by being honest about the gap, but only data can close it.

**Score: 6.5/10** -- well-calibrated writing with significant experimental gaps. Clear path to 7.5-8.0 with blocking experiments completed.

---

*Reviewed by: Sibyl Final Critic (Round 2)*
*Review standard: NeurIPS/ICML main track*
*Date: 2026-03-18*
*Previous review score: 5.5/10 (Round 1)*
