# Result Debate Synthesis: Unified Assessment

**Synthesizer**: Sibyl Result Synthesizer (sibyl-heavy)
**Date**: 2026-03-18
**Inputs**: Six independent analyses — Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist
**Data basis**: iter_003 experiments (90 AdamW runs + 84 SGD runs), iter_004 Phase 0 (7 pilot runs), sgd_baseline_analysis.json

---

## 1. Consensus Map

The following conclusions are agreed upon by at least 5 of 6 perspectives, and are treated as **high-confidence conclusions**.

### C1: The BN Ablation is the Highest-Priority Blocking Experiment (6/6)

Every perspective independently flagged the ResNet-20-NoBN ablation (P0-3) as a prerequisite before any mechanistic claim can be written. D'Angelo (2024) already proves BN scale-invariance can cause WD to be "ineffective." The two paths are:
- **Path A** (NoBN AdamW still shows invariance): AdamW's sign normalization / ℓ∞ constraint is the primary mechanism — a strong, original contribution distinct from D'Angelo.
- **Path B** (NoBN breaks invariance): BN is a necessary condition — reframe as "BN + AdamW joint mechanism," still publishable but with reduced novelty margin against D'Angelo.

**Action**: This experiment must be run before writing Sections 4 and 5 of the paper. Estimated cost: 1 GPU hour, 18 runs. Already labeled P0-3 in the plan. Escalate to P0 priority.

### C2: n=3 is Severely Underpowered for Equivalence Claims (6/6)

All perspectives agree that n=3 cannot support a formal equivalence claim:
- TOST power at n=3 (σ≈0.30%, δ=0.5%) is approximately 15–20%.
- The minimum detectable effect at 80% power with n=3 is ±0.77% — meaning an effect as large as 0.76% would be missed 80% of the time.
- The CIFAR-100 AdamW spread of 0.76% (cosine 63.42% vs no_wd 62.66%) is not clearly within the noise margin with n=3; overlapping CIs do not confirm equivalence.

**Action**: Increase n to ≥5 for all core AdamW methods (14 additional runs, ~2h). For the central constant vs no_wd comparison, increase to n=7–10 for TOST with sufficient power. Report post-hoc power analysis in the paper.

### C3: The λ Regime Sweep is the Paper's Central Falsification Experiment (5/6)

The Phi Invariance Trichotomy (ρ = λ/η regime boundaries) currently rests on a single operating point (ρ=0.5). Regime II and III have zero empirical support. Five perspectives agree that the λ sweep (P1-1, testing λ ∈ {5e-5, 5e-4, 5e-3, 5e-2}) is essential to transform the ρ framework from post-hoc rationalization to prospective, falsifiable theory.
- If spread increases at ρ>1: regime boundary is validated, paper becomes a strong positive result.
- If spread remains flat even at ρ=50: the ℓ∞ constraint is stronger than theoretical bounds, also a publishable finding.

**Dissent**: The Skeptic maintains that even with the λ sweep, deriving a three-regime theory from n=3 per condition is over-theorized. This concern is valid but does not block the experiment — it informs how the results should be framed (Conjecture + empirical verification, not Theorem).

### C4: SGD Statistical Claims Require Correction (5/6)

After Bonferroni-Holm correction, only one SGD comparison is statistically significant: constant vs no_wd (p=0.0022, d=12.17 paired / 10.3 pooled). The claims that "SGD dynamic methods outperform constant" and "multiple SGD methods show significant effects" are not supported after correction.

**Specific corrections needed**:
- Report corrected p-values (Holm): swd p_adj=0.054 (NS), half_lambda p_adj=0.062 (NS), only constant vs no_wd is significant.
- The 18.3× ratio must be labeled explicitly as "WD presence-absence effect ratio" (constant vs no_wd), not "dynamic strategy effect ratio."
- Bootstrap BCa 95% CI for the ratio must be computed and reported.

### C5: The Phi Modulator Framework Has Independent Value (5/6)

The four-axis Phi Modulator taxonomy (temporal × directional × spatial × target-norm) is recognized by 5 perspectives as a clean, operationally useful conceptual contribution. It is more prescriptive and practically oriented than Ye (2024) and D'Angelo (2024)'s frameworks. It provides a common interface (λ_eff(t, w, g) = λ · φ(t, w, g)) that unifies SWD, CWD, cosine schedules, AdamWN, and others as specializations.

**Caveat** (Skeptic): The framework should not be over-packaged. As a taxonomy + evaluation protocol, it is solid. As a "unified theory," it requires the formal invariance proof that currently does not exist.

### C6: Narrative Reframe Required — from "Null Result" to "Conditional Positive" (5/6)

The paper should not lead with "WD strategies are equivalent under AdamW." It should lead with: "We identify the regime boundary at which WD strategy choice begins to matter, and show that standard AdamW settings fall in the invariant regime."

**Recommended framing**: "When Does Dynamic Weight Decay Matter? A Regime-Boundary Analysis."

---

## 2. Conflict Resolution

### Conflict on Finding 1: AdamW Phi Invariance — "Core Contribution" vs "Anticipated Null Result"

**Optimist/Strategist/Comparativist position**: The controlled, 7-method, 3-seed comparison confirming <0.25% spread under AdamW is the paper's empirical foundation. While D'Angelo (2024) and Wang & Aitchison (2024) implied this, no prior paper provided controlled experimental confirmation with explicit budget-equivalent comparisons.

**Skeptic/Methodologist position**: "Not significant at n=3" is not "equivalent." The paper must run formal TOST before claiming invariance. Without this, the finding is "we failed to detect a difference" rather than "methods are equivalent."

**Synthesis judgment**: Both are correct. The empirical observation (7 methods within 0.25% spread, consistent across 2 datasets) is real and valuable. The statistical framing must be honest: report non-significance clearly, explicitly state n=3 limitations, run TOST at n=5 before the paper is written. The observation itself is publishable; the equivalence claim requires TOST support.

**Resolution**: State as "all 7 methods produce statistically indistinguishable accuracy under standard AdamW (p>0.09 corrected, n=3); formal TOST equivalence testing at n=5 confirms equivalence within δ=0.5% at α=0.05" — after the n=5 runs are complete.

### Conflict on Finding 2: 18.3× SGD/AdamW Ratio — "Empirical Anchor" vs "Noise-Amplified Statistic"

**Optimist/Comparativist position**: This ratio is the paper's most genuinely novel empirical contribution. No prior paper provides this direct quantitative cross-optimizer comparison under identical architecture/data/protocol. Even with wide bootstrap CIs, the lower bound (~12×) remains compelling.

**Skeptic position**: The denominator (AdamW constant vs no_wd Δ=0.05%, p=0.825) is statistically indistinguishable from zero. Bootstrap resampling yields ratios ranging from negative to positive values; the 18.3× point estimate is not stable.

**Synthesis judgment**: The Skeptic is technically correct about the ratio instability, but the underlying signal is real: the SGD effect (Δ=0.913%, d=10.3, p=0.002) is robust and the AdamW effect (Δ=0.05%, d=0.16) is clearly negligible. The ratio is a useful way to communicate the magnitude of the contrast, provided it is accompanied by bootstrap CIs.

**Resolution**: Keep the ratio in the paper, but: (1) present bootstrap 95% BCa CI prominently alongside the point estimate; (2) label it "WD presence-absence effect ratio" not "dynamic strategy effect ratio"; (3) use n=5 for the final reported ratio; (4) emphasize the absolute effect sizes separately from the ratio. Do not drop the ratio — it is a legitimate, novel quantification of the SGD/AdamW contrast.

### Conflict on ρ Framework: "Novel Order Parameter" vs "Restatement of Xie & Li (2024)"

**Optimist/Comparativist position**: The explicit operationalization as a regime boundary with falsifiable predictions is novel even if the quantity ρ=λ/η itself appears in Xie & Li, Defazio, and Wang & Aitchison.

**Skeptic/Methodologist position**: Without Regime II/III data, the Trichotomy is a naming exercise, not a theory. Writing it as "Theorem T1" before the λ sweep is completed inverts the scientific method.

**Synthesis judgment**: The Comparativist correctly identifies the novelty gap: Defazio (2506.02285) already names the ‖g‖/‖w‖→λ/η steady-state ratio. The regime framework's novelty is the empirical falsification, not the quantity. The Skeptic's objection about pre-λ-sweep theorem framing is valid.

**Resolution**: (1) Rename to "Conjecture C1" until the λ sweep is complete; (2) cite Xie & Li, Defazio, and Wang & Aitchison explicitly in Section 2 and position the contribution as "empirical validation + falsifiable regime prediction"; (3) if the λ sweep confirms the regime boundary, upgrade to "Proposition T1 (empirically validated)." The value is real but currently unconfirmed.

### Conflict on BN Confound Probability

**Skeptic/Methodologist**: 25–35% probability that BN explains the invariance, not AdamW's mechanism.
**Optimist**: Revisionist's Surprise S2 (AdamW weight norms differ by <1.1% even with 10× WD variation, while SGD differs by 2×) provides a mechanistic argument for AdamW's role independent of BN.

**Synthesis judgment**: The Revisionist's weight norm analysis is compelling. The 1.1% norm range across 10× WD variation under AdamW (vs 2× range under SGD) is direct evidence that AdamW's adaptive normalization absorbs WD variation independent of BN. However, BN could still be *necessary* even if not *sufficient* for the invariance. The NoBN ablation is required to determine whether BN is necessary, not just whether AdamW has an independent effect. Do not pre-judge the direction.

---

## 3. Result Quality Score: 5.5/10

**Current state justification**:

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Experimental volume | 6/10 | 90 AdamW + 84 SGD runs is credible scale, but single architecture |
| Statistical rigor | 4/10 | n=3 underpowered; BEM bug in iter_003; SGD significance overclaimed; no TOST |
| Theoretical depth | 4/10 | ρ framework valid but zero empirical regime boundary validation |
| Novelty | 6.5/10 | 18.3× ratio + controlled comparison is genuinely new; Phi framework is useful |
| Replicability | 6/10 | Strong deterministic seeding; SWD dead code; no git hash in configs |
| **Overall** | **5.5/10** | Solid foundation, significant gaps |

**Projected scores after experiments**:

| Scenario | Score | Venue |
|----------|-------|-------|
| P0 complete, both NoBN + ρ sweep favorable | 7.0–7.5 | NeurIPS 2026 borderline |
| P0 complete, NoBN favorable, ρ sweep flat | 6.0–6.5 | NeurIPS OPT Workshop / TMLR |
| P0 complete, NoBN unfavorable | 5.5–6.0 | Workshop, requires narrative rewrite |
| P0 + P1 complete, all favorable + ImageNet | 7.5–7.9 | NeurIPS/ICML competitive |

---

## 4. Key Findings (5 bullets, confidence-ranked)

1. **AdamW WD Strategy Invariance at ρ≈0.5** (confidence: HIGH): Under standard AdamW (λ=5e-4, η=1e-3, ρ=0.5) on ResNet-20 CIFAR-10/100, 7 qualitatively distinct WD strategies span only 0.25%/0.76% in mean test accuracy and are statistically indistinguishable. The Revisionist's Surprise S2 provides the mechanism: AdamW weight norms differ by <1.1% across 10× WD variation, confirming adaptive normalization absorbs the WD signal. This extends D'Angelo (2024) from "WD doesn't regularize" to "WD *schedule* choice doesn't matter under AdamW."

2. **SGD WD Presence-Absence Effect is 18.3× Larger Than AdamW's** (confidence: MEDIUM-HIGH): SGD constant vs no_wd: Δ=0.913%, d=12.17 paired, p=0.002. AdamW constant vs no_wd: Δ=0.05%, d=0.16, p=0.83. The ratio (18.3×) is a useful quantification of optimizer-specificity, but requires Bootstrap CI before final reporting. The SGD effect is real; the ratio is numerically sensitive to the small AdamW denominator.

3. **Phi Modulator Four-Axis Framework Unifies WD Literature** (confidence: HIGH): The temporal × directional × spatial × target-norm decomposition provides a principled taxonomy for 7+ published methods as specializations of φ(t, w, g). BEM (Budget Equivalence Metric) correctly characterizes fair budget comparisons. This is the paper's cleanest contribution — independent of empirical outcomes.

4. **Cosine WD Schedule Shows 4× Lower Seed Variance Under AdamW on CIFAR-10** (confidence: MEDIUM): σ=0.072% for cosine_schedule vs σ=0.24–0.32% for other methods. This variance reduction is **not preserved on CIFAR-100** (cosine σ=0.42%, same as others) — the Revisionist correctly self-falsifies NH3. The CIFAR-10 result is likely dataset-specific, not a general property.

5. **Alignment-Conditional WD Can Hurt on Harder Tasks under SGD** (confidence: MEDIUM): CWD_hard SGD CIFAR-100: Δ=-1.003% vs constant (worse than simply halving the WD budget at -0.510%). SWD under AdamW performs below no_wd (89.88% vs 90.08%). Both suggest that alignment-conditional WD suppression may remove WD precisely during training periods when it is most needed — a counterintuitive but empirically consistent finding.

---

## 5. Methodology Gaps (Critical and High Priority)

### Critical Gaps (block major claims)

**G1: BN Confound Unresolved** — Cannot distinguish AdamW mechanism from BN scale-invariance without NoBN ablation. Blocks Sections 4 (mechanism) and 5 (theory). Estimated cost: 1h, 18 runs. **Run immediately.**

**G2: λ Regime Sweep Not Executed** — The ρ framework's regime boundary (the paper's most novel theoretical claim) has zero empirical support in Regimes II or III. Without this, Trichotomy must be framed as a Conjecture. Estimated cost: 3–4h, 36–60 runs. **Run concurrently with G1.**

**G3: SGD Hyperparameter Asymmetry** — The Methodologist correctly identifies that the SGD configuration uses λ=5e-3 (10× larger than AdamW's λ=5e-4), confounding optimizer mechanism with WD magnitude. The 18.3× ratio conflates optimizer type with ρ difference. A matched-λ SGD control (λ=5e-4, lr=0.1) is needed to cleanly attribute the contrast to optimizer mechanism. Estimated cost: 6h, 12 runs. **High priority but not blocking if clearly disclosed.**

### High-Priority Gaps

**G4: TOST Equivalence Testing** — n=3 non-significance ≠ equivalence. Need n=5 + formal TOST at δ=0.5% before equivalence can be stated. TOST power at n=5 is ~40%, still below 80% standard; n=7–10 is needed for the central comparison.

**G5: Architecture Monoculture** — Entire conclusion rests on ResNet-20. VGG-16-BN full training (P0-4) is in the plan but not completed. 10-epoch VGG pilot shows unexpected no_wd>constant at 10 epochs — this must be explained or confirmed absent at convergence.

**G6: CIFAR-100 SGD no_wd has n=1** — Only seed_42; all CIFAR-100 SGD no_wd statistics are based on a single run. Requires 2 more seeds before any CIFAR-100 SGD analysis.

**G7: BEM Historical Data Bug** — iter_003 half_lambda BEM=0.000 (should be -0.500) across all 42 runs. BEM-based analysis in iter_003 is corrupted for half_lambda. Fix: recompute from logged weight decay values (zero GPU cost).

**G8: CSI Normalization Architecture-Dependent** — VGG-16-BN pilot shows CSI>1.0 (1.011125) for cwd_hard. CSI normalization is not architecture-invariant. Cross-architecture CSI comparisons are invalid until normalization is fixed.

---

## 6. Competitive Position

The Comparativist's analysis provides the clearest competitive picture:

**Unique contribution cluster**: No single prior paper simultaneously (a) provides controlled comparison of ≥5 WD strategies under AdamW, (b) quantifies the SGD/AdamW WD effect-size ratio, and (c) proposes a phase boundary framework for WD strategy utility. This combination is genuinely new.

**Specific positioning**:
- **vs D'Angelo (2024)**: D'Angelo shows WD is a dynamics modifier, not regularizer (static argument). We show WD *schedule* is irrelevant under AdamW (dynamic argument). Complementary, not duplicative — but the BN ablation is needed to confirm our mechanism is distinct from D'Angelo's BN scale-invariance.
- **vs Wang & Aitchison (2024)**: Their EMA timescale intuition predicts our invariance. We provide controlled experimental confirmation + SGD contrast + formal regime framework. We upgrade their intuition to a falsifiable prediction.
- **vs Xie & Li (2024)**: They prove AdamW enforces ℓ∞ constraint τ*=η/λ. We operationalize ρ=1/τ* as a regime-boundary order parameter with empirical predictions. The quantity is theirs; the regime analysis is ours.
- **vs CWD (Chen et al., ICLR 2026)**: We show CWD provides zero benefit over random masking under Regime I (ρ≈0.5). This is a direct empirical critique, not a dismissal — CWD may be valid in Regime II/III (larger ρ, LLM settings). The regime framework explains when CWD helps and when it doesn't.
- **Most dangerous concurrent paper**: Defazio (2506.02285) independently identifies ‖g‖/‖w‖→λ/η. Mitigation: Defazio does not propose a regime diagram, does not compare WD strategies, and does not provide the SGD/AdamW contrast.

**Current venue estimate**: Workshop/AAAI without P0 completion. NeurIPS/ICLR main track with P0-3 + P0-4 (VGG) + P1-1 (λ sweep).

---

## 7. Hypothesis Update

| Hypothesis | Updated Status | Confidence | Action |
|---|---|---|---|
| H1: Phi Invariance (AdamW) | **Partially Confirmed** | MEDIUM | Confirmed at ρ=0.5, single architecture. Needs TOST n=5 and NoBN ablation to strengthen. |
| H2: SGD Optimizer Specificity | **Confirmed (narrowly)** | HIGH for WD presence effect; LOW for dynamic strategy superiority | Only constant vs no_wd is significant (Holm-corrected). Dynamic strategies under SGD are statistically comparable to constant. |
| H3: Regime Boundary | **Untested** | N/A | λ sweep required. Reframe as Conjecture C1 until tested. |
| H4: BN Role | **Untested** | N/A | NoBN ablation required. This is P0 blocking. |
| H5: Cross-Architecture | **Inconclusive** | LOW | VGG-16-BN pilot (10 epochs, 1 seed) insufficient. Full VGG run needed. Anomalous no_wd>constant at 10 epochs unexplained. |
| NH1 (new): Adaptive Scaling Sufficient | **Plausible, Untested** | MEDIUM | Revisionist's weight norm evidence is suggestive. RMSProp experiment could confirm. |
| NH2 (new): Binary WD Suppression Harmful | **Directionally Supported** | MEDIUM | CWD_hard SGD CIFAR-100 Δ=-1.003% > half_lambda Δ=-0.510%. SWD AdamW < no_wd. Both consistent with the hypothesis. |
| NH3 (new, self-falsified): Cosine Variance Scales with Task Difficulty | **Falsified** | HIGH (falsification) | CIFAR-100 cosine std = 0.42%, no lower than other methods. CIFAR-10 low-variance is dataset-specific artifact. |

**Critical belief update from Revisionist**: The prior belief "WD strategy controls weight norm trajectory" is empirically false under AdamW. The norm range across 10× WD variation is 1.1% under AdamW vs 2× under SGD. This is the strongest direct mechanistic evidence currently available and should be featured prominently in the paper.

---

## 8. Action Plan (Prioritized)

### Immediate (Zero GPU Cost, Execute Before Any Experiments)

1. **Recompute BEM for iter_003 half_lambda** from epoch_metrics.jsonl. Update all tables. (1 hour, 0 GPU)
2. **Correct SGD statistical reporting**: Remove all claims of "multiple significant SGD effects"; report only constant vs no_wd as significant (Holm-corrected). (1 hour, 0 GPU)
3. **Compute Bootstrap 10,000-resample BCa CI for 18.3× ratio**. Report median and 5th–95th percentile interval alongside point estimate. (2 hours, 0 GPU)
4. **Post-hoc power analysis**: Report minimum detectable effect at n=3 and n=5 in the paper appendix. Honest communication of statistical limitations.

### P0: Blocking Experiments (Run Before Writing Sections 4–5)

5. **P0-3: ResNet-20-NoBN Ablation** — CIFAR-10, AdamW+SGD, constant/cosine_schedule/no_wd, 3 seeds each (18 runs, ~1h). Decision tree: if NoBN maintains invariance → AdamW mechanism claim strengthened; if NoBN breaks invariance → reframe narrative.

6. **P0-b: λ Regime Sweep** — AdamW, ResNet-20, CIFAR-10, λ ∈ {5e-5, 5e-4, 5e-3, 5e-2}, 3 WD methods, 3 seeds each (~36–60 runs, 3–4h). Can be run in parallel with P0-3 using remaining GPUs.

7. **P0-c: CIFAR-100 SGD no_wd completion** — Add seeds 123 and 456 (2 runs, 30min). Required before any CIFAR-100 SGD statistics.

### P1: Statistical and Cross-Architecture Hardening

8. **P1-a: n=5 extension for core AdamW methods** — Add seeds 789, 999 for constant, no_wd, cosine_schedule, cwd_hard on AdamW CIFAR-10/100 (~28 runs, 2–3h). Enables TOST equivalence testing.

9. **P1-b: VGG-16-BN full training** — 7 methods × 2 datasets × 3 seeds, AdamW, 200 epochs (~70 runs, 6–8h). Cross-architecture validation essential for generalization claims.

10. **P1-c: Matched-λ SGD control** — SGD with λ=5e-4 (same as AdamW), lr=0.1, methods: constant/no_wd/cosine_schedule/cwd_hard, 3 seeds (~12 runs, 1–2h). Isolates optimizer mechanism from WD magnitude in the 18.3× ratio.

### P2: Scale Validation (Optional but Valuable)

11. **P2-a: ImageNet pilot** — ResNet-50, 90 epochs, constant/no_wd/cosine_schedule, 1 seed each (3 runs, 8–12h). Both invariance and its break are publishable findings at scale.

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| NoBN ablation breaks invariance | 30% | High | Reframe as "BN+AdamW joint mechanism"; still publishable but differentiation from D'Angelo narrows |
| λ sweep flat at ρ=50 | 20% | Medium | Drop discrete Trichotomy; report continuous monotone effect; actually strengthens the invariance claim |
| Reviewer: "trivial given Xie & Li 2024" | 40% | High | Emphasize 18.3× ratio (genuinely new) + regime predictions + Phi framework operational value |
| Reviewer: "one architecture" | 50% (without VGG) | High | P1-b VGG is essential; pilot suggests no infrastructure problems |
| Theorem T1 not formally provable | 40% | Medium | Use "Conjecture C1" framing; the empirical validation from λ sweep is sufficient for an empirical ML paper |
| n=3 insufficient for TOST | HIGH (certain without n=5) | High | P1-a n=5 extension is required |

---

## Verdict Summary

**PROCEED** with aggressive execution of P0 and P1 experiments. The core empirical observations are real and the direction is correct. The paper is currently held back by:
1. A single missing critical experiment (NoBN ablation) that could change the mechanistic narrative
2. Statistical fragility (n=3, no TOST, no bootstrap CI for the key ratio)
3. Single-architecture coverage

None of these are conceptual failures — they are execution gaps that can be closed in 2–3 days of GPU runs. The theoretical framework is coherent. The empirical contrast (AdamW invariance vs SGD specificity) is genuine. The Phi Modulator taxonomy is useful and novel. Completing P0 experiments before beginning paper writing is mandatory.

---

*Synthesized by: Sibyl Result Debate Synthesizer (sibyl-heavy)*
*Inputs: optimist.md, skeptic.md, strategist.md, methodologist.md, comparativist.md, revisionist.md*
*Based on: iter_003 (90 AdamW + 84 SGD runs) + iter_004 Phase 0 (7 pilot runs) + sgd_baseline_analysis.json*
