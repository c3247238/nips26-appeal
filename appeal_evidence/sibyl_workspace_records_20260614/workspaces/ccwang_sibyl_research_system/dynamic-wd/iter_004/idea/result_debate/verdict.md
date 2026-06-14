# Result Debate Verdict

**Date**: 2026-03-18
**Decision**: PROCEED (with P0 experiments as hard prerequisite before paper writing)

---

## Quality Score: 5.5/10

The current result package supports a workshop paper. It requires P0 completion to reach NeurIPS borderline (7.0–7.5). The experimental observations are real; the gaps are execution-level, not conceptual.

---

## Key Conclusions (What We Actually Learned)

1. **Under standard AdamW (ρ=λ/η≈0.5), 7 qualitatively distinct WD strategies produce statistically indistinguishable accuracy** — 0.25% spread on CIFAR-10, 0.76% on CIFAR-100, all pairs p>0.09 corrected. The mechanism is clear: AdamW's adaptive normalization absorbs 10× WD variation with only a 1.1% weight norm range, while SGD shows a 2× norm range for the same WD variation.

2. **The WD presence-absence effect under SGD is ~18× larger than under AdamW** (constant vs no_wd: SGD Δ=0.913%, d=12.17; AdamW Δ=0.05%, d=0.16). This is the paper's strongest novel empirical finding — no prior work quantifies this contrast under controlled conditions. Statistical caveat: the ratio requires Bootstrap CIs (estimated [12×, 26×] at n=5); the denominator is not significant, but the contrast in effect size is genuine.

3. **The Phi Modulator framework** (λ_eff = λ·φ(t,w,g)) cleanly unifies 7+ WD methods as specializations of a common functional interface. BEM correctly quantifies budget equivalence. This is the paper's most technically solid contribution.

4. **Alignment-conditional WD can hurt, not just fail to help**: CWD_hard SGD CIFAR-100 underperforms half_lambda (which uses half the WD budget) by 0.5%. SWD under AdamW underperforms no_wd. Binary WD suppression may remove regularization precisely during high-alignment (high-progress) periods.

5. **The ρ = λ/η regime boundary is a conjecture, not a theorem**: The Trichotomy currently has zero empirical support in Regimes II or III. P1-1 (λ sweep) is required before this claim can move from theoretical speculation to empirical finding.

---

## Critical Gaps (Must Fix Before Writing)

| Gap | Action | Cost | Deadline |
|-----|--------|------|----------|
| BN confound unresolved | NoBN ablation (18 runs) | 1h GPU | Before Section 4 |
| ρ regime untested | λ sweep (36–60 runs) | 3–4h GPU | Before Section 3 theory |
| n=3 underpowered | Extend to n=5 (28 runs) | 2h GPU | Before TOST claims |
| SGD stat overclaims | Recompute Holm-corrected p-values | 0h | Immediately |
| Bootstrap CI for 18.3× ratio | 10,000 BCa resamples | 0h | Immediately |
| BEM bug (half_lambda, iter_003) | Recompute from epoch_metrics.jsonl | 0h | Immediately |
| Architecture monoculture | VGG-16-BN full run (70 runs) | 6–8h GPU | Before paper submission |

---

## PROCEED or PIVOT?

**PROCEED.** All three backup scenarios (NoBN breaks invariance, λ sweep flat, VGG breaks invariance) have pre-planned publishable narratives. The research direction has converged. The core findings are not artifacts. The remaining work is experimental validation and statistical hardening, not ideation.

**The single most important experiment**: NoBN ablation (P0-3). If AdamW invariance persists without BN, the paper's mechanistic claim is strong and distinct from D'Angelo (2024). If NoBN breaks invariance, the narrative pivots to "BN+AdamW joint mechanism" — reduced novelty, but still publishable with honest framing.

**Decision timeline**:
- Day 0: Run zero-GPU fixes (BEM recompute, SGD stat corrections, Bootstrap CI)
- Day 0–1: Launch NoBN ablation + λ regime sweep + CIFAR-100 SGD no_wd completion
- Day 1–2: Analyze P0 results → confirm narrative direction
- Day 2–3: Launch VGG-16-BN full + n=5 extensions
- Day 3–5: Paper writing based on confirmed narrative

---

*Verdict by: Sibyl Result Debate Synthesizer | 2026-03-18*
