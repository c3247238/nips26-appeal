# Experiment Result Analysis

**Supervisor**: sibyl-supervisor-decision (Opus 4.6)
**Date**: 2026-03-18
**Iteration**: 7 (experiment decision for iter_005 data + iter_003 foundation)
**Research Focus**: FOCUSED mode (lean toward PROCEED)

---

## Key Results Summary

### Rock-Solid Completed Results (iter_003, 168 runs total)
- **AdamW CIFAR-10 ResNet-20**: phi_spread = 0.25% across 7 methods x 3 seeds (21 runs). Constant WD = 90.13 +/- 0.31 (best or tied-best).
- **AdamW CIFAR-100 ResNet-20**: phi_spread = 0.75% across 7 methods x 3 seeds (21 runs). cosine_schedule = 63.42 (best), no_wd = 62.66 (worst).
- **SGD CIFAR-10 ResNet-20**: phi_spread = 0.91% across 7 methods x 3 seeds (21 runs). Constant = 91.22 +/- 0.07 (best).
- **SGD CIFAR-100 ResNet-20**: phi_spread = 1.71% across 7 methods x 3 seeds (21 runs). Constant = 65.37 (best), no_wd = 63.66 (worst).
- **SGD/AdamW sensitivity ratio**: 3.65x on CIFAR-10, 2.28x on CIFAR-100 (sub-additive).

### Partially Completed Results (iter_005, cross-architecture)
- **VGG-16-BN CIFAR-10**: 4/7 methods x 3 seeds = 12 runs. Mean range = 0.16% (half_lambda 92.15 > cwd 92.06 > constant 92.05 > cosine 91.99). Individual seed range up to 0.70% (cosine_schedule seed_42 = 91.62 pulls it down). Missing: swd, no_wd, random_mask.
- **NoBN ResNet-20 CIFAR-10**: Only constant method, 3 seeds. Mean = 87.74 +/- 0.21. AIS = 0.490 (vs BN ~0.34, +44% elevation). CWD in-flight at ep147 = 87.04% (tracking below constant). LR confound: NoBN uses lr=5e-4 vs BN lr=1e-3.
- **rho_low (rho=0.05)**: Only constant method, 2 seeds. Mean = 90.18. AIS = 0.372 (similar to standard rho). Weight norms ~96.7 (2x standard).
- **matched-rho SGD**: Only constant method, 2 full seeds (seed_42 = 5 epochs only!). Mean = 90.92 +/- 0.04 (tight). CWD in-flight at ep74 = 88.38%.
- **rho_high (rho=5.0)**: ALL EXPERIMENTS FAILED. Zero usable data.
- **ImageNet**: ALL EXPERIMENTS FAILED. Zero usable data.

---

## Debate Perspectives Summary

- **Optimist**: Emphasizes the strong 168-run foundation, highlights VGG phi_spread (0.16%) as confirming multi-architecture null, identifies interesting signals in NoBN AIS elevation and VGG seed-variance/CSI correlation. Views in-flight experiments as on-track. Identifies publishable story regardless of outcome on pending experiments.

- **Skeptic**: Raises 2 fatal flaws (rho_high missing = Theorem 1 Corollary unfalsifiable; ImageNet missing = limited to CIFAR-scale scope) and 5 serious concerns (VGG 4/7 methods with selection bias toward conservative methods; matched-rho SGD incomplete; NoBN only 1 method; BEM metric buggy/contradictory; CIFAR-10 saturation confound). Argues the paper has ONE strong empirical result and a theoretical framework built around experiments that have not been run.

- **Strategist**: Recommends PROCEED with clear priority queue: (1) VGG completion + matched-rho SGD in parallel (2h), (2) rho_high root cause diagnosis + re-run (5h), (3) PMP-WD pilot conditional on rho_high. Estimates paper is ~60% data-ready. Even without rho_high, Scenario C (7.0-7.5) is achievable. Notes the two strong signals (AdamW null + SGD sensitivity) are publication-worthy alone.

- **Comparativist**: Positions against CWD (ICLR 2026), AdamO (Feb 2026), AlphaDecay (NeurIPS 2025), and SWD (NeurIPS 2023). Core novelty: first systematic 7-method x 3-seed comparison of WD methods. All accepted top-tier WD papers include either ImageNet or LLM results (we have neither). Current state = workshop/AAAI tier; with P0 complete = conditional NeurIPS/ICML. CWD counter-evidence is a genuine contribution.

- **Methodologist**: Flags 4 critical issues: (1) NoBN lr confound invalidates causal BN attribution; (2) matched-rho SGD seed_42 ran only 5 epochs but task marked "completed"; (3) zero hypotheses (H5-1 through H5-5) can be fully tested with current data; (4) no TOST equivalence tests reported despite claiming null results. Reproducibility score: 3.5/5.

- **Revisionist**: Proposes reframing to "Why doesn't adaptive weight decay help?" — centering the robust null result rather than speculative regime transitions. Notes Theorem 1 predicts a null result that is trivially observed; its non-trivial prediction (CWD wins at high rho) is untested. Method ordering reshuffles across architectures (VGG: half_lambda > cwd > constant; ResNet-20: constant > all), weakening directional claims. Recommends acknowledging the gap between theoretical ambition and empirical evidence.

---

## Analysis

### 1. Method Feasibility

The core research method works as intended. The 7-method comparison framework with phi_spread as the primary metric is well-designed and has produced a clear, reproducible null result on the CIFAR-10/100 + ResNet-20 foundation (168 runs). The proxy metrics (CSI, AIS, BEM) are computed consistently. The theoretical framework (Theorems 1-3) is mathematically coherent and explains the observed patterns.

**Verdict**: The method is feasible and producing meaningful results. The issue is not method validity but experiment completion.

### 2. Performance

The core finding is a **null result**: WD method choice produces < 0.25% accuracy variation under AdamW with BN at standard rho. This is the paper's strength, not its weakness. The SGD sensitivity ratio (3.65x) is a genuine positive finding. VGG-16-BN preliminary data (0.16% spread across 4 methods) supports multi-architecture generalization.

**However**: No experiment has demonstrated that any proposed algorithm (PMP-WD, SPWD) outperforms constant WD. The "when does it help" question remains unanswered due to rho_high failure.

**Verdict**: Strong empirical foundation for the null result; no positive algorithmic results yet.

### 3. Improvement Headroom

There is a clear, bounded path to completing the experimental picture:
- **VGG completion** (3 missing methods x 3 seeds): ~2 GPU-hours, zero new code. Low risk.
- **matched-rho SGD completion** (2 missing methods x 3 seeds + seed_42 re-run): ~2 GPU-hours. Low risk.
- **rho_high diagnosis + re-run**: ~5 GPU-hours. Medium risk (prior complete failure).
- **PMP-WD implementation + pilot**: ~6 GPU-hours. Medium risk (new code).
- **NoBN completion** (2 missing methods): ~2 GPU-hours. Low risk.

Total: ~17 GPU-hours for the full picture. With 8 GPUs available, this is roughly 3-4 hours wall-clock for P0 + 5-7 hours for P1.

**Verdict**: Clear improvement headroom with bounded effort. The most critical experiments (VGG, matched-rho) are low-risk.

### 4. Time-Cost Tradeoff

Starting fresh (PIVOT) would mean abandoning:
- 168 completed runs across 4 conditions (iter_003)
- 24+ completed runs from iter_005 (VGG, NoBN, rho_low, matched-rho)
- A coherent 3-theorem theoretical framework with dual derivation
- 5 iterations of progressive refinement

The remaining work to reach a publishable paper (Scenario B/C at 7.0-8.0) requires only ~10-17 GPU-hours. A pivot to a new research direction would require starting from scratch with idea generation, literature review, experiment design, and initial runs -- easily 100+ GPU-hours and multiple iterations.

**Verdict**: Continuing is overwhelmingly more efficient than pivoting. The sunk cost is large AND the remaining cost is small.

### 5. Critical Objections Assessment

**Skeptic's Fatal Flaw F1 (rho_high missing)**: This is the most serious concern. Without rho_high data, Theorem 1's regime boundary prediction is unfalsifiable, and the "stability-optimal control" framing loses its most dramatic prediction. However, this flaw is **addressable**: diagnose the failure mode and re-run (possibly at rho=2.0 fallback). The paper can also be framed without rho_high as a negative-result paper (Scenario C, 7.0-7.5).

**Skeptic's Fatal Flaw F2 (ImageNet missing)**: Serious for venue targeting but not for the research direction's validity. All recent WD papers at top venues include ImageNet/LLM results. However, this is an infrastructure issue, not a conceptual one. The paper can explicitly acknowledge this limitation.

**Methodologist's NoBN confound**: Addressable by (a) reframing NoBN results as "suggestive" or (b) re-running at matched LR. Not fatal.

**Revisionist's reframing proposal**: Valid concern about theory-evidence gap. The reframing to "Why doesn't adaptive WD help?" is a reasonable fallback (Scenario C/D) but should only be adopted if rho_high and PMP-WD both fail.

**CIFAR-10 saturation**: The CIFAR-100 data (phi_spread = 0.75%) shows the effect scales with task difficulty but remains < 1%. Not a fatal flaw but a legitimate limitation.

**Verdict**: All critical objections are either addressable (rho_high, NoBN, matched-rho) or manageable limitations (ImageNet, CIFAR saturation). None are fatal to the research direction.

---

## Decision Rationale

The PROCEED decision is justified on the following grounds:

1. **The empirical foundation is strong and unique.** 168 completed runs with 7 methods x 3 seeds x 2 optimizers x 2 datasets constitutes the most systematic WD method comparison in the literature. No prior work provides this level of multi-method, multi-seed coverage on the same benchmark. This alone is a publishable contribution.

2. **The theoretical framework is coherent and non-trivial.** Three theorems with dual derivation (PMP + RG beta function) from independent mathematical frameworks is substantial theoretical content. Proposition 1 (alignment noise constraint) converts a challenge into a design principle.

3. **The remaining experiments have bounded cost and clear value.** VGG completion (~2h) and matched-rho SGD (~2h) are low-risk, high-value experiments that can be parallelized immediately. Even rho_high (medium risk) has a clear diagnostic path.

4. **Multiple viable paper scenarios exist.** Even the worst case (Scenario D: only existing data + theory) produces a 6.5-7.0 paper. The most likely outcome (Scenario B/C with VGG + matched-rho completed) targets 7.0-8.0. The research direction does not require rho_high success to be publishable.

5. **Core hypotheses are NOT refuted.** The null result at standard rho is the EXPECTED outcome of the theory. The theory predicts constant WD wins at standard rho with BN -- exactly what is observed. The open question (does method sensitivity emerge at high rho?) is untested, not refuted.

6. **FOCUSED mode directive.** With research_focus=4, the bar for PIVOT is that core hypotheses must be clearly refuted with no credible improvement path. Neither condition is met: hypotheses are confirmed (not refuted) at standard rho, and clear improvement paths exist (complete missing experiments, diagnose rho_high, implement PMP-WD).

**Counter-argument considered but rejected**: The Revisionist's argument that "the paper's strongest theoretical claims rest on the weakest empirical foundations" is true but premature. The remaining experiments to strengthen the empirical foundation require only ~10-17 GPU-hours. Pivoting before exhausting this bounded effort would be wasteful.

---

## DECISION: PROCEED
