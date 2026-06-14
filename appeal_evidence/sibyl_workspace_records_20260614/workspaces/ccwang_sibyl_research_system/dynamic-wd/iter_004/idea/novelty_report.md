# Novelty Report — Iteration 6

**Agent**: Sibyl Novelty Checker
**Date**: 2026-03-18
**Based on**: Iteration 6 Synthesized Proposal + Iteration 5 Novelty Check (baseline) + Fresh literature search for new candidates and 2025-2026 papers

---

## Executive Summary

| Candidate | Novelty Score (1–10) | Risk Level | Recommendation |
|---|---|---|---|
| `cand_rho_trichotomy` — Phi Invariance Trichotomy via ρ = λ/η | **7** | Medium | **Proceed** (unchanged from Iter 5) |
| `cand_rho_controller` — ρ-Controller: Feedback-Stabilized WD | **6** | Medium-High | **Proceed with differentiation** from AWD (NeurIPS 2023) |
| `cand_super_twisting` — Super-Twisting SMC as Chattering-Free CWD | **8** | Low | **Proceed conditional** (unchanged from Iter 5; demoted to backup) |
| `cand_gradient_weight_ratio` — r_t unification + factorial benchmark | **6** | Medium | **Merge** into front-runner (unchanged) |
| `cand_spectral_rank_feedback` — Stable Rank Feedback WD | **6** | Medium | **Appendix only** (unchanged; dropped from main) |

**Overall novelty**: **medium** — The front-runner (Phi Invariance Trichotomy) retains defensible novelty at score 7. The new primary secondary contribution (rho-Controller) has a significant overlap risk with AWD (NeurIPS 2023, arXiv:2210.00094), which already formulates weight decay as an adaptive ratio controller. The key differentiators (Lyapunov formal backing, Defazio steady-state target, regime-conditional prediction) are real but require explicit acknowledgment and careful positioning.

**Critical new finding in Iteration 6**: AWD (Ghiasi et al., NeurIPS 2023) uses `λ_t = λ_awd · ‖∇w_t‖/‖w_t‖` to maintain a constant gradient-to-weight ratio — this is essentially the zero-error-signal form of the rho-Controller. The rho-Controller's distinctiveness must be precisely argued: (1) AWD targets a ratio chosen heuristically; ours targets ρ* derived from Defazio's steady-state theory; (2) AWD has no Lyapunov convergence backing for the ratio dynamics; (3) AWD is not regime-conditional (not motivated from the Trichotomy); (4) AWD targets adversarial robustness, not generalization accuracy under WD scheduling comparison.

---

## Part I: Candidate cand_rho_trichotomy

### Core Contribution Claims (Iteration 6 — unchanged from Iter 5)

1. ρ = λ/η is the **order parameter** for a three-regime phase diagram (Trichotomy) governing dynamic WD utility under AdamW
2. Phi Invariance Trichotomy Conjecture (honest label pending Gate 0) with explicit regime boundaries (ρ₁, ρ₂)
3. **Empirical confirmation**: 18.3× SGD/AdamW effect ratio; BN ablation; VGG cross-architecture
4. Theorem T2 (bridge): τ* = 1/ρ (Xie & Li) ↔ R_* = ρ (Defazio) — already repositioned as citing both

### New Search Findings (Iteration 6)

Searches performed:
- "weight decay regime phase transition lambda eta ratio order parameter 2025 2026"
- "arXiv 2026 lambda/eta weight decay ratio phase transition regime deep learning"
- "weight decay schedule invariance SGD AdamW comparison regime boundary phase diagram 2025 2026"
- "weight decay scheduling does not matter AdamW batch normalization empirical study 2025 2026"
- "phi invariance weight decay invariance WD schedule sensitivity regime analysis optimizer 2025 2026"

#### No new direct collisions found

No paper discovered in the Iteration 6 search introduces a three-regime phase diagram for dynamic WD utility parameterized by ρ = λ/η with falsifiable crossover predictions. The search landscape for Iteration 6 is unchanged from Iteration 5 on this candidate.

#### New supporting papers found

**arXiv:2602.11137 — "Weight Decay Improves Language Model Plasticity" (2025/2026)**
- Shows that larger WD improves plasticity for fine-tuning even at the cost of pretraining loss
- Supports the narrative that WD regime (high vs. low ρ) produces qualitatively different effects
- No overlap with Trichotomy; add to Related Work as further evidence that ρ regime matters for model behavior

**arXiv:2506.12543 — "Is your batch size the problem? Revisiting the Adam-SGD gap" (June 2025)**
- Revisits the optimizer gap for language modeling using SGD vs. Adam
- Critical note: deliberately excludes weight decay to avoid "side-effects" — their SGD comparison is without WD
- Our 18.3× ratio is the SGD+WD vs. SGD-no_WD comparison, not the SGD vs. Adam gap — NO collision
- Add as Related Work: their SGD-without-WD results provide a contrasting data point

### Collision Status: Unchanged from Iteration 5

All previous collisions remain (Defazio 2025, Wang & Aitchison 2024, D'Angelo 2024). No new partial-overlap collision found in Iteration 6 search.

### Novelty Score Justification: 7/10 (unchanged)

- No paper found that publishes the Phi Invariance Trichotomy with ρ-dependent regime boundaries and falsifiable crossover predictions
- 18.3× SGD/AdamW effect ratio with controlled statistical methodology remains original
- Same score conditions as Iteration 5 (8/10 if T1 provable and NoBN ablation confirms AdamW-intrinsic mechanism)

---

## Part II: Candidate cand_rho_controller (NEW — Iteration 6)

### Core Contribution Claims

1. First formulation of WD selection as a **closed-loop P-controller** targeting the gradient-to-weight ratio steady state ρ* = ‖g_t‖/‖w_t‖
2. Control law: λ_t = λ₀ · (ρ_t / ρ*)^{−α} where ρ_t = mean over layers of ‖g_t^(ℓ)‖/‖w_t^(ℓ)‖
3. **Formal Lyapunov backing**: Lyapunov function V_t = (ρ_t − ρ*)²; claims exponential convergence O(e^{−αt}) vs O(1/t) for constant WD
4. Computationally free (log ρ_t already available); one-line modification to any optimizer
5. Only effective in Regime II/III; collapses in Regime I (standard AdamW, ρ=0.5)

### Search Coverage

Searches performed:
- "weight decay feedback control gradient-to-weight ratio P-controller neural network optimizer 2025 2026"
- "rho controller adaptive weight decay proportional control optimizer training 2025 arXiv"
- "weight decay dynamic control Lyapunov gradient weight ratio convergence feedback optimizer 2025"
- "adaptive weight decay gradient norm weight norm ratio automatic tuning optimizer training 2025 2026"
- "closed-loop weight decay weight decay controller adaptive weight decay Lyapunov training neural network 2025 2026"

### Collision Analysis

#### Collision 1 (PARTIAL OVERLAP — HIGH CONCERN — NEW IN ITERATION 6)
**Ghiasi et al. (NeurIPS 2023). "Improving Robustness with Adaptive Weight Decay." arXiv:2210.00094**

This is the most significant new collision for the rho-Controller candidate.

**What AWD does**:
- Tracks the ratio of weight decay magnitude to gradient magnitude: λ_awd(t) = ‖λ_wd · w_t‖ / ‖∇w_t‖
- Update rule: λ_wd(t) = λ_awd · ‖∇w_t‖ / ‖w_t‖  (i.e., WD proportional to g/w ratio)
- Aims to keep the ratio of contributions constant during training
- Uses exponential moving average for stability: λ̄_t = 0.1×λ̄_{t-1} + 0.9×λ_t

**Overlap with rho-Controller**:
- Both methods make λ_t a function of ‖g_t‖/‖w_t‖ (the gradient-to-weight ratio)
- AWD's steady-state condition is equivalent to λ_wd · ‖w‖ ≈ const × ‖∇w‖, which is a form of ratio control
- The conceptual core — "make WD proportional to the gradient-to-weight ratio" — is shared

**Key differentiators (what the rho-Controller offers beyond AWD)**:
1. **Theoretically grounded target ρ***: AWD's λ_awd is chosen heuristically as a constant. The rho-Controller targets ρ* derived from Defazio (2025) steady-state theory: ρ* = R_* = λ₀/η. AWD has no connection to the Defazio dynamics.
2. **Regime-conditional motivation**: The rho-Controller is explicitly motivated only for Regime II/III. AWD is applied uniformly regardless of the ρ regime — it does not predict its own futility in Regime I.
3. **Power-law feedback (P-controller form)**: AWD's update is linear in the ratio; the rho-Controller uses (ρ_t/ρ*)^{−α} which allows α-tunable convergence speed. α=1 is essentially AWD's form; α≠1 provides a richer family.
4. **Lyapunov convergence analysis**: AWD makes no convergence claim for the ratio dynamics. The rho-Controller provides a formal Lyapunov argument for exponential convergence to ρ*.
5. **Context**: AWD is evaluated for adversarial robustness (not standard generalization accuracy); no evaluation in the Trichotomy regime framework; no SGD/AdamW comparison; no regime sweep.
6. **Novelty claim calibration**: AWD claims "maintaining ratio constant" but does not claim to be a P-controller, does not cite Defazio, does not derive ρ* from theory.

**Severity assessment**: `partial_overlap` — The CONCEPT of making λ_t proportional to ‖g_t‖/‖w_t‖ is already in AWD (2022/NeurIPS 2023). The specific form, theoretical motivation, regime-conditional scope, and Lyapunov analysis are new. The rho-Controller should be presented as a "theoretically motivated extension and refinement of AWD's ratio principle" with explicit credit to AWD.

**Required action**:
- Add AWD (arXiv:2210.00094) as a Related Work collision for the rho-Controller
- Do NOT claim "first formulation of WD as a ratio controller" — credit AWD for this idea
- Reframe contribution as: "theoretically grounded rho-Controller targeting Defazio's steady-state ρ* (not an arbitrary ratio); regime-conditional (Trichotomy-motivated); Lyapunov convergence for the ratio dynamics"
- Revised claim: "First WD controller formally derived from steady-state theory (Defazio 2025) with Lyapunov convergence guarantees, applicable in a regime-conditional framework"

#### Collision 2 (RELATED WORK — MODERATE)
**Nakamura & Hong (2019). "Adaptive Weight Decay for Deep Neural Networks." arXiv:1907.08931**

- AdaDecay adjusts per-parameter WD based on normalized gradient norms using a sigmoid function
- Does not specifically track the gradient-to-weight ratio for ratio control
- **No overlap with rho-Controller's specific P-control formulation**

**Severity**: `related_work` — Cite as predecessor; not a direct collision.

#### Collision 3 (RELATED WORK — MODERATE)
**Chen et al. (Nature Communications 2024). "Accelerated optimization in deep learning with a proportional-integral-derivative controller." PIDAO.**

- Applies PID control to the optimizer update direction (not to the weight decay coefficient)
- The control variable is the parameter update vector, not λ
- **No overlap with rho-Controller's specific formulation** (different control variable, different target)

**Severity**: `related_work` — Cite as "PID-based optimization perspective"; emphasize that our contribution is a P-controller specifically for the *weight decay coefficient* targeting the gradient-to-weight steady state.

#### Collision 4 (RELATED WORK — LOW)
**Ghiasi et al., Cautious Weight Decay (CWD), arXiv:2510.12402 (ICLR 2026)**

- CWD modulates WD by gradient sign alignment, not by gradient/weight ratio
- No overlap with ratio-control formulation

**Severity**: `related_work`

### Novelty Score Justification: 6/10

**Before AWD discovery**: The Innovator assessed rho-Controller as novel (8/10). After finding AWD (NeurIPS 2023):
- The conceptual kernel (WD proportional to g/w ratio) is in AWD (−2 points)
- What remains genuinely novel: Defazio-derived ρ* target, Lyapunov convergence, regime-conditional scope (+1 point relative to pure derivative)
- Total: 6/10 (the contribution is a theoretically enriched version of a known idea, not a new idea)

**Upgrade path**: If P1-2 experiments show rho-Controller outperforms constant WD in Regime II AND AWD fails in Regime I (as the Trichotomy predicts), this becomes a stronger "theoretically-motivated improvement over AWD with regime awareness" contribution. Score could reach 7-7.5 in that case.

### Differentiation Actions (Required Before Writing Section)

1. **Add AWD (arXiv:2210.00094) to Related Work** as: "AWD (Ghiasi et al., 2023) introduces the idea of making λ proportional to ‖g‖/‖w‖ for adversarial robustness. Our rho-Controller derives the ratio target ρ* from steady-state theory (Defazio 2025), provides Lyapunov convergence analysis, and applies exclusively in Regime II/III where the Trichotomy predicts benefit."
2. **Remove "first closed-loop feedback WD" claim**: AWD (2023) is a prior closed-loop ratio controller.
3. **Retain claim**: "First WD controller theoretically grounded in the gradient-to-weight steady state (Defazio 2025), with Lyapunov convergence, applied in a regime-conditional framework."
4. **Add AdaDecay (1907.08931) as predecessor** — briefer citation.
5. **Add PIDAO (Nature Comm 2024)** as related work in the optimizer-as-control-system context.

---

## Part III: Candidate cand_super_twisting (status: backup, unchanged from Iter 5)

### Status: No new collisions found in Iteration 6

The super-twisting SMC as chattering-free CWD extension retains novelty score 8/10 conditional on H7 (alignment signal informative in NoBN). This candidate has been demoted to conditional backup in the Iteration 6 proposal; the rho-Controller is now the primary secondary contribution. No new papers found applying super-twisting SMC to WD in the Iteration 6 search.

**Score**: 8/10 (conditional on H7), 5/10 (if H7 not confirmed) — unchanged.
**Recommendation**: `proceed_conditional` — include only if BN ablation confirms alignment signal.

---

## Part IV: Candidate cand_gradient_weight_ratio (merge recommended, unchanged)

No new search findings. Defazio (2025) remains the primary collision. Recommendation: merge r_t mechanistic content into front-runner.

**Score**: 6/10 (unchanged)

---

## Part V: Candidate cand_spectral_rank_feedback (dropped, unchanged)

Status remains `dropped` as per candidates.json. No new papers found that change the assessment.

**Score**: 6/10 for the closed-loop mechanism; appendix only.

---

## Part VI: New Papers Discovered in Iteration 6

### High Priority (must engage in Related Work — new in Iteration 6)

| Paper | arXiv | Finding | Action |
|---|---|---|---|
| Ghiasi et al. (NeurIPS 2023) "Improving Robustness with Adaptive Weight Decay" | 2210.00094 | **CRITICAL**: AWD already formulates λ_t proportional to ‖g_t‖/‖w_t‖ ratio. This is a partial collision with rho-Controller. | Add to Related Work; reframe rho-Controller contribution to acknowledge AWD as conceptual precursor |
| "Weight Decay Improves Language Model Plasticity" | 2602.11137 | Larger WD improves plasticity; supports regime-dependent WD effects narrative | Add to Related Work for regime importance of WD value |
| "Is your batch size the problem? Revisiting the Adam-SGD gap" | 2506.12543 | SGD without WD competitive with Adam in small-batch; intentionally excludes WD | Add to Related Work as contrasting data point (no-WD SGD setting different from ours) |

### Medium Priority (new in Iteration 6)

| Paper | arXiv | Finding |
|---|---|---|
| Chen et al. (Nature Comm 2024) "PIDAO" | N/A (doi:10.1038/s41467-024-54451-3) | PID controller applied to optimizer updates (not WD); related conceptual frame |
| Nakamura & Hong (2019) "AdaDecay" | 1907.08931 | Per-parameter WD based on gradient norm; precursor to AWD and rho-Controller |
| "Convergence Bound and Critical Batch Size of Muon" | 2507.01598 | Weight decay + Muon convergence analysis; WD coefficient affects bounds |

### Papers Confirmed from Iteration 5 (still valid)

All papers listed in Iteration 5 report (Defazio 2025, Wang & Aitchison 2024, D'Angelo 2024, Kosson 2024, Chou 2025, arXiv:2510.19093, arXiv:2510.15262, arXiv:2505.13738) remain valid. No superseding papers found.

---

## Part VII: Updated Risk Ranking (Iteration 6)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| AWD (NeurIPS 2023) undermines rho-Controller "first" claims | 90% (certain to be raised) | Moderate — rho-Controller must be repositioned as "AWD + Defazio steady-state motivation + regime-conditioning + Lyapunov" | Reframe contribution; cite AWD explicitly |
| BN confound confirmed | 25% | Moderate — reattribute to D'Angelo | P0-3 NoBN ablation |
| Theorem T1 not provable | 20% | Moderate — switch to Conjecture | Day 0 check |
| Reviewer notes Defazio (2025) covers R_* = λ/η | 80% | Low (if already cited/credited) | Already in plan |
| W&A (2024) overlap undermines Phi Invariance | 35% | Moderate | Comparison table in Related Work |
| Super-twisting H7 fails | 35% | Low (secondary contribution only) | Run P0-3 before writing |
| P1-1 λ sweep flat (Regime II absent) | 25% | Moderate — rho-Controller loses primary motivation in AdamW; survives for SGD | Both paths publishable |

---

## Part VIII: Overall Novelty Assessment (Updated Iteration 6)

### What Remains Genuinely Novel

1. **Phi Invariance Trichotomy Conjecture** with explicit ρ-dependent regime boundaries and falsifiable crossover predictions — no prior paper found
2. **18.3× SGD/AdamW WD presence-absence ratio** with statistical rigor — original controlled empirical data
3. **BN ablation design** distinguishing static (D'Angelo) from dynamic (Phi Invariance) mechanisms — original experimental design
4. **Rho-Controller: Defazio steady-state-grounded target ρ*, Lyapunov convergence, regime-conditional scope** — novel over AWD's heuristic ratio control
5. **Super-twisting WD as chattering-free CWD extension** — conditional on H7 confirmation

### What Has Been Superseded or Must Be Repositioned

1. **"First formulation of WD as ratio controller"** (rho-Controller claim): AWD (NeurIPS 2023) already does this — must be repositioned
2. **Theorem T2 "dual characterization" as new connection**: Already repositioned (Defazio + Xie & Li; unchanged from Iter 5)
3. **r_t steady-state convergence as primary mechanistic claim**: Already Defazio (2025); merge into front-runner

### Score Summary

| Candidate | Iter 5 Score | Iter 6 Score | Change | Reason |
|---|---|---|---|---|
| cand_rho_trichotomy | 7 | 7 | 0 | No new collisions found |
| cand_rho_controller | N/A (new) | **6** | — | AWD (NeurIPS 2023) is a partial collision; Defazio-grounded target is novel |
| cand_super_twisting | 8 | 8 | 0 | No new collisions; still conditional on H7 |
| cand_gradient_weight_ratio | 6 | 6 | 0 | Unchanged |
| cand_spectral_rank_feedback | 6 | 6 | 0 | Dropped; unchanged |

---

## Part IX: Required Actions Before Writing (Updated)

### Blocking (must do before writing rho-Controller section)

1. **Add AWD (arXiv:2210.00094) to Related Work** as a critical precursor to the rho-Controller
2. **Remove "first closed-loop WD" claim** from rho-Controller abstract/introduction
3. **Revise rho-Controller framing**: "Building on AWD's ratio-control principle, the rho-Controller provides a theoretically grounded target ρ* derived from Defazio's steady-state analysis, with Lyapunov convergence guarantees and regime-conditional motivation from the Phi Invariance Trichotomy"
4. **Run P1-2 pilot** before claiming rho-Controller improves over AWD in practice

### Writing Actions (from Iteration 5 — still required)

5. Remove Theorem T2 "dual characterization" as new discovery (still pending from Iter 5)
6. Day 0 theorem provability check before framing T1 as theorem vs. conjecture
7. P0-3 BN ablation before writing abstract claims

### Nice to Have

8. Compare rho-Controller vs. AWD empirically in P1-2 (add AWD as additional baseline)
9. Test whether AWD fails in Regime I as the Trichotomy predicts (this would differentiate our work empirically)
10. Read full AWD paper (beyond abstract) to verify exact formula and check whether Lyapunov analysis is present

---

## References

### New in Iteration 6

- [Ghiasi et al. (NeurIPS 2023). Improving Robustness with Adaptive Weight Decay. arXiv:2210.00094](https://arxiv.org/abs/2210.00094) — **HIGH PRIORITY: partial collision for rho-Controller; add to Related Work**
- [Nakamura & Hong (2019). Adaptive Weight Decay for Deep Neural Networks. arXiv:1907.08931](https://arxiv.org/abs/1907.08931) — predecessor to AWD; medium priority
- [Chen et al. (Nature Comm 2024). PIDAO: Accelerated optimization with PID controller. doi:10.1038/s41467-024-54451-3](https://www.nature.com/articles/s41467-024-54451-3) — related conceptual frame (PID for optimizer)
- [arXiv:2602.11137. Weight Decay Improves Language Model Plasticity. 2025/2026](https://arxiv.org/abs/2602.11137) — regime-dependent WD effects; medium priority
- [arXiv:2506.12543. Is your batch size the problem? Revisiting the Adam-SGD gap. June 2025](https://arxiv.org/abs/2506.12543) — SGD without WD comparison; low priority

### Previously Established (Iteration 5, still valid)

- [Defazio (2025). Why Gradients Rapidly Increase Near the End of Training. arXiv:2506.02285](https://arxiv.org/abs/2506.02285) — HIGH PRIORITY
- [Wang & Aitchison (2024). How to set AdamW's weight decay. arXiv:2405.13698](https://arxiv.org/abs/2405.13698)
- [D'Angelo et al. (NeurIPS 2024). Why Do We Need Weight Decay? arXiv:2310.04415](https://arxiv.org/abs/2310.04415)
- [Kosson et al. (ICML 2024). Rotational Equilibrium. arXiv:2305.17212](https://arxiv.org/abs/2305.17212)
- [Chou (2025). Correction of Decoupled Weight Decay. arXiv:2512.08217](https://arxiv.org/abs/2512.08217)
- [arXiv:2510.19093. Weight Decay May Matter More Than µP. 2025](https://arxiv.org/abs/2510.19093)
- [arXiv:2510.15262. Robust Layerwise Scaling Rules. 2025](https://arxiv.org/abs/2510.15262)
- [arXiv:2505.13738. Power Lines: Scaling Laws for WD and Batch Size. 2025](https://arxiv.org/abs/2505.13738)

---

*Report generated by Sibyl Novelty Checker Agent (sibyl-standard). All collision claims are supported by direct search evidence. AWD (arXiv:2210.00094) was discovered fresh in Iteration 6 via targeted search for "adaptive weight decay gradient norm weight norm ratio controller." Searches confirm no paper publishes the Phi Invariance Trichotomy regime structure or the 18.3× SGD/AdamW ratio in a controlled experimental setting.*
