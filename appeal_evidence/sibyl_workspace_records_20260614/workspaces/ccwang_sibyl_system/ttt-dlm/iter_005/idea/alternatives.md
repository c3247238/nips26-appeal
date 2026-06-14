# Backup Research Alternatives for Pivot

**Synthesizer**: sibyl-synthesizer
**Date**: 2026-03-11
**Revision**: Round 3 (post-debate, structurally reoriented)

Note: The previous "Alternative A" (Information-Gain Adaptive Unmasking) has been promoted to a co-equal primary track (Track A in proposal.md). The alternatives below are fallback directions if the primary tracks fail.

---

## Alternative D: Systematic Diagnostic Study — "Extrinsic vs Endogenous: Why Schedule Optimization Outperforms Cross-Step Memory in DLM Inference Scaling"

### Status: ACTIVE PARALLEL TRACK (data collection ongoing)

### Pivot Trigger
- IGGD fails to replicate at n>=300 AND
- DaL fails after D0c+D0d pass AND
- COBA produces only incremental findings

OR (more likely scenario):
- DaL kills confirmed, IGGD works, and the diagnostic data enriches the IGGD paper

### Core Idea
Publish the 22+ iterations of systematic exploration as a **mechanistic analysis paper** explaining why different paradigms succeed or fail in DLM inference-time scaling:

1. **Extrinsic vs Endogenous information classification** — why Info-Gain Soft (+5pp) and A-CFG (+12.5pp at n=16) succeed while ReMDM (-9pp), DaL (-1pp), and MetaState-GRU (-6.25pp) fail
2. **SSL-task misalignment analysis** (D0c data) — formal measurement of decorrelation between self-supervised and downstream objectives
3. **Information-theoretic ceiling** (D0d Delta_I data) — upper bound on cross-step memory benefit
4. **Compute-fair Pareto analysis** — showing which methods dominate at each FLOPs budget
5. **Turbo coding framework** — extrinsic information principle applied to DLM denoising

### Evidence Available
- 22+ iterations, 20+ method variants
- P1-P3 pilot data (SSL feasibility, signal quality, task evaluation)
- alt_a_pilot (IGGD +5pp, Greedy -18pp, entropy analysis)
- D0c + D0d diagnostic data (to be collected Phase 0)
- COBA grid search data (collected Phase 0)

### Target Venue
NeurIPS 2026 (if mechanistic explanation is deep enough) or EMNLP 2026 main

### Advantages
- Paper is partially written from accumulated data
- Genuine contribution preventing others from repeating mistakes
- The extrinsic vs endogenous framework is a novel theoretical lens
- Honest science with high citation potential

### Risk
- Requires exceptionally clear mechanistic explanations, not just failure catalog
- Must include cross-backbone validation (Dream-7B + LLaDA-8B)
- Reviewers may see as "we tried and failed" without sufficient theoretical depth

### Estimated GPU Cost
- Additional experiments needed: cross-backbone validation (~8h), gap-filling analysis (~4h)
- Most data collected as part of primary tracks
- Total marginal: ~12-15 GPU-hours

### Success Probability
70% (publishable with proper mechanistic framing + cross-backbone validation)

---

## Alternative B: Enhanced Learned Optimizer with Noise-Level Conditioning

### Status: STANDBY (Low priority)

### Pivot Trigger
- DaL Phase 1 shows TTT-MLP ~ GRU (H1 falsified) — gradient steps too noisy
- AND D0c passes (SSL-task alignment exists)
- AND Delta_I > 0.1 (cross-step memory has headroom)

### Core Idea
Instead of replacing GRU with raw gradient steps, **enhance the learned optimizer**:
1. **Noise-level conditioned GRU**: Gate parameters conditioned on current mask ratio r_t
2. **Momentum + remasking-coupled decay** (Titans-style)
3. **Hybrid GRU-then-TTT**: GRU for early steps (sparse signal) -> gradient for late steps (rich signal)

### Risk
- MetaState-GRU itself underperformed vanilla in pilot (43.75% vs 50.0%)
- Incremental over MetaState
- Requires MetaState reproduction first

### Estimated GPU Cost: ~60 GPU-hours
### Success Probability: 30% (downgraded due to MetaState pilot failure)

---

## Alternative E: Associative Denoising Memory (ADM) — Hopfield Networks as Cross-Step Memory

### Status: STANDBY (Conditional on IGGD mechanism analysis)

### Pivot Trigger
- IGGD verified at scale
- Cross-step hidden state correlation > 0.1 (diagnostic from innovator)
- Interest in providing architectural strengthening of IGGD mechanism

### Core Idea (from Innovator)
Use modern Hopfield network layers as cross-step memory. At each step, revealed tokens query an associative memory storing patterns from all previous steps. No gradient computation — just energy minimization dynamics.

### Connection to IGGD
ADM's Hopfield retrieval is mathematically related to information-gain maximization: both identify the most "uncertain" patterns that would benefit most from additional context. ADM could serve as an architectural implementation of the IGGD principle within the backbone.

### Risk
- Hopfield-diffusion equivalence is asymptotic, may not hold practically
- May produce averaged/blurred patterns (mode collapse)
- Novel and untested

### Estimated GPU Cost: ~4-8 GPU-hours (pilot)
### Success Probability: 25%

---

## Alternative F: Self-Play Denoising (SPD) with External Verifier

### Status: STANDBY (Conditional on trajectory diversity)

### Pivot Trigger
- IGGD verified, looking for complementary W-axis method
- Trajectory diversity diagnostic shows Distinct-1 < 0.9 across 4 parallel trajectories

### Core Idea (from Innovator, revised)
Run N=4 parallel denoising trajectories, score with external verifier (not self-likelihood), use contrastive signal for test-time DPO update. Requires external verifier to avoid self-reward miscalibration.

### Kill Criterion
- Trajectory Distinct-1 > 0.9 -> trajectories too similar, SPD infeasible
- Winning vs losing trajectory answer-correct-rate difference < 5% -> contrast signal too weak

### Estimated GPU Cost: ~6-8 GPU-hours (pilot)
### Success Probability: 20%

---

## Priority Order for Pivots (Revised)

| Priority | Alternative | Trigger | Status | Effort | P(success) |
|----------|------------|---------|--------|--------|------------|
| 1 | **D: Diagnostic/Mechanistic Study** | DaL fails + diagnostic data rich | Data collecting | Low (3-5 days marginal) | 70% |
| 2 | **E: ADM (Hopfield)** | IGGD works + cross-step correlation exists | Standby | Medium (4-8h pilot) | 25% |
| 3 | **B: Enhanced Optimizer** | D0c+D0d pass + TTT~GRU | Standby | High (60h) | 30% |
| 4 | **F: Self-Play Denoising** | IGGD works + trajectory diversity exists | Standby | Medium (6-8h pilot) | 20% |

### Key Change from Round 2
- Previous "Alternative A" promoted to primary Track A
- Previous "Alternative C" (COBA Framework) integrated into primary proposal as Track C
- Alternative D strengthened with extrinsic vs endogenous framework
- Alternatives E and F (from innovator) added as conditional directions
- Alternative B deprioritized due to MetaState pilot failure
