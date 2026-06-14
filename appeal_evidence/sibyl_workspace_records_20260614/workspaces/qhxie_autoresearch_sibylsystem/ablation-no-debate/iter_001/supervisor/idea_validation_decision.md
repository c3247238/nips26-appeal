# Idea Validation Decision

## Pilot Evidence Summary

### H_Mech (2x2 Factorial Decomposition) - MAJOR NEW FINDING

| Condition | Absorption | Description |
|-----------|------------|-------------|
| A: Rand enc + Rand dec | 0.299 | Pure geometry baseline |
| B: Train enc + Rand dec | 0.490 | Encoder effect = +0.191 |
| C: Rand enc + Train dec | 0.299 | Decoder effect = 0.0 |
| D: Train enc + Train dec | 0.484 | Full training |

**CRITICAL FINDING**: Decoder geometry contributes ZERO to absorption. Absorption is entirely ENCODER-driven. This contradicts the original pilot interpretation (which said absorption was "primarily geometric"). The 0.191 effect comes entirely from encoder alignment with hierarchical feature structure.

### H1 (Multi-Child Proportional Absorption)

| Condition | Absorption Rate | Std | Notes |
|-----------|----------------|-----|-------|
| Trained SAE | 0.500 | 0.0 (deterministic) | All 3 seeds = 0.500 exactly |
| Random Baseline | 0.147 | 0.065 | Variable across seeds |
| Cohen's d | 8.94 | p < 10^-133 | Strongly significant |

### H3 (Steering Fix)

- Steering verified: delta_norm > 0 for all alpha > 0
- Absorbed sensitivity: 0.055 vs Non-absorbed: 0.034
- Sensitivity ratio: 1.62x (absorbed features are 62% more sensitive to steering)
- PASSED (implementation fixed)

### H_Safe (Safety-Critical Features)

- p = 0.695 (Mann-Whitney)
- Mean difference: 0.01 (safety 0.140 vs non-safety 0.130)
- **FAIL**: Pilot tested synthetic SAE, NOT Gemma Scope SAEs. This was the wrong experiment.

### H2 (Frequency Correlation) - ARCHIVED NEGATIVE RESULT

- rho = +0.171 (positive, not negative as hypothesized)
- FAILED (archived as negative result in prior decision)

---

## Decision Matrix

### cand_p1 (front-runner with H1, H_Mech, H3, H_Safe evidence)

| Criterion | Weight | Score | Evidence |
|-----------|--------|-------|----------|
| Pilot signal strength | 0.30 | 5 | H1: d=8.94 deterministic; H_Mech: encoder effect=0.191, decoder=0; H3: 1.62x steering ratio verified |
| Hypothesis survival | 0.25 | 4 | H1 survives. H_Mech REVEALS new mechanism (not falsified). H3 works. H_Safe FAILED on wrong experiment. |
| Path to full result | 0.20 | 4 | Clear: real SAEs + multi-seed factorial + logit-level steering + Gemma Scope (retry H_Safe) |
| Novelty | 0.15 | 5 | Encoder-driven absorption is a genuinely novel mechanism finding. Safety analysis is 9/10 if Gemma Scope works. |
| Resource efficiency | 0.10 | 4 | ~2h GPU total. Tasks within 1-hour budget. |
| **Weighted Score** | | **4.40** | |

### H3 (steering intervention as separate focus)

| Criterion | Weight | Score | Evidence |
|-----------|--------|-------|----------|
| Pilot signal strength | 0.30 | 3 | 1.62x sensitivity ratio is positive but modest. Basu et al. showed zero effect. |
| Hypothesis survival | 0.25 | 3 | Valid effect, but needs more seeds to be publishable. |
| Path to full result | 0.20 | 3 | Needs logit-level metrics and multi-seed validation. |
| Novelty | 0.15 | 3 | Basu et al. complicates claims (6/10). |
| Resource efficiency | 0.10 | 4 | 30 min GPU. |
| **Weighted Score** | | **3.20** | REFINE |

### H_Safe (safety-critical absorption)

| Criterion | Weight | Score | Evidence |
|-----------|--------|-------|----------|
| Pilot signal strength | 0.30 | 1 | p=0.695 on synthetic SAE = no evidence about real Gemma Scope SAEs. |
| Hypothesis survival | 0.25 | 3 | Gemma Scope not tested. Independent of H1/H_Mech. |
| Path to full result | 0.20 | 2 | Requires Gemma Scope installation (non-trivial). |
| Novelty | 0.15 | 5 | 9/10 if it works. |
| Resource efficiency | 0.10 | 2 | Gemma Scope loading is nontrivial. |
| **Weighted Score** | | **2.20** | PIVOT (from priority queue) |

---

## Decision Rationale

### ADVANCE cand_p1

1. **H1 confirmed with deterministic result**: Trained SAE absorption = 0.500 across all seeds. Random baseline = 0.147 with variance. Cohen's d = 8.94. This is not a weak signal.

2. **H_Mech is a MAJOR positive finding**: The 2x2 factorial produced a genuinely surprising and novel result -- absorption is entirely encoder-driven (effect = 0.191), with decoder geometry contributing zero. This contradicts the prior interpretation and is a stronger, more interesting paper contribution.

3. **H3 steering is now validated**: The implementation was fixed. Absorbed features show 1.62x more sensitivity to steering. This enables causal claims about absorption.

4. **H_Safe failed on wrong experiment**: The synthetic SAE pilot (p=0.695) is not informative about real Gemma Scope SAEs. This hypothesis needs Gemma Scope to be properly tested, but is not killed by the synthetic data.

5. **Resource efficient**: ~2 hours total GPU time across all experiments.

### Why PIVOT on H_Safe for now

The H_Safe pilot tested a synthetic SAE, not Gemma Scope. The p=0.695 result is meaningless for the real hypothesis. Gemma Scope installation is a prerequisite, not a "nice to have". Until Gemma Scope works, H_Safe cannot be advanced.

### Why REFINE H3

The 1.62x effect is real but modest. Basu et al. showed zero steering effect in their work. We need logit-level metrics (not feature activation level) and more seeds before making causal publication claims.

---

## Next Actions

1. **Full multi-seed factorial validation**: 5 seeds x 4 conditions x 200+ samples to confirm encoder effect is reproducible
2. **Update paper narrative**: Absorption is encoder-driven, not decoder geometry. This is the main contribution.
3. **Gemma Scope installation attempt**: Try loading Gemma Scope SAEs from HuggingFace. If successful, prioritize H_Safe (highest novelty).
4. **Logit-level steering validation**: Switch from feature activation to logit-level metrics per Basu et al. methodology.
5. **Retry H_Safe on Gemma Scope**: If Gemma Scope loads successfully.
6. **Document negative results**: H2 (wrong direction correlation), H_Safe synthetic failure (wrong experiment).

---

SELECTED_CANDIDATE: cand_p1
CONFIDENCE: 0.85
DECISION: ADVANCE
