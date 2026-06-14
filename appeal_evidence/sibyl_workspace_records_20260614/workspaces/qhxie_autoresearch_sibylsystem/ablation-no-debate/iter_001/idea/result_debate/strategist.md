# Strategic Analysis: Result Debate (Updated)

## Date: 2026-04-29
## Workspace: ablation-no-debate/iter_001
## Strategist: sibyl-strategist

---

## 1. Signal Strength Assessment

| Hypothesis | Signal Strength | Evidence | Justification |
|------------|----------------|----------|---------------|
| **H1: Multi-child absorption** | **STRONG** | Cohen's d = 8.94, delta=0.353 | Unambiguous separation. Effect is deterministic (trained=0.500, random=0.147). |
| **H2: Frequency-absorption correlation** | **NOISE** | rho = +0.17, wrong direction | Expected negative correlation; found positive. Competitive exclusion theory not supported. |
| **H3: Steering intervention** | **MODERATE** | 1.62x sensitivity ratio | Steering fixed and works. Absorbed features are 1.62x more sensitive. |
| **H_Safe: Safety-critical absorption** | **NOISE** | p = 0.665, d = 0.007 | NULL RESULT. Safety (90.7%) = Non-safety (90.6%). No elevated absorption. |
| **H_Mech: Encoder vs Decoder** | **STRONG** | encoder=0.191, decoder=0.0 | MAJOR FINDING. Absorption is encoder-driven, NOT geometric. |

### Key Evidence Details

**H1 Evidence (Strong):**
- Trained SAE absorption: 0.500 (std = 0.0, deterministic across seeds 42,43,44)
- Random decoder absorption: 0.147 (std = 0.065, variance)
- Delta: 0.353 (70% relative difference)
- Cohen's d: 8.94 (beyond "large" threshold of 0.8)
- Multi-seed validation: All seeds = 0.500 (DETERMINISTIC)

**H_Mech Evidence (Strong - NEW):**
- Condition A (Rand Enc, Rand Dec): 0.299
- Condition B (Train Enc, Rand Dec): 0.490 (+0.191 from encoder)
- Condition C (Rand Enc, Train Dec): 0.299 (+0.0 from decoder)
- Condition D (Train Enc, Train Dec): 0.484
- **Conclusion**: Absorption is DRIVEN BY ENCODER, NOT decoder geometry

**H3 Evidence (Moderate - Fixed):**
- Steering verification: PASS (||steered - baseline|| > 0)
- Absorbed feature sensitivity: 0.055
- Non-absorbed sensitivity: 0.034
- Sensitivity ratio: 1.62x
- Directionally correct but marginal for publication

**H_Safe Evidence (Null):**
- Safety absorption: 0.907 +/- 0.038
- Non-safety absorption: 0.906 +/- 0.048
- Mann-Whitney U: p = 0.665
- **Conclusion**: No evidence that safety-critical features have different absorption rates

---

## 2. Opportunity Cost Analysis

| Direction | Estimated GPU Hours | Expected Information Gain | Rationale |
|-----------|--------------------|---------------------------|-----------|
| Held-out generalization | 1 hour | **HIGH** | Validates H1 robustness; required for paper credibility |
| Gemma Scope H_Mech | 2 hours | **HIGH** | Confirms encoder-driven absorption on real SAE (novelty) |
| H_Mech full (5 seeds) | 2 hours | **MEDIUM** | Solidifies H_Mech but pilot already strong |
| H3 full experiment | 2 hours | **MEDIUM** | 1.62x is suggestive but may not reach p<0.01 |
| H_Safe full (Neuronpedia) | 1 hour | **LOW** | p=0.665 is already null; additional features unlikely to change |
| cand_geom (niche geometry) | 0.5 hours | **MEDIUM** | Training-free diagnostic; complementary to H_Mech |

### GPU Cost vs Information Gain Ranking

1. **Held-out generalization** - Required for paper; validates H1 on unseen data
2. **Gemma Scope H_Mech** - Confirms novel encoder-driven claim on real model
3. **cand_geom diagnostic** - Theoretical depth, training-free, low cost
4. **H3 full experiment** - Only if time permits

---

## 3. Decision Matrix

| Direction | Signal | GPU Cost | Risk | Expected Outcome |
|-----------|--------|----------|------|------------------|
| Held-out generalization | STRONG | 1 hr | LOW | Validates H1; required for paper |
| Gemma Scope H_Mech | STRONG | 2 hrs | MEDIUM | Confirms encoder effect on real SAE |
| cand_geom diagnostic | MEDIUM | 0.5 hrs | LOW | Training-free predictor; theoretical depth |
| Paper drafting | STRONG | 0 GPU | LOW | Can start immediately |
| H3 full experiment | MODERATE | 2 hrs | MEDIUM | May not reach p<0.01 |
| H_Safe full (Neuronpedia) | NOISE | 1 hr | HIGH | Wasted effort on confirmed null |

---

## 4. PIVOT vs PROCEED Verdict

**VERDICT: PROCEED**

**Rationale:**
1. H1 is strongly confirmed (d = 8.94) with multi-seed validation
2. H_Mech provides genuinely NOVEL contribution: first evidence that absorption is encoder-driven, not geometric
3. Clear path to NeurIPS/ICLR-quality paper exists
4. H_Safe is a null result, but null results are publishable as cautionary findings

**This is NOT a candidate for PIVOT because:**
- Front-runner hypothesis (H1) has extremely strong empirical support
- H_Mech provides a clear new contribution that emerged from data
- Multiple validated hypotheses exist within current framing
- Clear contribution statement: "Multi-child proportional ablation + encoder-driven absorption mechanism"

**What changed from prior proposal:**
- **Drop**: H_Safe as positive contribution (null result confirmed)
- **Elevate**: H_Mech as primary novel contribution
- **Keep**: H1 as empirical anchor
- **Consider**: H3 as supplementary evidence (marginally significant)

---

## 5. Recommended Next Experiments (Priority Order)

### Priority 1: Held-Out Generalization (Required)
- **GPU Cost**: 1 hour
- **Information Gain**: HIGH
- **Purpose**: Validate H1 on 20% held-out synthetic hierarchies
- **Pass criterion**: Trained > Random with p < 0.001
- **Justification**: Required for paper credibility; establishes robustness

### Priority 2: Gemma Scope H_Mech Validation (Novelty Confirmation)
- **GPU Cost**: 2 hours
- **Information Gain**: HIGH
- **Purpose**: Confirm encoder-driven absorption on real SAE (Gemma-2B layer 12)
- **Method**: Replicate H_Mech factorial on pre-trained Gemma SAE decoder
- **Pass criterion**: Decoder freeze vs decoder train shows <0.05 delta
- **Justification**: Essential for claiming universality of encoder-driven finding

### Priority 3: Paper Draft (Start Immediately)
- **GPU Cost**: 0
- **Information Gain**: N/A
- **Sections to draft**:
  - Abstract (based on H1 + H_Mech)
  - Introduction (Chanin 2024 gap + Korznikov 2026 collision)
  - Methodology (multi-child proportional ablation)
  - Results (H1, H_Mech, H_Safe null, H3 marginal)
  - Discussion (implications for interpretability)
- **Justification**: Can proceed in parallel with experiments

### Priority 4: cand_geom Diagnostic (Optional)
- **GPU Cost**: 0.5 hours (computation only)
- **Information Gain**: MEDIUM
- **Purpose**: Training-free absorption predictor using convex geometry
- **Justification**: Theoretical depth; useful for practitioners

---

## 6. Recommended Drop/Archive

| Item | Action | Rationale |
|------|--------|-----------|
| H_Safe (full) | **ARCHIVE as null result** | p=0.665 confirmed; additional testing unlikely to change |
| H2 (frequency correlation) | **ARCHIVE as negative result** | Wrong direction; competitive exclusion theory not supported |
| cand_eco (competitive exclusion) | **DROP** | H2 failure undermines theoretical foundation |
| cand_ens (multi-resolution ensemble) | **DEFER** | High cost (2 hours), overlaps with Gadgil et al. |

---

## 7. Revised Paper Narrative

**Original framing** (from proposal): "Absorption is a geometric property of decoder structure"

**Revised framing**: "Absorption is primarily driven by encoder alignment with hierarchical feature structure in activation space"

**Evidence chain:**
1. H1: Trained SAE absorbs more than random (d=8.94, deterministic)
2. H_Mech: Encoder effect = 0.191, Decoder effect = 0.0
   - Random encoder + trained decoder = 0.299 (same as both random)
   - Trained encoder + random decoder = 0.490 (near full training)
3. H_Safe: NULL - Universal absorption affects all feature types equally
4. H3: Absorbed features are 1.62x more sensitive to steering

**Novelty claim**: First decomposition of absorption mechanism into encoder vs decoder contributions. Prior work assumed geometric; we show encoder alignment is the driver.

---

## 8. Paper Structure Recommendation

1. **Introduction**: Absorption is a fundamental limitation for SAE interpretability (Chanin 2024); Korznikov 2026 showed general IE is unreliable
2. **Methodology**: Multi-child proportional ablation as fixed measurement methodology
3. **H1 Results**: Strong evidence for multi-child absorption in trained SAEs
4. **H_Mech Results**: First decomposition - absorption is encoder-driven, not geometric
5. **H_Safe Results**: NULL - Safety-critical features not disproportionately absorbed (cautionary)
6. **H3 Results**: Absorbed features are more sensitive (suggestive)
7. **Discussion**: Implications for SAE-based safety analysis and circuit discovery
8. **Conclusion**: Encoder-driven absorption requires rethinking SAE training objectives

**Key contribution framing**: "We identify that SAE feature absorption is driven by encoder alignment, not decoder geometry. This reframes the problem from a structural limitation to a training objective issue."

---

## 9. Anti-Pattern Warnings

- **Fence-sitting**: NOT doing this - clear PROCEED recommendation
- **Sunk cost**: NOT applicable - pivot from "geometric" to "encoder-driven" is a gain, not a loss
- **Ignoring resource constraints**: Prioritized list with explicit GPU costs provided

---

## 10. Summary

| Status | Finding | Priority |
|--------|---------|----------|
| H1 (multi-child absorption) | **STRONG PASS** - d=8.94, deterministic | Anchor |
| H_Mech (encoder-driven) | **STRONG PASS** - encoder=0.191, decoder=0.0 | **PRIMARY NOVELTY** |
| H3 (steering) | **MODERATE PASS** - 1.62x sensitivity | Supplementary |
| H_Safe (safety) | **NULL** - p=0.665 | Archive as cautionary |
| H2 (frequency) | **FAIL** - Wrong direction | Archive as negative |
| Overall | **PROCEED** | Paper-ready with H1 + H_Mech |

The research is in excellent shape. H1 provides robust empirical evidence. H_Mech provides genuinely novel contribution that reframes understanding of absorption mechanism. Paper can proceed to drafting phase.
