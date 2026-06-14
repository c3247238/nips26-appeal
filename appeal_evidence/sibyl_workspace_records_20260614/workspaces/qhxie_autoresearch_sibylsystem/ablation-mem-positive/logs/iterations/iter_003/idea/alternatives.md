# Backup Ideas for Potential Pivot

These alternatives are maintained as backup options if the primary proposal needs refinement or pivot based on experimental validation.

---

## Decision Logic for Backup Promotion

| Condition | Action |
|-----------|--------|
| Primary fails at steering validation (no CV effect) | Promote Backup 1 (probe projection for cross-layer) |
| Primary actionability connection shows universal failure | Promote Backup 2 (steering effectiveness — confirm Basu et al.) |
| Primary CV-steering works but mechanism unclear | Add mechanism investigation from Backup 3 |
| All primary hypotheses confirmed | Consider cross-architecture validation as future work |

---

## Backup 1: Projection-Based Cross-Layer Absorption Quantification

**Candidate ID**: backup_projection

**Status**: backup (promote if H3 retesting at λ_c fails)

### Hypothesis

SAEBench probe projection metric provides training-free, ablation-free measurement of absorption across ALL layers (not just early layers 0-17). Cross-layer absorption patterns can be characterized without layer-specific saturation issues that plagued H3 at λ=0.001.

### Why It Could Work

1. **No ablation required**: Probe projection works across all layers
2. **Clean metric**: Directly measures feature representation contribution vs ablation
3. **Standardized**: SAEBench provides methodology across SAE configurations
4. **Publishable in either direction**: Variation found or not, both are findings

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| Cross-layer at λ_c | 2-3 hours | Layers 0,3,6,9,11 at λ_c=5e-5 using probe projection |
| Cross-model replication | 2 hours | GPT-2 → Gemma-2 comparison |

### Falsification Criteria

- If ALL layers show identical absorption rates at λ_c → backup is falsified
- If only early layers show variation → layer saturation still limits interpretation

---

## Backup 2: Steering Effectiveness and Actionability Analysis

**Candidate ID**: backup_steering

**Status**: backup (confirms or refutes Basu et al. actionability paradox)

### Hypothesis

The actionability paradox (Basu et al., 2026) applies universally to SAE steering. Features with high absorption scores show zero steering effectiveness regardless of other properties. If universal, this is a strong negative result; if not universal, we need to explain which features are steerable.

### Why It Could Work

1. **Directly addresses field's critical question**: Does quantifying absorption help us steer models?
2. **Extends Basu et al.**: Clinical domain (Basu) → non-clinical domain (spelling task)
3. **Strong publication**: Universal failure confirms important boundary condition; partial success identifies exceptions
4. **Actionable in either case**: Either absorption metrics matter or they don't

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| Steering test: high-CV vs low-CV absorbed | 1-2 hours | Test steering on 20 high-CV vs 20 low-CV absorbed features |
| Steering test: absorbed vs non-absorbed | 1 hour | Compare overall steering effectiveness between absorption classes |
| Decoder orthogonality analysis | 30 min | Compute cosine similarity; test if orthogonal = steerable |

### Falsification Criteria

- If non-absorbed features show significantly higher steering effectiveness → absorption metric IS predictive
- If all features show zero steering → actionability paradox is universal
- If orthogonality correlates with steering (r>0.3) → alternative predictor identified

---

## Backup 3: Cross-Architecture Phase Transition Validation

**Candidate ID**: backup_cross_arch

**Status**: backup (generalization validation, future work if primary succeeds)

### Hypothesis

The finite-size scaling with ν=3 discovered on GPT-2 TopK SAEs generalizes to Gemma-2-2B JumpReLU SAEs, with architectural correction (different λ_c but same ν). Cross-architecture validation establishes phase transition as universal phenomenon.

### Why It Could Work

1. **Tests artifact hypothesis**: Contrarian raised concern that phase transitions are GPT-2/TopK artifact
2. **Universal scaling**: If ν is consistent across architectures, strengthens theoretical framework
3. **Architectural correction**: λ_c variation is itself informative
4. **Strong publication**: First cross-architecture validation of phase transition in SAEs

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| GPT-2 sparsity sweep | 45 min | Confirm λ_c=5e-5 on current setup |
| Gemma-2 sparsity sweep | 45 min | Identify λ_c for JumpReLU architecture |
| Scaling collapse comparison | 30 min | Test if ν=3 holds across architectures |

### Falsification Criteria

- If no critical threshold found in Gemma-2 → artifact hypothesis confirmed
- If ν ≠ 3 for Gemma-2 → scaling exponent is not universal

---

## Current Status

**Primary candidate**: CV-based actionability decomposition (cand_cv_actionability)

**Backup pool**:
1. **backup_projection** (high priority): Ready if cross-layer at λ_c shows uniform saturation
2. **backup_steering** (medium priority): Ready if actionability connection is critical
3. **backup_cross_arch** (lower priority): Generalization validation, future work

**Recommendation**: Execute primary proposal first. If steering validation fails (no CV effect), pivot to Backup 2 confirming Basu et al. universal failure. If H3 (cross-layer at λ_c) shows uniform saturation even at λ_c=5e-5, pivot to Backup 1 using SAEBench probe projection metric.