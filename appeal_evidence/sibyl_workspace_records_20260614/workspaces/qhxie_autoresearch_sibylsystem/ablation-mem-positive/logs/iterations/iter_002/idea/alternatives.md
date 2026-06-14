# Backup Ideas for Potential Pivot

These alternatives are maintained as backup options if the primary proposal needs refinement or pivot based on experimental validation.

---

## Backup 1: Projection-Based Cross-Layer Absorption Quantification

**Candidate ID**: backup_projection

**Status**: backup (ready to promote if primary encounters critical barriers)

### Hypothesis

SAEBench probe projection metric provides a training-free, ablation-free measurement of absorption across ALL layers (not just early layers 0-17). Cross-layer absorption patterns can be characterized without layer-specific saturation issues.

### Why It Could Work

1. **No ablation required**: Probe projection works across all layers (unlike ablation-based detection limited to early layers)
2. **Clean metric**: Directly measures how much a feature's representation contributes to probe predictions vs ablation
3. **Systematic**: SAEBench provides standardized methodology across SAE configurations
4. **Publishable in either direction**: If cross-layer variation found → validates layer heterogeneity; if no variation found → field correction

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| Cross-layer at λ_c | 2-3 hours | Layers 0,3,6,9,11 at λ_c=5e-5 using probe projection |
| Cross-model replication | 2 hours | GPT-2 → Gemma-2 comparison |

### Risk

- If no cross-layer variation found even at λ_c, the backup also fails
- If variation found but small, statistical power may be insufficient

### Falsification Criteria

- If ALL layers show identical absorption rates → backup is falsified
- If only early layers show variation → layer saturation still limits interpretation

---

## Backup 2: Steering Effectiveness and Actionability Analysis

**Candidate ID**: backup_steering

**Status**: backup (addresses actionability paradox directly)

### Hypothesis

The actionability paradox (Basu et al., 2026) applies universally to SAE steering—features with high absorption scores AND features with low absorption scores show zero steering effectiveness. The CV-based decomposition (high-CV absorbed vs low-CV non-absorbed) may predict which features retain steering potential.

### Why It Could Work

1. **Directly addresses field's critical question**: Does quantifying absorption help us steer models?
2. **Connects to Basu et al.**: Extends their clinical domain finding to non-clinical domain (spelling task)
3. **Tests CV hypothesis**: If high-CV absorbed features show ANY steering advantage, the variance paradox becomes actionable
4. **Strong negative result**: Universal actionability failure is itself a significant finding

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| Steering test: high-CV vs low-CV absorbed | 1-2 hours | Test steering on 20 high-CV vs 20 low-CV absorbed features |
| Steering test: absorbed vs non-absorbed | 1 hour | Compare overall steering effectiveness between absorption classes |
| Decoder orthogonality analysis | 30 min | Compute cosine similarity; test if orthogonal = steerable |

### Risk

- If universal actionability failure confirmed (all features zero steering), this is a negative result but may be too strong
- If steering works for non-absorbed features only, this supports using absorption metrics for filtering

### Falsification Criteria

- If non-absorbed features show significantly higher steering effectiveness (p<0.01, Cohen's d>0.5) → absorption metric IS predictive
- If all features show zero steering → actionability paradox is universal
- If orthogonality correlates with steering effectiveness (r>0.3) → alternative predictor identified

---

## Backup 3: Cross-Architecture Phase Transition Validation

**Candidate ID**: backup_cross_arch

**Status**: backup (validates generalizability of phase transition framework)

### Hypothesis

The finite-size scaling with ν=3 discovered on GPT-2 TopK SAEs generalizes to Gemma-2-2B JumpReLU SAEs, with architectural correction (different λ_c but same ν). Cross-architecture validation establishes the phase transition as a universal phenomenon, not a GPT-2/TopK artifact.

### Why It Could Work

1. **Tests artifact hypothesis**: The contrarian raised the concern that phase transitions are GPT-2/TopK artifact
2. **Universal scaling**: If ν is consistent across architectures, this strengthens the theoretical framework
3. **Architectural correction**: λ_c variation across architectures is itself informative (what determines λ_c?)
4. **Strong publication**: First cross-architecture validation of phase transition in SAEs

### Pilot Plan

| Experiment | Duration | Expected Outcome |
|------------|----------|-----------------|
| GPT-2 sparsity sweep | 45 min | Confirm λ_c=5e-5 on current setup |
| Gemma-2 sparsity sweep | 45 min | Identify λ_c for JumpReLU architecture |
| Scaling collapse comparison | 30 min | Test if ν=3 holds across architectures |

### Risk

- If Gemma-2 shows no phase transition (smooth absorption curve), artifact hypothesis is confirmed
- If λ_c varies dramatically, explaining the variation requires additional theory

### Falsification Criteria

- If no critical threshold found in Gemma-2 → artifact hypothesis confirmed, phase transition is architecture-specific
- If ν ≠ 3 for Gemma-2 → scaling exponent is not universal

---

## Decision Logic for Backup Promotion

| Condition | Action |
|-----------|--------|
| Primary fails at λ_c identification | Promote Backup 1 (probe projection for cross-layer) |
| Primary CV decomposition shows no steering advantage | Promote Backup 2 (steering effectiveness) |
| Primary H1/H2 fail replication on Gemma-2 | Promote Backup 3 (cross-architecture) |
| All primary hypotheses confirmed but actionability unknown | Add Backup 2 as supplementary experiment |

---

## Current Status

**Primary candidate**: Phase transition framework with variance paradox (CV_reversed)

**Backup pool**:
1. **backup_projection** (high priority): Ready if cross-layer at λ=0.001 remains saturated
2. **backup_steering** (medium priority): Ready if actionability connection is critical
3. **backup_cross_arch** (lower priority): Generalization validation, can be future work

**Recommendation**: Execute primary proposal first. If H3 (cross-layer at λ_c) shows uniform saturation even at λ_c=5e-5, immediately pivot to Backup 1 using SAEBench probe projection metric.
