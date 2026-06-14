# Backup Ideas for Potential Pivot

These alternatives are maintained as backup candidates in case the front-runner encounters fatal issues (falsification, resource constraints, or unexpected null results).

---

## Alternative 1: Encoder Regularization to Reduce Absorption (cand_encreg)

**Status**: backup
**Novelty**: 7/10 (differentiated from OrtSAE by encoder-only intervention)

### Core Idea

If H_Mech confirms encoder-driven absorption, modifying the encoder architecture or training objective could reduce absorption. Prior work (OrtSAE) modifies both encoder and decoder; encoder-only modification is a cleaner intervention that leverages the B>D finding.

### Hypothesis

H_EncReg: Encoder regularization penalizing parent-child activation correlation reduces absorption >30% with <5% reconstruction degradation.

### Why This is a Strong Backup

1. **Theoretically grounded**: The B>D finding suggests decoder acts as regularizer. An encoder-only regularizer could achieve the same effect more directly.

2. **Differentiated from OrtSAE**: OrtSAE modifies both encoder and decoder. Our approach targets encoder only, which is a cleaner test of the mechanism.

3. **Practical value**: If successful, this is a constructive intervention the community can use.

### Pilot Focus

1. Implement encoder-only orthogonality penalty
2. Compare absorption reduction vs. OrtSAE
3. Verify reconstruction degradation < 5%

### Resource Estimate

~1.5 hours on 1 GPU

### Drop Trigger

If regularization degrades reconstruction >10% or reduces absorption <10%

---

## Alternative 2: Training-Free Absorption Proxy Validation (from Empiricist)

**Status**: backup
**Novelty**: 7/10 (systematic validation of proxy metrics)

### Core Idea

The field needs training-free absorption detection because ablation-based methods only work for layers 0-17. Validate encoder-decoder asymmetry as a proxy against SynthSAEBench ground truth.

### Hypothesis

H_Proxy: Encoder-decoder asymmetry (specifically reconstruction fidelity divergence) predicts absorption status with AUC > 0.75 against ablation-based ground truth.

### Why This is a Strong Backup

1. **Direct field need**: SAEBench shows proxy metrics don't predict practical performance -- but nobody has systematically validated which proxies work.

2. **Falsifiable**: Clear success criterion (AUC > 0.75). Either the proxy works or it doesn't.

3. **Practical impact**: If validated, practitioners can screen pretrained SAEs without expensive ablation.

### Pilot Focus

1. Compute asymmetry metrics on SynthSAEBench (ground truth absorption labels)
2. Evaluate AUC-ROC against binary absorption label
3. Compare multiple candidate proxies: ECD, RER, AED, FSS

### Resource Estimate

~1 hour on 1 GPU

### Drop Trigger

If no proxy achieves AUC > 0.70 against ground truth

---

## Alternative 3: Multi-Resolution SAE Ensemble (cand_ens)

**Status**: deferred
**Novelty**: 5/10 (partial overlap with Gadgil et al.)

### Core Idea

Train ensemble of SAEs with varying L0 targets (16, 64, 256) to collectively recover hierarchical features. High-sparsity SAEs capture coarse parent features; low-sparsity SAEs capture fine child features.

### Why This is a Weak Backup

1. **Overlap with prior work**: Gadgil et al. (2025) covers SAE ensemble concept
2. **High cost**: 2 hours training time
3. **Uncertain payoff**: Ensemble recovery rate may be < 20%

### When to Pivot Here

Only if:
- H_Downstream shows absorption DOES matter for downstream tasks
- Encoder regularization fails
- The L0 diversity mechanism shows clear differentiation from Gadgil

### Resource Estimate

~2 hours on 1 GPU

### Drop Trigger

If ensemble recovery rate < 20%

---

## Alternative 4: Information-Theoretic Collision Framework (from Theoretical)

**Status**: conceptual backup

### Core Idea

Absorption occurs when mutual information gap favors child features:

```
I(child; X) > I(parent; X) → child gets encoder direction → parent absorbed
```

### Why This is Conceptual

1. **Cannot directly measure I(F; X)** -- requires knowing true feature distribution
2. **Can approximate with activation frequency** -- but this is a proxy
3. **H2 result contradicts simple frequency prediction** -- observed +0.171 correlation

### When to Use This

As a theoretical framing for the paper's discussion section, not as a standalone experiment.

---

## Pivot Priority Order

If front-runner (cand_p1 + H_Downstream) fails:

1. **First pivot**: Encoder Regularization (cand_encreg) -- highest practical value
2. **Second pivot**: Training-Free Proxy Validation (empiricist) -- direct field need
3. **Third pivot**: Multi-Resolution Ensemble (cand_ens) -- only if resources allow

---

## Dropped Alternatives

### cand_geom: Encoder Geometry Diagnostic

**Reason**: H_Mech shows decoder geometry contributes nothing. Original geometry-based prediction was about decoder directions. Would need to retool for encoder directions only, but Tang 2512.05534 covers theoretical ground.

### cand_eco: Efficient Coding Framing

**Reason**: This is a reframing, not a new empirical contribution. Barlow 1961 is the origin. Use as discussion framing, not standalone contribution.