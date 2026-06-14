# Backup Ideas for Potential Pivot

## Priority Pivot Candidates

### Pivot A: Encoder Architecture Modifications to Reduce Absorption (PRIORITY 1)

**Origin**: H_Mech factorial result + Theoretical perspective

**Status**: NEW -- The H_Mech finding that absorption is encoder-driven opens a direct intervention target.

**Why Now**: If the encoder is the sole driver of absorption, modifying the encoder architecture or training objective could reduce absorption without touching the decoder. This is a cleaner intervention than prior work (OrtSAE modifies both encoder and decoder).

**When to Pivot Here**:
- If H_Mech is confirmed across multiple seeds and hierarchy strengths
- If the paper needs a constructive contribution beyond diagnosis
- If H_Safe shows null results and a mitigation story is needed

**Intervention Ideas**:
1. **Hierarchical encoder regularization**: Penalize encoder weights that create parent-child activation correlations
2. **Multi-head encoder**: Separate heads for parent vs. child features, combined at decoder
3. **Frequency-aware encoder initialization**: Initialize encoder columns to be orthogonal for hierarchical feature groups

---

### Pivot B: Real-World Absorption on Gemma Scope (PRIORITY 2)

**Origin**: Innovator + Pragmatist perspectives

**Status**: BLOCKED -- Requires SAELens + Gemma Scope installation

**Why Now**: All current evidence is on synthetic data (d_model=128, synthetic hierarchy). The field needs evidence on real LLM SAEs.

**When to Pivot Here**:
- If Gemma Scope SAEs become available
- If the paper is rejected for lacking real-model evidence
- As a parallel validation track

**Experiments**:
1. Measure absorption on Gemma 2 2B layer 12 SAE (16k features)
2. Compare absorption rates across layers (8, 12, 18)
3. Compare across SAE widths (16k, 65k, 1M)
4. Test whether absorption correlates with feature interpretability (Neuronpedia annotations)

---

### Pivot C: Absorption as a Feature Quality Diagnostic (PRIORITY 3)

**Origin**: Contrarian + Empiricist perspectives

**Status**: EMERGING -- H3 fix shows absorbed features are 1.62x more sensitive to steering

**Why Now**: The unexpected H3 result (absorbed features MORE sensitive) suggests absorption is not purely a failure mode. Absorbed features may retain parent information in a "steerable" form.

**When to Pivot Here**:
- If H3 result holds on real models
- If the paper needs a positive framing (absorption as signal, not noise)
- If mitigation experiments fail

**Claim**: Absorption rate can be used as a diagnostic for feature "steerability" -- highly absorbed features are more responsive to interventions.

---

## Deferred Candidates

### Deferred: Competitive Exclusion Framing

**Origin**: Interdisciplinary perspective

**Status**: DEFERRED -- H2 failure (positive frequency correlation) undermines theoretical foundation.

**H2 Result**: Positive correlation (rho=+0.171) contradicts competitive exclusion prediction.

**Why Deferred**: The theoretical framework predicting low-frequency vulnerability is not supported. The new encoder-driven mechanism suggests efficient coding, not competitive exclusion.

**When to Revisit**:
- Only if real-world data shows negative frequency correlation
- As supplementary discussion, not primary framing

---

### Deferred: Multi-Resolution SAE Ensemble (MRSAE)

**Origin**: Interdisciplinary perspective

**Status**: DEFERRED -- High cost (2 hours), overlaps with Gadgil et al.

**Novelty Issue**: Gadgil et al. already showed ensembles capture more features. L0 diversity is differentiated but needs proof.

**When to Revisit**:
- If H1 and H3 both confirmed on real models
- If practical mitigation is needed beyond diagnosis
- As a mitigation proposal, not primary contribution

---

### Deferred: Niche Geometry Diagnostic

**Origin**: Interdisciplinary perspective

**Status**: DEFERRED -- H_Mech result shows decoder geometry is irrelevant.

**Why Deferred**: The niche geometry diagnostic (containment_ratio) operates on decoder directions. The H_Mech factorial shows decoder geometry contributes nothing to absorption. A geometry-based predictor would need to operate on encoder directions instead, which is less interpretable.

**When to Revisit**:
- If encoder-direction geometry proves predictive
- If a training-free diagnostic is needed

---

## Dropped Candidates

### Dropped: Decoder Geometry Theory

**Status**: DROPPED -- H_Mech factorial shows decoder contributes nothing.

**Result**: Condition C (random encoder + trained decoder) = 0.299, same as random/random.

**When to Revisit**:
- Never, unless contradictory evidence emerges

### Dropped: Encoder-Decoder Asymmetry (cand_asym)

**Status**: DROPPED -- Pilot data shows no separation.

**Result**: Asymmetry index (norm ratio) is 0.487 (SAE) vs 0.471 (baseline), essentially no separation.

---

## Experimental Validation Checklist

Before pivoting to any backup, verify:

1. **Encoder-driven mechanism confirmed**: H_Mech must hold across seeds {42, 43, 44, 45, 46}
2. **Zero-variance resolution**: Stochastic hierarchy generation must produce variance > 0.05
3. **H3 replication**: Steering sensitivity ratio must hold across seeds
4. **Gemma Scope availability**: SAELens must load real pretrained SAEs

---

## Pivot Decision Tree

```
New Pilot Results Review
       |
       v
H_Mech confirmed (encoder-driven)?
 /          \
No            Yes
 |            |
H_Safe       H3 replicated?
analysis     /              \
            |               No
            |                |
            v                v
     Proceed with     H_Safe on Gemma
     encoder         Scope?
     framing        /        \
                   |          |
                   v          v
            Proceed with   Proceed with
            H1 + H_Mech   H1 + H_Mech
            + H3          + H_Safe
```
