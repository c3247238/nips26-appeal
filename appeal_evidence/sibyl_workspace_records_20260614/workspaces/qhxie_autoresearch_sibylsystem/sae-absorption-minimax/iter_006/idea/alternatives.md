# Alternative Research Candidates

Backup ideas for pivot if the Sensitivity Floor hypothesis is falsified.

---

## Alternative A: Sensitivity-First (Pragmatist) - Pivot if SF1 Fails

### Title
**"Feature Sensitivity Failures Alone Explain Why Random Baselines Match Sparse Autoencoders"**

### Core Claim
Sensitivity failures (NOT absorption) are the primary driver. Q3 dominates at 65% of features. Absorption adds no predictive value after controlling for sensitivity.

### Key Evidence
- Q3 = 28/43 features (65%) - low-absorption + low-sensitivity
- Q1 = 15/43 features (35%) - high-absorption + low-sensitivity
- Q2 + Q4 = 0/43 (0%) - NO high-sensitivity features
- 85% of features are low-sensitivity regardless of absorption status

### Hypotheses
**H-A**: Low-sensitivity features show steering indistinguishable from random baseline.
**H-B**: Absorption does NOT predict steering after controlling for sensitivity.

### Pilot (~15 min)
1. Load Q3 features (28 features, already classified)
2. Perform steering at beta=5
3. Compare Q3 steering vs random baseline (0.73 from Korznikov)

### Why Pivot Here
Simpler explanation (one failure mode, not two). Directly falsifiable in 15 minutes. If Sensitivity Floor fails, this is the most parsimonious replacement.

---

## Alternative B: Geometric Density Mediator (Empiricist) - Pivot if SF3/SF4 Fail

### Title
**"Geometric Density as the Common Cause of Feature Absorption and Sensitivity Failure"**

### Core Claim
Both failure modes share geometric density as common cause. Dense regions cause decoder overlap (absorption) AND vague activation patterns (low sensitivity). Density mediates r=0.59.

### Key Evidence
- r(absorption, sensitivity) = 0.59 requires common cause explanation
- OrtSAE orthogonality reduces absorption 65% (consistent with reduced overlap)
- Density measurable via k-NN on activation data

### Hypotheses
**H-B1**: Absorbed features have higher geometric density than non-absorbed.
**H-B2**: Partial r(absorption, sensitivity | density) < 0.3.

### Pilot (~15 min)
1. Collect activations on WikiText-103 (~5K tokens)
2. Compute k=20 NN cosine similarity for 100 features
3. Test: r(density, absorption) > 0

### Why Pivot Here
Explains the positive correlation mechanistically. Generates actionable intervention (sparsify dense regions). Testable in 15 minutes of CPU time.

---

## Alternative C: Layer-wise UAS Saturation Mapping (Empiricist + Contrarian)

### Title
**"Measuring the Measurers: Layer-wise UAS Saturation Reveals Critical Gaps in SAE Feature Metrics"**

### Core Claim
UAS saturation is layer-dependent, not universal. Some layers (4, 6, 10) show UAS variance while layer 8 shows saturation. This determines where absorption CAN be measured.

### Key Evidence
- UAS=1.0 at layer 8 (all features absorbed or metric saturates)
- Layer 8 may be architecturally special (deeper = more hierarchical compression)
- Sensitivity (Tian protocol) may work where UAS fails

### Hypotheses
**H-C1**: UAS variance differs across layers (some show std > 0.05, others don't).
**H-C2**: At non-saturated layers, sensitivity correlates with UAS (r > 0.4).

### Pilot (~20 min)
1. Measure UAS at layers 4, 8, 10 (30 features each)
2. Compute mean and std UAS per layer
3. If layers 4 and 10 show variance → saturation is layer-specific

### Why This First
If saturation is universal, the Sensitivity Floor cannot be tested via absorption. This determines the measurement path forward for ALL subsequent experiments.

---

## Alternative D: Steering Diffusion (Theoretical)

### Title
**"Steering Diffusion Explains Beta=20 Reversal: Why Low-Absorption Features Steer More at High Magnitude"**

### Core Claim
Beta=20 reversal (p=0.015) is caused by steering diffusion. High-absorption features have decoder directions that overlap with MORE neighbors. At high beta, steering activates multiple neighbors simultaneously, diluting the effect.

### Key Evidence
- iter_004 found beta=20 reversal
- H6 FALSIFIED: L2 norm ratio = 1.0 (no saturation mechanism)
- At beta <= 10, no absorption-steering correlation (null result)

### Hypotheses
**H-D1**: Absorbed features have higher decoder-neighbor overlap.
**H-D2**: Diffusion is non-linear in beta; null at beta <= 10, effect at beta = 20.

### Pilot (~15 min)
1. Measure decoder-neighbor overlap for 50 features (k=20 NN in decoder space)
2. Test: absorbed features have higher overlap
3. Fit diffusion model to iter_004 beta sweep

### Why This Matters
The beta=20 reversal has no current explanation. If diffusion is confirmed, this is a novel mechanism that reframes steering magnitude effects.

---

## Decision Tree

```
Pilot: Q2+Q4 emptiness at N=200 (30 min)
|
+-- Q2+Q4 > 10% --> Pivot to Alternative A (Sensitivity-First, 15 min)
|
+-- Q2+Q4 <= 10% --> Sensitivity Floor confirmed
    |
    +-- Test U-shape, frequency, density mediation (~45 min)
    |
    +-- If all mediation tests fail --> Consider Alternative B (Geometric Density)
```

---

## Why All Alternatives Are Grounded in Evidence

| Finding | Implication |
|---------|-------------|
| r = 0.59 positive correlation | Requires common cause; rules out independence |
| Q4 = 0/43 | Rules out "best-case quadrant exists" |
| ratio = 1.0 | Rules out decoder norm saturation for beta=20 |
| UAS=1.0 at layer 8 | Measurement limitation, not universal truth |
| Beta=20 reversal unexplained | Requires steering diffusion mechanism |

All four alternatives address specific empirical findings without assuming independence or compound failure.
