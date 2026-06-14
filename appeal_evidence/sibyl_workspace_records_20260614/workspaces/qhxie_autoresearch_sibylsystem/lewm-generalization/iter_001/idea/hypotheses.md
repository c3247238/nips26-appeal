# Testable Hypotheses

## Primary Hypotheses

### H1: Compositional Generalization Gap
**Claim**: LeWM exhibits a significant compositional generalization gap when tested on held-out combinations of physical parameters.

**Operationalization**:
- Train LeWM on 18/27 DMControl Walker combinations (friction × gravity × mass, 3 levels each)
- Test zero-shot on 9 held-out combinations
- Primary metrics: linear probing R², CEM planning success rate

**Expected outcome**: Linear probing R² drops ≥20% relative; CEM success rate drops ≥25% relative; gap scales super-linearly from 1-factor to 2-factor to 3-factor holdout.

**Falsification**: If R² drops <10% relative on two-factor holdouts across ≥5 of 7 seeds, the hypothesis is falsified — LeWM compositionally generalizes.

**Statistical test**: Paired t-test (in-distribution vs. holdout) across 7 seeds; bootstrap 95% CI; Cohen's d > 0.8.

---

### H2: SIGReg Promotes Latent Factorization
**Claim**: SIGReg's isotropic Gaussian prior promotes the linear orthogonal factorization required for compositional generalization (Uselis et al., 2026).

**Operationalization**:
- Compare LeWM-SIGReg vs. LeWM-VICReg on the same training split
- Measure: principal angle orthogonality between per-factor latent subspaces; displacement vector consistency (parallelogram test)

**Expected outcome**: SIGReg model shows mean cosine similarity between per-factor subspaces < 0.15; VICReg model shows > 0.25. Orthogonality score correlates with holdout accuracy (Pearson r > 0.6).

**Falsification (neutral)**: If both models show similar orthogonality (difference < 0.05), Gaussian prior is neutral with respect to factorization.

**Falsification (negative)**: If SIGReg model shows *lower* orthogonality than VICReg (contrarian prediction), Gaussian prior actively competes with factorization — a strong negative result.

---

### H3: Predictor is the Generalization Bottleneck
**Claim**: The predictor (not the encoder) encodes domain-specific dynamics and is the primary bottleneck for cross-domain transfer.

**Operationalization**:
- Apply LoRA (rank 4, 8) to: (a) encoder Q/V only, (b) predictor Q/V only, (c) both
- Fine-tune on 50 target-domain trajectories from held-out combinations
- Metric: probing R² and CEM success rate recovery

**Expected outcome**: Predictor-only LoRA recovers ≥70% of in-distribution probing R² with 50 trajectories; encoder-only LoRA recovers ≤40%.

**Falsification**: If encoder-only and predictor-only LoRA perform within 10% of each other, the decomposition hypothesis fails.

---

### H4: Displacement Vector Consistency Predicts Transfer
**Claim**: The parallelogram test (displacement vector consistency) is an independent predictor of compositional transfer beyond principal angle analysis.

**Operationalization**:
- For each factor, compute displacement vector at ≥3 reference points in factor space
- Displacement consistency score: cosine similarity of delta vectors across reference points
- Regress holdout probing accuracy on (1) principal angles alone, (2) principal angles + displacement consistency

**Expected outcome**: Adding displacement consistency increases R² by > 0.1 in the regression, indicating it captures unique variance not in principal angles.

**Falsification**: If partial R² < 0.05, displacement consistency is redundant with principal angles.

---

### H5: Physical Coupling Predicts Regime-Dependent Failure
**Claim**: Physical coupling between factors (not just model quality) is a primary predictor of compositional generalization failure. Failures in strongly-coupled regimes persist across model architectures.

**Operationalization**:
- Estimate coupling strength: |∂²S/∂f_i∂f_j| for trajectory summary statistic S
- Classify holdout combinations as weakly/strongly coupled (threshold c*)
- Test whether coupling predicts holdout failure better than probing accuracy or Grassmannian distance
- Check if strongly-coupled failures persist across LeWM-SIGReg, LeWM-VICReg, and DINO-WM

**Expected outcome**: Coupling strength is a significant predictor (Pearson r > 0.5, p < 0.05) of holdout failure. Strongly-coupled failures persist across all three model variants (inter-model concordance in failure patterns).

**Falsification**: If coupling strength explains no additional variance beyond Grassmannian distance, or if failure patterns are model-specific rather than regime-specific.

---

## Secondary Hypotheses

### H2a: SIGReg Lambda Optimal for Generalization ≠ Optimal for In-Distribution
**Claim**: The SIGReg regularization strength (lambda) that optimizes in-distribution probing accuracy is not the same lambda that optimizes compositional holdout accuracy.

**Operationalization**: Train with lambda ∈ {0.01, 0.05, 0.1, 0.2} (full sweep); measure in-distribution and holdout performance separately.

**Expected outcome**: Optimal lambda for holdout > optimal lambda for in-distribution (stronger Gaussian enforcement helps composition even at the cost of in-distribution performance).

### H2b: Interventional Independence Score Tracks Factorization
**Claim**: The IIS (Interventional Independence Score) correlates with geometric factorization metrics (principal angles) but adds predictive power for behavioral generalization.

**Operationalization**: Compute IIS matrix for all concept pairs. Compute Pearson correlation between IIS and (a) principal angles, (b) holdout probing R², (c) CEM success rate.

**Expected outcome**: IIS correlates moderately with principal angles (r > 0.5), but has stronger predictive power for behavioral generalization (CEM success rate) than principal angles alone.

### H3a: LoRA Rank = Intrinsic Dimensionality of Domain Gap
**Claim**: The minimum LoRA rank for efficient adaptation correlates with the intrinsic dimensionality of the changed factors' latent subspace.

**Operationalization**: Compare minimum rank for 90% recovery vs. number of significant eigenvalues in the per-factor latent subspace covariance.

**Expected outcome**: Pearson r > 0.5 between minimum adaptation rank and factor subspace intrinsic dimensionality.

---

## Pilot Hypotheses (Decision Rules)

### P1: Framework Validation
**Test**: Train LeWM on friction levels {0.5x, 2.0x}, probe on {1.0x} (interpolation) and {3.0x} (extrapolation).

**If linear probe R² drops > 15% on 1.0x**: Single-factor interpolation is challenging → proceed to full multi-factor study.

**If linear probe R² drops < 5% on 1.0x**: Single-factor generalization is robust → multi-factor holdout or extrapolation will be the interesting test; proceed with more aggressive holdout.

**If linear probe R² drops > 40% on 1.0x**: Generalization failure is severe → reduce factor range or focus on interpolation-type holdouts only.

### P2: LoRA Feasibility
**Test**: Apply LoRA-r4 to predictor Q/V only; fine-tune on 50 trajectories from held-out friction level; measure R² recovery.

**If R² recovery > 50%**: LoRA is viable for adaptation diagnostic; proceed with full LoRA sweep.

**If R² recovery < 20%**: LoRA may be too constrained for ViT-Tiny predictor; increase rank range to 32, 64; or focus on full fine-tuning as ceiling.
