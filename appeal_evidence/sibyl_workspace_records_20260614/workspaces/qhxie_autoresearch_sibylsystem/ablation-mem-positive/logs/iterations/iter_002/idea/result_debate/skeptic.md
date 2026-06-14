# Skeptic Analysis: Phase Transitions in SAE Feature Absorption (iter_005)

## 1. Statistical Risk Inventory

### Risk 1: Finite-Size Scaling Has Only 3 Degrees of Freedom (SERIOUS)
**Concern**: H2 claims scaling collapse with ν=3, R²=0.951 using **only 3 dictionary sizes** (6144, 12288, 24576). Fitting a power-law scaling relation with 2 free parameters (ν, proportionality constant) on n=3 points guarantees R²≈1 regardless of the true physics.

**Specific Evidence**: From `dict_size_sweep.json` — 3 data points, best collapse at ν=3.

**Why This Is Unreliable**: With 3 points you can always rotate a 2-parameter model to pass through all 3 exactly. The R²=0.951 is not a measure of predictive accuracy — it's a measure of fit to the same data used to select ν.

**Concrete Issue**: If the paper claims "first quantitative measurement of scaling law in SAE absorption", this requires out-of-sample validation. A 4th dictionary size (e.g., 32k or 8k) would test whether the ν=3 scaling generalizes.

**Verdict**: **Serious concern** — the scaling law is a mathematical artifact of fitting 2 parameters to 3 points.

---

### Risk 2: Pilot-to-Full λ_c Instability — 10x Shift (SERIOUS)
**Concern**: Pilot identified λ_c ≈ 5e-4 (susceptibility χ=1.38). Full experiment finds λ_c = 5e-5 (susceptibility χ=11.19). The critical point shifted by 10x and susceptibility increased 8x when sample size went from n=100 to n=1000.

**Specific Evidence**:
- Pilot (`pilot_summary.json`): peak_lambda = 0.0005, susceptibility_peak = 1.38
- Full (`sparsity_sweep_full.json`): critical_lambda = 5e-05, max_susceptibility = 11.189

**Why This Is Unreliable**: The "critical point" is supposedly a physical property of the SAE, not a function of sample size. If λ_c truly represents a phase transition in the feature space, it should be stable across sample sizes. The 10x instability suggests the "peak" is an artifact of the finite-difference calculation or noise sensitivity.

**Concrete Issue**: χ = dm/dλ is a derived quantity — the finite difference between adjacent λ values. With n=1000, the smoothing is much greater than n=100, which could shift where the discrete peak appears.

**Verdict**: **Serious concern** — the "critical lambda" is not a well-defined physical quantity; it's sensitive to sample size and discrete approximation method.

---

### Risk 3: Revised Formula for H5 Is Post-Hoc (SERIOUS)
**Concern**: H5 claims r=0.647 for "revised formula" vs baseline r=-0.52. The baseline formula presumably failed to produce a positive correlation, so the formula was revised post-data. This is data-snooping — the revision was chosen to make the correlation positive after observing the data.

**Specific Evidence**:
- "revised formula achieves r=0.647 vs baseline r=-0.52" (proposal.md)
- p_value = 3.58e-261

**Why This Is Unreliable**: p=3.58e-261 is effectively p=0 — numerical underflow. With n=314 (E2 meta-analysis), any correlation will achieve "significance" at the expense of practical meaning. The improvement of Δr=1.167 (from -0.52 to +0.647) is implausibly large for a formula revision.

**Concrete Issue**: Without pre-registration of the formula, r=0.647 could be entirely due to overfitting to the specific E2 dataset. A hold-out validation would test this.

**Verdict**: **Serious concern** — r=0.647 is a post-hoc result requiring hold-out validation to be credible.

---

## 2. Alternative Explanations for Key Claims

### For H1 (Quasi-Critical Threshold, chi_ratio=1.88):
**Alternative 1**: The susceptibility peak is an artifact of the finite-difference approximation. When you compute χ = dm/dλ on a discrete grid, any smooth function will have local maxima depending on step size.

**Alternative 2**: The "peak" at λ=5e-5 reflects optimization stability at certain sparsity levels (TopK gating threshold), not a physical phase transition in feature space.

**Alternative 3**: chi_ratio=1.88 < 3.0 means the transition is gradual. Calling it "quasi-critical" is framing a falsified hypothesis (sharp transition) as a discovery.

---

### For H2 (Finite-Size Scaling with ν=3, R²=0.951):
**Alternative 1**: n=3 dictionary sizes provide zero degrees of freedom for testing — any 2-parameter model will fit perfectly.

**Alternative 2**: Dictionary size is confounded with training dynamics (more capacity → different absorption profiles), not a physical "system size" in the statistical physics sense.

**Alternative 3**: The "collapse quality" metric is ad hoc — no theoretical basis for why this particular R² metric measures scaling validity.

---

### For H4 (CV Reversed — Absorbed Have Higher CV):
**Alternative 1**: CV = σ/μ is inflated when μ ≈ 0. Absorbed features may have near-zero mean activation but high variance (sparse, context-dependent firing), inflating CV by division by a small number.

**Alternative 2**: The steering test (pilot_steering_cv) uses only ±3 steering strength — this does not replicate Basu et al.'s setup (98.2% AUROC → 0% output change). The "2x effect" may not translate to actionability.

**Alternative 3**: Selection artifact — features classified as "absorbed" are rare/specialized, which inherently creates high variance (spike-like behavior).

---

## 3. Proxy Metric Audit

| Claimed Contribution | What Is Actually Measured | Gap |
|----------------------|---------------------------|-----|
| "Critical threshold at λ_c ≈ 5e-5" | Peak of dm/dλ on 12-point sparsity sweep (n=1000) | Peak location shifts 10x between pilot (n=100) and full (n=1000) |
| "Finite-size scaling ν=3, R²=0.951" | Scaling collapse with only 3 dictionary sizes | n=3 provides no out-of-sample test |
| "CV_reversed (absorbed CV=7.33 >> non-absorbed CV=0.01)" | CV across 1000 samples per feature group | CV inflated for low-mean features; non-absorbed n=0 at λ=0.001 |
| "Steering effectiveness: high-CV 2x more steerable" | Mean steering effect at ±3 strength | Single steering strength; does not replicate Basu setup |

**Critical Gap**: The proposal claims connection to Basu et al.'s actionability paradox (98.2% AUROC → 0% output change), but the steering experiment does not measure this paradox directly. The CV-based decomposition is a hypothesis about mechanism, not a measurement.

---

## 4. Severity Classification

| Issue | Hypothesis | Severity | Evidence |
|-------|------------|----------|----------|
| λ_c instability (10x pilot→full shift) | H1 | **Serious** | pilot λ_c=5e-4, full λ_c=5e-5 |
| n=3 for scaling law fit | H2 | **Serious** | 3 dictionary sizes, 2 parameters |
| Post-hoc formula revision | H5 | **Serious** | r=0.647 vs baseline r=-0.52 |
| CV inflated for near-zero-mean features | H4 | **Minor** | CV=σ/μ undefined when μ≈0 |
| Steering test doesn't replicate Basu | H4 (actionability) | **Serious** | Single ±3 strength, not multi-probe |
| chi_ratio below threshold (1.88 < 3.0) | H1 | **Minor caveat** | Reframed as "quasi-critical" |

**No Fatal Flaws** — unlike iter_004 (which had H4 compromised by absorption_rate=1.0 saturating all layers), iter_005 has:
- Activation patching validation (67.3% mean recovery) confirming genuine causal effects
- Steering test showing high-CV features are 2x more steerable (pilot_steering_cv passed)

These are real signals, not methodological artifacts.

---

## 5. Concrete Remediation

### For H1 (λ_c Instability): Multi-Seed Validation
**Experiment needed**:
- Run the full sparsity sweep (12 λ values, 1000 samples) with 5 different random seeds
- Compute λ_c for each seed independently
- Report: mean λ_c, std(λ_c), and range

**Expected outcome**: If σ(λ_c) < 0.5 decades, the critical point is stable. If σ(λ_c) > 1 decade, the "peak" is noise-driven.

---

### For H2 (Scaling Law): Out-of-Sample Dictionary Size
**Experiment needed**:
- Add a 4th dictionary size (e.g., 32,768 or 8,192) outside the training range
- Compute transition width δλ for this new size
- Verify δλ ∝ N^(-1/3) holds including the new point

**Expected outcome**: If new point falls within 95% prediction band, scaling law is validated. If not, n=3 fit was overfitted.

---

### For H5 (Post-Hoc Formula): Pre-Registration or Hold-Out
**Experiment needed**:
- Pre-specify the exact co-occurrence formula before analysis
- OR split E2 data (n=314) into train/test (n=157 each)
- Test on held-out data only

**Expected outcome**: If pre-specified formula achieves r > 0.5 on held-out data, the mechanism is credible.

---

### For H4 (CV vs Actionability): Direct Basu Replication
**Experiment needed**:
- Use SAEBench probe projection metric (same as Basu et al.) for feature detection
- Measure both detection AUROC and steering effectiveness for the same features
- Compute actionability gap: (steering effect) / (AUROC - 0.5)

**Expected outcome**: If high-CV shows both high detection AND high steering, CV-based decomposition explains Basu's paradox. If detection is high but steering is zero (as Basu found), the paradox remains.

---

## 6. Summary Assessment

| Hypothesis | Status | Main Concern |
|------------|--------|--------------|
| H1 (Critical Threshold) | WEAKLY SUPPORTED | λ_c shifts 10x between pilot and full (instability) |
| H2 (Finite-Size Scaling) | UNVERIFIED | n=3 provides no out-of-sample validation |
| H3 (Layer Critical) | NOT_SUPPORTED | Correctly reported (absorption saturates at λ=0.001) |
| H4 (CV Difference) | SUPPORTED (with caveats) | Direction reversed confirmed; CV may be inflated for low-mean features |
| H5 (Co-occurrence) | SUSPICIOUS | Post-hoc formula revision, no hold-out validation |
| H6 (Graph Topology) | NOT_SUPPORTED | Correctly reported |

**Key Improvement Over iter_004**: The activation patching validation (67.3% mean recovery across 9 words) is genuine causal evidence, not a metric artifact. This validates the core research direction.

**Overall Verdict**: Proceed with full experiments, but the theoretical contributions (critical phenomena framework, finite-size scaling) rest on weak statistical foundations. Focus on empirical discoveries (CV reversal validated by steering, activation patching confirming genuine absorption) for a mid-tier venue with honest scope.

**Pre-registration requirement**: Before claiming scaling law ν=3 or information bottleneck mechanism, run the remediation experiments (multi-seed λ_c stability, out-of-sample dictionary size, hold-out co-occurrence). These are tractable within the ~3 hour GPU budget.