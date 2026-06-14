# Methodology -- Iteration 11: Data Integrity & Verification (Revised)

## Overview

Iteration 11 is a **data integrity and verification iteration**. Ten prior iterations accumulated substantial empirical evidence but also accumulated data integrity debt. The iter_010 review (score 6.5) identified 5 critical and 8 major issues.

**Phase 0 (Data Integrity) -- COMPLETED:**
- validate_integration.py implemented: 51/53 checks PASS (2 minor FAILs: city-continent strict/compensatory pct within tolerance)
- Aggregation unified: per-token canonical chosen, 21.6% vs 27.1% discrepancy explained (different probes, different experiments)
- Table 3 CI inversions FIXED: 6/6 inversions resolved with per-token bootstrap
- Contributions restructured: #1 probe degradation, #2 causal mechanism, #3 cross-domain characterization
- Headline numbers corrected: 4.1x replaced with 2.7x (quality-gated)

**Phase 1.1 (Patching Spot-Check) -- COMPLETED:**
- 20 city-continent entities sampled (seed=42, stratified by continent)
- Mean recovery: 62.7% (expected 61.9%, diff = +0.8 pp)
- Cohen's d = 2.04, p < 0.001
- VERDICT: Sign reversal correction CONFIRMED

**Remaining Work:**
- Phase 1.2: City-country patching (or documented exclusion)
- Phase 1.3: First-letter patching at L24 for Figure 4
- Phase 2: Writing revisions (probe degradation reframe, Section 7.2 repetition, figures, abstract)
- Phase 2.7: Final validation pass
- Phase 3 (optional): City-continent probe degradation, rate-distortion validation

---

## Phase 1: Verification Experiments (Remaining ~1-1.5 GPU-hours)

### 1.2 City-Country Activation Patching (or Documented Exclusion)

**Rationale:** City-country has the highest absorption (45.1%) but was excluded from patching without explanation (F1=0.73 below quality gate).

**Method (Option A -- preferred if time permits):**
1. Run city-country activation patching at L24 despite low probe quality (F1=0.73).
2. Report result with explicit caveat: "probe quality below gate; recovery measurement may be confounded."
3. If recovery is positive (d > 0.3): strengthens universal claim.
4. If recovery is null: documented as consequence of probe quality, not as absence of mechanism.

**Method (Option B -- fallback if time-constrained):**
1. Add explicit paragraph in Section 5.2 explaining exclusion.
2. Document: "City-country was excluded from activation patching because its probe quality (F1=0.73) falls below the relaxed gate. False negatives in the probe itself confound the recovery measurement."

**Pass criteria (Option A):** Patching completes. Result reported with appropriate caveats.

### 1.3 First-Letter Activation Patching at L24

**Rationale:** Figure 4 has layer mismatch: first-letter at L12, RAVEL at L24. Undermines visual comparison.

**Method:**
1. Re-run first-letter activation patching at L24 (currently only L12 data exists) using Gemma Scope L24 16k SAE.
2. Use same protocol as RAVEL patching: zero top child feature, measure parent probe recovery.
3. Generate updated Figure 4 with all three panels at L24.

**Pass criteria:** Patching completes. Recovery rate consistent with L12 result (qualitatively: positive recovery, d > 0.5). Figure 4 updated with consistent layers.

---

## Phase 2: Writing Revisions (zero GPU, ~6 hours)

### 2.1 Reframe Probe Degradation Section

**Method:**
1. Present linear fit as primary result; quadratic as exploratory only.
2. Emphasize perfect monotonicity (rho=-1.0) over R^2 (7 points, 2 params, overfitting risk).
3. Explicitly note binary-to-multiclass extrapolation limitation.
4. Reframe from "quantitative decomposition" to "directional evidence that probe quality is a major confound."

### 2.2 Eliminate Section 7.2 Repetition

**Method:**
- Section 5: keep full numerical detail (recovery rates, effect sizes, p-values).
- Section 7.2: discuss implications only, reference Section 5 for numbers.
- Section 8: one-sentence summary without per-hierarchy numbers.

### 2.3 Fix Remaining Figures

**Method:**
- Figure 5: ensure both distributions visible (first-letter and city-continent), reconcile N=1471 vs N=1464.
- Figure 6: reconcile p=0.063 (figure) vs p=0.041 (text).

### 2.4 Reduce Abstract Density

**Method:** Limit to 5-7 key numerical results. Move remaining numbers to introduction.

### 2.5 Minor Fixes

**Method:**
- Remove hype words ("dominant" -> "widely adopted", "amplify" -> "are relevant to").
- Add SAEBench dual-inclusion acknowledgment in Related Work.
- Add code availability statement.
- Add version pinning for all libraries (SAELens v6, TransformerLens, sae-spelling, RAVEL).

### 2.6 Update Proposal Numbers

**Method:** Reconcile proposal.md with paper.md on all numerical claims using data_manifest.json.

### 2.7 Final Validation Pass

**Method:**
1. Run validate_integration.py on final paper.md.
2. If any mismatches remain, fix them.
3. Generate final validation_report.json documenting: total claims checked, claims passing, claims fixed this iteration, remaining known limitations.

---

## Phase 3: Optional Strengthening Experiments (~1-2 GPU-hours)

### 3.1 City-Continent Probe Degradation (Optional)

**Rationale:** Validates cross-domain transferability of probe degradation curve.

**Method:**
1. Degrade city-continent multi-class probes to 3-4 F1 levels using same noise injection protocol.
2. Measure absorption at each degraded F1 level.
3. Compare slope with first-letter degradation curve.

**Pass criteria:** Slope within 2x of first-letter curve validates extrapolation. If slope differs substantially, reframe Section 4.6.

### 3.2 Rate-Distortion Full Validation (Optional)

**Rationale:** Tests H9 (Spearman rho > 0.5 for per-pair absorption prediction).

**Method:**
1. Compute per-pair predictors (cos_sim, co-occurrence, reconstruction importance) across all hierarchy types.
2. Fit three-factor model: cos_sim x co-occurrence / reconstruction_importance.
3. Evaluate Spearman rho on pooled data.

**Pass criteria:** rho > 0.5 and p < 0.05. If rho < 0.3, document as additional negative result.

---

## Model & SAE Specifications

| Component | Specification |
|-----------|---------------|
| Language Model | Gemma 2 2B (google/gemma-2-2b) |
| SAEs | Gemma Scope pre-trained (various L0/layers) |
| Primary SAE | L24 16k (layer 24, 16384 features) |
| Probe Training | Linear probes, sklearn LogisticRegression |
| Activation Patching | TransformerLens hook-based intervention |
| RAVEL Dataset | hij/ravel (HuggingFace) |
| First-Letter Task | sae-spelling library |
| Libraries | SAELens v6, TransformerLens, sae-spelling |

## Baselines

All experiments are CHARACTERIZATION, not improvement. The baseline is the existing SAE behavior without intervention:
- **Absorption measurement baseline:** Raw absorption rate = 1 - (SAE feature firing rate for parent when child active)
- **Activation patching baseline:** Control condition = zero a random non-child feature (same procedure, random target)
- **Probe degradation baseline:** F1=1.0 (original probe) absorption rate as the undegraded reference

## Evaluation Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| Absorption Rate | % of absorbed tokens (parent fails to fire when child active) | H1, H2', H10 |
| Cohen's d | Effect size for patching recovery vs. control | H7 |
| Spearman rho | Rank correlation for predictor validation | H9, H10 |
| Bootstrap 95% CI | Per-token stratified bootstrap | All rates |
| Kruskal-Wallis H | Non-parametric test for cross-domain differences | H1 |
| Bonferroni-corrected p | Pairwise significance with family-wise error control | H1 |
| R^2 | Variance explained by probe degradation model | H10 |

## Expected Visualizations

- **Figure 4 (Updated):** Three-panel activation patching comparison, ALL panels at L24 (first-letter, city-continent, city-language). Paired dot plot: child-zeroed recovery vs. control.
- **Table 2 (Updated):** Cross-domain absorption rates with per-token aggregation. CI contains point estimate.
- **Table 3 (Updated):** Probe degradation F1 levels. All CIs recomputed with per-token aggregation.
- **Figure 5 (Fixed):** Dual-distribution histogram (first-letter + city-continent decoder entanglement).
- **Figure 6 (Fixed):** Architecture comparison bar chart with reconciled p-value.

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| First-letter L24 patching contradicts L12 | Medium | 15% | Report both; note layer-dependence as finding. |
| City-country patching fails (no absorbed pairs) | Low | 10% | Fall back to documented exclusion paragraph. |
| validate_integration.py reveals new errors | Medium | 30% | Fix all before declaring complete. |

## Time and Resource Budget

| Phase | Tasks | GPU-hours | Wall-clock | Priority |
|-------|-------|-----------|------------|----------|
| Phase 0 | Data integrity (COMPLETED) | 0 | ~5 hr (done) | P0 (DONE) |
| Phase 1.1 | Patching spot-check (COMPLETED) | 0.5 | 0.5 hr (done) | P1 (DONE) |
| Phase 1.2 | City-country patching | 0-0.5 | 0.5 hr | P1 |
| Phase 1.3 | First-letter L24 patching | 0.5 | 0.5 hr | P1 |
| Phase 2 | Writing revisions (2.1-2.7) | 0 | ~6 hr | P1 |
| Phase 3 | Optional experiments | 1-2 | ~2 hr | P3 |
| **Total Remaining** | | **~1-3** | **~9 hr** | |

**Hardware:** Single GPU >= 24GB VRAM (Gemma 2 2B fits comfortably). compute_backend: local.
