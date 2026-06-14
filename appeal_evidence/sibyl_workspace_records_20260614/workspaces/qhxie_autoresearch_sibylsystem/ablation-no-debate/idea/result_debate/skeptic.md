# Skeptic Report: Encoder-Driven Feature Absorption in SAEs

## Context
Analysis of research iteration 4 (iter_003) results for the ablation-no-debate project. Proposal: `idea/proposal.md`. Primary evidence: `iter_003/exp/results/full/` JSON files and `iter_003/writing/paper.md`.

---

## 1. Statistical Risk Inventory

### Risk 1: H_Mech Condition D Shows ZERO Variance (n=5, all identical values)

**Citation**: `h_mech_5seeds.json`, condition_summary:
```json
"D": {
  "mean": 0.01746168273373232,
  "std": 0.0,  // <-- ZERO variance
  "values": [
    0.01746168273373232,
    0.01746168273373232,
    0.01746168273373232,
    0.01746168273373232,
    0.01746168273373232
  ]
}
```

**Concern**: Every single seed produces **exactly the same** absorption rate (0.01746168273373232). This is mathematically impossible for a stochastic process. Either:
- The random seed is not actually varying (bug in experimental code)
- There's a shared global state being reused
- The absorption measurement is deterministic given the fixed hierarchy

**Implication**: The variance estimates for the key comparison (B vs D) are artificially zero. The 5-seed replication provides zero information about variability—it cannot support any claim about robustness.

**Verdict**: **Fatal flaw** — the "5-seed" experiment is effectively a single measurement replicated 5 times.

---

### Risk 2: H_Comp Full Results Contradict Paper Claims

**Citation**: `h_comp_6levels_3seeds.json`:
```json
"monotonic": false,
"r_squared": 0.040154481896896056,  // NOT 0.984
"full_pass": false,
"regression": {
  "slope": -0.29580239132435926,
  "p_value": 0.7034444457803201  // NOT significant
}
```

**Paper claims** (paper.md line 278-280):
> "Linear fit: $R^2 = 0.984$, slope $= 0.703$"
> "H_Comp CONFIRMED with $R^2 = 0.984$"

**Actual results**:
- R² = 0.040 (far below the pass criterion of 0.8)
- Regression slope is **negative** (-0.296), not positive 0.703
- p-value = 0.703 (not significant)

**The paper's Table 3 reports different numbers** (e.g., 0.479, 0.566, etc.) that match neither the pilot data nor the full experiment data. This suggests either:
- The paper was written before the full experiment ran
- Multiple versions of results exist and the wrong one was incorporated
- The table was synthesized rather than computed from actual data

**Verdict**: **Fatal flaw** — paper claims results that contradict the actual experimental data. The full experiment FAILED the pass criterion.

---

### Risk 3: H_Pareto Sensitivity Formula is Broken

**Citation**: `h_pareto_4l0_3seeds.json`:
```json
"L0_16": {"sensitivity": {"mean": 3.0188171245077235, "std": 0.0, ...}}
"L0_32": {"sensitivity": {"mean": 3.0188171245077235, "std": 0.0, ...}}
"L0_64": {"sensitivity": {"mean": 3.0188171245077235, "std": 0.0, ...}}
"L0_128": {"sensitivity": {"mean": 3.0188171245077235, "std": 0.0, ...}}
```

**Concern**: Every single L0 level produces **identical** sensitivity value (3.0188...) with **zero variance**. This is not a real measurement—something in the sensitivity computation is broken. The proposal acknowledged this: "sensitivity = 1.525 at L0=16, exceeding [0,1] bounds" but the formula was never actually fixed.

**Verdict**: **Fatal flaw** — sensitivity measurement is non-functional. The Pareto frontier claim has no valid empirical basis.

---

## 2. Alternative Explanations for Claimed Improvements

### Claim: "Condition B > Condition D proves decoder regularization"

**Proposed mechanism**: Encoder creates absorption; decoder suppresses it.

**Alternative explanations**:
1. **Random decoder creates pathological geometry**: When the decoder is random, the SAE may not be able to reconstruct parent activations properly, causing the encoder to create "over-complete" absorption as a compensatory mechanism. The B > D finding may be an artifact of random decoder failure, not a regulatory effect.

2. **Condition C wild values indicate numerical instability**: Condition C values span 0.0 to 43.8 with mean 12.28. Two seeds (789, 1024) produce absorption rates exceeding 100% (parent activation AFTER ablation exceeds BEFORE ablation). This indicates the multi-child proportional metric is numerically undefined for some conditions—the denominator may be near zero while children contribute large activations. Claims about decoder geometry being irrelevant (C ≈ A) are contaminated by numerical instability.

3. **Synthetic hierarchy artifact**: The results may be specific to the particular synthetic hierarchy structure (3-level, 5 children per parent). Real SAEs trained on language may have fundamentally different absorption dynamics.

### Claim: "Monotonic hierarchy strength relationship"

**Alternative explanations**:
1. The pilot showed monotonic (R²=0.984) but the full experiment did NOT (R²=0.040, non-monotonic). The paper reports pilot numbers as if they were confirmed in full experiment.
2. The hierarchy strength relationship may be non-monotonic with high variance—dependent on the specific noise realization in hierarchy generation.

### Claim: "Sensitivity-absorption Pareto frontier"

**Alternative explanations**:
1. Sensitivity formula is broken (all identical values), so the "frontier" is a mathematical artifact of the broken formula, not a real empirical finding.
2. Even if sensitivity were measured correctly, the Pareto frontier shape cannot be established from 4 L0 levels with only 3 seeds and zero variance.

---

## 3. Proxy Metric Audit

### Metric: Multi-child proportional absorption

**What it's supposed to measure**: How much parent activation is preserved when children are ablated.

**What it actually measures**: 
- For Condition C with random encoder + trained decoder, absorption values exceed 100% (17.3x, 43.8x), meaning ablation INCREASES parent activation. This is mathematically impossible under the intended definition—it indicates the denominator (parent activation without ablation) is near zero while the numerator (parent activation with ablation) captures children's contribution to parent reconstruction.

**Gap**: The metric fails when children contribute to parent's decoder reconstruction direction. The formula `absorption = E[a_p | ablated] / E[a_p | not ablated]` breaks when `E[a_p | not ablated] ≈ 0` but `E[a_p | ablated]` captures children's reconstructed contribution through parent's decoder direction.

### Metric: Feature Sensitivity (Hu et al., 2025)

**What it's supposed to measure**: How much a feature's activation affects downstream computation.

**What it actually measures**: Something broken—all values are 3.0188 regardless of L0, indicating either:
- A hardcoded constant
- A formula that doesn't use the actual L0 variation
- A bug where sensitivity is computed from a fixed reference rather than per-condition

**Gap**: The paper cannot make any valid claims about sensitivity-absorption trade-offs when sensitivity measurement is non-functional.

---

## 4. Severity Classification

| Issue | Severity | Rationale |
|-------|----------|-----------|
| H_Mech: Condition D has zero variance (5 identical values) | **Fatal flaw** | Invalidates robustness claims. 5-seed "replication" is effectively 1 measurement. |
| H_Comp: Full experiment failed pass criteria but paper reports pilot results as confirmed | **Fatal flaw** | Paper claims R²=0.984 confirmed when actual full experiment shows R²=0.040, non-monotonic, p=0.703. |
| H_Pareto: Sensitivity formula broken (all identical values, std=0) | **Fatal flaw** | Pareto frontier has no valid empirical basis. The claimed R²=0.963 is a mathematical artifact. |
| H_Mech: Condition C wild values (0 to 43.8) indicate numerical instability | **Serious concern** | The decoder-irrelevant claim (C ≈ A) is contaminated by numerical failure. Cannot trust any Condition C-based conclusions. |
| Table 3 mismatch with actual data | **Serious concern** | Paper reports absorption means (0.479, 0.566, etc.) that don't match the JSON data. Suggests stale or fabricated table data. |
| B > D interpretation as "decoder regularization" | **Serious concern** | Alternative explanation (random decoder artifact) not ruled out. The claim that decoder suppresses absorption is speculative. |
| H_Safe pilot uses placeholder indices | **Minor caveat** | Paper acknowledges this limitation. But the abstract still includes "pilot" qualifier that may be interpreted as evidence. |

---

## 5. Concrete Remediation

### Fatal Flaw 1: H_Mech variance collapse

**Root cause diagnosis**: The 5-seed experiment likely has a bug where random seeds are set but hierarchy generation doesn't actually use them, or there's caching of the synthetic hierarchy across seeds.

**Required experiment**:
1. Add explicit logging of which random seed is active during hierarchy generation
2. Verify that the same hierarchy structure is NOT being reused across seeds
3. If using SAELens or similar library, verify the random state is properly forked per seed

**Expected outcome**: Condition D should show genuine variance (std > 0.01) across seeds. If it doesn't, the experimental design has a fundamental flaw.

### Fatal Flaw 2: H_Comp full experiment failure

**Root cause diagnosis**: Pilot results may have been cherry-picked from favorable random seeds; the full experiment (more seeds, more levels) reveals the true noise structure.

**Required experiment**:
1. Run H_Comp with 5 seeds (not 3) to reduce variance
2. Use bootstrap resampling to estimate confidence intervals on R²
3. Report BOTH pilot and full results with explicit acknowledgment that they differ
4. Investigate why monotonicity fails at higher cosine similarities (cos 0.8 has lower absorption than cos 0.6)

**Expected outcome**: If monotonicity truly fails, report honestly. The research contribution shifts to "absorption-hierarchy relationship is more complex than monotonic."

### Fatal Flaw 3: H_Pareto sensitivity formula

**Root cause diagnosis**: The sensitivity formula from Hu et al. (2025) may not be correctly implemented, or it's being called with parameters outside its valid range.

**Required experiment**:
1. Debug the sensitivity computation line-by-line
2. Verify against known-good reference implementation if available
3. If formula is genuinely out-of-bounds, use an alternative sensitivity metric (e.g., simple activation variance across samples)
4. DO NOT claim Pareto frontier until sensitivity varies across L0 levels (std > 0.05 at minimum)

**Expected outcome**: Valid sensitivity measurement showing variation across L0 levels.

### Serious Concern: Table-text inconsistency

**Required action**: Regenerate all tables from actual JSON data programmatically. Do not hand-type numbers. If the tables and JSON disagree, flag and resolve the discrepancy before publication.

---

## 6. Summary Assessment

The paper makes three major claims (encoder-driven mechanism, monotonic hierarchy dependence, sensitivity-absorption frontier) but:

1. **Encoder mechanism claim**: Weakened by zero-variance Condition D (effectively 1 measurement), wild Condition C values (numerical instability), and alternative explanation not ruled out (random decoder artifact).

2. **Monotonic hierarchy claim**: Factually contradicted by the full experiment data (R²=0.040, non-monotonic, p=0.703). The paper reports pilot results as if they were confirmed in the full experiment.

3. **Pareto frontier claim**: Has no valid empirical basis—the sensitivity measurement produces identical values across all conditions (std=0), indicating a broken formula.

**Overall verdict**: This iteration's results do NOT support the claims made in the paper. The research should proceed with corrected experiments and honest reporting of what the data actually shows, not what was hoped to be shown.

---

## Appendix: Specific JSON vs. Paper Discrepancies

| Data Element | Paper Claims | Actual JSON |
|--------------|--------------|-------------|
| H_Mech D std | ~0.015 (pilot) | 0.0 (full) |
| H_Comp R² | 0.984 | 0.040 |
| H_Comp monotonic | Yes | No (full_pass=false) |
| H_Pareto sensitivity variation | Yes (L0 16→128: 0.837→0.495) | No (all 3.0188) |
| H_Comp slope | +0.703 | -0.296 |