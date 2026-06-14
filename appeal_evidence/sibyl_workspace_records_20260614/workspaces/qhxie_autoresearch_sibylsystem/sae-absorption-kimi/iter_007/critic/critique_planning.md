# Planning Critique: Iteration 009 Methodology

## Overall Assessment

The methodology is well-structured and addresses key confounds, but contains several design flaws that limit the validity of conclusions.

## Critical Issues

### 1. L0-Matching Protocol Incomplete (CRITICAL)

The protocol states: "Train Baseline L1 with tuned lambda to approach [each variant's] L0."

**Problem**: The lambda sweep in the full experiment only went up to 0.002, achieving L0~995. The pilot data shows lambda=0.005, 0.01, and 0.02 all achieve L0=50.0. The protocol was not followed through to its logical conclusion.

**Impact**: The paper falsely claims L0-matching is "impossible" when the pilot data shows it is possible. This undermines the central contribution.

**Fix**: Extend the lambda sweep to include 0.005, 0.01, and 0.02. Report the L0-matched comparison at these levels.

### 2. Downstream Metrics Inadequate (CRITICAL)

The methodology lists three downstream metrics:
1. Sparse probing F1
2. Steering efficacy
3. Circuit-tracing precision

**Problem**: Only MCC is reported, and MCC is at chance level for all variants including Random. The methodology does not explain why MCC was chosen over the listed metrics, or what MCC measures.

**Impact**: The causal inference rests on a single metric that cannot discriminate trained from random SAEs.

**Fix**: Either (a) report all three promised metrics, or (b) justify MCC as the primary metric and demonstrate its validity.

### 3. Statistical Power Analysis Misleading (MAJOR)

The proposal states: "With 5 seeds x 4 variants = 20 data points per condition, detectable effect size d >= 0.9 at 80% power."

**Problem**: This power analysis assumes 20 independent observations, but the 5 seeds per variant are replicates of the same condition, not independent conditions. The effective sample size for cross-variant comparisons is 5 (seeds) per variant, not 20.

**Impact**: The study is underpowered for detecting small-to-medium effect sizes (d < 0.9).

**Fix**: Correct the power analysis. With n=5 per variant, detectable effect size at 80% power is d >= 2.0 (very large). Acknowledge limited power.

### 4. Control Design Issues (MAJOR)

**Random SAE**: The Random control is described as "untrained dictionary" but the methodology does not specify how absorption is computed for it. If Random achieves MCC=0.222 (same as trained), it is not a useful control.

**Shuffled labels**: The methodology promises shuffled label controls but no results are reported.

**Fix**: Clarify Random SAE purpose (is it for absorption baseline or MCC baseline?). Report shuffled label results.

### 5. Data Integrity Pipeline Not Evidenced (MAJOR)

The methodology promises five mandatory checks:
1. Feature count validation
2. Convergence verification
3. Cross-seed independence (MD5 hashes)
4. Output file audit
5. Numerical provenance

**Problem**: No evidence of these checks is present in the outputs. No MD5 hashes, no convergence diagnostics, no audit trail.

**Fix**: Provide the data integrity check output as supplementary material.

### 6. Scope Reduction Not Acknowledged (MINOR)

The proposal plans for 16k features in full experiments, but all experiments use 1024 features. This is a significant scope reduction.

**Fix**: Explicitly state the scope reduction and discuss generalizability implications.

## Design Strengths

1. **Ground-truth synthetic data**: Eliminates probe calibration bias
2. **5 random seeds**: Provides replication (though power is limited)
3. **Pilot-first approach**: Methodologically sound (though not fully executed)
4. **Pre-registered analysis plan**: Reduces p-hacking risk

## Recommendations

1. Complete the L0-matching protocol with lambda up to 0.02
2. Add valid downstream metrics or demonstrate MCC validity
3. Correct the power analysis
4. Provide evidence of data integrity checks
5. Report all promised controls and ablations
