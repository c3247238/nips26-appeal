# Result Debate Synthesis: Activation Energy Theory (Round 4)

**Date**: 2026-04-29
**Model**: Qwen2.5-Math-7B-Instruct on MATH dataset
**Sample**: n=50 (full), n=30 (pilot)

---

## 1. Consensus Map: Where All 6 Perspectives Agree

These are high-confidence conclusions that survive scrutiny across all viewpoints:

### 1.1 H1 (Arrhenius Kinetics at Aggregate Level) — AGGREGATE-ONLY CONFIRMED
- **All agree**: The exponential saturation model `P_k = P_inf * (1 - exp(-k/k_0))` fits the aggregate accuracy curve well (R^2 = 0.924).
- **All agree**: This is an aggregate-level phenomenon only. Per-problem fit is catastrophic (median R^2 = 0.000, mean = 0.077, 80% of problems fail to fit).
- **All agree**: The mathematical form is equivalent to Yang et al. (2025) — no mathematical novelty can be claimed.

### 1.2 H2 (Ea Correlates with MATH Level) — MARGINAL CONFIRMATION
- **All agree**: Spearman(Ea, Level) = 0.448, p = 0.001 is statistically significant.
- **All agree**: The effect size is modest, and Ea values are highly concentrated (bimodal at ~9.47 and ~10.0), limiting discriminative power.
- **All agree**: Level 5 Ea values are numerically compressed (std ~1.9e-6), suggesting ceiling effects or algorithmic artifacts.

### 1.3 H3 (Low-Ea Routing) — FALSIFIED
- **All agree**: Despite the "75.0%" threshold pass, Ea has zero predictive power for single-pass correctness.
- **All agree**: AUC = 0.436 < 0.5 is decisive evidence of failure.
- **All agree**: Spearman(Ea, accuracy) = -0.063, p = 0.66 confirms no relationship.
- **All agree**: The threshold search was post-hoc (data leakage), making the 75% figure unreliable.

### 1.4 H5 (Ea vs k_0 Consistency) — FALSIFIED
- **All agree**: Spearman(Ea, k_0) = -0.219 with only 10/50 valid pairs means the two "activation energy" measures are unrelated.
- **All agree**: This undermines the theoretical unity of the framework.

### 1.5 Methodological Weaknesses — UNIVERSALLY ACKNOWLEDGED
- **All agree**: Sample sizes are too small for physical-law claims (n=50 for aggregate, n=30 for pilots).
- **All agree**: Sample mismatch across G1/G2/G3 compromises cross-experiment validity.
- **All agree**: Answer extraction has systematic errors (truncated answers, empty extraction_methods).
- **All agree**: Model switching (DeepSeek -> Qwen) invalidates cross-round comparisons.

---

## 2. Conflict Resolution: Where Perspectives Disagree

### 2.1 Optimist vs Skeptic/Methodologist on H1

| Perspective | Position |
|-------------|----------|
| **Optimist** | "R^2=0.924 confirms the theory framework. This is physically reasonable and statistically significant." |
| **Skeptic** | "R^2=0.924 is a mathematical artifact of fitting 2 parameters to 5 points. Per-problem median R^2=0 proves the theory fails at the individual level." |
| **Methodologist** | "H1 CONFIRMED should be downgraded to WEAK. Aggregate fit smooths away individual variation — this is ecological fallacy." |

**Resolution**: The Skeptic and Methodologist are correct. The aggregate R^2 is not evidence of a physical law; it is a statistical smoothing of 50 problems' noise. The per-problem failure rate (80% cannot fit) is the decisive evidence. However, the aggregate pattern itself is real and replicable — it just should not be over-interpreted as a physical law.

**Judgment**: H1 is **CONFIRMED as a descriptive aggregate pattern**, but **REJECTED as a universal physical law**.

### 2.2 Optimist vs Strategist on H3

| Perspective | Position |
|-------------|----------|
| **Optimist** | "Low-Ea accuracy = 75.0% meets the threshold. Routing strategy is feasible with 25pp gap." |
| **Strategist** | "H3 is BORDERLINE at best. Pilot showed 68.4% (FAIL). The 75% is a statistical artifact of bimodal Ea distribution." |

**Resolution**: The Strategist is right. The 75% figure is a post-hoc threshold optimization on bimodal data. The AUC = 0.436 and Spearman = -0.063 are the true indicators, and both show zero predictive power. The 25pp gap between low-Ea (75%) and high-Ea (50%) is driven by the threshold cutting between two clusters, not by genuine predictive signal.

**Judgment**: H3 is **FALSIFIED**.

### 2.3 Comparativist vs Skeptic on Contribution Value

| Perspective | Position |
|-------------|----------|
| **Comparativist** | "We occupy a unique position in the literature: first Qwen2.5-Math-7B saturation parameters, first Ea-MATH level correlation, first systematic quantification of Ea routing ceiling." |
| **Skeptic** | "These 'contributions' are parameter estimates on n=50 with broken extraction infrastructure. The AUC<0.5 'ceiling' is not a contribution — it's a failure." |

**Resolution**: Both have valid points. The Comparativist correctly identifies that even negative results have value when they cross-validate ACAR and quantify a known ceiling. The Skeptic correctly notes that the empirical foundation is too weak to support strong claims. The synthesis: the **narrative framing** ("limits of Ea routing") is valuable and well-positioned in the literature, but the **empirical claims must be heavily qualified**.

**Judgment**: The contribution is **valid as a negative-result / boundary-delineation paper**, not as a theory-proposing paper.

### 2.4 Strategist vs Revisionist on Next Steps

| Perspective | Position |
|-------------|----------|
| **Strategist** | "Proceed with Plan A: negative-result paper + H4 error classification + light H5 exploration." |
| **Revisionist** | "Introduce new hypotheses H6-H8 (bimodal distribution, ensemble vs individual, power-law alternative)." |

**Resolution**: The Strategist's Plan A is the correct immediate path. The Revisionist's H6-H8 are interesting theoretical directions but require additional experiments that may not be justified given the weak empirical foundation. H4 (error classification) is the highest-value next step because it directly explains *why* H3 failed.

**Judgment**: **Proceed with Strategist Plan A**, incorporating Revisionist's theoretical framing (Ea measures stability, not correctness) into the paper narrative.

---

## 3. Result Quality Score: 4/10

### Scoring Breakdown

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Statistical Rigor | 2/10 | Sample mismatch, data leakage in H3, per-problem fit failures, no cross-validation |
| Reproducibility | 3/10 | Missing temperature, seed, extraction method metadata; answer extraction errors |
| Theoretical Coherence | 4/10 | Aggregate pattern is real, but physical analogy is overextended; Ea != k_0 |
| Novelty | 5/10 | No mathematical novelty (Yang et al. collision), but negative-result framing is underexplored |
| Literature Positioning | 6/10 | Good alignment with ACAR, Li, RASC; clear differentiation from training-based methods |
| Actionability | 5/10 | H4 could yield actionable insights; H1/H2 provide descriptive value |

### Why Not Lower
- The aggregate Arrhenius pattern (R^2=0.924) is replicable and aligns with Yang et al.
- H2's Spearman=0.448 is a genuine finding, even if modest.
- The negative-result framing ("Ea cannot predict single-pass accuracy") is a real contribution that saves community resources.

### Why Not Higher
- 80% per-problem fit failure undermines any "physical law" claim.
- AUC=0.436 is decisive evidence of predictive failure.
- Data leakage in H3 threshold search is a serious methodological flaw.
- Sample sizes (n=30-50) are inadequate for the claims made.

---

## 4. Key Findings: What We Actually Learned

1. **Aggregate Arrhenius saturation is real but not a physical law**: The exponential saturation model describes group-average behavior well (R^2=0.924), but 80% of individual problems cannot be fit. This is a statistical regularity, not a universal mechanism.

2. **Ea correlates with MATH Level but has no predictive power for correctness**: Spearman(Ea, Level)=0.448 confirms Ea captures some difficulty information, but Spearman(Ea, accuracy)=-0.063 and AUC=0.436 prove it cannot predict whether a single pass will be correct.

3. **Consistency measures stability, not correctness**: The Revisionist's H6 is the key insight. Ea's bimodal distribution (~9.47 and ~10.0) reflects answer stability (consistency of responses), not answer correctness. Stable wrong answers exist.

4. **Two "activation energy" measures are unrelated**: Consistency-derived Ea and saturation-derived k_0 are uncorrelated (Spearman=-0.219), meaning they measure different constructs. The theoretical framework lacks internal consistency.

5. **The "agreement-but-wrong" ceiling is ~25pp for Ea routing**: This is higher than ACAR's 8pp for variance-based routing, suggesting the signal type significantly affects the irreducible error floor.

---

## 5. Methodology Gaps: Critical Improvements Needed

### 5.1 Immediate (Required for Any Paper)

| Gap | Severity | Fix |
|-----|----------|-----|
| **Data leakage in H3 threshold** | CRITICAL | Re-run H3 with pre-registered threshold or cross-validation |
| **Sample mismatch G1/G2/G3** | CRITICAL | Use identical problem sets across all experiments |
| **Answer extraction audit** | HIGH | Record extraction method; manually audit 10% of cases |
| **Missing metadata** | HIGH | Report temperature, sampling method, seed, judge criteria |
| **Per-problem fit failure** | HIGH | Report per-problem R^2 distribution, not just aggregate |

### 5.2 For Future Experiments

| Gap | Fix |
|-----|-----|
| Sample size | Minimum 250 problems (50 per MATH Level) |
| Cross-validation | All thresholds optimized on train set, reported on test set |
| Alternative models | Compare Arrhenius vs binomial `1-(1-p)^k` vs power-law |
| Effect sizes | Report confidence intervals and Cohen's d, not just p-values |
| Error classification | Systematic annotation of failure modes (H4) |

---

## 6. Competitive Position: Where We Stand vs SOTA

### 6.1 What We Cannot Claim (Already Covered)

| Claim | Covered By |
|-------|-----------|
| Exponential saturation formula | Yang et al. (2025) — exact mathematical equivalence |
| Self-consistency voting | Wang et al. (2022) |
| Confidence/entropy routing | CGES, LEASH, CoDE-Stop |
| Early stopping | RASC, CGES, CoDE-Stop |

### 6.2 What We Can Claim (Literature Gaps)

| Claim | Evidence | Value |
|-------|----------|-------|
| Qwen2.5-Math-7B saturation parameters | P_inf=0.835, k_0=0.613 (n=50) | First reported for this model |
| Ea-MATH Level correlation | Spearman=0.448, p=0.001 | First on MATH difficulty分层 |
| Ea routing ceiling quantification | Low-Ea accuracy 75%, AUC=0.436 | First systematic measurement |
| Ea vs k_0 decoupling | Spearman=-0.219 | Reveals two "difficulty" constructs |

### 6.3 Literature Dialogue Strategy

- **Yang et al.**: "We adopt the exponential saturation framework established by Yang et al. and provide the first model-specific parameter estimates for Qwen2.5-Math-7B."
- **ACAR**: "While ACAR reports an 8pp agreement-but-wrong ceiling for variance-based routing, we find a larger ~25pp ceiling for convergence-rate-based routing, suggesting signal type significantly affects the irreducible floor."
- **Li (2026)**: "Our finding that 25% of low-Ea problems remain unsolvable aligns with Li's Error Depth Hypothesis — these may correspond to 'deep errors' that resist sampling-based correction."

---

## 7. Hypothesis Update: Survived, Revised, Abandoned

### 7.1 Abandoned Hypotheses

| Hypothesis | Status | Reason |
|------------|--------|--------|
| H3: Low-Ea single-pass accuracy >75% | **FALSIFIED** | AUC=0.436, Spearman=-0.063 |
| H5: Ea matches k_0 | **FALSIFIED** | Spearman=-0.219, p=0.54 |
| Ea as routing signal | **FALSIFIED** | Zero predictive power for correctness |
| Arrhenius as physical law | **ABANDONED** | 80% per-problem fit failure |

### 7.2 Revised Hypotheses

| Original | Revised |
|----------|---------|
| H1: Arrhenius is a universal law | H1': Arrhenius is an aggregate statistical pattern |
| H2: Ea precisely reflects difficulty | H2': Ea coarsely reflects difficulty (Level-discriminative, not within-Level) |
| Ea measures "activation energy" | Ea measures "answer stability" (consistency of responses) |

### 7.3 New Hypotheses (From Revisionist)

| Hypothesis | Evidence | Test |
|------------|----------|------|
| H6: Ea bimodality reflects two reasoning modes | Ea clusters at ~9.47 and ~10.0 | Gaussian mixture model fit |
| H7: Aggregate != Individual | R^2_agg=0.924 vs R^2_per-prob=0.000 | Quantify ensemble-average deviation |
| H8: Power-law may fit individuals better | Power-law wins 24/50 BIC | Formal model comparison on larger sample |

---

## 8. Action Plan: Prioritized Next Steps

### 8.1 Recommended Path: Negative-Result Paper (Plan A)

**Paper Title**: *The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning*

**Core Contributions**:
1. Independent validation of aggregate Arrhenius kinetics on Qwen2.5-Math-7B (R^2=0.924)
2. First systematic falsification of Ea-based problem-level routing (AUC=0.436)
3. Diagnosis: Ea measures stability, not correctness (H6)
4. Decoupling evidence: consistency-Ea and saturation-k_0 are unrelated (H5)

### 8.2 Immediate Action Items

| Priority | Task | Time | Success Criteria | Go/No-Go |
|----------|------|------|------------------|----------|
| **P0** | H4 Error Classification | 30 min | Classify 50 low-Ea failures into execution/conceptual/extraction errors | If conceptual errors >50% -> supports deep-error narrative; else -> pivot |
| **P1** | Fix H3 with pre-registered threshold | 20 min | Re-run H3 with threshold=median(Ea) or Level-based split; report AUC | If AUC still <0.5 -> confirms failure; if AUC >0.5 -> re-evaluate |
| **P2** | Answer extraction audit | 20 min | Manually check 10% of extractions; fix extraction pipeline | Required for any paper |
| **P3** | Paper outline | 30 min | Draft outline with negative-result narrative | Proceed to writing if P0-P2 support narrative |

### 8.3 Decision Gates

**Gate 1 (After P0)**:
- If H4 shows execution errors dominate (>50% of low-Ea failures): **PROCEED** with "deep error" narrative linking to Li (2026)
- If H4 shows conceptual errors dominate: **PROCEED** with "Ea measures stability not understanding" narrative
- If H4 is inconclusive: **PIVOT** to broader routing signal comparison (Plan B)

**Gate 2 (After P1)**:
- If pre-registered H3 still shows AUC < 0.5: **CONFIRM** negative-result framing
- If pre-registered H3 shows AUC > 0.5: **RE-EVALUATE** — the post-hoc threshold may have been masking real signal

### 8.4 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Reviewers reject negative result | Cite NeurIPS negative results track; frame as "saving community resources" |
| Yang et al. novelty concern | Explicitly acknowledge; emphasize "independent validation + extension" |
| Sample size criticism | Acknowledge limitation; frame as pilot study; propose larger follow-up |
| H4 subjectivity | Use explicit heuristic rules; report inter-rater agreement if possible |

---

## 9. Final Verdict

**Overall Assessment**: The Activation Energy Theory experiments produced a **genuine but limited finding**: Arrhenius-like aggregate saturation is observable, but the framework fails at the individual-problem level where it matters for routing. The most valuable output is not a new theory but a **systematic falsification** of a plausible hypothesis, with clear diagnosis of why it failed.

**Quality Score**: 4/10 — Weak empirical foundation, but salvageable as a negative-result paper.

**Recommendation**: **PROCEED with caution** under the negative-result framing. Do not claim theoretical novelty. Do claim empirical boundary-delineation. Run H4 immediately as the decisive next step.
