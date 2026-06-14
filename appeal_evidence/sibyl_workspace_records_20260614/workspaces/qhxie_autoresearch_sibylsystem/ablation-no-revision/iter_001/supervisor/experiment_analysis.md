# Experiment Result Analysis

## Key Results Summary

Round 4 Activation Energy Theory experiments (n=50, Qwen2.5-Math-7B-Instruct) produced the following results:

| Experiment | Metric | Result | Status |
|------------|--------|--------|--------|
| G0 Baseline | Accuracy | 47% (n=100) / 68% (n=50) | PASS |
| G1 Saturation | R2 (exponential fit) | 0.924 (aggregate) | CONFIRMED |
| G1 Saturation | Per-problem R2 | mean=0.077, median=0.000 | FAIL (individual) |
| G2 Consistency | Spearman(Ea, MATH level) | 0.448, p=0.001 | CONFIRMED |
| G3 Routing | Low-Ea single-pass accuracy | 75.0% (threshold 75%) | TECHNICALLY CONFIRMED |
| G3 Routing | AUC (Ea as predictor) | 0.436 (< 0.5) | FALSIFIED (predictive power) |
| G3 Routing | Spearman(Ea, accuracy) | -0.063, p=0.66 | FALSIFIED |
| H5 Matching | Spearman(Ea, k0) | -0.219, p=0.54 (valid pairs: 10/50) | FALSIFIED |

**Critical Discovery**: The H3 "CONFIRMED" status (75.0% low-Ea accuracy) is contradicted by AUC=0.436 (< 0.5, worse than random) and Spearman(Ea, accuracy)=-0.063 (no correlation). The 75% threshold was achieved via post-hoc threshold optimization on the same data (threshold = 9.999999999903855), constituting data leakage. Ea distribution is bimodal (~9.47 vs ~10.0), making the threshold a trivial split between two clusters rather than a meaningful predictor.

**Asymptotic Ceiling**: P_inf = 0.835, k0 = 0.613 (from aggregate fit on 5 data points)

## Debate Perspectives Summary

- **Optimist**: Three hypotheses confirmed (H1/H2/H3). Theory framework validated. Exponential saturation is physically meaningful. Ea provides zero-cost difficulty estimation. Routing strategy is feasible with 75% low-Ea accuracy and 25pp delta.

- **Skeptic**: Severe internal contradictions. H1 aggregate R2=0.924 is overfitting (5 points, 2 parameters). Per-problem median R2=0, 80% fit failure. H3 pilot (68.4%) contradicts full (75.0%). AUC=0.436 < 0.5 directly contradicts "CONFIRMED". H5 falsification destroys theoretical coherence. Ea measures stability, not correctness. Sample size (n=50) absurdly small for physical law claims.

- **Strategist**: Do not pursue Activation Energy Theory as a predictive theory. Evidence does not support it. Instead, use confirmed H1/H2 as background and build the paper on negative findings (H3/H5) with diagnostic analysis. Recommended paper: "On the Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning."

- **Comparativist**: Clear literature positioning. Yang et al. collision acknowledged (same formula). Unique contributions: (1) first systematic quantification of Ea routing ceiling (AUC=0.436), (2) Ea-k0 decoupling reveals two "difficulty" concepts, (3) cross-validation with ACAR's 8pp ceiling. Negative result is publishable.

- **Methodologist**: Fatal methodological flaws. Sample mismatch across G1/G2/G3. Data leakage in H3 threshold search. Per-problem exponential model inferior to power-law (mean R2: 0.077 vs 0.220). Ea distribution too concentrated (std ~0.25). Level 5 Ea compressed to 10.0 (std=1.9e-6). Must fix methods before any paper.

- **Revisionist**: Theory must be downgraded from "predictive physical analogy" to "descriptive statistical framework." H1 holds at aggregate level only. H2 holds as coarse-grained difficulty signal. H3 falsified (AUC < 0.5 proves no predictive power). H5 falsified (Ea and k0 measure different concepts). New hypotheses needed: H6 (bimodal Ea = two reasoning modes), H7 (aggregate != individual), H8 (power-law may be better individual model).

## Analysis

### 1. Method Feasibility: PARTIAL / DEGRADED

The core Arrhenius framework works as a **descriptive tool** at the aggregate level (R2=0.924 on 5 points), but fails as a **predictive theory** at the individual problem level:
- Per-problem exponential fit: mean R2=0.077, median=0.000, 80% of problems cannot produce valid k0
- Per-problem model selection (AICc/BIC): power-law wins 20/50 and 24/50 respectively, exponential wins only 8/50 and 4/50
- The aggregate "success" is likely an ensemble averaging effect, not a universal physical law

The Ea estimation method has severe construct validity issues:
- Ea values are bimodal and highly concentrated (only two distinct values: ~9.47 and ~10.0)
- Level 5 Ea is numerically compressed (std=1.9e-6), suggesting algorithmic saturation rather than physical measurement
- c0 estimates are identical across many problems (0.10565106800135829), indicating boundary convergence

### 2. Performance: MIXED WITH FATAL CONTRADICTIONS

| Claim | Evidence | Verdict |
|-------|----------|---------|
| H1: Arrhenius kinetics | Aggregate R2=0.924 | True (aggregate only) |
| H2: Ea correlates with difficulty | Spearman=0.448, p=0.001 | True (coarse-grained) |
| H3: Ea predicts single-pass solveability | 75% at optimized threshold | False (data leakage) |
| H3: Ea as classifier | AUC=0.436 | Worse than random |
| H3: Ea-accuracy correlation | Spearman=-0.063 | No correlation |
| H5: Ea matches k0 | Spearman=-0.219, p=0.54 | Unrelated measures |

The H3 "CONFIRMED" label is the most problematic. The analysis script reports "GO" based on 75.0% accuracy at a threshold of 9.999999999903855, but simultaneously reports AUC=0.436 and "useful_predictor: false". These are logically incompatible. A classifier with AUC < 0.5 performs worse than coin-flipping -- it cannot be a "useful routing signal."

### 3. Improvement Headroom: LIMITED IN CURRENT DIRECTION

The Strategist's assessment is correct: continuing to optimize Activation Energy Theory as a predictive framework is not justified. The evidence shows:
- Ea does not predict single-pass solveability (AUC < 0.5)
- Ea and k0 measure different concepts (Spearman=-0.219)
- The Arrhenius analogy lacks physical foundation (k is discrete sampling count, not temperature)
- A simpler 1-parameter binomial model 1-(1-p)^k can explain the saturation without invoking physical parameters

However, there IS a viable path forward if we **reframe the contribution**:
- Use H1/H2 as empirical background (confirming Yang et al. on Qwen2.5-Math-7B)
- Make H3/H5 falsification the central contribution (negative result)
- Add H4 error classification and H5 entropy comparison as diagnostic experiments
- Position paper as "boundary diagnosis" rather than "theory proposal"

### 4. Time-Cost Tradeoff: FAVORABLE FOR REFRAMED DIRECTION

| Next Step | Time | Value if Reframed |
|-----------|------|-------------------|
| H4 Error classification | 30 min | Explains WHY Ea fails |
| H5 Entropy routing | 45 min | Finds alternative signal |
| Full validation (n=200) | 120 min | Strengthens negative result |
| Paper writing | 2-3 hours | Negative result + diagnostic |

Total additional investment: ~3-4 hours for a complete negative-result paper. This is efficient because:
- No GPU training needed (pure inference)
- No API keys needed (local model)
- Experimental infrastructure already built
- Clear paper angle (Strategist's "boundary diagnosis")

### 5. Critical Objections: FATAL FOR PREDICTIVE THEORY, ADDRESSABLE FOR NEGATIVE RESULT

**Skeptic's concerns about H3 are fatal** for the predictive theory claim:
- AUC=0.436 < 0.5 is decisive evidence that Ea cannot predict single-pass solveability
- The 75% threshold was data-leaked (post-hoc optimization)
- Pilot (68.4%) and full (75.0%) results contradict each other
- Spearman(Ea, accuracy)=-0.063 proves no linear relationship

**However**, these same concerns become **valuable contributions** when framed as a negative result:
- First systematic quantification of Ea routing ceiling: AUC=0.436
- First demonstration that consistency-Ea and saturation-k0 are decoupled
- Diagnostic analysis of why Ea fails (stability != correctness)

**Methodologist's concerns must be addressed** in the paper:
- Acknowledge sample size limitations (n=50)
- Acknowledge data leakage in H3 threshold (label as "exploratory analysis")
- Report per-problem fit failures explicitly
- Report effect sizes and confidence intervals, not just p-values

## Decision Rationale

The evidence overwhelmingly supports **reframing, not abandonment**. The original goal -- proving Activation Energy Theory as a predictive framework for LLM reasoning optimization -- has failed. H3 (routing) is falsified, H5 (measurement coherence) is falsified, and the per-problem predictive power is nonexistent.

However, the empirical findings (H1 aggregate saturation, H2 difficulty correlation) are real statistical phenomena. The negative results (H3/H5) are novel and publishable. The Strategist's recommended paper angle -- "On the Limits of Consistency-Based Activation Energy for Problem-Level Routing" -- is viable and well-supported by the evidence.

The Comparativist's literature analysis confirms:
- Yang et al. collision acknowledged (not claiming formula novelty)
- ACAR cross-validation (our ~25pp ceiling vs their 8pp ceiling)
- Li (2026) connection (error depth explains low-Ea failures)
- Unique position in "dynamics signal" quadrant

The Revisionist's framework downgrade is the correct theoretical move:
- From "predictive physical analogy" to "descriptive statistical framework"
- From "theory proposal" to "boundary diagnosis"
- From "routing optimization" to "routing signal failure analysis"

**Arguments for PROCEED (reframed)**:
1. H1/H2 are solid empirical findings that serve as paper background
2. H3/H5 falsification is a novel, publishable negative result
3. Clear diagnostic path forward (H4 error classification, H5 entropy)
4. Low additional cost (~3-4 hours total)
5. Well-defined literature positioning

**Arguments for PIVOT (to entirely new direction)**:
1. Core practical application (routing) is definitively falsified
2. Must acknowledge significant methodological flaws
3. Negative result papers face higher scrutiny
4. If H4/H5 also fail, contribution narrows further

**Assessment**: The negative-result reframing is strong enough to justify continued investment. The Strategist's paper outline is comprehensive and addresses all major objections. The time-cost is favorable. The only reason to PIVOT would be if the team is unwilling to write a negative-result paper -- but the evidence clearly supports that as the most honest and valuable contribution.

## DECISION: PROCEED
