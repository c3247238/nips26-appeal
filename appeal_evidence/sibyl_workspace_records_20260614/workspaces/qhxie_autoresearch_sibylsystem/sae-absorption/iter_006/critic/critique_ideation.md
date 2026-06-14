# Ideation Critique -- Iteration 6

## Overall Assessment

The research proposal represents a correct strategic pivot from the unfixable causal-mediation framing (iterations 4-5) to a JumpReLU metric audit on Gemma 2 2B. The pivot was driven by evidence (within-width Gamma=1.0, GPT-2 dead features, control failure patterns) rather than premature abandonment. The proposal is well-structured, with pre-registered decision gates and three backup alternatives -- Alternative 1 (metric decomposition focus) and Alternative 3 (CMI-focused paper) turned out to be more aligned with the actual results than the primary plan. However, the proposal made three fundamental miscalibrations that shaped the iteration's weaknesses.

---

## Critical Issues

### 1. Rate-Distortion Theory Was Over-Weighted

The proposal elevates the successive refinement / CMI diagnostic to "co-primary contribution" based on a novelty score of 8/10. This was strategically risky:

- The successive refinement theorem's formal conditions (stationary ergodic source, successive descriptions at increasing rates) do not map precisely onto SAE feature interactions (discrete latent activations, JumpReLU thresholds, finite vocabulary)
- The CMI estimator's reliability in high-dimensional settings with small samples was a known concern
- No pilot data supported the CMI-absorption correlation -- the proposal committed significant experimental resources to an untested theoretical prediction
- The d'=10 selection was not pre-registered as primary, leaving room for post-hoc researcher degrees of freedom

**Root cause**: The Theoretical perspective's elegance was weighted more heavily than the Empiricist's and Contrarian's caution. The novelty score (8/10) captured theoretical originality but not empirical risk. The proposal should have designated the CMI analysis as exploratory from the start, with explicit pre-registration of d'=10 if it was to be primary.

### 2. Cross-Domain Characterization Was Not Adequately Risk-Assessed

The proposal rated the risk of "cross-domain absorption < 3% everywhere after controls" at 25% probability. The actual outcome was worse: not only were cross-domain rates low (0-6.6%), but shuffled controls exceeded measured rates in ALL five domains, rendering the entire cross-domain characterization uninterpretable.

The proposal did not anticipate that the Chanin metric might fail basic validity checks on JumpReLU SAEs -- a possibility that the Contrarian perspective raised but that was not given adequate weight. The decision gates ("After Exp 1a: If first-letter controls not calibrated...") were not enforced as blocking gates in the execution sequence, allowing the full experimental suite to run before the control failure was discovered to be universal.

### 3. Confound Decomposition Definition Was Not Stress-Tested

The proposal describes the hedging classification as "operational" but does not consider that the definition is near-tautological when the L0 range is large. The multi-L0 sweep design (L0=22 to 176, 8x range) ensures that any token whose false-negative status resolves at any higher L0 is classified as hedging -- which is nearly all of them at L0=176.

The proposal should have pre-specified a STRICT classification criterion (parent latent specifically fires at higher L0) alongside the permissive one (any resolution at higher L0), anticipating that the permissive definition would produce inflated hedging rates.

---

## Major Issues

### 4. Hypothesis Calibration Was Optimistic

- H1 (cross-domain >= 5%): Pilot data showed 0-6.6%, but controls were not validated. Setting the threshold at 5% was reasonable, but not conditioning on control credibility was an error.
- H2 (>80% hierarchy-driven at L0=22): The pilot showed 96.9%, but this used within-L0 analysis rather than cross-L0 persistence. The proposal did not recognize the methodological difference.
- H3 (CMI rho < -0.3): No pilot data existed. Setting this as a primary hypothesis without pilot validation was high-risk.
- H5 (scaling law width-L0 interaction): Result was ambiguous (OLS significant, GAM not).

### 5. Proposal Scope Was Too Broad

7 hypotheses, 14+ experiments, 6-7 hours estimated GPU time across 6 methodological stages. This breadth-first approach produced many experiments of moderate quality rather than fewer experiments of high quality. The three highest-value experiments (activation patching, control diagnosis, CMI replication at L0=22) were not included in the plan at all, despite being more informative than several experiments that were included (cross-domain on animal-class, hierarchy predictor analysis with n=5 domains, unsupervised detection).

---

## Positive Findings

### 1. Strategic Pivot Capability (Exemplary)

The pivot from epidemiological methods (iterations 4-5) to metric audit (iteration 6) was correctly identified, evidence-driven, and well-executed. The system correctly recognized that the within-width Gamma=1.0 finding meant the causal mediation story was unfixable and that the control failure pattern pointed toward a stronger contribution.

### 2. Pre-Registered Decision Gates (Good Practice)

Three explicit decision gates ("After Exp 1a," "After Exp 3a," "After Exp 6") with specific quantitative criteria. These were not enforced as blocking in execution order, which limited their value, but their existence is good practice.

### 3. Backup Alternatives Were Prescient

Alternative 1 (metric decomposition focus) and Alternative 3 (CMI-focused paper) both turned out to be closer to the actual findings than the primary plan. The fact that these were articulated before the experiments demonstrates good research planning.

### 4. Pilot-Driven Calibration

H1 threshold adjusted from 10% to 5% based on pilot data. H4 target lowered from rho > 0.5 to > 0.3 based on ITAC pilot weakness. Unsupervised detection de-prioritized based on pilot evidence. This evidence-driven calibration is correct practice.

### 5. Comprehensive Domain Coverage

Five hierarchy domains (first-letter, city-country, city-continent, city-language, animal-class) spanning syntactic, geographic, linguistic, and taxonomic relationships. While the control failure invalidates the absolute rates, the breadth of domains strengthens the universality of the control failure finding itself.
