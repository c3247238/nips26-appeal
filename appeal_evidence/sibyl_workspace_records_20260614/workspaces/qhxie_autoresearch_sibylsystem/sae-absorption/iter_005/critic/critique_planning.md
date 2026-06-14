# Planning Critique: Iteration 5

## Overall Assessment

The experimental plan is well-structured with clear phase dependencies, reasonable time estimates, and an appropriate pilot design. The four-phase structure correctly sequences the go/no-go test first. The total resource budget (5h CPU + 6.5h GPU) is appropriate. However, the plan had a critical gap: the fallback for Gemma 2B access restriction was not adequately designed, resulting in an underpowered Phase 2 that could not answer its own research question. The plan also omitted an architecture covariate in the Phase 3 GAM design, missed a data pipeline validation step, and did not anticipate the need to reconcile contradictory absorption rate definitions.

---

## Strengths

### 1. Go/no-go structure

Phase 1, Step 1.1 (L0 as covariate) as a critical go/no-go test is correct. If all four partial correlations dropped below |0.2|, the causal chain hypothesis would be falsified early, avoiding wasted GPU time on Phase 2. This is good experimental design.

### 2. Appropriate phase ordering

P0 (confound go/no-go) before P1 (full analysis + probe training + scaling surface) before P2 (full cross-domain + comparison). Dependencies are correctly identified: Phase 2.2 depends on 2.1 (probe training), Phase 1.2-1.6 depend on Phase 1.1 (go/no-go). Independent phases (1.x, 2.1, 3.x) can run in parallel.

### 3. Control conditions well-specified

Phase 2 includes three controls (shuffled hierarchy, random probe direction, first-letter baseline). The shuffled control proved essential -- it exposed the metric failure. The threshold sweep (cosine and dominance thresholds) provides robustness characterization.

### 4. Falsification criteria are explicit

Each hypothesis has a concrete falsification condition specified before experiments. The "paper killed if H1 AND H2 simultaneously falsified" threshold with a planned pivot to honest audit framing is honest.

---

## Weaknesses

### 1. GPT-2 Small fallback was not fit for purpose

The risk table lists "Gemma 2B model access blocked" at 15% likelihood with mitigation "Fall back to GPT-2 Small for cross-domain; use Gemma Scope SAEs only for scaling surface." This mitigation does not preserve the experiment's scientific value. GPT-2 Small (124M) with a 24k SAE (98% dead features on city prompts) cannot test whether absorption occurs on knowledge hierarchies -- it can only test whether a tiny SAE has knowledge features at all.

A better plan would have been:
- (a) Use Pythia-1.4B or Pythia-2.8B (fully open, no access gates) with SAELens-compatible SAEs
- (b) Design the cross-domain experiment around syntactic features (e.g., subject-verb agreement hierarchies) that are well-represented even in small models
- (c) Secure Gemma 2B HuggingFace access BEFORE committing to the cross-domain contribution

### 2. Architecture covariate missing from Phase 3 design

The methodology specifies the GAM as: `absorption ~ s(log_width) + s(log_L0) + ti(log_width, log_L0) + layer`. With 360 L1 SAEs and 54 JumpReLU SAEs in the 420-SAE dataset, architecture is a confound that should have been in the GAM specification from the start. The plan assumes homogeneous architecture, which is false.

### 3. No data pipeline validation step

The plan includes experiment execution and results collection but no validation step to verify that summary files (final_results.json, final_results_summary.md) correctly propagate results from individual experiment JSONs. The two data pipeline failures (wrong Sobel z values, wrong taxonomy rates) would have been caught by a simple automated validation script that compares summary values against their source files.

### 4. Taxonomy correction plan underspecified

The plan for Phase 4 says "For each letter with n_comparison_tokens=0, use tokens from the same log-frequency band." But the executed experiment used three distinct strategies (A: non-letter-context activations, B: global when-active baseline, C: Chanin absorption rate). The plan does not anticipate that these strategies would produce dramatically different rates (19.2% vs 73.1%), nor does it specify how to reconcile them. A well-designed plan would have pre-registered which metric is "primary."

### 5. Mediation analysis design limitations

The plan specifies Baron-Kenny mediation but does not discuss:
- What to do if the total effect is non-significant (which happened for SP-F1)
- How to handle proportion mediated > 1.0 (which happened for SCR)
- Whether to compute the reverse direction test as a validity check
- Power analysis for mediation with n = 48

These are foreseeable issues with small-sample mediation analysis that should have been addressed in the methodology.

### 6. Missing within-width analysis in matching design

The Rosenbaum analysis plan specifies matching on width (exact) and L0 (nearest neighbor). But the plan does not specify within-width matching as a separate analysis strategy. The three within-width strategies (exact: 4 pairs, median split: 23 pairs, tertile: 16 pairs) appear to have been added during execution rather than planned. This is not necessarily bad (exploring multiple strategies is good), but the plan should have anticipated that within-width matching would be the critical test of whether absorption has an independent effect.

---

## Resource Budget Assessment

| Phase | Planned Time | Actual Time | GPU-Hours | Assessment |
|-------|-------------|-------------|-----------|------------|
| Phase 1 (confound) | ~3h CPU | ~3h | 0 | On target |
| Phase 2 (cross-domain) | 4-6h GPU | ~1h | ~1h | Under budget (GPT-2 Small is much faster than Gemma 2B) |
| Phase 3 (scaling surface) | ~1h CPU | ~0.5h | 0 | On target |
| Phase 4 (taxonomy) | 1-2h GPU | ~0.5h | ~0.5h | Under budget |
| **Total** | **9-12h** | **~5h** | **~1.5h** | **Under budget** |

The plan was under-budget because the GPT-2 Small fallback is much faster to run than the planned Gemma 2B experiments. This is a false efficiency -- the experiments ran faster because they tested a weaker setting, not because the plan was well-optimized.

---

## Score

Planning quality: **6/10**. The overall structure is sound (go/no-go first, parallel execution of independent phases, explicit falsification criteria). But the critical gap -- inadequate GPT-2 Small fallback design for Phase 2, missing architecture covariate in Phase 3, and no data pipeline validation -- means the execution fell short of what the plan promised.
