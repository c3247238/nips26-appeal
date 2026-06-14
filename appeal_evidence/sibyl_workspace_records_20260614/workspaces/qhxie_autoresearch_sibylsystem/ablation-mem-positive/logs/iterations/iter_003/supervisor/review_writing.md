# Supervisor Review: CV Predicts Steering Heterogeneity in Absorbed SAE Features

**Score: 6.5 / 10 (Borderline Reject)**
**Verdict: REVISE**

---

## Executive Summary

The paper makes a legitimate and potentially publishable primary finding: coefficient of variation (CV) predicts steering effectiveness within absorbed SAE features, with high-CV absorbed features showing 1.47x larger effects than low-CV (p < 0.01 BH-corrected). Activation patching validates genuine causal structure (67.3% mean recovery). These are real contributions.

However, three major issues prevent a higher score:

1. **Cross-architecture validation is incomplete** — all quantitative results are GPT-2 only, yet the abstract and introduction claim generalization
2. **The "Variance Paradox" (H4) framing is circular** — absorbed features have higher CV by construction of the classification scheme
3. **Non-absorbed baseline comparison is unreliable** — used 3 prompts vs main experiment's 5 prompts, making "parity" claim unfounded

The mechanism interpretation (bypass vs context-sensitive routing) is hypothetical without direct evidence.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 7 | First CV-based predictor for steering feasibility within absorbed features. Addresses a timely open problem (Basu et al. actionability paradox). Incremental but genuine contribution. |
| **Soundness** | 6 | H1 (CV predicts steering) is well-supported. H4 framing is circular. Mechanism interpretation is unsupported hypothesis. H6 falsification acknowledged but not integrated. |
| **Experiments** | 6 | Main finding (H1) is rigorously validated with BH-corrected statistics. Missing: cross-architecture results, proper non-absorbed comparison, mechanism differentiation experiments. |
| **Reproducibility** | 6 | Steering protocol documented. CV computation depends on specific input distribution (1000 samples) — not specified in enough detail. CV threshold is model/layer-specific. |

---

## Major Issues

### 1. Cross-Architecture Validation Incomplete (Critical)

**Location**: Section 4.6 (Cross-Architecture Validation)

The paper explicitly states "detailed integration of cross-architecture results remains future work" yet the abstract claims CV "provides actionable guidance for interpretability practitioners" and the introduction frames the contribution as establishing CV as a "practical predictor." All quantitative results are GPT-2 Small only.

**Impact**: Without Gemma-2-2B results, this is a single-architecture finding. The contribution claim of establishing CV as a general predictor is not supported by evidence.

**Fix**: Either remove generalization claims or provide the actual Gemma-2-2B results. If pending, clearly label as preliminary.

---

### 2. Variance Paradox Framing is Circular (Major)

**Location**: Section 3.1, Section 4.7 (H4)

The paper classifies features as "high-CV" (CV > 1.0) and "low-CV" (CV <= 1.0) using the coefficient of variation, then presents H4 as the discovery that absorbed features have CV ~7.33 vs non-absorbed CV ~0.01 (733x ratio). This is expected by construction — absorbed features were classified BY their CV, so of course they have high CV in aggregate.

The genuine finding is that CV predicts steering within absorbed features (H1). H4 is a tautology given the classification scheme.

**Impact**: A reviewer will immediately flag this as circular reasoning, undermining the paper's credibility.

**Fix**: Drop the "paradox" framing. Reframe H4 as confirming that absorbed features (classified by absorption_score) have higher CV than non-absorbed features — a descriptive observation, not a discovery. Focus the narrative on H1 as the primary finding.

---

### 3. Non-Absorbed Baseline Comparison Unreliable (Major)

**Location**: Section 4.5

The paper claims absorbed high-CV features are "approximately equal" to non-absorbed features (0.097 vs 0.102). However:
- Non-absorbed baseline used only 3 prompts; main experiment used 5 prompts
- Main experiment high-CV at +5 strength shows 0.5222 effect — 5x larger than the 0.102 non-absorbed baseline
- These are different experimental conditions, not comparable

**Impact**: The "parity" finding is a key piece of evidence that absorption per se does not destroy steering potential. If it reverses under controlled comparison, the paper's central narrative weakens.

**Fix**: Rerun non-absorbed baseline with identical conditions (5 prompts, +3/+5/+7 strengths). Or qualify the claim as "pilot comparison at +5 strength with 3 prompts."

---

### 4. Mechanism is Hypothetical (Major)

**Location**: Sections 3.2-3.3

The paper proposes bypass vs context-sensitive routing to explain why CV predicts steering. Activation patching confirms causal structure exists (67.3% recovery) but does not differentiate between the two routing regimes. No experiment directly tests this mechanism.

**Impact**: The theoretical contribution is a hypothesis, not an established explanation. The paper presents it with more confidence than the evidence warrants.

**Fix**: Either run experiments to differentiate the mechanisms (e.g., activation patching timecourse, context-dependent child activation analysis) or clearly frame this as a hypothesis to be tested in future work.

---

## Minor Issues

| Issue | Category | Description |
|-------|----------|-------------|
| Domain specificity claim | novelty | Paper attributes Basu et al.'s failure to clinical features being predominantly low-CV, but no CV data from their features is provided. Post-hoc explanation. |
| H6 falsification not integrated | experiment | Decoder orthogonality was falsified (r=-0.136) but paper does not discuss what this implies for the routing mechanism. |
| Abstract overreach | writing | Claims "absorption per se does not destroy steering potential" but non-absorbed comparison is unreliable. |
| Post-hoc threshold | experiment | CV=1.0 threshold was empirically selected from pilot data. Should be explicitly noted as such. |

---

## What Would Raise the Score to 8.0+

1. **Complete Gemma-2-2B cross-architecture validation** with quantitative results — establishes generalization
2. **Run non-absorbed baseline with identical conditions** (5 prompts, +3/+5/+7) — properly establishes parity claim
3. **Prospective held-out validation of CV=1.0 threshold** — establishes CV as a predictor, not just post-hoc correlation
4. **At least one experiment differentiating bypass from context-sensitive routing** — supports mechanism interpretation

---

## Risks

- If Gemma-2-2B replication fails, generalization claims collapse entirely
- The non-absorbed comparison could reverse under controlled conditions (3 vs 5 prompts is a large difference)
- Circular H4 framing will invite reviewer criticism that colors perception of the entire paper
- Without cross-architecture validation, this is a GPT-2-only finding of limited general interest

---

## Evidence Gaps

1. No Gemma-2-2B quantitative results — cross-architecture validation explicitly incomplete
2. Non-absorbed baseline run with different prompt set (3 vs 5) — not comparable to main experiment
3. No direct evidence distinguishing bypass routing from context-sensitive routing mechanism
4. No CV data from Basu et al. clinical features to support domain-specificity claim
5. No held-out validation of CV=1.0 threshold — post-hoc selection

---

## Recommendation

**REVISE** — the core finding (CV predicts steering heterogeneity, H1) is real and worth publishing, but the paper has significant gaps in experimental coverage and has circular reasoning in the H4 framing. These are addressable within another iteration.

Priority fixes:
1. Remove or provide Gemma-2-2B results
2. Drop "Variance Paradox" framing as a discovery; acknowledge it's expected given the classification
3. Rerun non-absorbed baseline with identical conditions, or qualify the claim appropriately
4. Frame mechanism section as hypothesis, not established explanation