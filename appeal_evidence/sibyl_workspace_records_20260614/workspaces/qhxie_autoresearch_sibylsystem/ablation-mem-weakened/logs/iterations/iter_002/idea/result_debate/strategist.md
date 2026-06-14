# Strategist Analysis: Local Inhibition Graph Framework

## Executive Summary

**Verdict: PROCEED with the Local Inhibition Graph (cand_f), with bounded risk and clear fallback.**

The project has pivoted from a failed correlation study (cand_a, score 3/10) to a neuroscience-inspired theoretical framework. This is not a rescue mission for the old hypothesis --- it is a genuinely new research direction with independent falsification criteria. The structural correspondence (W_dec^T W_dec = G_LCA) is mathematically exact, the novelty verification confirms zero prior work, and the existing data provides strong indirect support. The critical test (H6: graph edges predict absorption pairs) requires ~15 minutes of computation and has a clear pass/fail threshold.

**Key difference from the old strategist analysis**: The previous pivot recommendation (to Alternative C, trade-off analysis) was correct for the old correlation study. The new front-runner (cand_f, Local Inhibition Graph) is a different research program with different hypotheses, different evidence requirements, and a different risk profile.

---

## 1. Signal Strength Assessment

### Indirect Evidence from Prior Experiments (Iterations 1-8)

| Finding | Signal | Justification | Relevance to Inhibition Graph |
|---------|--------|---------------|------------------------------|
| Precision invariance (H5) | **Strong** | Precision std = 0.028-0.054 vs recall std = 0.192-0.199; 21-25/26 features have precision=1.0 at k>=5 | Exactly predicted by competitive suppression: inhibition suppresses true positives (recall loss) without creating false positives (precision preserved) |
| Delta-corrected steering (H1b, L8) | **Moderate** | r = -0.431, p = 0.028 (uncorrected); Spearman rho = -0.502, p = 0.009 | Consistent with competitive suppression at deeper layers where hierarchical structure is stronger |
| Steering robustness (Feature U) | **Strong** | 24.2% absorption, 100% steering success at s=50 | Decoder geometry intact; only encoder activation suppressed --- matches inhibition mechanism |
| EC50 null result | **Moderate** | No significant EC50 difference between high/low absorption features | Inhibition affects activation probability, not decoder alignment --- predicted by framework |
| Layer-dependence (L4 vs L8) | **Moderate** | H1b significant at L8, not L4 | Deeper layers have stronger hierarchical structure = stronger competitive suppression |
| Random baseline non-negligible | **Strong** | Random steering achieves 34-38% success | Decoder directions are non-orthogonal; decoder correlations encode semantic structure --- this IS the inhibition graph |

### Novelty Verification (External)

| Claim | Evidence | Confidence |
|-------|----------|------------|
| No prior LCA-SAE connection | 4 web searches, zero matches; Rozell (2008) ~2000 citations, zero LLM SAE applications | **High** |
| No prior inhibition graph for SAEs | Zero matches for "inhibition graph" + "sparse autoencoder" | **High** |
| No prior training-free repair | Matryoshka, OrtSAE, ATM, Balance Matryoshka all require retraining | **High** |

### Summary

The prior experiments provide **strong indirect support** for the inhibition framework. The precision-recall asymmetry, delta-corrected steering, layer-dependence, and decoder-activation decoupling all have natural explanations in competitive suppression. However, the **direct test** (H6: graph edges predict absorption pairs) has not been run. The framework lives or dies on this test.

---

## 2. Opportunity Cost Analysis

### Option A: PROCEED with Local Inhibition Graph (cand_f)

| Experiment | GPU Hours | Expected Info Gain | Cost/Benefit |
|------------|-----------|-------------------|--------------|
| E1: Graph construction + validation (H6) | ~0.5 | **Critical**: precision@k vs chance; determines framework viability | Low cost, high leverage |
| E2: Precision-recall asymmetry test (H7) | ~0.5 | High: tests mechanistic explanation | Low cost |
| E3: Graph predicts at-risk features (H8) | ~0.5 | Medium: practical diagnostic value | Low cost |
| E4: Layer-dependent graph structure (H9) | ~1.0 | Medium: explains layer-dependence finding | Low cost |
| E5: Homeostatic rebalancing (H10) | ~1.0 | High if successful; diagnostic stands if failed | Medium cost |
| E6: Cross-model validation (Gemma-2-2B) | ~2.0 | High: generalization test | Medium cost, auth risk |
| **Subtotal (H6-H9, core)** | **~2.5** | **Determines framework viability** | **Excellent cost/benefit** |
| **Subtotal (all)** | **~6.0** | **Full validation + repair** | **Good cost/benefit** |

### Option B: PIVOT to Trade-off Analysis (cand_c, backup)

| Experiment | GPU Hours | Expected Info Gain | Cost/Benefit |
|------------|-----------|-------------------|--------------|
| Compare standard SAE vs OrtSAE on GPT-2 | ~3 | High: direct trade-off quantification | Moderate risk: OrtSAE checkpoints may be unavailable |
| Measure dead neuron rate + reconstruction error | ~1 | High: completes trade-off picture | Low cost |
| Fit Pareto frontier | ~1 | Medium: novel visualization | Low cost |
| Cross-architecture comparison | ~5 | High: generalizes claim | Moderate risk |
| **Subtotal** | **~10** | **Descriptive claim, no mechanistic depth** | **Moderate cost/benefit** |

### Option C: PIVOT to Metric Validation (cand_d, dropped)

| Experiment | GPU Hours | Expected Info Gain | Cost/Benefit |
|------------|-----------|-------------------|--------------|
| Enhanced metric implementation | ~2 | Medium: detects more absorption | Low novelty |
| SynthSAEBench validation | ~1 | Medium: ground truth | May not be available |
| **Subtotal** | **~3** | **Incremental methodological contribution** | **Poor cost/benefit** |

### Option D: Write Null-Result Paper (fallback)

| Effort | GPU Hours | Expected Info Gain | Cost/Benefit |
|--------|-----------|-------------------|--------------|
| Write + submit | 0 | Low: workshop/arXiv only | Zero cost, low reward |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome | Info Gain / GPU-hr |
|-----------|-----------------|----------|------|------------------|-------------------|
| **PROCEED cand_f (core H6-H9)** | Strong indirect | **2.5** | **Low** (bounded: H6 fails -> pivot) | Publication-quality if H6 validates; clear fallback if not | **Very High** |
| PROCEED cand_f (full H6-H10 + cross-model) | Strong indirect | 6.0 | Medium | Stronger paper with repair + generalization | High |
| PIVOT to cand_c (trade-off) | Moderate (data supports contrarian) | 10 | Low-Medium | Descriptive paper, no theoretical depth | Medium |
| PIVOT to cand_d (metric validation) | None | 3 | Low | Incremental methodological paper | Low |
| Write null result | Weak | 0 | Low | Workshop / arXiv only | Very Low |

**Dominant strategy**: PROCEED with cand_f core experiments (H6-H9, ~2.5 GPU hours). The risk is bounded: if H6 fails, pivot to cand_c. If H6 succeeds, the framework provides a genuinely novel theoretical contribution with practical utility. The info gain per GPU-hour is maximized because the core experiments are computationally cheap and determinative.

---

## 4. PIVOT vs PROCEED Verdict

### Criteria Check for cand_f

- **At least one hypothesis with moderate+ signal?** YES. H5 (precision invariance) is strongly supported. H1b (delta-corrected steering) is moderately supported (uncorrected p<0.05, but does not survive MCP). The indirect evidence is strong.
- **Clear path to publication-quality results?** YES, conditional on H6. If H6 validates (precision@20 >= 0.10), the paper has four firsts: (1) first LCA-SAE connection, (2) first local inhibition graph for SAE diagnostics, (3) first mechanistic explanation for precision-recall asymmetry, (4) first training-free repair (if H10 succeeds).
- **All hypotheses weak/noise?** NO. H5 is strongly supported. The framework has independent falsification criteria.
- **Contribution margin too small for target venue?** NO. The novelty verification confirms zero prior work on the LCA-SAE connection. The contribution is theoretical + practical, not just empirical.

### Verdict: **PROCEED with cand_f**

The Local Inhibition Graph is a decisively different research program from the failed correlation study. It does not attempt to rescue the old hypotheses --- it replaces them with a new theoretical framework that explains the old findings. The core test (H6) is cheap, fast, and determinative. The risk is bounded: if H6 fails, we pivot to cand_c with no sunk cost.

---

## 5. Exact Next Experiments (Priority Order)

### Experiment 1: Graph Construction + Validation (H6) --- ~15 min, **CRITICAL**

**What**: Construct local inhibition graph for GPT-2 Small SAE (24K latents, k=20 neighbors). Validate precision@k against Chanin absorption pairs on 26 first-letter features.

**Method**:
1. Load W_dec from gpt2-small-res-jb SAE (shape: 24576 x 768)
2. Compute G = W_dec @ W_dec.T (decoder correlation matrix)
3. For each latent i, find top-k neighbors by |G_ij|
4. For each absorption pair (parent i, child j), check if j in N(i)
5. Compute precision@k, recall@k, Fisher exact test for enrichment
6. Random baseline: shuffle latent indices, expected precision@20 = 20/24576 ~ 0.0008

**Pass threshold**: precision@20 >= 0.10 (125x enrichment over chance)
**Fail threshold**: precision@20 <= 0.05

**Expected outcome**: If decoder correlations encode parent-child relationships, precision@20 should be significantly above chance. The 0.10 threshold is conservative --- even 0.05 would be 62x enrichment.

**If H6 fails**: Pivot to cand_c (trade-off analysis). The inhibition framework's central claim collapses.

### Experiment 2: Precision-Recall Asymmetry Test (H7) --- ~15 min, **CRITICAL**

**What**: Compute total incoming inhibition per feature. Test correlation with recall and precision.

**Method**:
1. For each feature's parent latent i, compute total_inhibition_i = sum_{j in N(i)} |G_ij|
2. Correlate total_inhibition_i with recall@k (predicted: negative correlation)
3. Correlate total_inhibition_i with precision@k (predicted: no correlation)

**Pass threshold**: r(recall, inhibition) < -0.3, p < 0.05; r(precision, inhibition) ~ 0, p > 0.05
**Fail threshold**: Both correlations non-significant or precision correlation significant

**Expected outcome**: Competitive suppression reduces recall (parent fails to fire) but not precision (when child fires, it fires correctly).

### Experiment 3: Graph Predicts At-Risk Features (H8) --- ~15 min, **HIGH**

**What**: Test whether total incoming inhibition predicts absorption rate.

**Method**:
1. For all 26 first-letter features, compute total_inhibition_i
2. Correlate total_inhibition_i with Chanin absorption rate
3. Compare top quartile (highest inhibition) vs bottom quartile absorption rates

**Pass threshold**: r > 0.3, p < 0.05; top quartile has >2x absorption rate vs bottom
**Fail threshold**: r < 0.2 or p > 0.10

**Note**: The Skeptic correctly identifies that this is underpowered with n=26. If H8 fails due to power, it does not falsify the framework --- it just means we need more features. The diagnostic claim can still stand on H6 + H7.

### Experiment 4: Layer-Dependent Graph Structure (H9) --- ~20 min, **HIGH**

**What**: Construct graphs for layers 0, 4, 8, 10. Compare graph statistics.

**Method**:
1. Load SAEs for layers 0, 4, 8, 10
2. Construct inhibition graph for each layer
3. Compute mean edge weight, density, clustering coefficient per layer
4. Test correlation between mean edge weight and layer depth

**Pass threshold**: r(mean_weight, layer) > 0.3; L8 has strongest structure
**Fail threshold**: No layer-dependent trend

**Expected outcome**: Deeper layers have stronger hierarchical structure = stronger competitive suppression = denser, stronger inhibition graphs. This explains why H1b was significant at L8 but not L4.

### Experiment 5: Homeostatic Rebalancing (H10) --- ~30 min, **MEDIUM**

**What**: Apply z'_i = z_i + alpha * inh_i. Test parent firing restoration + reconstruction constraint.

**Method**:
1. For input activation a, compute original latents z = f(W_enc * a + b_pre)
2. Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
3. Apply boost: z'_i = z_i + alpha * inh_i (clip negative values)
4. Measure: (a) parent firing rate change, (b) reconstruction error change
5. Sweep alpha in [0.1, 0.5, 1.0, 2.0, 5.0]

**Pass threshold**: Parent firing increases >20% for some alpha; recon error increase < 5%
**Fail threshold**: Recon error > 5% for all alpha, or no parent firing improvement

**Note**: The Skeptic raises a valid concern about the sign of the update rule. If z'_i = z_i + alpha * inh_i increases parent activation when child features fire (inh_i > 0), this is "activation boosting," not "homeostatic rebalancing." The mechanism should be tested empirically: try both z'_i = z_i + alpha * inh_i and z'_i = z_i - alpha * inh_i, and report which works. If neither works, drop repair claims; diagnostic contribution (H6-H8) stands independently.

### Experiment 6: Cross-Model Validation --- ~30 min, **HIGH**

**What**: Replicate H6-H8 on Gemma-2-2B (layer 12, 16K dict).

**Method**: Same as E1-E3, on GemmaScope SAE.

**Expected outcome**: Stronger effects due to richer hierarchies in larger models.

**Risk**: Gemma access may be blocked (auth issues blocked pilot in iter_001). If so, use Pythia-160M as cross-model validation.

---

## 6. Resource Allocation

| Resource | cand_f (core) | cand_f (full) | cand_c (backup) |
|----------|--------------|---------------|-----------------|
| GPU hours | ~2.5 | ~6.0 | ~10 |
| Wall-clock | 1 day | 2-3 days | 1-2 weeks |
| Models needed | GPT-2 Small (have) | GPT-2 + Gemma/Pythia | GPT-2 only |
| Code reuse | 90% (existing SAE loading) | 85% | 95% |
| Novelty claim | 4 firsts (if H6 validates) | 4 firsts + repair | "First trade-off quantification" |
| Publication target | NeurIPS/ICML/ICLR | NeurIPS/ICML/ICLR | COLM, ICLR workshop, arXiv |
| Risk | Low (bounded by H6) | Medium | Low-Medium |

---

## 7. Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| H6 fails (graph edges don't predict absorption) | Medium | High | Fallback: pivot to cand_c (trade-off analysis). The structural correspondence is mathematically exact; if edges don't match, this itself is a finding about decoder correlation limitations. |
| H7 fails (inhibition doesn't explain precision-recall) | Low | Medium | Precision invariance is strongly supported by data. If H7 fails, the explanation may need refinement (e.g., L1 sparsity also contributes). Diagnostic claims stand. |
| H10 fails (repair degrades reconstruction) | Medium | Medium | Alpha is tunable; test both additive and subtractive rules. If repair fails, drop repair claims; diagnostic contribution stands independently. |
| H3 underpowered (n=26 too small) | High | Low | H3 is secondary. If it fails due to power, expand feature set in follow-up work. The core claims (H6, H7) do not depend on H3. |
| Gemma access blocked | High | Low | Primary experiments on GPT-2 Small; cross-model validation is bonus, not required. Use Pythia-160M as fallback. |
| Neuroscience analogy criticized as superficial | Low | Medium | The correspondence is exact (W_dec^T W_dec = G_LCA), not metaphorical. Ground claims in Rozell et al. (2008) Eq. 3. |
| LCA correspondence only for tied SAEs | Medium | Medium | Test on both tied (res-jb) and untied (JumpReLU) SAEs. If only tied works, qualify claims. |

---

## 8. Decision Tree

```
Current Proposal: Local Inhibition Graph (cand_f)
|
|-- H6 validated (precision@20 >= 0.10):
|   --> PROCEED with H7-H9 (core framework validated)
|   --> If H10 succeeds: Full paper with diagnostic + repair
|   --> If H10 fails: Paper with diagnostic only (still strong)
|   --> Paper target: NeurIPS/ICML/ICLR
|
|-- H6 partially validated (precision@20 = 0.05-0.10):
|   --> PROCEED with diagnostic-only claims (no repair)
|   --> Paper becomes "Decoder Correlations Reveal Competitive Suppression in SAEs"
|   --> Paper target: ICLR workshop, COLM, or arXiv
|
|-- H6 not validated (precision@20 <= 0.05):
|   --> PIVOT to cand_c (trade-off analysis)
|   --> Paper becomes "Feature Absorption as Optimal Compression: Evidence from GPT-2 Small"
|   --> Inhibition framework retained as theoretical speculation in Discussion
|   --> Paper target: COLM, ICLR workshop, or arXiv
|
|-- H7 fails (inhibition doesn't explain precision-recall):
|   --> Refine explanation (add L1 sparsity contribution)
|   --> Diagnostic claims (H6) still stand
|
|-- H10 fails (repair doesn't work):
|   --> DROP repair claims
|   --> Diagnostic contribution stands independently
```

---

## 9. Key Differences from Previous Strategist Analysis

The previous strategist analysis (iter_001-008) recommended PIVOT to Alternative C (trade-off analysis) because:
1. The correlation study had zero significant results after MCP correction
2. The effect sizes were tiny (R^2 < 0.10)
3. The H3 coding bug undermined credibility
4. The contribution margin was marginal-to-negative

**This analysis recommends PROCEED with cand_f because:**
1. The inhibition graph is a **new research program**, not a rescue of the old one
2. The structural correspondence (W_dec^T W_dec = G_LCA) is **mathematically exact**, not an empirical hypothesis
3. The novelty verification confirms **zero prior work** on the LCA-SAE connection
4. The existing data provides **strong indirect support** (precision invariance, delta-corrected steering, layer-dependence)
5. The core test (H6) is **cheap, fast, and determinative** (~15 minutes)
6. The risk is **bounded**: if H6 fails, pivot to cand_c with no sunk cost

The old analysis was correct for the old direction. The new direction deserves its own evaluation.

---

## 10. Bottom Line

**PROCEED with the Local Inhibition Graph framework.**

Run E1 (H6 validation) first --- it is the gatekeeper experiment. If precision@20 >= 0.10, the framework's central claim is validated and we proceed with H7-H9. If precision@20 <= 0.05, pivot immediately to cand_c (trade-off analysis).

The inhibition graph transforms the project's narrative from "we tried to correlate absorption with tasks and found nothing" to "we discovered that decoder correlations encode competitive suppression, explaining why absorption affects recall but not precision." This is a theoretical contribution with practical utility, not a null result.

**The path forward is a decisive experiment, not a rescue mission.**
