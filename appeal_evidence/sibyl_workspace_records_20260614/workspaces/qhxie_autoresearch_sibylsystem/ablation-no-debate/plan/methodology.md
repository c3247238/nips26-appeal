# Methodology: Encoder-Driven Feature Absorption in SAEs (iter_004)

## 1. Setup and Infrastructure

### 1.1 Environment
- **Compute backend**: SSH MCP (config: `ssh_server=default`)
- **Remote base**: `/home/qhxie/sibyl_system`
- **Python environment**: `.venv` with `sae-lens`, `transformer-lens`, `torch`
- **GPU allocation**: max 2 GPUs, 1 GPU per task (parallel execution)

### 1.2 Shared Resources
- **Remote registry**: `shared/registry.json` (dataset/model paths)
- **Experiment DB**: `shared/experiment_db.jsonl` (append results)
- **SAE releases**:
  - Synthetic experiments: custom 3-level hierarchy generator
  - Real SAE experiments: `gemma-2b-res` (gemma-2b, layer 12) from SAELens

### 1.3 Model Choices
| Experiment | Model | SAE | Notes |
|------------|-------|-----|-------|
| H_Mech (synthetic) | Custom hierarchy generator | Custom SAE | 3-level stochastic hierarchy |
| H_Comp (synthetic) | Custom hierarchy generator | Custom SAE | Vary cosine similarity |
| H_Pareto (synthetic) | Custom hierarchy generator | Custom SAE | Suspended - formula bug |
| H_Safe (real) | Gemma 2B | Gemma Scope layer 12 | Neuronpedia-validated features |
| H_Downstream (real) | Gemma 2B | Gemma Scope layer 12 | Matched pair steering |

---

## 2. Experiment Design

### 2.1 H_Mech: Encoder-Driven Mechanism Validation (Priority 1)

**Objective**: Confirm encoder alignment is the sole driver of absorption across stochastic hierarchies.

**Design**: 2x2 factorial with stochastic hierarchy noise (epsilon ~ N(0, 0.05)).

| Condition | Encoder | Decoder | Hypothesis |
|-----------|---------|---------|------------|
| A | Random | Random | Baseline (geometry only) |
| B | Trained | Random | Encoder alignment only |
| C | Random | Trained | Decoder geometry only |
| D | Trained | Trained | Full training |

**Key Finding to Validate**: B > D anomaly (trained encoder + random decoder produces MORE absorption than full training).

**Prior result** (deterministic hierarchy, 3 seeds):
- B ≈ D (0.490 vs 0.484): encoder sufficient
- C ≈ A (0.299 vs 0.299): decoder irrelevant
- **ANOMALY**: B > D suggests decoder acts as implicit regularizer

**Full experiment**:
- 5 seeds × 4 conditions = 20 runs
- Each run: 100 hierarchy samples × 5 ablations = 500 measurements

### 2.2 H_Comp: Hierarchy Strength Dependence (Priority 2)

**Objective**: Test monotonic absorption-hierarchy strength relationship.

**Design**: Parent-child cosine similarity sweep.

| Strength Level | Cosine Overlap | Pilot Absorption |
|---------------|----------------|------------------|
| 1 | 0.50 | TBD |
| 2 | 0.60 | 0.585 |
| 3 | 0.70 | TBD |
| 4 | 0.80 | 0.673 |
| 5 | 0.90 | TBD |
| 6 | 0.95 | 0.802 |

**Pilot results** (3 levels, iter_003): Confirmed monotonic increase.

**Full experiment**: 6 levels × 3 seeds = 18 runs.

### 2.3 H_Safe: Safety-Critical Feature Absorption (Priority 3)

**Objective**: Test whether safety-critical features are disproportionately absorbed in real SAEs.

**CRITICAL CHANGE**: Must use Neuronpedia-validated features ONLY.

**Design**:
1. Load Gemma Scope SAE (gemma-2b, layer 12) via SAELens
2. Select **validated** safety-relevant features from Neuronpedia (deception, jailbreak, harm, manipulation)
3. Match 20 non-safety features by activation frequency ±10% and same layer
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test comparing distributions

**Prior INVALID pilot** (iter_003): Used arbitrary indices (1024, 2048, etc.) - NOT validated.

**Full experiment**: 20 safety + 20 non-safety features, 100 samples each.

### 2.4 H_Downstream: Downstream Utility Test (Priority 0 - NEW)

**Objective**: Test whether absorption degrades steering/circuit utility.

**Why This Matters**: If absorbed vs non-absorbed features show < 10% difference in steering success, absorption may not matter for interpretability -- the field may be optimizing wrong metrics.

**Design**:
1. Classify features as "absorbed" vs "non-absorbed" via multi-child proportional ablation
2. Match pairs on confounders (activation frequency, magnitude, interpretability score from Neuronpedia)
3. Test both on:
   - Steering accuracy: behavior change correlation
   - Circuit completeness: edge coverage comparison
4. Statistical test: McNemar's test for paired proportions

**Sample size**: 30 absorbed + 30 matched non-absorbed features (from Gemma Scope layer 12).

**Falsification Criterion**: < 10% difference in steering success rate → absorption does not have practically significant downstream impact.

### 2.5 H_Pareto: Sensitivity-Absorption Frontier (Priority 4 - SUSPENDED)

**Status**: SUSPENDED until sensitivity formula bug is fixed.

**Issue**: Pilot data shows sensitivity = 1.525 at L0=16, exceeding [0,1] bounds. Formula bug in sensitivity measurement.

---

## 3. Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Multi-child absorption rate | Parent activation after ablating k children / parent activation before | Compare distributions |
| Mann-Whitney U | Non-parametric test for distribution difference | p < 0.05 |
| McNemar's test | Paired proportion test for steering accuracy | > 10% difference |
| Spearman correlation | Hierarchy strength vs absorption | rho > 0.8 |
| R² (monotonic fit) | Goodness of fit for H_Comp curve | > 0.8 |

---

## 4. Falsification Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H_Mech | Condition B ≈ D, C ≈ A (B>D anomaly explained) | NOT TESTED |
| H_Comp | R² > 0.8 for monotonic fit | PASSED (pilot) |
| H_Safe | Mann-Whitney p < 0.05 | NOT TESTED |
| H_Downstream | > 10% difference in steering success | NOT TESTED |
| H_Pareto | Detectable frontier shape | SUSPENDED |

---

## 5. Expected Visualizations

| Figure | Type | Content | Paper Section |
|--------|------|---------|---------------|
| Figure 1 | Bar chart | H_Mech 2x2 factorial results (5 seeds) | method |
| Figure 2 | Line plot | H_Comp absorption vs hierarchy strength | results |
| Figure 3 | Violin plot | H_Safe safety vs non-safety absorption | results |
| Figure 4 | Scatter + bar | H_Downstream steering accuracy by absorption status | results |
| Table 1 | Summary table | All hypotheses: status, metric, threshold | results |

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| H_Downstream: No effect found | Medium | HIGH (landmark negative) | Document as landmark result |
| H_Safe: No difference | Medium | Medium | Document as negative result |
| H_Mech: B>D not replicated | Medium | High | More seeds, investigate |
| Sensitivity formula unfixable | Low | Medium | Use alternative metric |

---

## 7. Task Dependencies

```
[setup_env]
       ↓
   ┌───┴───┬───────────┬─────────┐
   ↓       ↓           ↓         ↓
[h_mech] [h_comp]  [h_safe] [h_downstream]
   │       │           │         │
   └───────┴───────────┴─────────┤
                    ↓            │
              [held_out_validation]◄┘
```

**Parallel execution**: h_mech, h_comp, h_safe, h_downstream can run in parallel after setup (2 GPUs).

---

## 8. Iteration History

From `idea/hypotheses.md`:
- **H1**: Multi-child proportional absorption - PASSED
- **H_Mech**: Encoder-driven mechanism - PASSED + anomaly (B>D)
- **H2**: Frequency correlation - FAILED (+0.171)
- **H3**: Steering sensitivity - PASSED (1.62x ratio)
- **H_Comp**: Monotonic hierarchy strength - PASSED (pilot)
- **H_Downstream**: NEW - contrarian's challenge
- **H_Safe**: Safety-critical absorption - NOT TESTED on validated features
- **H_Pareto**: Sensitivity-absorption frontier - SUSPENDED