# Methodology: Encoder-Driven Feature Absorption in SAEs

## 1. Setup and Infrastructure

### 1.1 Environment
- **Compute backend**: SSH MCP (config: `ssh_server=default`)
- **Remote base**: `/home/qhxie/sibyl_system`
- **Python environment**: `.venv` with `sae-lens`, `transformer-lens`, `torch`
- **GPU allocation**: max 2 GPUs, 1 GPU per task

### 1.2 Shared Resources
- **Remote registry**: `shared/registry.json` (populate with dataset/model paths)
- **Experiment DB**: `shared/experiment_db.jsonl` (append results for traceability)
- **SAE releases**:
  - Synthetic experiments: custom 3-level hierarchy generator
  - Real SAE experiments: `gemma-2b-res` (gemma-2b, layer 12) from SAELens

### 1.3 Model Choices
| Experiment | Model | SAE | Notes |
|------------|-------|-----|-------|
| H_Mech (synthetic) | Custom hierarchy generator | Custom SAE | 3-level stochastic hierarchy |
| H_Comp (synthetic) | Custom hierarchy generator | Custom SAE | Vary cosine similarity |
| H_Pareto (synthetic) | Custom hierarchy generator | Custom SAE | Vary L0 {16, 32, 64, 128} |
| H_Safe (real) | Gemma 2B | Gemma Scope layer 12 | Neuronpedia-annotated features |

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

**Measurement**: Multi-child proportional ablation (k=5)
```
absorption_rate = parent_activation_after_ablating_top_k_children / parent_activation_before
```

**Falsification**: If Condition D >> Condition B (decoder contributes significantly).

**Prior result** (deterministic hierarchy, 3 seeds):
- B ≈ D (0.490 vs 0.484): encoder sufficient
- C ≈ A (0.299 vs 0.299): decoder irrelevant

**Full experiment**:
- 5 seeds × 4 conditions = 20 runs
- Each run: 100 hierarchy samples × 5 ablations = 500 measurements
- Pilot: 100 samples, seed 42, timeout 900s

### 2.2 H_Comp: Hierarchy Strength Dependence (Priority 2)

**Objective**: Test monotonic absorption-hierarchy strength relationship.

**Design**: Parent-child cosine similarity sweep.

| Strength Level | Cosine Overlap | Expected Absorption |
|---------------|----------------|-------------------|
| 1 | 0.50 | Low |
| 2 | 0.60 | |
| 3 | 0.70 | |
| 4 | 0.80 | |
| 5 | 0.90 | |
| 6 | 0.95 | Near 1.0 |

**Measurement**: Mean absorption rate per strength level.

**Falsification**: R² < 0.8 for monotonic fit.

**Pilot**: 3 strength levels (0.6, 0.8, 0.95), 100 samples, seed 42.

### 2.3 H_Pareto: Sensitivity-Absorption Frontier (Priority 3)

**Objective**: Quantify irreducible trade-off between sparsity and feature quality.

**Design**: L0 target sweep with paired absorption + sensitivity measurement.

| L0 Target | Sparsity | Absorption | Sensitivity |
|-----------|----------|------------|-------------|
| 16 | High | Low? | Low? |
| 32 | Medium | ? | ? |
| 64 | Medium-Low | ? | ? |
| 128 | Low | High? | High? |

**Metrics**:
- **Absorption**: Multi-child proportional (k=5)
- **Sensitivity**: From Hu et al. 2025 (steering coefficient variance)

**Falsification**: No detectable Pareto frontier shape.

**Pilot**: L0 ∈ {16, 64}, 100 samples, seed 42.

### 2.4 H_Safe: Safety-Critical Feature Absorption (Priority 4)

**Objective**: Test whether safety-critical features are disproportionately absorbed in real SAEs.

**Design**:
1. Load Gemma Scope SAE (gemma-2b, layer 12) via SAELens
2. Select 20 safety-relevant features from Neuronpedia annotations (deception, jailbreak, harm, manipulation)
3. Match 20 non-safety features by activation frequency ±10% and same layer
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test comparing distributions

**Falsification**: Mann-Whitney p > 0.05 (no difference).

**Novelty**: 9/10 - No prior work examines safety-critical feature absorption.

**Pilot**: 5 safety + 5 control features, 100 samples per feature, seed 42.

---

## 3. Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Multi-child absorption rate | Parent activation after ablating k children / parent activation before | Compare distributions |
| Feature sensitivity | Steering coefficient variance (Hu et al. 2025) | Pareto frontier |
| Mann-Whitney U | Non-parametric test for distribution difference | p < 0.05 |
| Spearman correlation | Hierarchy strength vs absorption | rho > 0.8 |
| R² (monotonic fit) | Goodness of fit for H_Comp curve | > 0.8 |

---

## 4. Falsification Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H_Mech | Condition B ≈ Condition D, C ≈ A | PASSED (pilot) |
| H_Comp | R² > 0.8 for monotonic fit | NOT TESTED |
| H_Pareto | Detectable frontier shape | NOT TESTED |
| H_Safe | Mann-Whitney p < 0.05 | NOT TESTED |
| H2 (archive) | rho < -0.3 | FAILED (+0.171) |

---

## 5. Expected Visualizations

| Figure | Type | Content | Paper Section |
|--------|------|---------|---------------|
| Figure 1 | Bar chart | H_Mech 2x2 factorial results | 3 (Method) |
| Figure 2 | Line plot | H_Comp absorption vs hierarchy strength | 4 (Results) |
| Figure 3 | Scatter + frontier fit | H_Pareto sensitivity-absorption Pareto | 4 (Results) |
| Figure 4 | Violin plot | H_Safe safety vs non-safety absorption | 4 (Results) |
| Table 1 | Summary table | All hypotheses: status, metric, threshold | 4 (Results) |

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| H_Safe: SAELens Gemma Scope load failure | Low | High | Pre-install check; fallback to GPT-2 Small |
| H_Pareto: Sensitivity metric implementation complex | Medium | Medium | Use simplified steering coefficient |
| H_Comp: Stochastic noise obscures curve | Medium | Medium | 5 seeds for robust fit |
| GPU OOM on Gemma 2B | Low | High | Batch size = 1, sequence length = 128 |

---

## 7. Task Dependencies

```
[setup_sae_env]
       ↓
   ┌───┴───┬─────────────┐
   ↓       ↓             ↓
[h_mech] [h_comp]    [h_pareto]
   │       │             │
   └───────┴─────────────┤
            ↓            │
       [h_safe_gemma]◄───┘
              ↓
       [held_out_validation]
```

All tasks can run independently after setup, enabling parallel execution on 2 GPUs.

---

## 8. Iteration History

From `idea/hypotheses.md`:
- **H1**: Multi-child proportional absorption - PASSED (pilot)
- **H_Mech**: Encoder-driven mechanism - PASSED (pilot, deterministic hierarchy)
- **H2**: Frequency correlation - FAILED (wrong direction: +0.171)
- **H3**: Steering sensitivity - PASSED (1.62x ratio)

This iteration focuses on:
1. Stochastic validation of H_Mech (full 5 seeds)
2. Hierarchy strength dependence (H_Comp)
3. Sensitivity-absorption frontier (H_Pareto)
4. Real SAE safety validation (H_Safe on Gemma Scope)
