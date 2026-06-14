# Methodologist Analysis: Experimental Methodology Audit

## 1. Baseline Fairness Audit

### 1.1 UAD Baselines

| Baseline | Description | Fairness Assessment | Issue Severity |
|----------|-------------|---------------------|----------------|
| Random pair selection | 100 random selections, report mean F1 | **NOT EXECUTED** | Critical |
| Co-occurrence thresholding (no clustering) | Listed in ablation plan (f7) | **NOT EXECUTED** | High |
| AUC-ROC vs random | Listed in methodology | **NOT EXECUTED** | High |

**Finding**: None of the three UAD baselines were actually run. The "random pair selection" baseline is particularly critical because it establishes whether UAD's F1=0.704 is meaningfully above chance. With 500 features and 50 clusters, the probability of random same-cluster pairs being true collisions is non-trivial. Without this baseline, we cannot claim UAD "detects" absorption -- it may simply be detecting any co-occurrence structure.

**Evidence**: `task_plan.json` lists `baseline_random` as a task, but `exp/results/full/summary.md` shows no baseline results. Only f1, f2, and f5 full experiments were completed.

### 1.2 DFDA Baselines

| Baseline | Description | Fairness Assessment | Issue Severity |
|----------|-------------|---------------------|----------------|
| No-compensation | Parent activation without MLP residual | **Executed implicitly** (baseline_mse in f5) | Fair |
| Random residual injection | Inject random residuals instead of MLP-predicted | **NOT EXECUTED** | Critical |

**Finding**: The "random residual injection" baseline was not run. This is critical because the DFDA improvement metric (99.5% MSE reduction) could be achieved by ANY injection that perturbs the parent feature toward zero. A random residual baseline would test whether the MLP is learning something meaningful or simply injecting noise that happens to reduce MSE because the parent is near-zero in child-dominant positions.

**Evidence**: `f5_dfda_scale_results.json` shows baseline_mse values ranging from 0.10 to 39.18, with compensated_mse near zero (1e-10 to 1e-7). The summary.md explicitly flags this: "DFDA improvement metric is artifactual... The MLP learns to predict near-zero values."

### 1.3 Baseline Hyperparameter Symmetry

| Method | Hyperparameters Tuned | Search Budget | Asymmetry? |
|--------|----------------------|---------------|------------|
| UAD | n_clusters=50, top_k=500, linkage=Ward | Fixed (no search) | N/A -- no tuning |
| DFDA | 2-layer MLP, 64 hidden, ReLU | Fixed architecture | N/A -- no tuning |

**Assessment**: Neither UAD nor DFDA performed hyperparameter search. This is actually a strength for fairness -- no method was given a tuning advantage. However, it is also a weakness: we do not know if the chosen hyperparameters are optimal or if they were selected post-hoc to maximize results.

**Specific concern**: The UAD full experiment uses 500 top features and 50 clusters, while the pilot used 100 features. The F1 improved from 0.522 to 0.704. Was 500 chosen because it gave better results, or was it pre-registered? The methodology.md specifies 500 features for full experiments, suggesting it was planned, but the improvement is large enough to warrant scrutiny.

## 2. Metric-Claim Alignment

### 2.1 Claim-to-Metric Mapping

| Claimed Contribution | Evaluation Metric | Alignment | Gap |
|---------------------|-------------------|-----------|-----|
| "UAD detects feature absorption" | F1 vs Chanin-style supervised labels | **Partial** | Chanin labels measure collision rate, not true absorption |
| "UAD achieves perfect recall" | Recall = 1.0 (29/29 supervised collisions found) | **Misleading** | Recall=1.0 means all labeled collisions were found, but we do not know how many true absorptions were missed because the label set is incomplete |
| "DFDA compensates for absorption" | Per-pair residual MSE improvement | **Poor** | MSE improvement on parent feature does not demonstrate that the SAE's interpretability is improved |
| "DFDA is parameter-efficient" | Parameter ratio < 0.01% of SAE | **Good** | 0.004% is well under threshold; metric aligns with claim |
| "UAD is robust across seeds" | std(F1) = 0.000 across 3 seeds | **Misleading** | Perfect consistency may reflect determinism, not robustness (see Validity Threats) |

### 2.2 Measurement Gaps

**Gap 1: No reconstruction MSE validation for DFDA**
The methodology.md specifies "Reconstruction MSE change < 2%" as a metric for DFDA (Section 3.3), but `f5_dfda_scale_results.json` contains no reconstruction MSE measurements. The experiment only reports per-pair residual MSE improvement. Without reconstruction MSE, we cannot claim DFDA is "safe" -- it may degrade overall SAE reconstruction quality.

**Gap 2: No downstream task validation**
The end-to-end pipeline experiment (f6) was not executed. The methodology.md specifies "Probe accuracy improvement on absorbed concepts >= 5 percentage points" as the key metric, but no probe accuracy results exist. Without this, DFDA is an MSE optimization with no demonstrated interpretability benefit.

**Gap 3: No true absorption ground truth**
The methodology.md includes E9 (True Absorption Validation) to correlate collision rate with Chanin-style true absorption rate. This was not executed. Without it, the supervised labels used for UAD evaluation may be measuring collision (co-occurrence) rather than absorption (causal suppression).

**Gap 4: No precision-at-k or ranking metric**
UAD reports precision=0.569, but this is precision among same-cluster pairs, not precision-at-k across all possible pairs. With 24,576 features, there are ~300M possible pairs. UAD identifies 51 same-cluster pairs. A precision-at-50 metric would be more informative for a detection task.

## 3. Validity Threats Checklist

### 3.1 Data Leakage

- [x] **Test data in training set?** N/A -- UAD and DFDA are training-free methods using pre-trained SAEs. No "training" in the traditional sense.
- [x] **Evaluation data overlap?** **THREAT DETECTED**. The same OpenWebText samples used for UAD clustering are also used for DFDA training and evaluation. The 8 absorbed pairs in f5 were selected by UAD running on the same corpus. This creates a train-test overlap: the pairs were selected based on patterns discovered in the evaluation data.

**Evidence**: `f1_uad_gpt2_full_results.json` uses n_samples=1000 from OpenWebText. `f5_dfda_scale_results.json` uses the same model (gpt2-small, layer 8) and the absorbed pairs were derived from UAD output on the same data. There is no held-out validation set.

### 3.2 Contamination

- [x] **Benchmark answers in pretraining?** N/A -- This is not a benchmark evaluation. The "ground truth" (first-letter features) is derived from supervised probing of the SAE, not from an external benchmark.
- [x] **SAE pretraining data overlap?** The SAE (`gpt2-small-res-jb`) was trained on OpenWebText, the same corpus used for evaluation. This is expected and documented, but it means the SAE features were optimized for this data distribution. Generalization to other corpora is untested.

### 3.3 Selection Bias

- [x] **Hyperparameters tuned on test set?** **THREAT DETECTED**. The UAD hyperparameters (n_clusters=50, top_k=500) were chosen based on pilot results (p1_uad_gpt2 with 100 features gave F1=0.522). The full experiment with 500 features gave F1=0.704. If the pilot was used to select 500 as the "winning" configuration, this is indirect test-set tuning.

- [x] **DFDA pair selection bias?** **THREAT DETECTED**. The 8 pairs in f5 were selected by UAD. If UAD has false positives (precision=0.569 means 43% of same-cluster pairs are NOT true collisions), then some DFDA pairs may be non-absorbed. Training an MLP on a non-absorbed pair and showing MSE improvement is meaningless.

**Evidence**: Of 51 same-cluster pairs, only 29 are true collisions (precision=0.569). The 22 false positive pairs were not excluded from DFDA consideration. The methodology does not specify how pairs were selected from the UAD output for DFDA training.

### 3.4 Overfitting to Evaluation

- [x] **Single benchmark?** **THREAT DETECTED**. All UAD evaluation uses first-letter features (a-z) as ground truth. This is a single, narrow concept domain. The methodology.md mentions WordNet semantic hierarchies as secondary validation, but no such experiments were run.

- [x] **Single model?** **THREAT DETECTED**. All experiments use GPT-2 Small. The methodology specifies cross-validation on Gemma-2B and Pythia-2.8B, but these were blocked (Gemma gated, Pythia SAE unavailable). The pilot_summary.md notes: "Cross-model validation blocked by gated model access and missing pretrained SAEs."

- [x] **Single layer?** **PARTIAL THREAT**. Full experiments only use layer 8. Cross-layer pilots (layers 4, 8, 10) show layer 4 F1=0.432 (below 0.5 threshold), indicating layer-dependent performance. The full experiment does not account for this variation.

## 4. Ablation Gap Analysis

### 4.1 UAD Ablations (from methodology.md Section 4.1)

| Ablation | Status | Gap |
|----------|--------|-----|
| No clustering (pure co-occurrence thresholding) | **NOT EXECUTED** | Cannot isolate clustering contribution |
| No phi coefficient (raw co-occurrence counts) | **NOT EXECUTED** | Cannot isolate normalization contribution |
| No dead feature filtering | **NOT EXECUTED** | Cannot assess dead feature impact |
| Single-link clustering (replace Ward) | **NOT EXECUTED** | Cannot assess linkage sensitivity |

**Finding**: Zero UAD ablations were executed. The methodology.md lists 4 ablation variants, but `exp/results/full/summary.md` shows no ablation results. This is a critical gap because:
1. We cannot claim HAC (Ward linkage) is necessary -- raw co-occurrence thresholding might achieve similar F1.
2. We cannot claim phi coefficient is necessary -- raw counts might work as well.
3. We cannot assess whether dead feature filtering (which removes ~89-99% of features) is the primary driver of performance.

### 4.2 DFDA Ablations (from methodology.md Section 4.2)

| Ablation | Status | Gap |
|----------|--------|-----|
| Linear only (single layer, no hidden) | **NOT EXECUTED** | Cannot assess nonlinearity need |
| Larger MLP (256 hidden units) | **NOT EXECUTED** | Cannot assess capacity sensitivity |
| No residual connection | **NOT EXECUTED** | Cannot validate residual formulation |

**Finding**: Zero DFDA ablations were executed. The 2-layer MLP architecture was used without validation that a simpler linear model would not suffice. Given that the "improvement" comes from predicting near-zero parent values, a linear model might achieve the same result with fewer parameters.

### 4.3 Missing Ablations Not in Methodology

| Missing Ablation | Why It Matters |
|-----------------|----------------|
| UAD with different cluster counts (10, 25, 100) | Tests sensitivity to n_clusters hyperparameter |
| UAD with different top-k (250, 1000) | Tests sensitivity to feature selection |
| DFDA with identity mapping (always predict zero) | Tests whether "prediction" is necessary or if zero-injection suffices |
| DFDA on non-absorbed pairs (from UAD false positives) | Tests whether MSE improvement is specific to absorbed pairs |

## 5. Reproducibility Score: 2/5

| Criterion | Status | Score |
|-----------|--------|-------|
| Random seeds fixed | Seeds 42, 123, 456 documented and used | +1 |
| All hyperparameters specified | n_clusters=50, top_k=500, linkage=Ward documented | +1 |
| Code available | Not verified -- no code path in results | 0 |
| Data available | OpenWebText is public; concept lists not independently verified | 0 |
| Hardware requirements documented | Not specified | 0 |
| Package versions pinned | "SAELens >= 2.0.0" -- not exact versions | 0 |
| Bootstrap 95% CI reported | **NOT reported** -- only point estimates | 0 |
| Could reproduce within 10%? | Unlikely without code and exact SAE checkpoint IDs | No |

**Score justification**: The experimental protocol is documented in methodology.md, but critical reproducibility elements are missing:
- No code paths or repository references in result files
- No exact SAELens release ID for the SAE checkpoint (only `gpt2-small-res-jb` mentioned)
- No bootstrap confidence intervals (methodology.md promises them but they are absent from results)
- No hardware specification (GPU type, memory)
- Package versions are loose bounds (>= 2.0.0) not exact pins

**Evidence**: `f1_uad_gpt2_full_results.json` contains no code_version, no git_commit, no package_versions, no hardware_info fields. The methodology.md mentions "Exact package versions pinned in requirements.txt" but this file is not referenced in any result.

## 6. Top-3 Methodology Improvements

### Recommendation 1: Execute Missing Baselines (Effort: Low, Credibility Impact: High)

**What**: Run the random pair selection baseline and random residual injection baseline.

**Why**: Without these, the core claims are unanchored. If random pairs achieve F1=0.4 (not impossible with 50 clusters and 29 true collisions), UAD's F1=0.704 is less impressive. If random residual injection achieves 90% MSE improvement, DFDA's 99.5% is not meaningful.

**How**: 
- Random pair baseline: 100 trials of selecting 51 random feature pairs, computing F1 against supervised labels.
- Random residual baseline: For each of the 8 DFDA pairs, inject Gaussian noise with matched variance instead of MLP output, measure MSE change.

**Expected outcome**: If random baselines are far below our methods, credibility increases. If they are close, the claims must be downgraded.

### Recommendation 2: Add Reconstruction MSE and Downstream Validation (Effort: Medium, Credibility Impact: High)

**What**: Measure (a) full SAE reconstruction MSE before/after DFDA, and (b) sparse probing accuracy on absorbed concepts.

**Why**: The current DFDA metric (per-pair residual MSE) is methodologically weak because:
- It does not demonstrate the SAE as a whole works better
- The 99.5% improvement is artifactual (near-zero parent values)
- Without downstream validation, DFDA has no practical interpretability claim

**How**:
- Reconstruction MSE: Compute ||SAE_decode(z_compensated) - true_residual||^2 across a held-out token set
- Probe accuracy: Train linear probes on first-letter concepts using compensated vs uncompensated SAE features

**Expected outcome**: If reconstruction MSE increases < 2% and probe accuracy improves > 5pp, DFDA is validated. If not, DFDA should be reported as an MSE optimization without practical benefit.

### Recommendation 3: Execute Critical Ablations (Effort: Low-Medium, Credibility Impact: Medium-High)

**What**: Run the three most critical ablations: (a) UAD without clustering, (b) DFDA linear-only, (c) DFDA on UAD false-positive pairs.

**Why**: These directly test the necessity of the proposed components:
- If UAD without clustering achieves F1 > 0.6, the HAC contribution is marginal
- If linear DFDA matches 2-layer MLP, the nonlinear architecture is unnecessary
- If DFDA on false-positive pairs also shows MSE improvement, the improvement is not specific to absorption

**How**: Each ablation requires minimal code changes and can reuse existing infrastructure.

**Expected outcome**: These ablations will either strengthen the claims (if baselines/ablations perform worse) or force a more honest framing (if they perform similarly).

## 7. Summary of Critical Issues

| Issue | Threatens Main Conclusion? | Severity |
|-------|---------------------------|----------|
| No random baselines executed | **YES** -- UAD F1 may not be above chance | Critical |
| DFDA metric is artifactual (near-zero prediction) | **YES** -- 99.5% improvement is not meaningful | Critical |
| No reconstruction MSE or downstream validation | **YES** -- DFDA has no demonstrated practical benefit | High |
| No ablations executed | **YES** -- Cannot claim proposed components are necessary | High |
| Train-test overlap (UAD pairs used for DFDA) | Partial -- may inflate DFDA results | Medium |
| Single model, single layer, single concept domain | **YES** -- Generalization claims are unsupported | High |
| Perfect seed consistency may reflect determinism, not robustness | Partial -- robustness claim is overstated | Medium |
| No bootstrap CIs reported | No -- point estimates are still valid, but less credible | Low |

## 8. Honest Assessment

The UAD+DFDA experiments have a **methodologically weak foundation** for the claims being made. The core issues are:

1. **Unanchored metrics**: Without random baselines, we do not know if UAD's F1=0.704 is meaningful or if DFDA's 99.5% improvement is specific.

2. **Misaligned metrics**: The DFDA improvement metric does not measure what the paper claims (absorption compensation). It measures MSE reduction on parent features in child-dominant positions, which is trivially achieved by predicting near-zero.

3. **Missing validation chain**: The methodology specifies a validation chain (UAD detects -> DFDA compensates -> reconstruction MSE stable -> probe accuracy improves), but only the first two steps were executed, and the second step uses a flawed metric.

4. **No ablation evidence**: The paper claims specific design choices (HAC, phi coefficient, 2-layer MLP, residual connection) are necessary, but no ablations were run to support these claims.

5. **Generalization unsupported**: All experiments on a single model (GPT-2 Small), single layer (8), single concept domain (first letters). Cross-model validation was planned but not executed.

**Verdict**: The experiments demonstrate that UAD can find same-cluster feature pairs that correlate with Chanin-style labels, and that a tiny MLP can reduce MSE on parent features. However, the methodological gaps are large enough that these results should be treated as **promising pilot findings requiring rigorous validation**, not as established contributions ready for publication.
