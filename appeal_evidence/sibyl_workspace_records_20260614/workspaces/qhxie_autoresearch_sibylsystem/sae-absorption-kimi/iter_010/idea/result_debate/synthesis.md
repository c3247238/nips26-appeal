# Result Debate Synthesis: Iteration 10 Pilot Experiments

## 1. Consensus Map: What All 6 Perspectives Agree On

The following conclusions enjoy broad agreement across all perspectives and represent high-confidence findings:

| # | Consensus Finding | Supporting Evidence |
|---|-------------------|---------------------|
| 1 | **Absorption rate is a discriminative metric** | Monotonic ordering: Random (0.452) > Baseline (0.226) > TopK (0.033) / MultiScale (0.042). Successfully distinguishes trained from random, and sparse from non-sparse. |
| 2 | **TopK and MultiScale both achieve dramatically lower absorption than Baseline** | Magnitude: 85.5% reduction (TopK) and 81.6% reduction (MultiScale). Effect size is enormous (Cohen's d = 5.51). |
| 3 | **MCC is insensitive to the manipulations** | Variation of only ~0.006 across all variants, including Random. All perspectives agree MCC should not be relied upon as a primary metric. |
| 4 | **Explained variance is negative for all trained models** | All variants show negative values (-0.93 to -0.34), though Random is far worse (-118.6). Training improves reconstruction relative to random, but not above the mean-prediction baseline. |
| 5 | **Dead neurons are a significant issue** | TopK: 83.2%, MultiScale: 58.4%. This is a real tradeoff that must be transparently reported. |
| 6 | **The current result is from a single-seed pilot** | All agree this provides directional evidence but not statistical proof. Multiple seeds are needed. |

## 2. Conflict Resolution: Where Perspectives Disagree

### Conflict A: Is the 85.5% reduction real or an artifact?

- **Optimist**: Real and transformative. Cross-architecture validation (TopK + MultiScale both work) proves the effect is not architecture-specific.
- **Skeptic**: Potentially driven by dead neurons. With 83% dead neurons, the pool of co-activation candidates is mechanically reduced.
- **Methodologist**: Both views have merit. The effect is real in the sense that it replicates (iter_005 showed 78% reduction), but the dead-neuron mechanism must be isolated via active-neuron-only analysis.
- **Resolution**: The effect is **directionally real** but the exact magnitude may be inflated by dead neurons. The active-neuron-only analysis (no training required) will resolve this.

### Conflict B: Is "absorption is a sparsity phenomenon" the right story?

- **Optimist / Comparativist**: Yes. Both TopK and MultiScale converge to L0=50 and achieve similar absorption rates. Sparsity level, not architecture, is the driver.
- **Skeptic / Methodologist**: Unproven. The comparison confounds architecture and sparsity. Baseline was not L0-matched. Until a Baseline with L0=50 is tested, "sparsity causes lower absorption" is a hypothesis, not a conclusion.
- **Revisionist**: Intermediate position. Sparsity is likely the primary factor, but architecture may have secondary effects via activation distribution shape.
- **Resolution**: The "sparsity story" is the **leading hypothesis** but requires the L0-matched control experiment for confirmation. If L0-matched Baseline also shows low absorption, the story is confirmed. If not, the story becomes "architecture matters, and some architectures happen to enforce sparsity."

### Conflict C: How strong is the competitive position?

- **Comparativist**: Moderate novelty. The finding aligns with Anthropic's qualitative observations but adds quantification and cross-architecture validation. The dose-response relationship (k-sweep) would be genuinely novel.
- **Skeptic**: Low novelty risk. "Sparsity reduces co-activation" may be seen as obvious by reviewers.
- **Resolution**: The **dose-response curve** (absorption vs. k) and the **active-neuron analysis** are the strongest novelty claims. The core "sparsity reduces absorption" finding needs framing as "quantitative evidence for a previously qualitative observation" plus "architectural independence."

### Conflict D: Should we proceed or pivot?

- **Optimist**: Full steam ahead. The results are strong enough to justify full experiments.
- **Skeptic**: Proceed with caution. Multiple methodological issues must be resolved before claiming discovery.
- **Strategist**: ADVANCE with structured follow-up. The upside (workshop paper guaranteed, conference possible) justifies the investment.
- **Resolution**: **PROCEED with the full experiment plan**, but with explicit go/no-go criteria after each stage.

## 3. Result Quality Score: 6.5 / 10

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Effect size | 9/10 | 85.5% reduction is enormous and replicates across iterations. |
| Metric validity | 7/10 | Absorption rate discriminates well, but threshold sensitivity untested. |
| Internal validity | 5/10 | L0 confound, dead-neuron artifact, single seed all threaten causal inference. |
| External validity | 4/10 | Only synthetic data. Real LLM validation pending. |
| Reproducibility | 6/10 | Code and data are structured, but 5 seeds needed for variance estimation. |
| Theoretical grounding | 6/10 | Connection to polysemanticity/superposition is plausible but not formally proven. |

**Overall: 6.5/10** — Directionally compelling but methodologically incomplete. The score reflects that the core finding is likely real, but multiple control experiments are needed before it can be claimed with confidence.

## 4. Key Findings: What We Actually Learned

1. **Sparse SAE architectures (TopK, MultiScale) achieve 80-85% lower absorption rates than standard ReLU SAEs** on synthetic data. This is a large, consistent effect that replicated from iter_005 to iter_010.

2. **Absorption rate successfully discriminates trained models from random weights**, and sparse models from non-sparse models. The monotonic ordering (Random > Baseline > Sparse) validates the metric as capturing meaningful structure.

3. **MCC is not a useful metric for this experimental setting**. Its near-constant value across all conditions (including Random) suggests it measures something orthogonal to the manipulations, or that the "one-neuron-one-feature" assumption underlying MCC is violated.

4. **All trained models have negative explained variance**, meaning SAE reconstruction is worse than mean prediction. This is concerning for the interpretability value of the learned features, but does not invalidate the absorption findings since absorption compares trained models to each other.

5. **Dead neurons are a major side effect** (83% for TopK, 58% for MultiScale) and may mechanically contribute to lower absorption. The active-neuron-only analysis is critical to disentangle this.

## 5. Methodology Gaps: Critical Experimental Improvements Needed

| Priority | Gap | Experiment Required | Effort |
|----------|-----|---------------------|--------|
| **P0** | L0 confound | L0-matched Baseline (force L0=50 via L1 penalty) | 1 training run |
| **P0** | Dead neuron artifact | Active-neuron-only absorption rate analysis | No training; analysis only |
| **P0** | Statistical power | 5-seed replication (already planned) | 15 training runs |
| **P1** | Dose-response | k-sweep (k in {10, 25, 50, 100, 200, 500}) | 6 training runs |
| **P1** | External validity | Gemma Scope 2B validation | 2-3 training runs |
| **P2** | Metric robustness | Threshold sensitivity analysis | No training; analysis only |
| **P2** | Mechanism | Training dynamics (absorption vs. epoch) | 1 training run with logging |

## 6. Competitive Position: Where We Stand vs. SOTA

### Relationship to Key Papers

| Paper | Their Contribution | Our Addition |
|-------|-------------------|--------------|
| Bricken et al. (2023) "Towards Monosemanticity" | Qualitative: larger dictionaries -> more monosemantic features | Quantitative: absorption rate as a computable proxy for polysemanticity |
| Anthropic (2024) "Scaling Monosemanticity" | Qualitative: sparser SAEs -> more monosemantic | Dose-response curve + architectural independence proof |
| Gao et al. (2024) TopK SAE | TopK improves reconstruction-sparsity tradeoff and interpretability | **New**: TopK also reduces feature absorption (unreported benefit) |
| JumpReLU / Matryoshka SAE | Reduce dead neurons via learned thresholds | MultiScale achieves lower dead-neuron rate than TopK with slightly higher absorption |

### Novelty Assessment

| Claim | Novelty | Confidence |
|-------|---------|------------|
| Absorption rate as a metric | Medium | Systematization of co-activation concept |
| Sparsity reduces absorption | Medium | Quantitative confirmation of qualitative prior art |
| Cross-architecture validation | Medium | Both TopK and MultiScale work |
| Dose-response (absorption ~ k) | **High** | Continuous relationship not previously reported |
| Dead-neuron / absorption tradeoff | Medium | Quantification of known issue |

### Target Venues

- **High confidence**: ACL/EMNLP interpretability workshop (with 5 seeds + L0 control)
- **Medium confidence**: NeurIPS/ICML XAI track (with + k-sweep + real LLM validation)
- **Low confidence**: Full conference paper (requires + mechanism analysis + large-scale validation)

## 7. Hypothesis Update: What Survived, What Needs Revision

### Survived (supported by evidence)

| Hypothesis | Prior Confidence | Current Confidence | Evidence |
|------------|-----------------|-------------------|----------|
| H1: Sparse SAEs have lower absorption than dense SAEs | 0.70 | **0.85** | Pilot strongly supports; replicates iter_005 |
| H2: Absorption rate is a valid discriminative metric | 0.80 | **0.75** | Discriminates trained/random/sparse, but MCC failure suggests caution |

### Needs Revision

| Hypothesis | Prior Confidence | Current Confidence | Revision Needed |
|------------|-----------------|-------------------|-----------------|
| H3: Architecture does not matter (only sparsity) | 0.30 | **0.60** | L0-matched control required before upgrading to "confirmed" |
| H4: Results generalize to real LLMs | 0.50 | **0.50** | No evidence yet; Gemma Scope test pending |

### Rejected / Deprecated

| Hypothesis | Status | Reason |
|------------|--------|--------|
| MCC is a useful metric for this setting | **Rejected** | Near-zero variation across all conditions including Random |
| Explained variance is a useful quality signal | **Deprecated** | All models negative; may reflect synthetic data properties rather than model quality |

### Updated Causal Model

```
Pre-pilot (v1.0):  absorption <-- architecture choice
                    "TopK is better because it's TopK"

Post-pilot (v2.0): absorption <-- sparsity level <-- architecture choice
                    "TopK is better because it enforces sparsity"
                    (architecture effect is indirect)

Target (v3.0):     absorption <-- f(sparsity, activation_distribution, feature_correlation)
                    "Sparsity is primary, but distribution shape matters"
                    (to be validated by L0-matched control)
```

## 8. Action Plan: Prioritized Next Steps

### Verdict: **PROCEED** with structured go/no-go gates.

The core finding (sparsity reduces absorption) is directionally supported by strong effect sizes and cross-architecture replication. The methodological gaps are addressable with ~2 hours of additional GPU time. The risk of the full experiment failing to replicate is low enough to justify the investment.

### Stage 1: Immediate (Next Iteration) — P0 Experiments

1. **5-seed full experiment** (Baseline, TopK, MultiScale on SynthSAEBench-16k)
   - Expected: 15 training runs, ~45 min GPU time
   - Go/no-go: If 5-seed mean effect size < 50% reduction, reconsider story

2. **L0-matched Baseline control** (force Baseline L0=50 via L1 penalty)
   - Expected: 1 training run, ~3 min GPU time
   - Go/no-go: If L0-matched Baseline absorption < 0.10, the "sparsity story" is confirmed. If > 0.15, the story must shift to "architecture matters."

3. **Active-neuron-only absorption analysis**
   - Expected: No training; post-hoc analysis on existing pilot data
   - Go/no-go: If active-only absorption still shows > 60% reduction, dead neurons are not the primary driver.

### Stage 2: Following Iteration — P1 Experiments

4. **TopK k-sweep** (k in {10, 25, 50, 100, 200, 500})
   - Expected: 6 training runs, ~18 min GPU time
   - Purpose: Establish dose-response curve; strongest novelty claim

5. **Gemma Scope 2B validation**
   - Expected: 2-3 training runs, ~30 min GPU time
   - Purpose: External validity; critical for conference-level claims
   - Go/no-go: If effect direction does not replicate, scope paper to synthetic-only with explicit limitation

### Stage 3: Writing & Submission

6. **Workshop submission** (if Stages 1-2 succeed partially)
   - Target: ACL/EMNLP interpretability workshop
   - Requirements: 5 seeds + L0 control + active-neuron analysis

7. **Conference submission** (if Stage 2 succeeds fully)
   - Target: NeurIPS/ICML XAI track
   - Requirements: + k-sweep + Gemma Scope validation + mechanism discussion

### Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| 5-seed effect size smaller than pilot | Medium | Pre-register L0-matched control as alternative story |
| Gemma Scope does not replicate | Medium | Report as synthetic-only with explicit external validity limitation |
| Reviewer finds "obvious" | High | Emphasize (a) quantitative evidence where prior work was qualitative, (b) architectural independence, (c) dose-response curve |
| Dead neurons dominate discussion | Medium | Lead with active-neuron-only analysis; transparently report tradeoff |

---

*Synthesized from 6 perspectives: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist*
*Iteration: 10 | Date: 2026-04-26*
