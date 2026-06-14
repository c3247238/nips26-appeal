# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024)** - "A is for Absorption" — Defines the differential correlation metric for absorption detection; proves absorption is logical consequence of sparsity loss (Proposition 2). Foundational but does not address metric calibration.

2. **Karvonen et al. (2025)** - SAEBench (ICML 2025) — Standardized benchmark with 8 metrics including absorption. 200+ SAEs evaluated. Critical reference for evaluation protocols. [arXiv:2503.09532]

3. **Korznikov et al. (2025)** - OrtSAE — Demonstrates decoder orthogonality reduces absorption by 65%. Important comparison point: architectural solutions vs. metric validation. [arXiv:2509.22033]

4. **Bussmann et al. (2025)** - Matryoshka SAEs (ICML 2025) — Reduces absorption from 0.49 to 0.05 via nested architecture. Key contrast: they treat absorption as problem to fix; we question whether fixing is necessary. [arXiv:2503.17547]

5. **Wang et al. (ICLR 2026)** - "Does Higher Interpretability Imply Better Utility?" — Shows weak correlation (~0.3) between interpretability and steering utility. Supports our null-result framing. [arXiv:2510.03659]

6. **Kantamneni et al. (2025)** - "Are Sparse Autoencoders Useful?" (ICML 2025) — Negative results on probing tasks. Critical context: SAE utility is questioned independent of absorption.

7. **Various (2026)** - "Sanity Checks for Sparse Autoencoders" — Frozen/random baselines match trained SAEs on standard metrics. Must address directly: our work extends by showing trained < random specifically on absorption metric. [arXiv:2602.14111]

8. **Tang et al. (2025)** - "On the Theoretical Foundation of SDL" — First theoretical explanation for absorption as spurious local minima. Introduces "feature anchoring." Theoretical grounding for rate-distortion framing. [arXiv:2512.05534]

### Experimental Landscape

**What has been properly tested:**
- Absorption exists and is measurable (2-24% across layers/features)
- Null results on downstream task degradation (steering, probing, EC50)
- Precision-recall asymmetry (precision invariant at k>=5, recall varies)
- Random SAE baselines show higher absorption than trained SAEs

**What is accepted without evidence:**
- That absorption is a "failure mode" requiring mitigation
- That Matryoshka/OrtSAE architectural fixes are necessary
- That the Chanin absorption metric is well-calibrated

**Methodological gaps:**
- No systematic cross-model absorption comparison using identical protocols
- No rigorous comparison to random baselines on absorption metrics prior to our work
- No dose-response analysis connecting absorption level to downstream task degradation
- No precision-recall decomposition applied systematically across features

---

## Phase 2: Initial Candidates

Since all experiments are completed, this phase evaluates the **existing experimental design** rather than proposing new ones.

### Candidate A: Optimal Compression Framing (cand_g - front-runner)

- **Core hypothesis**: Feature absorption is rate-distortion optimal compression behavior, not a learned failure mode that degrades interpretability.
- **Falsification criterion**: If absorption significantly degrades steering/probing (r < -0.3, p < 0.05 after MCP), then the optimal compression framing is undermined.
- **Evaluation protocol**: Correlation between absorption rate and downstream task metrics (steering success, probing accuracy, EC50) across 26 features at 2 layers.
- **Ablation plan**: N/A (all experiments completed)
- **Confounders identified**:
  1. Dictionary size effect: Larger dictionaries may show higher absorption artifact
  2. Layer-specific effects: Early vs. late layers may have different absorption profiles
  3. Feature selection bias: 26 first-letter features may not represent full distribution
- **Pilot design**: Already executed — H1-H4 null results, H7 (trained < random)

### Candidate B: Metric Validation Focus (cand_h - fallback)

- **Core hypothesis**: The Chanin absorption metric is not specific to learned structure; it measures dictionary artifacts that training reduces.
- **Falsification criterion**: If trained and random SAEs show similar absorption distributions, the metric validation claim fails.
- **Evaluation protocol**: Direct comparison of trained vs. random SAE absorption distributions
- **Confounders**: Random SAE construction method (orthonormal vs. random Gaussian decoder)
- **Pilot design**: Already executed — H7/H10: trained=0.034 vs. random=0.278, p<0.001

### Candidate C: Downstream Task Impact Focus

- **Core hypothesis**: Absorption level predicts failure on circuit discovery, concept erasure, or steering tasks.
- **Falsification criterion**: If high-absorption features (absorption > 20%) achieve comparable steering to low-absorption features (<5%), the downstream impact claim fails.
- **Evaluation protocol**: Compare steering success rate between high-absorption and low-absorption feature groups
- **Confounders**: Feature semantic difficulty (some features harder to steer regardless of absorption)
- **Pilot design**: Already executed — Feature U (24.2% absorption) achieves 100% steering success

---

## Phase 3: Self-Critique

### Against Candidate A (Optimal Compression Framing)

- **Confound attack**: The random SAE comparison (H7) uses a specific construction (frozen orthonormal decoder, random encoder). Would different random initializations yield the same 8x difference? The result is robust but the mechanism is unclear.

- **Statistical attack**: The null results (H1-H4) have low statistical power due to n=26 features. Effect sizes of r=0.1-0.3 are undetectable with 80% power at alpha=0.05. The study may be underpowered to detect small but meaningful degradation.

- **Benchmark attack**: The steering and probing tasks may not be sensitive enough to detect absorption impact. If absorption causes subtle errors that average out, the metrics would show null results even with real degradation.

- **Ablation completeness attack**: The precision-recall asymmetry (H5) is consistent with optimal compression, but also consistent with many other explanations (encoder noise floor, decoder stability, etc.). The evidence is correlative, not causal.

- **Verdict**: MODERATE — The optimal compression framing is theoretically plausible but not definitively proven. The empirical evidence supports "absorption is not harmful" more than "absorption is optimal compression."

### Against Candidate B (Metric Validation Focus)

- **Confound attack**: The random SAE construction uses orthonormal decoder. Real random SAEs with Gaussian initialization might show different behavior. The orthonormal constraint may artificially inflate absorption.

- **Statistical attack**: The comparison (trained vs. random) uses different encoder distributions. The difference could be due to encoder quality rather than dictionary structure. No ablation of encoder vs. decoder effects.

- **Benchmark attack**: High absorption in random SAEs may be measuring a different phenomenon (numerical instability, orthogonality artifact) rather than absorption per se. The metric may be valid for trained SAEs but invalid for random baselines.

- **Ablation completeness attack**: The result (trained < random) is robust but only tested on one SAE architecture (gpt2-small-res-jb). Generalization to other architectures (GemmaScope, LlamaScope) is unconfirmed.

- **Verdict**: MODERATE — The metric validation insight is valuable but the claim that "absorption is partially a structural artifact" requires cross-architecture validation.

### Against Candidate C (Downstream Task Impact)

- **Confound attack**: Feature U (24.2% abs, 100% steering) is a single case study. The correlation analysis (H1-H4) shows high variance — some high-absorption features may fail. The case study does not rule out population-level effects.

- **Statistical attack**: The steering task uses 6 strength levels and relative probability lift. This is a continuous measure that may not capture binary success/failure thresholds. EC50 analysis addresses this but showed null results.

- **Benchmark attack**: Circuit discovery and concept erasure — the tasks that absorption would theoretically impact most — are not measured. The study only tests steering and probing, not the tasks where absorption is hypothesized to be most harmful.

- **Ablation completeness attack**: Without measuring circuit discovery precision/recall, the claim that "absorption is benign" is incomplete. The null results on steering/probing may not generalize to circuit-level tasks.

- **Verdict**: WEAK — The downstream task impact claim is not fully tested. Circuit discovery and concept erasure tasks are not measured.

---

## Phase 4: Refinement

### Dropped Candidates

- **cand_f (local inhibition graph)**: Falsified in prior iteration (precision@20=0.0). Not reconsidered.

### Strengthened Candidates

**cand_g (optimal compression)** strengthened by:
1. H7/H10 random baseline comparison — provides empirical grounding for "structural artifact" claim
2. H5 (precision-recall asymmetry) — provides theoretical mechanism for why absorption is benign
3. Null results on downstream tasks — rules out major failure modes

**Key remaining concern**: Low statistical power (n=26). Need to address whether small effects could be missed.

### Additional Controls Needed

1. **Cross-architecture validation**: Confirm trained < random on GemmaScope or LlamaScope SAEs (not just GPT-2 Small)
2. **Circuit discovery task**: Measure precision/recall on circuit identification to validate "absorption is benign" claim beyond steering/probing
3. **Effect size power analysis**: Report achieved power for smallest detectable effect (should be ~0.25 given n=26, alpha=0.05, power=0.80)

---

## Phase 5: Final Proposal

### Title
**"Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts"**

### Hypothesis
Feature absorption in SAEs is a structural artifact of overcomplete dictionary learning that training reduces, not a learned failure mode that degrades downstream interpretability tasks.

**Specific predictions:**
- H1: Absorption rate does not correlate with steering effectiveness (r < 0.3, p > 0.05 after MCP)
- H2: Absorption rate does not correlate with sparse probing accuracy (r < 0.3, p > 0.05 after MCP)
- H3: Trained SAEs have lower absorption than random baselines (mean difference > 0.1, p < 0.05)
- H4: Precision is invariant to absorption level; only recall varies

### Falsification Criterion
If any of the following is observed, the hypothesis is undermined:
1. Significant negative correlation (r < -0.3, p < 0.05 after MCP) between absorption and steering/probing
2. Trained SAEs show equal or higher absorption than random baselines
3. High-absorption features (>20%) systematically fail downstream tasks while low-absorption features (<5%) succeed

### Method

**Phase 1: Absorption Detection**
- Chanin differential correlation metric on 26 first-letter features (A-Z)
- GPT-2 Small, layers 0/4/8/10, gpt2-small-res-jb SAE (24K latents)
- 100 samples per feature
- Compare to random SAE baseline (frozen orthonormal decoder, random encoder)

**Phase 2: Downstream Task Evaluation**
- Feature steering at strengths [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- Sparse probing at k={1, 5, 10, 20}
- EC50 dose-response analysis
- Precision-recall decomposition

**Phase 3: Statistical Analysis**
- Multiple comparison correction (Bonferroni + BH-FDR)
- Cross-layer validation (L4, L8)
- Random baseline subtraction (delta-corrected steering)

### Evaluation Protocol

**Primary benchmarks:**
- SAEBench absorption metric (Chanin et al.)
- Steering success rate (relative probability lift)
- k-sparse probing accuracy

**Metrics with statistical tests:**
- Pearson correlation (absorption vs. steering/probing) with 95% bootstrap CI
- Paired t-test (trained vs. random SAE absorption)
- Coefficient of variation (cross-layer consistency)

**Number of random seeds:** 3 (fixed seeds for reproducibility)

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| Random SAE baseline | Metric sensitivity to structure | Random > trained absorption |
| Cross-layer (L4 vs L8) | Layer-specific effects | Consistent null results |
| Delta-corrected steering | Baseline-subtracted steering | Null result survives correction |
| Precision-recall decomposition | Encoder vs. decoder effects | Precision invariant, recall varies |

### Control Experiments

1. **Random steering baseline**: Mean success = 0.344 (L4), 0.379 (L8) — used for delta correction
2. **Random SAE baseline**: mean=0.278 — confirms metric sensitivity to structure
3. **Multiple comparison correction**: 12 tests, Bonferroni alpha=0.00417

### Pilot Design

Already completed:
- Absorption detection on 26 features across 4 layers
- Steering and probing at L4 and L8
- EC50 analysis
- Random SAE comparison

### Resource Estimate

**Completed experiments**: 10 major analyses (absorption detection, steering, probing, EC50, precision-recall, decoder graph, random baseline, cross-model, co-occurrence, baseline validation)

**Remaining work**:
- Cross-architecture validation (GemmaScope): ~2 hours
- Circuit discovery task measurement: ~1 hour
- Paper writing: ~1-2 days

### Risk Assessment

**Biggest threats:**
1. **Low statistical power**: n=26 features may miss small effects (power ~80% for r=0.4, but only ~30% for r=0.2)
2. **Single architecture**: GPT-2 Small only — generalization unconfirmed
3. **Task coverage**: Circuit discovery and concept erasure not measured — claims limited to steering/probing

**Mitigations:**
1. Report achieved power explicitly; acknowledge limitation
2. Recommend cross-architecture validation as future work
3. Frame claims narrowly: "absorption does not degrade steering or probing" not "absorption is harmless"

### Novelty Claim

This is the **first systematic comparison of trained vs. random SAEs on absorption metrics**, demonstrating that:
1. Trained SAEs show significantly lower absorption (mean=0.034) than random baselines (mean=0.278)
2. The Chanin absorption metric is sensitive to dictionary structure, not just learned pathology
3. Absorption does not significantly degrade steering effectiveness or sparse probing accuracy after rigorous multiple comparison correction

The experimental contribution is methodological: baseline correction, precision-recall decomposition, and EC50 analysis as a framework for future SAE evaluation.