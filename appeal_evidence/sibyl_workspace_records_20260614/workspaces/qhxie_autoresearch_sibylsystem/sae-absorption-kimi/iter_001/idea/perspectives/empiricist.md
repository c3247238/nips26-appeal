# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024)** — *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders* (arXiv:2409.14507). Introduces the canonical absorption metric via first-letter classification, k-sparse probing, and integrated-gradients ablation. Key methodological insight: absorption is detected when main latents fail on probe-true tokens and a compensating latent aligns with the probe direction (cosine similarity >= 0.025) and dominates the runner-up in ablation effect by >= 1.0. [Source](https://arxiv.org/abs/2409.14507)

2. **Karvonen et al. (2025)** — *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability* (arXiv:2503.09532). The dominant comprehensive benchmark with 8 evaluations. Critical methodological finding: "The sparsity-fidelity frontier does not reliably indicate performance on downstream tasks." Architecture rankings differ dramatically by metric—Matryoshka BatchTopK leads on concept detection/absorption while ReLU SAEs show highest absorption after proper training. [Source](https://arxiv.org/abs/2503.09532)

3. **Chanin & Garriga-Alonso (2025)** — *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders* (arXiv:2505.11756). Establishes that narrow SAEs systematically merge correlated features into compromised combinations due to reconstruction (MSE) pressure—distinct from absorption (sparsity-driven) but creating a width-dependent confound that any multi-objective evaluation must control for. [Source](https://arxiv.org/abs/2505.11756)

4. **Neuronpedia SAEBench Blog (Dec 2024)** — Revealed a severe evaluation confound: dead features can artificially lower absorption metrics. After improving ReLU training from "Towards Monosemanticity" to "Anthropic April Update," ReLU SAEs showed the *highest* absorption levels. Quote: "the decreased feature absorption in our original results was due to a high percentage of dead features." This means dead-neuron rate must be treated as a covariate, not an independent metric. [Source](https://www.neuronpedia.org/sae-bench/info)

5. **Korznikov et al. (2026)** — *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* (arXiv:2602.14111). Introduces frozen-decoder and random-decoder baselines that match trained SAEs on AutoInterp, sparse probing, and RAVEL. Strikingly, they did NOT measure absorption on these baselines. This creates an uncontrolled comparison: if random baselines also show absorption, the metric may reflect geometric structure rather than training quality. [Source](https://arxiv.org/abs/2602.14111)

6. **Wang et al. (2025)** — *Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders* (arXiv:2510.03659). Finds Kendall's tau_b ≈ 0.30 between SAEBench interpretability scores and AxBench steering utility. After selecting top steering features, the correlation collapses to ~0 or negative. Methodological implication: downstream "interpretability utility" is weakly coupled to absorption and other proxy metrics; causal claims require direct intervention experiments, not correlation. [Source](https://arxiv.org/abs/2510.03659)

7. **Minegishi et al. (2025)** — *Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words* (arXiv:2501.06254). Critiques the narrow MSE-L0 Pareto frontier as insufficient for interpretability. Found ranking for monosemantic feature extraction is ReLU > JumpReLU > TopK—opposite of the MSE-L0 ranking. This demonstrates that metric choice determines conclusion, making multi-metric Pareto analysis essential but also treacherous if metrics are incommensurable. [Source](https://arxiv.org/abs/2501.06254)

8. **Chanin & Garriga-Alonso (2025)** — *Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders* (arXiv:2508.16560). Shows that at low L0, incorrect SAEs with mixed features can achieve *better* reconstruction than correct SAEs. Proposes decoder pairwise cosine similarity (c_dec) as a proxy for correct L0. Methodological implication: L0 is not a free hyperparameter; comparing SAEs at mismatched L0s confounds architecture effects with correctness effects. [Source](https://arxiv.org/abs/2508.16560)

9. **Pacela et al. (2026)** — *Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation* (arXiv:2603.28744). Reframes SAE failure as a dictionary learning challenge. SAE-learned dictionaries point in wrong directions under OOD compositional shifts; the problem is not amortisation but dictionary quality. Cautionary for any pilot using in-distribution probes to validate SAE quality. [Source](https://arxiv.org/abs/2603.28744)

10. **Chanin & Garriga-Alonso (2026)** — *SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data* (arXiv:2602.14687). Provides ground-truth synthetic benchmark with controlled hierarchy and superposition. Key finding: Matching Pursuit SAEs achieve best explained variance but poor feature recovery (MCC)—a new failure mode where reconstruction overfits without learning true features. Validates that reconstruction metrics can be actively misleading. [Source](https://arxiv.org/abs/2602.14687)

### Experimental Landscape

**What has been properly tested:**
- Feature absorption exists across hundreds of LLM SAEs and is caused by sparsity optimization under hierarchical features (Chanin et al., 2024).
- Specific architectures (OrtSAE, Matryoshka) reduce absorption on the first-letter metric relative to Standard/TopK baselines.
- SAEBench provides a reproducible, automated evaluation pipeline for absorption and 7 other metrics.

**What is accepted without proper evidence:**
- That absorption reduction translates to improved downstream interpretability utility. The Wang et al. (2025) result suggests this link is weak (tau ≈ 0.30).
- That absorption is a training pathology rather than a geometric inevitability. Korznikov et al. (2026) showed random baselines match trained SAEs on standard metrics but did NOT test absorption.
- That the first-letter absorption metric generalizes to arbitrary semantic hierarchies. No task-agnostic validation exists.

**Where methodological gaps exist:**
1. **Dead-neuron confounding:** Absorption metrics are systematically biased by dead-feature rates, yet most comparisons do not control for this.
2. **L0 mismatch confounding:** Architecture comparisons often span different L0s, making it impossible to separate architecture effects from "correctness" effects (Sparse but Wrong).
3. **Metric incommensurability:** MSE-L0, absorption, hedging, and downstream utility often rank architectures differently. No prior work systematically maps the full Pareto front across all metrics simultaneously.
4. **Missing random baseline for absorption:** The sanity-check result has not been extended to absorption, leaving open whether the metric measures training quality or geometric structure.
5. **Weak causal evidence for downstream harm:** Most claims about absorption's practical impact rest on correlation or intuition, not controlled intervention.

---

## Phase 2: Initial Candidates

### Candidate A: Multi-Objective Pareto Evaluation of Absorption-Mitigation Methods

- **Hypothesis:** Absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, Masked Regularization) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Each method occupies a distinct tradeoff region.
- **Falsification criterion:** If one architecture family shows statistically significant stochastic dominance (Mann-Whitney U test, p < 0.05) across >= 4 out of 5 metrics, the hypothesis is falsified.
- **Evaluation protocol:**
  - Assemble 50-100 pretrained SAE checkpoints per model (Gemma-2-2B or GPT-2 Small / Pythia-160M) across 5+ architecture families.
  - Compute: absorption (sae-spelling / SAEBench), hedging (Chanin et al. 2025), explained variance, CE loss recovered, dead-neuron fraction, sparse probing F1, RAVEL Cause/Isolation.
  - Normalize metrics to [0,1] within model family.
  - Compute empirical Pareto fronts and test stochastic dominance with Mann-Whitney U.
  - Control for L0 by including it as a covariate in regression; cluster SEs by architecture family.
- **Ablation plan:**
  - Remove hedging metric: does the Pareto conclusion change? (Tests sensitivity to the width-narrowness confound.)
  - Stratify by L0 quartile: does any architecture dominate within a narrow L0 band? (Tests Sparse but Wrong confound.)
  - Exclude checkpoints with >30% dead neurons: does absorption ranking reverse? (Tests dead-neuron confound.)
- **Confounders identified:**
  - L0 mismatch: architectures may have different sparsity distributions.
  - Dead-neuron rate: can artificially improve absorption.
  - Layer depth: deeper layers show different metric baselines.
  - Dictionary width: wider dictionaries have more capacity but also more absorption opportunities.
- **Pilot design:** Run the full metric pipeline on 10 checkpoints (2 per architecture family) on GPT-2 Small with 50k-100k tokens. Expected duration: 10-15 minutes.

### Candidate B: Task-Agnostic Absorption Metric via Automated Hierarchy Discovery

- **Hypothesis:** A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with causal ablation, will correlate moderately-to-strongly (Pearson r > 0.4) with the original first-letter absorption benchmark while enabling absorption measurement across arbitrary semantic domains.
- **Falsification criterion:** If Pearson r < 0.3 between the task-agnostic metric and the first-letter benchmark across 20-50 SAEs, the automated hierarchy discovery is insufficiently reliable and the hypothesis is falsified.
- **Evaluation protocol:**
  - Select GemmaScope 16K (Gemma-2-2B, layer 12) or equivalent open-model SAE.
  - Use LLM judge to label top-N active features and organize into validated hierarchies for 2-3 domains (geography, biology, colors).
  - Train logistic regression probes for each parent-child concept.
  - Run absorption detection: k-sparse probing -> false negatives -> integrated-gradients ablation -> absorption classification.
  - Correlate new metric with SAEBench first-letter absorption scores.
- **Ablation plan:**
  - Vary the cosine-similarity threshold for absorber detection (0.015, 0.025, 0.05).
  - Vary the number of LLM-labeled hierarchies (5, 10, 20).
  - Test on multiple SAE widths (4K, 16K, 65K) to check metric stability.
- **Confounders identified:**
  - LLM hallucination in hierarchy construction.
  - Domain-specific absorption rates (some domains may be more hierarchical than others).
  - Probe quality variation across domains.
- **Pilot design:** Run hierarchy discovery and absorption detection on one SAE x one domain (geography: continent -> country) with 5-10 parent-child pairs. Expected duration: 10-15 minutes.

### Candidate C: Random-Decoder Baseline Test for Absorption

- **Hypothesis:** Feature absorption in sparse autoencoders is primarily a geometric consequence of sparse dictionary learning on hierarchically structured data, not a pathology of flawed training dynamics. Randomly initialized, frozen-decoder SAEs matched for L0 will exhibit absorption rates comparable to fully trained SAEs (within 20% relative).
- **Falsification criterion:** If random-decoder SAEs show <50% of the trained SAE absorption rate (relative), training dynamics are the dominant cause and the hypothesis is falsified.
- **Evaluation protocol:**
  - Load trained SAE (gpt2-small-res-jb, layer 8).
  - Initialize random-decoder SAE with frozen decoder; train encoder to matched L0 using TopK for exact control.
  - Run sae-spelling absorption metric on both.
  - Report mean absorption rate, full absorption rate, feature splitting rate, L0, MSE, dead-neuron fraction.
- **Ablation plan:**
  - Test soft-frozen decoder (cosine similarity >= 0.8 constraint) vs. hard-frozen decoder.
  - Test multiple random seeds for decoder initialization.
  - Test at multiple target L0s (16, 32, 64).
- **Confounders identified:**
  - Random decoder may fail to converge to reasonable MSE, making the comparison unfair.
  - TopK enforces exact L0 but changes the optimization landscape compared to L1-regularized training.
  - The first-letter task may not be representative of general absorption behavior.
- **Pilot design:** One trained SAE vs. one random-decoder SAE on GPT-2-small, layer 8, trained to matched L0 on ~1M tokens. Expected duration: ~15 minutes GPU time.
- **Constraint note:** This candidate **violates the project's training-free constraint** and can only proceed if explicitly relaxed.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Confound attack:** The pilot already revealed a critical confound: Gemma-2-2B is gated and inaccessible, forcing a fallback to GPT-2 Small. GPT-2 Small has far fewer pretrained SAE families available, especially absorption-mitigation architectures like OrtSAE and Matryoshka, which were primarily evaluated on Gemma-2-2B. This creates a **selection bias** where the most interesting architectures cannot be included. Additionally, L0 and dead-neuron rate are deeply entangled with absorption (Neuronpedia finding), and the pilot's simplified absorption proxy returned 0.0 for most checkpoints—a warning that the metric may be degenerate at small scales or for certain families.
- **Statistical attack:** With 50-100 checkpoints and 5+ architecture families, we have ~10-20 checkpoints per family. Mann-Whitney U tests with n=10-20 per group have limited power to detect dominance. The expected effect size is moderate (architecture differences of 10-30% on normalized metrics), but with 5 metrics and multiple comparisons, familywise error rate inflation is a real concern. A Bonferroni correction across 5 metrics x 10 pairwise comparisons would require p < 0.001 for significance.
- **Benchmark attack:** SAEBench metrics are proxy metrics. The Wang et al. (2025) result shows interpretability-utility correlation is only ~0.30. Even if we find no architecture dominates on SAEBench, this does not prove the field should abandon absorption research—it only proves the proxies are inconsistent. The practical relevance depends on whether the proxies predict real interpretability utility, which is itself disputed.
- **Ablation completeness attack:** Removing hedging or dead neurons tests sensitivity, but what if two components compensate? For example, Matryoshka's multi-scale design might reduce absorption at the coarse scale while increasing it at the fine scale. A single aggregate absorption score could wash out this structure. We are not planning to ablate by scale or by hierarchy depth.
- **Verdict:** MODERATE. The idea is well-motivated and novel, but the pilot exposed serious feasibility issues (Gemma gating, proxy degeneracy, limited architecture diversity on open models). The statistical power is borderline, and the causal leap from proxy inconsistency to "intrinsic tradeoff" is larger than the data can support.

### Against Candidate B

- **Confound attack:** The biggest confound is the **LLM judge**. Hierarchy discovery depends on an LLM correctly labeling and organizing feature descriptions. If the LLM is inconsistent or biased toward certain semantic domains, the metric reflects LLM behavior, not SAE behavior. There is no ground-truth validation for the hierarchies. The pilot requires GemmaScope 16K, which is gated. Pivoting to GPT-2 Small means no equivalent scope release exists, making the pilot impossible without training a new SAE—violating the training-free constraint.
- **Statistical attack:** Correlation of r > 0.4 with 20-50 SAEs requires moderate effect size. With 30 SAEs, the 95% CI for r=0.4 is approximately [0.08, 0.65]. The lower bound is below the falsification threshold of 0.3, meaning even if the true correlation is 0.4, we could easily observe r < 0.3 by chance. The sample size is too small for reliable inference.
- **Benchmark attack:** The first-letter benchmark itself may be unrepresentative. If the new metric correlates weakly, is that because the new metric is bad, or because first-letter absorption is a narrow task? The falsification criterion (r < 0.3) assumes first-letter absorption is the gold standard, but the literature explicitly critiques it as task-specific. This creates a paradox where the "falsification" result could actually be a valuable negative result.
- **Ablation completeness attack:** We plan to ablate thresholds and hierarchy counts, but we are not ablating the LLM judge itself (e.g., comparing GPT-4 vs. Claude vs. human labels). If judge variance is high, the metric is not reproducible. We also do not plan to test whether the metric predicts downstream utility—a critical validity check.
- **Verdict:** WEAK. The engineering complexity is high, the Gemma gating blocks the pilot, the statistical power is insufficient, and the validation criterion is conceptually shaky because it treats the task-specific benchmark as ground truth.

### Against Candidate C

- **Confound attack:** The trained SAE and random-decoder SAE may not be comparable even with matched L0. Random decoders often achieve lower explained variance (Korznikov et al. report 0.79 vs. 0.85 for JumpReLU). If reconstruction quality differs, any absorption difference could be attributed to MSE rather than training dynamics. TopK enforces exact L0 but is a different training objective than the L1-regularized baseline, introducing its own confound.
- **Statistical attack:** With only 1 trained SAE and 1 random-decoder SAE per condition, n=1. Even with multiple seeds, we are comparing distributions of random initializations against a single trained baseline. The trained SAE itself has variance across layers; we would need multiple layers/checkpoints to estimate it.
- **Benchmark attack:** The first-letter absorption metric is task-specific. If random decoders show comparable absorption, does that generalize to other hierarchies? The result would be suggestive but not conclusive. Moreover, if the result falls in the 50-80% range, interpretation is ambiguous—we pre-registered "within 20% = geometry dominates; below 50% = training dominates," but the middle ground has no clear interpretation.
- **Ablation completeness attack:** We are not testing whether random *encoder* SAEs (frozen encoder, train decoder) show absorption. If absorption is determined by decoder geometry, both frozen-decoder and frozen-encoder designs should show it. Testing only one direction leaves the mechanism unclear.
- **Verdict:** WEAK. The candidate is intellectually interesting but explicitly violates the training-free constraint, has severe sample-size limitations, and the interpretation thresholds are ambiguous.

---

## Phase 4: Refinement

### Dropped
- **Candidate C is dropped** due to the training-free constraint violation and weak experimental design.
- **Candidate B is dropped as a front-runner** due to the Gemma gating issue, insufficient statistical power, and dependence on an unvalidated LLM judge. It is retained as a long-term backup if an open-model hierarchy corpus becomes available.

### Strengthened: Candidate A

The front-runner survives but requires three critical fixes before scaling:

1. **Fix the absorption metric:** The pilot's simplified proxy returned degenerate 0.0 values for most checkpoints. We must integrate the proper `sae-spelling` absorption metric or SAEBench adaptation. This is non-negotiable for publication-ready numbers.

2. **Fix the token budget for dead-neuron detection:** The pilot used ~2k tokens, yielding 33-88% dead-neuron rates that are unreliable. Dead-neuron estimates require >=50k-100k tokens for stability. Because dead features confound absorption (Neuronpedia finding), this is a methodological prerequisite, not a nice-to-have.

3. **Fix the model anchor:** Gemma-2-2B is gated and inaccessible. We must commit to **GPT-2 Small** or **Pythia-160m** as the open-model anchor. This reduces architecture diversity (fewer OrtSAE/Matryoshka checkpoints available) but preserves the core claim: no architecture dominates across the full metric suite.

### Additional Controls Added
- **L0 stratification:** Compare architectures only within narrow L0 bands (e.g., L0 20-40, 40-80) to control for the "Sparse but Wrong" confound.
- **Dead-neuron regression:** Include dead-neuron fraction as a covariate in all absorption comparisons. Report results both with and without dead-neuron adjustment.
- **Layer fixed effects:** Include layer dummies to control for depth-dependent baseline differences.
- **Replication seeds:** Where checkpoints vary by seed, average across seeds and report standard errors.

### Selected Front-Runner
**Candidate A (Multi-Objective Pareto Evaluation)** remains the front-runner. It is the only candidate that is fully training-free, directly testable on open models, and addresses a novel empirical question: whether absorption-mitigation methods trade one pathology for another on a multi-objective front.

---

## Phase 5: Final Proposal

### Title
**The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods**

### Hypothesis
Absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, Masked Regularization, BatchTopK) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Instead, each architecture family occupies a distinct tradeoff region, and apparent superiority on any single metric is an artifact of selective reporting.

### Falsification Criterion
If one architecture family shows statistically significant stochastic dominance (Mann-Whitney U test, p < 0.05, Bonferroni-corrected for multiple comparisons) across >= 4 out of 5 metrics within a model family, the hypothesis is falsified.

### Method
We conduct a training-free, systematic analysis of existing pretrained SAE checkpoints using SAELens and SAEBench tooling.

**Checkpoint corpus:**
- GPT-2 Small: `gpt2-small-res-jb` (Standard, multiple layers), `gpt2-small-resid-post-v5-32k/128k` (TopK), `gpt2-small-mlp-out-v5-32k` (TopK_MLP), `gpt2-small-attn-out-v5-32k` (TopK_Attn).
- Pythia-160M: SAEBench released SAEs (Standard, TopK, JumpReLU, BatchTopK) if available via open-source release.
- Target: 20-30 checkpoints per model, spanning >= 4 architecture families.

**Metrics:**
1. **Absorption:** Chanin et al. first-letter metric via `sae-spelling` or SAEBench adaptation.
2. **Hedging:** Chanin et al. (2025) correlated-pair proxy or SAEBench implementation.
3. **Reconstruction fidelity:** Explained variance, CE loss recovered.
4. **Dead-neuron rate:** Fraction of latents with near-zero activation frequency on >=50k tokens.
5. **Downstream probing:** Sparse probing F1 from SAEBench or equivalent.

**Analysis plan:**
1. Normalize each metric to [0,1] within model family.
2. Compute empirical Pareto fronts per architecture family.
3. Test stochastic dominance using Mann-Whitney U with Bonferroni correction.
4. Run OLS regressions with architecture dummies, L0, dead-neuron rate, and layer fixed effects; cluster SEs by architecture family.
5. Report pairwise tradeoff curves (absorption vs. hedging, absorption vs. reconstruction, etc.) with bootstrap 95% CIs.

### Evaluation Protocol
- **Primary benchmarks:** SAEBench evaluations (absorption, sparse probing) and `sae-spelling` canonical absorption metric.
- **Statistical tests:** Mann-Whitney U for dominance; bootstrap percentile CIs for Pareto front visualization; OLS with cluster-robust SEs for covariate-adjusted comparisons.
- **Random seeds:** Average across available seed variants per checkpoint; minimum 3 seeds where multiple checkpoints exist.
- **Sample size:** 20-30 checkpoints per model family, >=4 architecture families.

### Ablation Schedule
| Ablation | What it tests | Expected outcome |
|----------|---------------|------------------|
| Exclude hedging metric | Sensitivity to width-narrowness confound | Pareto conclusion should be robust; if not, hedging is a critical axis |
| Stratify by L0 quartile | Control for "Sparse but Wrong" confound | Dominance claims should not reverse within narrow L0 bands |
| Exclude >30% dead-neuron checkpoints | Control for dead-neuron bias in absorption | Absorption rankings may shift; if so, dead neurons were masking true absorption |
| Separate analysis by layer depth | Control for depth-dependent baselines | Middle/late layers typically show highest absorption |

### Control Experiments
1. **Dead-neuron adjusted absorption:** Recompute absorption scores after regressing out dead-neuron rate. Tests whether architecture differences are driven by feature death rather than true absorption.
2. **L0-matched pairwise comparison:** For each architecture family, identify the checkpoint with L0 closest to the grand median L0; compare only these "L0-matched" representatives. Tests whether apparent architecture differences are L0 artifacts.
3. **Reconstruction-only ranking:** Rank checkpoints by explained variance alone. If the reconstruction leader is not the absorption leader, this confirms the multi-objective nature of the problem.

### Pilot Design
Run the full metric pipeline on 5-10 GPT-2 Small checkpoints (2 per architecture family: Standard, TopK, TopK_MLP, TopK_Attn, plus one Gated or JumpReLU if available) with:
- Proper `sae-spelling` or SAEBench absorption metric integration.
- 50k-100k tokens for dead-neuron detection.
- All 5 metrics computed end-to-end.

**Success criteria:**
- Absorption metric returns non-degenerate, non-zero values for >80% of checkpoints.
- Dead-neuron rates fall in a stable 10-60% range (not 90%+ as in the 2k-token pilot).
- Metric pipeline completes without errors in <=15 minutes.

If the pilot passes, proceed to full-scale evaluation (20-30 checkpoints). If absorption remains degenerate, diagnose whether the issue is the metric implementation, the model scale, or the checkpoint selection.

### Resource Estimate
- **Models:** GPT-2 Small (117M), Pythia-160M (160M).
- **SAEs:** Existing pretrained releases from SAELens and SAEBench.
- **Compute:** Single RTX 4090 / A100 equivalent. Per-checkpoint metric batch takes ~5-15 minutes. Total analysis: ~2-4 hours per model family, parallelizable across GPUs.
- **Target per task:** <=1 hour per independent subtask (per-checkpoint metric batch).

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Proper absorption metric integration fails or is too slow for GPT-2 | Medium | Fallback to SAEBench absorption evaluation (~26 min/SAE). If both fail, pivot to analyzing the metric pipeline failure as a reproducibility contribution. |
| GPT-2 Small lacks sufficient architecture diversity (no OrtSAE/Matryoshka) | High | Reframe the claim: "even among standard architectures, no single design dominates" rather than testing specific absorption-mitigation methods. Source Pythia-160M SAEBench releases for more diversity. |
| Downstream correlation is confounded by unobserved architecture differences | Medium | Include architecture dummies, width, L0, and dead-neuron rate as controls; use cluster-robust SEs; explicitly discuss causal limitations in the paper. |
| SAEBench metrics are too noisy for clean Pareto analysis | Medium | Average across multiple layers/checkpoints; report confidence intervals; use non-parametric tests that are robust to outliers. |
| Pilot reveals absorption is degenerate for all GPT-2 checkpoints | Low-Medium | If true, this is a valuable negative result suggesting absorption is either absent in small models or the metric is scale-dependent. Document and pivot to studying metric scale dependence. |

### Novelty Claim
This would be the **first systematic multi-objective Pareto evaluation of absorption-mitigation methods across the full suite of SAE quality metrics using existing checkpoints in a training-free setting**. Prior work (OrtSAE, Matryoshka, SAEBench) compares architectures on multiple metrics but does not frame this as a Pareto analysis showing no architecture dominates. If supported, the contribution reframes the SAE research agenda from "fixing absorption" to "navigating unavoidable tradeoffs."

### What the Empiricist Added
The empiricist perspective insisted on three non-negotiable methodological upgrades:
1. **Dead-neuron confound control:** The Neuronpedia finding that dead features artificially lower absorption means dead-neuron rate must be treated as a covariate, not an independent metric.
2. **L0 stratification:** The "Sparse but Wrong" result means architecture comparisons at mismatched L0s are uninterpretable. All dominance claims must be tested within narrow L0 bands.
3. **Pre-registered falsification criterion:** The hypothesis is only falsified if one architecture dominates on >=4/5 metrics with Bonferroni-corrected significance. This raises the evidential bar and protects against cherry-picking.

These controls transform a provocative but fragile contrarian claim into a rigorous, skeptical empirical test.
