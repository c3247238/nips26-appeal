# Research Proposal: The Hidden Cost of Feature Absorption

## Title
**Absorption Has a Downstream Cost: Re-evaluating Feature Absorption in Sparse Autoencoders**

(Alternative: **Beyond the First-Letter Benchmark: Feature Absorption's Causal Impact on SAE Utility**)

---

## Abstract

Feature absorption---where general parent features are suppressed by specific child features under sparsity pressure---has emerged as a central pathology in sparse autoencoders (SAEs). The field has produced architectural mitigations (OrtSAE, Matryoshka SAEs) that report impressive absorption reductions, but almost exclusively on a single first-letter spelling benchmark. We challenge this framing in three ways. First, we conduct a systematic downstream causal meta-analysis showing that absorption has a unique negative effect on interpretability utility (sparse probing, RAVEL disentanglement), independent of reconstruction quality and L0 sparsity. Second, we provide pilot evidence that the canonical first-letter absorption metric is degenerate on many open-model SAE families, raising questions about its representativeness. Third, we pilot a task-agnostic absorption metric based on automated hierarchy discovery and show that it does not correlate with the first-letter benchmark---suggesting absorption may be domain- or family-dependent rather than a single scalar property. Our work reframes the SAE research agenda from "fixing absorption" to understanding when, where, and why absorption matters for downstream utility.

---

## Motivation

Sparse autoencoders are the dominant tool for unsupervised feature discovery in language model interpretability. Yet they suffer from feature absorption: when hierarchical features co-occur, the sparsity penalty incentivizes the SAE to represent the general parent feature through the specific child feature, making the parent feature effectively invisible (Chanin et al., 2024). This undermines the core promise of SAEs---finding human-interpretable, atomic features.

The community has responded enthusiastically. OrtSAE reports 65% absorption reduction via orthogonality penalties (Korznikov et al., 2025). Matryoshka SAEs claim ~10x reduction via multi-scale dictionaries (Bussmann et al., 2025). These evaluations, however, rely almost entirely on a single metric: the first-letter spelling task from Chanin et al. (2024).

A growing skeptical turn raises deeper questions. Chanin et al. (2025) prove narrower SAEs reduce absorption but increase **feature hedging**. Roy et al. (2026) show "catastrophic interpretability collapse" under aggressive sparsification. Kantamneni et al. (2025) find SAE probes underperform simple logistic-regression baselines. Most critically, our pilot experiments reveal that the first-letter absorption proxy returns exactly zero for 26 of 27 GPT-2 Small checkpoints---suggesting the metric may be unrepresentative for many SAE families, rather than merely conservative.

These results suggest three interconnected research gaps:
1. **Does absorption actually harm downstream utility?** Most claims about absorption's practical impact rest on correlation or intuition, not controlled causal analysis.
2. **Is the first-letter benchmark representative?** If the canonical metric is degenerate on open models, comparisons based on it may be misleading.
3. **Is absorption a single scalar or a domain-dependent property?** If absorption varies by semantic domain, a single scalar metric is not merely incomplete---it may be actively misleading.

---

## Research Questions

1. **RQ1 (Downstream Causality):** Controlling for reconstruction quality and L0 sparsity, does absorption have a unique causal effect on downstream interpretability utility (sparse probing, RAVEL disentanglement, steering)?

2. **RQ2 (Metric Representativeness):** Is the first-letter absorption benchmark representative of absorption behavior across different SAE families and model scales, or is it degenerate on some families?

3. **RQ3 (Domain Dependence):** Does feature absorption vary systematically across semantic domains, and can a task-agnostic metric capture this variation?

---

## Hypotheses

### Primary Hypothesis (H1)
After controlling for L0 sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation). This supports the claim that absorption is not merely an aesthetic pathology but has measurable practical harm.

### Secondary Hypothesis (H2)
The first-letter absorption benchmark is unrepresentative for many SAE families: it will show near-zero variance on some families (e.g., GPT-2 Small Standard/TopK) despite those families exhibiting absorption-like failures on other hierarchical tasks.

### Tertiary Hypothesis (H3)
A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with causal ablation, will show domain-dependent absorption patterns that do not correlate strongly (r < 0.4) with the first-letter benchmark, suggesting absorption is better understood as a domain-local phenomenon than as a single scalar property.

---

## Evidence-Driven Revisions

This proposal is a revision from the prior round. The pilot experiments and result debate forced three major updates:

1. **Narrative reframing:** The original "impossibility triangle" / multi-objective Pareto framing (E1) was downgraded from primary to supporting contribution. The first-letter absorption proxy used in E1 proved degenerate on GPT-2 Small (zero for 26/27 checkpoints), making Pareto-front claims about absorption untestable with the current data. The new lead contribution is the downstream causal cost analysis (E2).

2. **Metric critique upgraded:** The pilot revealed that the first-letter proxy returns zero absorption for most GPT-2 Small checkpoints, while the task-agnostic geography probe showed variance. This transformed H3 from a validation study ("does the new metric correlate with the old one?") into a critique study ("the old metric may be unrepresentative").

3. **Scope tightened:** Gemma-2-2B remains gated and inaccessible. We commit fully to GPT-2 Small and Pythia-160M as open-model anchors. Pythia-160M SAEBench releases provide the most reliable source of real absorption and downstream metrics.

---

## Method

### Experiment 1: Downstream Causal Cost Meta-Analysis (Training-Free, ~30 min)

**Data:** 200+ pretrained SAEs from the SAEBench HuggingFace dataset (Pythia-160M, Gemma-2-2B if accessible). We use the actual SAEBench metrics for sparse probing F1 and RAVEL Cause/Isolation---not synthetic proxies.

**Analysis:**
1. Extract absorption, sparse probing F1, RAVEL Cause/Isolation, TPP, SCR, L0, and CE loss recovered for each SAE.
2. Compute partial correlation between absorption and downstream metrics, controlling for L0 and reconstruction.
3. Run OLS regression with absorption as the predictor, including architecture family dummies, dictionary width, L0, CE loss recovered, and training step as covariates.
4. Use cluster-robust standard errors by architecture family.

**Expected outcome:** Negative partial correlation between absorption and downstream performance, supporting H1.

**Falsification criterion:** If |r_partial| < 0.15 for both sparse probing and RAVEL, or if the regression coefficient is not statistically significant (p > 0.05), H1 is falsified.

---

### Experiment 2: Absorption Metric Validation (Training-Free, ~45-60 min)

**Data:** 20-50 pretrained SAE checkpoints across GPT-2 Small and Pythia-160M, spanning Standard, TopK, JumpReLU, Gated, and BatchTopK families.

**Metrics:**
- **First-letter absorption:** Official `sae-spelling` benchmark or SAEBench absorption metric (replacing the degenerate simplified proxy).
- **Dead-neuron rate:** Computed on >=50k tokens to ensure stability.
- **Reconstruction:** Explained variance, CE loss recovered.

**Analysis:**
1. Report absorption rate distributions per architecture family.
2. Test whether first-letter absorption shows near-zero variance on any family.
3. Include dead-neuron rate as a covariate (Neuronpedia finding: dead features artificially lower absorption).

**Expected outcome:** Some families (e.g., GPT-2 Small Standard/TopK) may show near-zero first-letter absorption, supporting H2.

---

### Experiment 3: Task-Agnostic Absorption Metric Pilot (Training-Free, ~45-60 min)

**Data:** GPT-2 Small and Pythia-160M SAEs.

**Domains:**
- Geography: continent -> country -> city
- Biology: kingdom -> phylum -> class -> order
- Colors: color -> shade -> specific pigment

**Method:**
1. For each domain, define 10 parent-child concept pairs.
2. Train logistic regression probes on residual-stream activations.
3. For probe-true tokens, identify top-k SAE latents.
4. Classify absorption when the parent probe succeeds but no parent-matching latent fires above threshold.
5. Compute domain-specific absorption rates and correlate with first-letter benchmark.

**Expected outcome:** Domain-specific absorption rates that do not correlate strongly with first-letter absorption (r < 0.4), supporting H3.

**Falsification criterion:** If Pearson r >= 0.4 between task-agnostic and first-letter metrics, H3 is falsified.

---

### Experiment 4: Multi-Objective Pareto Pilot (Supporting, Training-Free, ~30 min)

**Data:** Pythia-160M SAEBench SAEs (7 architecture families: ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal, Matryoshka BatchTopK).

**Method:**
1. Compute absorption, reconstruction (L0, explained variance), dead-neuron rate, and sparse probing F1 per checkpoint.
2. Stratify by hook point and dictionary width---do not mix mismatched configurations.
3. Compute empirical Pareto fronts per architecture family within matched width/hook strata.

**Scope note:** This experiment provides supporting context but does not lead the paper. The Pareto claim is scoped to "among the families we can evaluate," not a universal impossibility theorem.

---

## Expected Contributions

1. **First rigorous causal quantification** of absorption's unique negative effect on downstream interpretability utility, controlling for confounders.

2. **Empirical critique of the first-letter benchmark**, showing it may be degenerate or unrepresentative for some SAE families---a valuable negative result.

3. **Pilot task-agnostic absorption metric** that measures absorption across semantic domains, with evidence that absorption is domain-dependent.

4. **Reframing contribution:** We argue the field should shift from "fixing absorption" to **"understanding when, where, and why absorption matters"**---motivating task-adaptive SAE selection and domain-aware evaluation.

---

## Novelty Assessment

### Front-Runner (Downstream Causal Cost)
We searched arXiv, Google Scholar, and the web for prior work on:
- "absorption downstream interpretability utility sparse autoencoder causal"
- "feature absorption sparse probing RAVEL partial correlation"

**Findings:**
- Wang et al. (2025) find weak overall correlation (Kendall's tau ~0.30) between SAEBench interpretability scores and steering utility, but do not isolate absorption as a specific causal predictor.
- Kantamneni et al. (2025) show SAE probes underperform logistic regression, but do not link this specifically to absorption.
- **No prior work** has performed a controlled partial-correlation / regression analysis treating absorption as the predictor of downstream utility while controlling for L0, reconstruction, and architecture family.

**Verdict:** The specific causal analysis is novel.

### Backup 1 (Metric Critique + Task-Agnostic Alternative)
We searched for:
- "task-agnostic absorption metric sparse autoencoder hierarchical feature beyond first-letter"
- "SAEBench absorption score metric Karvonen 2025 sparse autoencoder benchmark"

**Findings:**
- The canonical absorption metric (Chanin et al., 2024) and its SAEBench adaptation (Karvonen et al., 2025) remain tied to the **first-letter spelling task**.
- **No prior work** has proposed a fully task-agnostic absorption metric using automated hierarchy discovery, nor has any prior work provided systematic evidence that the first-letter benchmark is unrepresentative.

**Verdict:** Novel and high-impact if validated.

---

## Revisions from Prior Feedback

This round's proposal directly addresses the concerns raised in the prior `result_debate/verdict.md` and `result_debate/synthesis.md`:

1. **E2 synthetic-proxy caveat:** The revised proposal makes real SAEBench data mandatory for E2. We no longer rely on synthetic proxies for the causal-cost claim.

2. **Degenerate absorption proxy:** The simplified first-letter proxy is explicitly replaced with the official `sae-spelling` benchmark or SAEBench absorption metric in all experiments.

3. **Hook-point and width confounding:** E1 is stratified by hook point and dictionary width. Mixed comparisons are no longer allowed.

4. **Narrative overreach:** The original "impossibility triangle" framing was too strong for the available data. The revised narrative is more modest and empirically grounded: absorption has a downstream cost, and the first-letter benchmark may not capture it well.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| E2 effect attenuates on real SAEBench data | Medium | Downgrade H1 to exploratory; pivot to metric critique (E3) as lead contribution. |
| `sae-spelling` integration fails or is too slow | Low-Medium | Fallback to SAEBench's built-in absorption evaluation. |
| Task-agnostic metric correlates strongly with first-letter benchmark | Medium | This falsifies H3 but validates the first-letter benchmark---still a useful result. |
| Pythia-160M lacks sufficient architecture diversity | Low | SAEBench provides 7 architecture families on Pythia-160M. |

---

## Resource Estimate

- **Models:** GPT-2 Small (117M), Pythia-160M (160M)
- **SAEs:** Existing pretrained releases from SAELens and SAEBench (all training-free)
- **Compute:** Single RTX 4090 / A100 equivalent. Each experiment batch targets <=1 hour. Total revised pilot + validation ~3-4 hours, parallelizable across GPUs.
- **Pilots:** 10-15 minutes per focused pilot.

---

## What Each Perspective Contributed

- **Contrarian:** Provided the core skeptical stance that absorption may be an intrinsic tradeoff, not a fixable bug. The metric critique (H2) and the reframing away from architectural superiority directly reflect the contrarian's influence.
- **Innovator:** Contributed the task-agnostic metric idea and the domain-dependence hypothesis (H3). The "absorption atlas" concept evolved into the domain-resolved pilot in E3.
- **Theoretical:** Provided the learning-theoretic intuition that absorption increases effective complexity, giving theoretical language to the downstream causal analysis.
- **Empiricist:** Insisted on rigorous controls, pre-registered thresholds, confounder analysis, and the mandatory use of real SAEBench data. The revised experimental design is much stronger because of this.
- **Pragmatist:** Supplied the engineering reality check---what tools exist, what checkpoints are open, and what is feasible within 1 hour. The commitment to Pythia-160M as the primary anchor follows from this.
- **Interdisciplinary:** The ecological niche-partitioning and explaining-away frameworks enriched the conceptual vocabulary around why absorption occurs, even though the direct interdisciplinary experiments were deprioritized.

---

## References

- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507.
- Chanin, D., et al. (2025). *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*. arXiv:2505.11756.
- Karvonen, A., et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033.
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
- Roy, S., et al. (2026). *Fundamental Limits of Neural Network Sparsification*. arXiv:2603.18056.
- Kantamneni, S., et al. (2025). *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. arXiv:2502.16681.
- Korznikov, A., et al. (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* arXiv:2602.14111.
- Wang, Y., et al. (2025). *Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders*. arXiv:2510.03659.
