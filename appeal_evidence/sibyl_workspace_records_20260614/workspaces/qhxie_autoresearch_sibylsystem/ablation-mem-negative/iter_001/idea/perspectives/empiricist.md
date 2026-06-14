# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **[SAEBench] Karvonen et al. (2025)** — "A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability" (arXiv:2503.09532, ICML 2025)
   - **Key insight**: 8+ metrics across 4 capability dimensions (Concept Detection, Interpretability, Reconstruction, Feature Disentanglement). Critical finding: "Small gains in sparsity-fidelity trade-off do not necessarily translate into qualitatively better representations." Proxy metrics (L0, Loss Recovered) do NOT reliably predict practical performance. This directly undermines any experiment that uses only reconstruction metrics as evidence.
   - **Methodology strength**: Standardized pipeline, 200+ SAEs, reproducible. Weakness: LLM-as-judge introduces non-determinism; cross-model comparisons are "potentially misleading."

2. **[CE-Bench] Gulko et al. (2025)** — "Towards a Reliable Contrastive Evaluation Benchmark" (BlackBoxNLP 2025, arXiv:2509.00691)
   - **Key insight**: LLM-free, fully deterministic alternative to SAEBench's LLM-based evaluation. Uses 5,000 curated story pairs. 70%+ Spearman correlation with SAEBench rankings but without prompt sensitivity and generation noise. This is the methodological gold standard for reproducibility.
   - **Methodology strength**: Deterministic, no external LLM calls, lower resource overhead. Weakness: Limited metric coverage (no absorption-specific evaluation).

3. **[Chanin et al. (2024)]** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507)
   - **Key insight**: First systematic absorption detection using controlled ground-truth setup (first-letter task with full label access). K-sparse probing + integrated gradients ablation for causal verification. Metric limited to early layers (0-17 in Gemma-2-2B) because attention moves information after layer ~18.
   - **Methodology strength**: Ground truth access enables precise measurement. Weakness: Single task (first-letter), conservative underestimate, focused on GPT-2, cannot evaluate late layers.

4. **[Chanin et al. (2025)]** — "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders" (arXiv:2505.11756)
   - **Key insight**: Identified hedging as the complement to absorption. Showed Matryoshka trades absorption for hedging. Proposed balanced loss coefficients (beta_m ~ 0.75). Limited to empirical analysis; no theoretical characterization of trade-off.
   - **Methodology strength**: Controlled toy model experiments + real LLM validation. Weakness: No statistical significance testing reported; single model (Gemma-2-2B).

5. **[Bussmann et al. (2025)]** — "Learning Multi-Level Features with Matryoshka Sparse Autoencoders" (arXiv:2503.17547)
   - **Key insight**: ~90% absorption reduction (0.49 -> 0.05 at L0=40) via hierarchical nesting. But introduces hedging in inner levels. Higher computational cost.
   - **Methodology strength**: Systematic comparison across dictionary sizes and sparsity levels. Weakness: No random seed variation reported; absorption measured only on first-letter task.

6. **[Korznikov et al. (2025)]** — "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features" (arXiv:2509.22033)
   - **Key insight**: ~65% absorption reduction via orthogonality constraints. ~50% less compute than Matryoshka. Ablation studies on chunk number (K = 4, 8, 16, 32, 64). Tested on Gemma-2-2B and Llama-3-8B.
   - **Methodology strength**: Multiple models, ablation studies, SAEBench evaluation. Weakness: No seed variation; single layer (12) as primary focus.

7. **[Paulo et al. (2025)]** — "Transcoders Beat Sparse Autoencoders for Interpretability" (arXiv:2501.18823)
   - **Key insight**: Skip transcoders Pareto-dominate SAEs on reconstruction vs. interpretability. Different objective (input-output vs. self-reconstruction). Tested on Pythia 160M/410M, Llama 3.2 1B, Gemma 2 2B.
   - **Methodology strength**: Multi-model, multi-scale, same evaluation pipeline. Weakness: Not directly comparable for all use cases; absorption not explicitly measured.

8. **[Korznikov et al. (2026)]** — "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" (arXiv:2602.14111)
   - **Key insight**: SAEs recover only 9% of true features despite 71% explained variance. Random baselines (frozen decoder, soft-frozen decoder, frozen encoder) match or nearly match fully-trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). This is a devastating methodological critique.
   - **Methodology strength**: Three rigorous random baselines. Weakness: Synthetic ground truth may not generalize to real LLM activations.

9. **[Chen et al. (2025)]** — "Provable Feature Recovery via Sparse Autoencoders" (arXiv:2506.14002, ICLR 2025)
   - **Key insight**: First SAE algorithm with theoretical recovery guarantees. Group Bias Adaptation (GBA) validated on 2B parameter LLMs. Guarantees require specific statistical model assumptions.
   - **Methodology strength**: Theory + empirical validation. Weakness: Statistical assumptions may not hold for real LLM activations; no absorption-specific evaluation.

10. **[Cui et al. (2025)]** — "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (arXiv:2506.15963)
   - **Key insight**: First closed-form optimal solution analysis. Proved standard SAEs fail unless features are extremely sparse. Identified feature shrinking/vanishing. Proposed Weighted SAE (WSAE).
   - **Methodology strength**: Rigorous theoretical analysis. Weakness: Limited empirical validation; no real LLM experiments reported.

11. **[Kulkarni et al. (2025)]** — "Interpretable and Steerable Concept Bottleneck Sparse Autoencoders" (arXiv:2512.10805)
   - **Key insight**: Only ~18.8% of SAE neurons are both interpretable AND steerable. 36.3% are neither. This directly challenges the assumption that SAE features are useful for downstream control.
   - **Methodology strength**: Systematic categorization of neuron utility. Weakness: Vision-language focus; may not generalize to text-only models.

12. **[EleutherAI (2025)]** — "Sparse Autoencoders Trained on the Same Data Learn Different Features" (arXiv:2501.16615)
   - **Key insight**: Only ~30% feature overlap across random seeds in 131K latent SAE. TopK SAEs are MORE seed-dependent than ReLU SAEs. Increasing SAE size DECREASES overlap.
   - **Methodology strength**: Hungarian algorithm for feature matching, systematic seed variation. Weakness: No absorption-specific analysis.

### Experimental Landscape

**What has been properly tested:**
- Absorption EXISTS in JumpReLU/TopK SAEs (Chanin et al., 2024) — well-established with ground-truth controls
- Matryoshka reduces absorption ~90% (Bussmann et al., 2025) — but only on first-letter task
- Orthogonality reduces absorption ~65% (Korznikov et al., 2025) — tested on SAEBench
- Absorption-hedging trade-off exists (Chanin et al., 2025) — but only empirically, no theory
- SAEs are highly seed-sensitive (EleutherAI, 2025) — most papers ignore this

**What is accepted without evidence:**
- That absorption rates generalize beyond first-letter tasks (NO systematic validation on other concept hierarchies)
- That lower absorption directly improves downstream interpretability (NO causal evidence; only correlations)
- That cross-architecture comparisons are fair (different architectures optimize different objectives; comparing at "same L0" may be apples-to-oranges)
- That pre-trained SAEs (GemmaScope) are representative (trained with specific hyperparameters; may not generalize)

**Where methodological gaps exist:**
1. **No random seed control**: Most absorption studies report single-seed results. Given 30% feature overlap across seeds, this is a major threat to validity.
2. **No random baselines**: Only Korznikov et al. (2026) compare to random. Our study MUST include random SAE baselines.
3. **Single-task validation**: First-letter task is convenient but may not represent general absorption behavior. Need multi-domain validation.
4. **No statistical significance testing**: Most papers report point estimates without confidence intervals or significance tests.
5. **Confounders uncontrolled**: Reconstruction quality, sparsity, dead feature ratio, and training data all vary across architectures. Partial correlations are rarely reported.
6. **Layer limitation**: Chanin et al.'s IG ablation only works for early layers. Late-layer absorption is unmeasured.
7. **No causal design**: All existing studies are correlational. No one has done the experiment: "If I force absorption to decrease, does downstream performance improve?"

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Architecture Absorption Benchmark with Seed Control and Random Baselines

- **Hypothesis**: Different SAE architectures exhibit significantly different absorption rates when controlling for random seed, reconstruction quality, and sparsity. JumpReLU shows the highest absorption; Matryoshka shows the lowest.
- **Falsification criterion**: If, after controlling for seed and sparsity, the architecture with highest absorption differs from the lowest by < 5 percentage points (p > 0.05, paired t-test across seeds), H1 is falsified. If random baselines achieve absorption rates within 50% of trained SAEs, the effect is artifactual.
- **Evaluation protocol**:
  - Primary: Chanin et al. absorption metric (k-sparse probe + integrated gradients ablation) on first-letter task
  - Secondary: SAEBench absorption metric (probe projection criteria, applicable at all layers)
  - Statistical: Bootstrap 95% CI across 5 random seeds per architecture; paired t-test for pairwise comparisons; Bonferroni correction for multiple comparisons
  - Control: Report partial correlation between architecture type and absorption, controlling for reconstruction MSE and L0 sparsity
- **Ablation plan**:
  - Ablation 1: Vary dictionary size (4k, 16k, 65k) to test width-absorption relationship
  - Ablation 2: Vary sparsity level (L0 = 20, 40, 100) to test sparsity-absorption monotonicity
  - Ablation 3: Remove orthogonality penalty (for OrtSAE) to isolate its contribution
  - Ablation 4: Test inner vs. outer Matryoshka levels separately to quantify hedging trade-off
- **Confounders identified**:
  - Training data distribution (different architectures may need different training regimes)
  - Dead feature ratio (affects effective capacity)
  - Feature shrinking (Cui et al., 2025) — may masquerade as absorption
  - Seed-dependent feature discovery (EleutherAI, 2025) — single-seed results may be idiosyncratic
- **Pilot design**: Train 2 architectures (JumpReLU vs TopK) on GPT-2 Small layer 8, 2 seeds each, 500K tokens. Run Chanin absorption detection. Target: < 15 min, detect > 10% absorption rate difference.

### Candidate B: Controlled Causal Experiment — Absorption Manipulation and Downstream Impact

- **Hypothesis**: Reducing feature absorption (via architecture choice) causally improves downstream interpretability tasks (sparse probing accuracy, steering efficacy), independent of reconstruction quality and sparsity.
- **Falsification criterion**: If partial correlation between absorption rate and downstream performance (controlling for reconstruction quality, sparsity, dead feature ratio) is not significantly negative (p > 0.05, one-tailed), OR if effect size (R^2) < 0.1, H2 is falsified. If random baselines show the same downstream performance as low-absorption SAEs, the effect is not causal.
- **Evaluation protocol**:
  - Primary benchmarks: Sparse probing on 100 concepts across 5 semantic domains (animals, countries, professions, emotions, scientific terms)
  - Secondary: Steering efficacy (logit difference shift) on sentiment and topic directions
  - Tertiary: Feature attribution consistency (integrated gradients stability across paraphrases)
  - Statistical: Partial correlation with bootstrap CI; mixed-effects model with architecture as fixed effect and seed/concept as random effects
  - Number of random seeds: Minimum 5 per architecture
- **Ablation plan**:
  - Ablation 1: Hold architecture fixed, vary sparsity penalty to manipulate absorption rate
  - Ablation 2: Hold sparsity fixed, vary architecture to manipulate absorption rate
  - Ablation 3: Zero-ablate absorbing features to measure direct causal impact
  - Ablation 4: Compare against random baseline SAEs at same sparsity/reconstruction
- **Confounders identified**:
  - Reconstruction quality (better reconstruction may improve downstream tasks regardless of absorption)
  - Sparsity (sparser features may be more interpretable but less complete)
  - Dead feature ratio (affects effective feature set)
  - Concept selection bias (concepts may vary in inherent detectability)
  - Steering direction quality (hand-crafted directions may not align with true feature axes)
- **Pilot design**: Train 2 TopK SAEs on GPT-2 Small (k=10 vs k=50, giving different absorption rates). Run sparse probing on 20 concepts. Measure correlation between absorption rate and probe accuracy. Target: < 15 min, detect correlation |r| > 0.3.

### Candidate C: "Nobody Has Run This Control" — Absorption in Randomly Initialized Transformers

- **Hypothesis**: Feature absorption occurs at similar rates in randomly initialized transformers as in trained transformers, suggesting absorption is an artifact of data statistics and architecture rather than learned computations.
- **Falsification criterion**: If absorption rate in trained transformers is > 20 percentage points higher than in random transformers (p < 0.05, Welch's t-test), the hypothesis is falsified. If the difference is < 10 pp, absorption is likely an architectural artifact.
- **Evaluation protocol**:
  - Primary: Chanin et al. absorption metric on first-letter task, comparing trained GPT-2 Small vs. randomly initialized GPT-2 Small
  - Secondary: SAEBench metrics (sparse probing, SCR, TPP) on both
  - Statistical: Welch's t-test (unequal variances expected); bootstrap CI for absorption rate difference; report effect size (Cohen's d)
  - Number of random seeds: 5 trained vs. 5 random initializations
- **Ablation plan**:
  - Ablation 1: Vary initialization scale to test if absorption is initialization-dependent
  - Ablation 2: Compare at same layer depth to control for architecture
  - Ablation 3: Test on multiple concept hierarchies (not just first-letter)
- **Confounders identified**:
  - Random initialization may produce different activation distributions
  - First-letter task may be solvable by random models (as shown by "SAEs interpret random transformers" paper)
  - Training data statistics may differ from random initialization statistics
- **Pilot design**: Load random GPT-2 Small, train TopK SAE for 500K tokens, run first-letter absorption detection. Compare with trained GPT-2 Small SAE from Candidate A pilot. Target: < 15 min, detect any absorption in random model.
- **Why this matters**: If absorption exists in random models, the entire field's framing of absorption as a "problem to solve" may be misguided. This is the most falsifiable and high-impact experiment.

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Architecture Benchmark)

- **Confound attack**: The architectures optimize different objectives. JumpReLU trains L0 directly; TopK uses k-sparse activation; Matryoshka has nested reconstruction constraints. Comparing them at "same L0" is comparing apples to oranges because the training dynamics differ. A fair comparison might require matching on reconstruction quality AND sparsity, but this creates a multi-dimensional matching problem.
  - *Search support*: SAEBench found "improvements from Gated, TopK, and JumpReLU over vanilla SAEs are often difficult to differentiate on practical metrics even when they excel on proxy metrics."
- **Statistical attack**: With 5 seeds and 5 architectures, we have 25 data points. Detecting a 5 pp difference with reasonable power (0.8) requires estimating variance across seeds. If seed variance is large (as EleutherAI suggests, with 30% feature overlap), we may need 10+ seeds per architecture. This blows up the resource budget.
  - *Power analysis*: If sigma_seed ~ 0.08 (plausible given feature overlap), n=5 gives power ~0.5 for 5 pp difference. Need n=10-15 for power=0.8.
- **Benchmark attack**: First-letter task is the standard but may not generalize. Chanin et al. used it because it has ground truth. But absorption may manifest differently for other hierarchies (e.g., semantic categories, syntactic structures). Without multi-domain validation, results may not generalize.
- **Ablation completeness attack**: Removing orthogonality penalty from OrtSAE tests one component. But Matryoshka has multiple interacting components (nested dictionaries, independent reconstruction constraints, loss coefficients). Ablating one at a time may miss interaction effects. And the "balanced Matryoshka" (Chanin et al., 2025) adds another tuning parameter (beta_m) that could dominate results.
- **Verdict**: MODERATE. The experiment is well-defined and addresses a real gap, but the confound of different training objectives, seed sensitivity, and single-task validation are serious threats.

### Against Candidate B (Causal Downstream Impact)

- **Confound attack**: The biggest confound is that architectures with lower absorption may also have better reconstruction, different feature distributions, or fewer dead features. Even with partial correlation, we cannot rule out that some unmeasured architectural property (not absorption per se) drives downstream performance. A true causal experiment would require intervening on absorption directly (e.g., via feature surgery) while holding everything else constant — which is technically very difficult.
  - *Search support*: Kulkarni et al. (2025) found only 18.8% of SAE neurons are both interpretable AND steerable, suggesting the link between feature quality and downstream control is weak.
- **Statistical attack**: Steering efficacy is notoriously noisy. Small changes in steering coefficient or direction can produce large, unstable effects. Logit difference shifts may not be normally distributed. Non-parametric tests (Mann-Whitney U) may be needed but reduce power. With 5 seeds and 100 concepts, we have 500 measurements, but concept-level clustering violates independence assumptions.
- **Benchmark attack**: Sparse probing on 100 concepts sounds rigorous, but concept selection is arbitrary. Different concept sets may yield different correlations. The "5 semantic domains" framing is post-hoc — there's no principled reason these domains are the right ones. This opens the door to p-hacking via domain selection.
  - *Search support*: OpenAI's scaling paper found "normalized MSE does not reliably predict downstream loss impact" and "overall loss difference has no clear relation with normalized MSE across layers."
- **Ablation completeness attack**: Zero-ablating absorbing features tests direct causal impact, but ablation may have off-target effects (other features compensate). The integrated gradients method assumes linearity, which may not hold. And "feature attribution consistency" across paraphrases is a weak signal — it measures stability, not correctness.
- **Verdict**: WEAK. The causal claim is too strong for the available methods. Correlational evidence is achievable, but claiming causality requires intervention studies that are beyond current capabilities.

### Against Candidate C (Random Transformer Control)

- **Confound attack**: Randomly initialized transformers have different activation statistics than trained ones. The SAE training may converge differently. If absorption rates differ, it could be due to training dynamics, not the presence/absence of learned features. Need to control for activation distribution (mean, variance, covariance structure).
  - *Search support*: The "SAEs interpret randomly initialized transformers" paper (arXiv:2501.17727) already showed SAEs find similar features in trained and untrained models, suggesting this direction is partially explored.
- **Statistical attack**: If absorption in random models is non-zero but lower, the difference may be small. With 5 seeds per condition, power to detect a 10 pp difference is marginal. Also, the "random" baseline is itself a distribution — different random seeds produce different initializations.
- **Benchmark attack**: First-letter task may be trivial for any model with English vocabulary statistics. The task doesn't require "understanding" — it requires letter-position mapping, which could be learned from token embeddings alone. A more demanding hierarchy (e.g., semantic categories) would be stronger.
- **Ablation completeness attack**: This experiment asks whether absorption is "real" or an artifact. But even if absorption exists in random models, it could still be problematic for interpretability. The experiment doesn't directly test whether absorption harms interpretability — it only tests its origin.
- **Verdict**: MODERATE. The experiment is conceptually clean and high-impact, but the "random transformers" angle is partially explored. The real contribution would be showing that absorption rates are SIMILAR (not just non-zero) in random vs. trained models, which would challenge the field's framing.

---

## Phase 4: Refinement

### Dropped

- **Candidate B (Causal downstream impact)**: The causal claim is too strong. Current methods cannot rule out confounds sufficiently to claim causality. We can measure correlation, but calling it "causal" would be misleading. DROPPED as a primary candidate.

### Strengthened

- **Candidate A (Cross-Architecture Benchmark)**:
  - **Added control**: Include random SAE baselines (frozen decoder, following Korznikov et al., 2026). If trained SAEs don't beat random baselines by > 20% on absorption metrics, the architectural differences are not meaningful.
  - **Added seed control**: Increase to minimum 5 seeds per architecture (accepting lower power). Report seed variance prominently.
  - **Added multi-domain validation**: Beyond first-letter, test on semantic hierarchies (WordNet hypernyms: animal -> mammal -> dog). This tests generalization.
  - **Added layer coverage**: Use SAEBench's probe projection metric for late layers where IG ablation fails.
  - **Tightened falsification criterion**: If random baselines are within 50% of the best architecture's absorption rate, the architectural differences are not practically significant.
  - **Analysis plan**: Report Pareto frontier (absorption vs. reconstruction vs. sparsity). Use mixed-effects model with architecture as fixed effect, seed and layer as random effects.

- **Candidate C (Random Transformer Control)**:
  - **Strengthened**: Frame as a "sanity check" rather than a primary contribution. The key question is not "does absorption exist in random models?" (already partially answered) but "are absorption RATES comparable?" If yes, the field's urgency around absorption may be misplaced.
  - **Added control**: Match activation statistics (mean, covariance) between random and trained models via whitening/dewhitening. This isolates the "learned vs. random" variable.
  - **Combined with A**: Run the cross-architecture benchmark on BOTH trained and random GPT-2 Small. This gives a 2x2 design (architecture x initialization) with richer inference.

### Selected Front-Runner

**Candidate A+**: Cross-Architecture Absorption Benchmark with Seed Control, Random Baselines, and Multi-Domain Validation

This is the strongest candidate because:
1. It addresses Gap 10 (cross-architecture comparison) — the most actionable and well-defined gap
2. It includes controls that NO existing paper has (random baselines + seed variation + multi-domain)
3. The experimental design is falsifiable: either architectures differ or they don't
4. The pilot can validate feasibility in < 15 minutes
5. Results are publishable regardless of outcome: if architectures differ, we have a ranking; if they don't, we have a negative result that challenges architectural innovation

---

## Phase 5: Final Proposal

### Title
"Do SAE Architectures Actually Differ in Feature Absorption? A Controlled Benchmark with Random Baselines and Seed Variation"

### Hypothesis
Different SAE architectures (JumpReLU, TopK, BatchTopK, Gated, Matryoshka, OrtSAE) exhibit significantly different mean absorption rates when evaluated with controlled random seeds, matched sparsity levels, and multi-domain concept hierarchies. Specifically, JumpReLU will show the highest absorption rate and Matryoshka the lowest, with a difference of at least 10 percentage points (p < 0.05, paired t-test across seeds).

### Falsification Criterion
The hypothesis is falsified if ANY of the following hold:
1. The difference in mean absorption rate between the highest and lowest architecture is < 5 percentage points (p > 0.05)
2. Random SAE baselines achieve absorption rates within 50% of the best trained architecture
3. Seed-to-seed variance exceeds architecture-to-architecture variance (indicating results are idiosyncratic, not systematic)
4. Results on semantic hierarchies (WordNet) contradict results on first-letter task

### Method
We train 6 SAE architectures on GPT-2 Small residual stream activations at layers 5, 10, and 15. Each architecture is trained with 5 random seeds. We evaluate absorption using:
1. Chanin et al.'s k-sparse probe + integrated gradients ablation (early layers)
2. SAEBench's probe projection criteria (all layers)
3. Random SAE baselines (frozen decoder, soft-frozen decoder) for sanity checking

### Evaluation Protocol

**Primary Benchmarks**:
- First-letter hierarchy (a-z, ground truth from Chanin et al.)
- WordNet semantic hierarchy (animal -> mammal -> dog, 50 concepts)
- Both evaluated on sparse probing accuracy and absorption rate

**Metrics with Statistical Test Plan**:
- Absorption rate: mean +/- bootstrap 95% CI across seeds
- Pairwise comparisons: paired t-test with Bonferroni correction (alpha = 0.05 / 15 comparisons = 0.0033)
- Effect size: Cohen's d for architecture differences
- Mixed-effects model: architecture (fixed) + seed + layer (random)
- Number of random seeds: 5 per architecture (25 total SAEs per layer)

**Control Experiments**:
1. Random baseline SAEs (frozen decoder, soft-frozen decoder) at same sparsity/reconstruction
2. Seed-sensitivity analysis: report feature overlap (Hungarian matching) across seeds
3. Partial correlation: architecture vs. absorption, controlling for reconstruction MSE and L0
4. Late-layer validation: SAEBench probe projection metric for layers where IG ablation fails

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|--------------|------------------|
| Vary dictionary size (4k, 16k, 65k) | Width-absorption relationship | Absorption increases with width (Chanin et al., 2024) |
| Vary sparsity (L0 = 20, 40, 100) | Sparsity-absorption monotonicity | Absorption decreases with higher L0 |
| Remove orthogonality penalty (OrtSAE) | Contribution of orthogonality constraint | Absorption increases toward baseline |
| Inner vs. outer Matryoshka levels | Hedging trade-off quantification | Inner levels show higher hedging; outer levels lower absorption |
| Random vs. trained transformer | Absorption as learned vs. artifact | If similar, absorption is architectural, not learned |

### Control Experiments (Ruling Out Alternative Explanations)

1. **Random baseline control**: If trained SAEs don't outperform random baselines, architectural differences are not meaningful (following Korznikov et al., 2026).
2. **Seed control**: If results vary dramatically across seeds, the effect is not reproducible (following EleutherAI, 2025).
3. **Multi-domain control**: If results only hold for first-letter task, they don't generalize.
4. **Sparsity-matching control**: If architectures differ only because they achieve different sparsity, the comparison is unfair.
5. **Late-layer control**: If absorption metrics fail for late layers, results are layer-dependent.

### Pilot Design

**Pilot P1**: Train JumpReLU and TopK on GPT-2 Small layer 8, 2 seeds each, 500K tokens. Run Chanin absorption detection on first-letter task. Target: < 15 min, detect > 10% absorption rate difference.

**Pilot P2**: Train random baseline (frozen decoder) on same setup. Compare absorption rate with trained SAEs. Target: < 10 min, establish baseline.

**Success criteria**:
- Both pilots complete within 30 minutes total
- Absorption rates are measurable (not all zero or all one)
- Difference between architectures > 5% (directional signal)
- Random baseline is clearly worse than trained SAEs (validation signal)

### Resource Estimate

| Component | GPU-hours | Notes |
|-----------|-----------|-------|
| Pilot (2 architectures x 2 seeds) | ~0.1 | GPT-2 Small, single layer |
| Full training (6 architectures x 5 seeds x 3 layers) | ~3.0 | GPT-2 Small, 500K tokens each |
| Evaluation (absorption detection) | ~0.5 | Batch processing |
| Random baselines (2 types x 5 seeds x 3 layers) | ~0.3 | Minimal training |
| **Total** | **~4.0 GPU-hours** | Well within 1-hour-per-task budget |

Model size: GPT-2 Small (124M parameters) — small enough for fast iteration, large enough for meaningful results. If time permits, extend to Gemma-2-2B.

### Risk Assessment

| Threat | Level | Mitigation |
|--------|-------|------------|
| Seed variance dominates architecture variance | HIGH | Report seed variance prominently; increase seeds if needed |
| Random baselines match trained SAEs | HIGH | This is a valid finding; paper pivots to "architectures don't matter" |
| Chanin absorption metric fails (no detectable absorption) | MEDIUM | Use SAEBench probe projection as backup; test on multiple concept sets |
| SAELens API compatibility issues | MEDIUM | Pin SAELens version; test loading before full run |
| Late-layer metrics unreliable | MEDIUM | Focus on early/mid layers where IG ablation works |
| Multi-domain concept selection bias | LOW | Use established hierarchies (WordNet); pre-register concept list |

### Novelty Claim

This experiment answers the following empirical question for the first time:
> **"When controlling for random seed, reconstruction quality, and evaluation domain, do SAE architectures actually differ in feature absorption rates, and do these differences exceed random baseline performance?"**

No existing paper has simultaneously:
1. Compared all major architectures (JumpReLU, TopK, BatchTopK, Gated, Matryoshka, OrtSAE) on absorption
2. Controlled for random seed variation
3. Included random SAE baselines
4. Validated across multiple concept hierarchies
5. Reported statistical significance with correction for multiple comparisons

The result will either:
- **Confirm architectural differences**: Provide the first rigorous ranking of architectures on absorption, with confidence intervals and significance tests. This directly fills Gap 10.
- **Reject architectural differences**: Show that absorption is dominated by seed variance or that random baselines match trained SAEs. This would be a high-impact negative result that challenges the field's focus on architectural innovation.

Either outcome is publishable and advances the field.

---

## Methodology Resources (Detailed)

### Papers Read for Methodology

1. [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders](https://arxiv.org/abs/2503.09532) — Karvonen et al., ICML 2025
2. [CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark](https://arxiv.org/abs/2509.00691) — Gulko et al., BlackBoxNLP 2025
3. [A is for Absorption](https://arxiv.org/abs/2409.14507) — Chanin et al., 2024
4. [Feature Hedging](https://arxiv.org/abs/2505.11756) — Chanin et al., 2025
5. [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) — Bussmann et al., 2025
6. [OrtSAE](https://arxiv.org/abs/2509.22033) — Korznikov et al., 2025
7. [Transcoders Beat SAEs](https://arxiv.org/abs/2501.18823) — Paulo et al., 2025
8. [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) — Korznikov et al., 2026
9. [Provable Feature Recovery](https://arxiv.org/abs/2506.14002) — Chen et al., ICLR 2025
10. [On the Limits of SAEs](https://arxiv.org/abs/2506.15963) — Cui et al., 2025
11. [Concept Bottleneck SAEs](https://arxiv.org/abs/2512.10805) — Kulkarni et al., 2025
12. [SAEs Trained on Same Data Learn Different Features](https://arxiv.org/abs/2501.16615) — EleutherAI, 2025
