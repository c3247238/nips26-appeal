# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Gradient-weight cosine alignment is an informative signal for dynamic WD control**
   - Evidence challenging it:
     - arXiv:2510.25480 (Hölzl et al., 2025): "The expected cosine similarity between vectors with independent noise components diminishes rapidly with increasing dimensions." The full-network gradient-weight alignment degrades sharply in high dimensions; alignment scores computed on full networks are dominated by noise, forcing practitioners to restrict computation to the final linear head.
     - SimiGrad (NeurIPS 2021): For small batch sizes (e.g., 4), the cosine similarity between consecutive minibatch gradients is "almost always 0" due to gradient noise. This directly threatens the viability of minibatch alignment proxies δ̂_t in the alignment-aware WD proposal.
     - Scale-invariant networks (Jane Street Blog; Van Laarhoven 2017; Arora et al., ICLR 2019): In BN/LayerNorm networks, the gradient direction is invariant to weight scaling. This means the cosine similarity between g_t and w_t is determined purely by the optimizer trajectory, NOT by any regularization-relevant signal about which parameters are redundant vs. important.

2. **Assumption: Dynamic WD scheduling is genuinely better than well-tuned fixed WD**
   - Evidence challenging it:
     - Choi et al. (2019, arXiv:1910.11758): "The hyperparameter search space may be the single most important factor explaining the rankings." Many claims of improved dynamic WD can be attributed to comparing against poorly tuned fixed WD baselines.
     - UNDERSTANDING AND SCHEDULING WEIGHT DECAY (OpenReview): SGD with λ=0.0001 (common in papers) is "not a good baseline, as λ=0.0005 often shows better generalization on CIFAR-10 and CIFAR-100." This creates an inflation effect for dynamic methods.
     - D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415): Weight decay is "never useful as an explicit regularizer" — it operates through training dynamics. If the mechanism is dynamics, then tuning dynamics via LR schedule may be equivalent to WD scheduling, making the "scheduling" contribution confounded.

3. **Assumption: A unified framework for WD sub-approaches is meaningful and will yield insights**
   - Evidence challenging it:
     - WD in BN networks is effectively just LR scheduling (Van Laarhoven 2017; Li & Arora 2019). WD scheduling and LR scheduling are conflated. A "unified WD framework" that ignores this conflation may be unifying phenomena that are actually identical (different parameterizations of the same effect).
     - Kosson et al. (arXiv:2510.19093, 2025) show WD matters more than μP, but conclude "real networks lack the perfect scale-invariance typically assumed by works on weight decay." The theoretical toolkit of alignment-aware and norm-matched WD assumes scale invariance that does not hold in practice.
     - Generalization bounds from alignment-based analyses tend to be vacuous for overparameterized deep nets (Nagarajan & Kolter 2019; OPT-ML 2025 paper). Building a "theoretical framework" on alignment quantities may yield bounds too loose to be practically informative.

4. **Assumption: Alignment-aware WD (CWD-style or continuous cosine modulation) improves generalization because it is alignment-aware**
   - Evidence challenging it:
     - CWD (ICLR 2026, arXiv:2510.12402): One real-world user reports that CWD in the Conda optimizer "turned out worse than with normal weight decay." CWD convergence proof acknowledges that "the inner product ⟨m, x⟩ has unknown sign," which is precisely why the mask is introduced — but this also means the masked decay may be non-monotone and create spurious dynamics.
     - CWD uses sign alignment only. The entire theoretical motivation rests on binary masking, but the convergence to a *non-unique* limit point depending on initialization means CWD's selectivity does not have provably better generalization — just different basins.
     - The "Radial Tug-of-War" framing in AdamO (arXiv:2602.05136) implies that CWD's sign alignment does NOT solve the core problem (it only partially masks the radial conflict). This suggests CWD's gains may be coming from a different mechanism than the stated one.

5. **Assumption: The proposed standardized metrics (BEM, CSI, AIS) will solve the evaluation fragmentation problem**
   - Evidence challenging it:
     - AlgoPerf (Dahl et al., 2023; Kasimbeg et al., 2024): Even the most rigorous benchmarking framework for optimizers cannot eliminate "cherry-picking via tuning protocols." Introducing three new metrics (BEM, CSI, AIS) does not resolve the fundamental problem: each metric definition contains hidden assumptions that favor particular methods.
     - The "Alignment Informativeness Score" is only meaningful if alignment carries genuine information. If alignment is noisy at practical batch sizes (SimiGrad result) and scale-invariant (BN effect), then AIS may measure noise, not information.
     - Budget Equivalence Metric (BEM): If WD scheduling is confounded with LR scheduling, then "equal compute" comparisons under different WD schedules do not control for the confound. BEM may obscure rather than reveal true differences.

6. **Assumption: Gradient-weight alignment (δ_t = ⟨g_t, w_t⟩/(‖g_t‖‖w_t‖)) is a useful proxy for when to apply strong regularization**
   - Evidence challenging it:
     - Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340, 2025): "L2 regularization induces parameter-gradient alignment, norm preservation, low-rank bias at stationary points." This creates circularity: WD itself causes δ_t to increase toward 1 at convergence. If alignment is a consequence of WD rather than a ground truth signal, using it to modulate WD creates a feedback loop with unclear stability properties.
     - Galanti et al. (arXiv:2206.05794): WD induces low-rank bias precisely by shrinking small singular values — the alignment signal δ_t aggregates over all parameter dimensions, masking this structurally important but individually invisible effect.

### Landscape of Doubt

The field is in a peculiar state: there is overwhelming empirical evidence that WD matters profoundly, but the theoretical explanations are fragmented and individually weak. Three core tensions define the landscape of doubt:

**Tension 1: The scale-invariance trap.** In the majority of modern architectures (ResNets, VGGs with BN, Transformers with LayerNorm), WD does not operate as a classical regularizer. Its effect is primarily to modulate the effective learning rate. This means that alignment-aware WD methods — which compute ⟨g_t, w_t⟩ — are computing a quantity that does not measure "how aligned the gradient is with the weight for regularization purposes." Instead, they measure a confounded quantity that depends on the effective LR, the batch noise level, and the normalization layer behavior. The alignment signal may be a ghost — numerically nonzero but mechanistically meaningless for the stated purpose.

**Tension 2: The hyperparameter tuning confound.** Dynamic WD methods consistently show gains over "fixed WD baselines," but these baselines are systematically under-tuned. The standard SGD baseline with λ=0.0001 used in many papers is suboptimal by 0.4–0.8% on CIFAR. When baselines are properly tuned (λ=0.0005 for CIFAR, or λ proportional to γ² for stable norms), the gains from dynamic scheduling shrink or disappear. This is not unique to WD — it is the original sin of the optimizer literature (Choi et al., 2019) now infecting the WD sub-literature.

**Tension 3: The unification mirage.** The proposal to unify WD scheduling, alignment-aware WD, decoupled WD, and norm-matched WD into a single framework is intellectually appealing but potentially misleading. These four sub-approaches address different failure modes of different underlying optimization problems. WD scheduling primarily addresses the final-phase gradient norm explosion (Xie et al., 2023). Decoupled WD fixes the L2-gradient interaction in adaptive optimizers (Loshchilov & Hutter, 2019). Norm-matched WD controls the weight-scale/LR coupling (Loshchilov, 2023). Alignment-aware WD targets directionality of the regularization force (Chen et al., 2025). Unifying them under a single mathematical formula λ(t, w, g) = f(...) risks producing a framework that is so general as to be trivially true, with no predictive or actionable content.

---

## Phase 2: Initial Candidates

### Candidate A: The Alignment Signal Is Noise, Not Information

**Challenged assumption:** The gradient-weight cosine similarity δ_t = ⟨g_t, w_t⟩/(‖g_t‖‖w_t‖) carries actionable information about when to apply strong WD.

**Evidence against:**
- In high dimensions, the cosine similarity between two vectors with independent noise components decays as O(1/√d) (GWA paper, arXiv:2510.25480). At d=10^6 (typical ResNet-20 parameter count), this is ~10^-3. The signal is essentially 0 for non-trivially correlated directions.
- In BN/LayerNorm networks, ‖∇f(w)‖/‖w‖ → steady state driven by the ratio of learning rate to weight norm, not by the alignment between update direction and weight direction. The cosine similarity δ_t is a downstream effect of this ratio, not an independent informative signal.
- WD itself induces alignment at stationary points (Kuzborskij & Abbasi-Yadkori, 2025). The proxy is causally downstream of WD, not upstream.
- The minibatch gradient ĝ_t ≈ ∇f(w_t) + noise. For batch size B, the noise variance is O(σ²/B). The alignment proxy ⟨ĝ_t, w_t⟩ has variance σ²‖w‖²/B. At batch size 128 (standard CIFAR), this noise variance can exceed the signal variance.

**Contrarian hypothesis:** The Alignment Informativeness Score (AIS) will be close to zero for most practical architectures and batch sizes, because the alignment signal δ̂_t is dominated by scale-invariant effective-LR dynamics and minibatch noise. Alignment-aware WD methods improve over fixed WD not because of alignment informativeness, but because they accidentally implement a form of gradient-norm-aware WD (when ‖g‖ is large, the sign alignment mask tends to fire more often), which approximates the well-known SWD heuristic.

**Exploitation plan:** Empirically measure AIS across architectures (with vs. without BN), batch sizes (32 to 4096), and training phases. Show that AIS ≈ 0 for BN networks at practical batch sizes. Then demonstrate that alignment-aware WD reduces to gradient-norm-aware WD as a special case, and that proper gradient-norm scheduling (SWD) matches or exceeds continuous alignment-aware methods.

**Novelty estimate:** 7/10 — challenges the core premise of alignment-aware WD while providing a constructive alternative explanation.

### Candidate B: Dynamic WD Is a Hyperparameter Search Proxy, Not a Mechanism

**Challenged assumption:** Scheduled or adaptive WD methods have a mechanistic advantage over fixed WD with proper tuning.

**Evidence against:**
- Optimizer benchmarking literature (Choi et al., 2019; AlgoPerf, 2023-2024): fixed-baseline comparisons are always biased toward dynamic methods when baselines are under-tuned. A comprehensive fair comparison in the WD literature does not exist.
- CPR (NeurIPS 2024): a principled method that outperforms AdamW on CIFAR-100, ImageNet, and GPT-2. But CPR uses only 2 hyperparameters and adapts WD per-parameter-matrix. The question is whether CPR's gains come from adaptation (mechanistic) or from the fact that its 2-hyperparameter search automatically finds better optima in the hyperparameter landscape.
- Ferbach et al. (arXiv:2602.05298): log-time WD schedules improve training. But they explicitly claim "40% compute efficiency gain" from a log-time schedule. If the gain is purely from the schedule shape, not the WD signal, then any reasonable schedule shape might achieve similar gains.

**Contrarian hypothesis:** After controlling for compute-equivalent hyperparameter search (using AlgoPerf-style fair tuning), the gain attributable specifically to "dynamic adaptation" of WD (as opposed to better average WD level) will be less than 0.5% accuracy on CIFAR-10/100 and within noise on ImageNet. The appearance of large gains in the literature is a systematic artifact of the under-tuned fixed baseline problem.

**Exploitation plan:** Run a head-to-head comparison: (1) fixed WD with Bayesian hyperparameter optimization over 20 trials; (2) dynamic WD methods with the same 20-trial budget. Measure gains on CIFAR-10, CIFAR-100 with ResNet-20, VGG-16-BN. Adjust for the hyperparameter search budget. Report the "honest gain" after budget equalization.

**Novelty estimate:** 6/10 — methodologically important but potentially incremental; similar points have been made about adaptive optimizers generally.

### Candidate C: The Unification Is Trivially True and Practically Useless

**Challenged assumption:** Unifying WD scheduling, alignment-aware WD, decoupled WD, and norm-matched WD into a single formula λ(t, w, g) = f(alignment, norm, schedule, target) reveals non-trivial theoretical structure.

**Evidence against:**
- Any functional form f(·) that includes alignment, norm, schedule, and target as inputs will trivially contain all four methods as special cases — this is just parameterization. A polynomial can fit any finite set of points; a sufficiently general function f fits any WD method. The unification claim requires demonstrating that the resulting family has a non-trivial mathematical structure that generates predictions beyond interpolation between existing methods.
- The methods address fundamentally different problems: decoupled WD fixes a bug in Adam (gradient contamination); scheduling fixes final-phase gradient explosion; norm-matching fixes the LR-WD coupling under scale invariance; alignment-aware WD adds selectivity. These four problems are not special cases of each other — they exist in different optimizer contexts, different training phases, and different architectural settings.
- Historical precedent: Similar unification frameworks for adaptive optimizers (AMSGrad, AdaBound, Padam) failed to produce lasting insights and were largely abandoned. Their "unifications" turned out to be parameterizations that did not transfer across tasks.

**Contrarian hypothesis:** A "unified dynamic WD framework" expressed as λ(t, w, g) = f(δ, ‖w‖, t, τ) will, when practically instantiated, either (a) reduce to one of the four existing methods at the optimal hyperparameter setting, or (b) fail to outperform the best single-method baseline on even one benchmark. The theoretical framework will generate predictions (e.g., "alignment-scheduled WD should help when δ_t < δ_threshold") that are either already captured by gradient-norm scheduling or are falsified by experiment.

**Exploitation plan:** Implement the unified framework with 4 input dimensions and train a meta-optimizer to search over the function f. If the optimal f found by meta-learning recovers an existing method (e.g., gradient-norm-aware WD or pure time decay), this constitutes evidence against non-trivial unification.

**Novelty estimate:** 5/10 — philosophically interesting but hard to make definitive; risks becoming a negative result without constructive alternative.

---

## Phase 3: Self-Critique

### Against Candidate A (Alignment Signal Is Noise)

**Steelman of conventional view:**
- CWD (ICLR 2026) was accepted to a top venue with positive results on language modeling and ImageNet. The sign-alignment signal, even if the cosine value is small, could carry O(1) binary information (positive vs. negative inner product) that remains informative even in high dimensions.
- Kuzborskij & Abbasi-Yadkori (2025) show that L2 induces alignment at *stationary points*, not during training. The dynamic δ̂_t during training may still be informative about the optimization trajectory's phase.
- AdamO (arXiv:2602.05136) explicitly decomposes gradient into radial and tangential components and shows gains from controlling only the radial component — this is a controlled experiment that supports the alignment-information hypothesis.

**Cherry-picking check:**
- I found that GWA paper and SimiGrad support the noise argument, but these papers are about gradient-gradient alignment (between samples), not gradient-weight alignment. The degradation result from GWA applies to the full-network computation, but CWD operates per-coordinate (element-wise sign), not over full-vector cosine similarity. This is a significant difference.
- The BN scale-invariance argument applies to WD's effect on the loss function, but does not directly falsify that the ⟨g, w⟩ inner product carries alignment information — it might carry different information than hypothesized.

**Confounding check:**
- The batch-size noise argument has a confound: larger batch sizes would reduce the noise, and CWD was validated in language model pretraining with large batch sizes where the noise is smaller.

**Actionability check:**
- Even if alignment is somewhat noisy, this leads to a constructive proposal: measure and report AIS as a diagnostic, and design experiments that separate alignment-informativeness from gradient-norm effects.

**Verdict: MODERATE**. The alignment-as-noise hypothesis is overstated for full-vector cosine but has real bite for per-coordinate sign alignment at small batch sizes. The key insight is valid but needs more precision: it applies most strongly to continuous modulation methods (like the project's δ̂_t proxy), not to CWD's binary sign test.

### Against Candidate B (Dynamic WD Is Hyperparameter Proxy)

**Steelman of conventional view:**
- CPR (NeurIPS 2024) outperforms AdamW with only 2 hyperparameters versus AdamW's 1 WD hyperparameter. If the gain were purely from better hyperparameter search, CPR's 2-parameter search should not outperform AdamW's well-tuned 1-parameter search. The adaptation mechanism must be contributing something.
- SWD (NeurIPS 2023) is compared against fixed WD *at the same value* (not under-tuned baselines) and still shows gains. The paper explicitly tunes the baseline.
- Log-time WD schedule (Ferbach et al., arXiv:2602.05298) shows gains even when the average WD value integrated over training is equalized to the fixed baseline.

**Cherry-picking check:**
- I'm citing Choi et al. (2019), which was about *adaptive optimizers* (Adam vs. SGD), not specifically about WD scheduling. The extension to WD may not hold directly.

**Confounding check:**
- Many "dynamic WD" methods also change the WD-LR coupling implicitly. If the gain comes from the coupling change (schedule shape), not the dynamic signal, the conclusion is nuanced: the method has a mechanism, but the mechanism is LR-WD co-design, not adaptive WD per se.

**Actionability check:**
- Even if the gains are partially confounded, this leads to a constructive proposal: the Budget Equivalence Metric should be defined precisely as "equal compute with equal hyperparameter search budget" to test whether the gains survive.

**Verdict: MODERATE**. The hyperparameter-proxy concern is valid and under-addressed in the literature, but it does not fully explain the gains from CPR or SWD. A cleaner claim: *some* gains in the WD scheduling literature are hyperparameter artifacts, and the field lacks a fair comparison protocol to distinguish genuine mechanism gains from search gains.

### Against Candidate C (Unification Is Trivially True)

**Steelman of conventional view:**
- Unification frameworks in ML have produced genuine insights even when seemingly trivial at first glance. The generalized linear model unifies regression/classification/Poisson regression — not trivially, because it reveals the role of link functions and exponential family structure.
- The unified WD framework could produce non-trivial insights if it identifies *necessary and sufficient conditions* for each sub-approach to be optimal. For example: "norm-matched WD is optimal if and only if the target norm τ equals the SGD equilibrium norm" is a non-trivial claim that could be tested.
- The standard of "generating new predictions" can be met: if the framework predicts that alignment-aware WD degrades to norm-matched WD in BN networks (because alignment is confounded by scale invariance), this is a non-trivial falsifiable prediction.

**Actionability check:**
- Even if the unification is sometimes trivial, the attempt generates new hypotheses worth testing. The risk is framing, not feasibility.

**Verdict: WEAK**. The "trivially true" concern is valid as a framing risk but not as a fatal objection. The concern is more about HOW the framework is presented than whether a framework is worth building. The actionable version: the framework must make falsifiable, non-trivial predictions, not just unify existing methods as special cases.

---

## Phase 4: Refinement

**Dropped:** Candidate C (Unification is trivially true) — this framing is more about presentation than a deep methodological flaw. Reframe as a constraint on how to build the framework, not a reason not to build it.

**Strengthened survivors:**

**Candidate A (refined):** The critique should be made precise:
- The claim is NOT that alignment is always noise. It is that: (1) in BN/LN networks, gradient-weight cosine similarity is driven primarily by scale-invariant dynamics, making δ̂_t a proxy for effective-LR-over-norm rather than alignment-quality; (2) at practical batch sizes (≤256), minibatch cosine similarity is dominated by noise; (3) the gain from alignment-aware WD likely comes from the effective gradient-norm adaptation implicit in the alignment mask, not from alignment per se.
- The constructive proposal: measure the AIS formally. Define AIS = mutual information between δ̂_t and the optimal WD direction (as determined by oracle WD that minimizes generalization error). Show empirically that AIS is near-zero for BN networks but non-zero for networks without BN. Then show that for non-BN networks, alignment-aware WD indeed shows larger gains.
- This turns the critique into a rich characterization paper: "When Does Alignment-Aware Weight Decay Work?"

**Candidate B (refined):** The key experimental design is the "equal-budget fair comparison":
- Use Bayesian optimization (e.g., Optuna) over the same number of trials (20) for both fixed WD and dynamic WD methods.
- Measure the distribution of test accuracy over 20 trials for each. Compare median, not best-run.
- If the median gain from dynamic WD survives, it is a genuine mechanism gain. If it disappears, it is a search artifact.
- This is a methodology paper as much as a WD paper.

**Additional corroboration for Candidate A (alignment noise):**
- arXiv:2510.25480 (GWA, 2025): "Computing alignment scores using the linear classifier head mitigates dimensionality issues such as decreasing mean of the alignment distribution and increasing tailedness. Using the full model, or including more layers of the network, leads to a worse learning signal."
- This directly implies that full-network gradient-weight alignment (as used in the project's δ̂_t formulation) is a worse signal than layer-restricted alignment.

**Selected front-runner:** Candidate A refined — "The Alignment Signal Degrades with Scale Invariance and High Dimensionality" with constructive proposal for layer-restricted alignment.

---

## Phase 5: Final Proposal

### Title

**Rethinking Gradient-Weight Alignment in Dynamic Weight Decay: When Scale Invariance and Dimensionality Make the Signal Vanish**

### Challenged Assumption

The gradient-weight cosine alignment δ̂_t = ⟨ĝ_t, w_t⟩/(‖ĝ_t‖‖w_t‖) is an informative signal for modulating WD strength across modern deep learning architectures.

### Evidence (Both For and Against)

**Evidence supporting the assumption (mainstream view):**
- CWD (ICLR 2026): sign alignment consistently improves final test accuracy on language models and ImageNet at million-to-billion parameter scales.
- AdamO (arXiv:2602.05136): explicit radial/tangential decomposition shows gains from radial (norm) control conditioned on alignment structure.
- Sun et al. (CVPR 2025): theoretical proof that WD improves generalization via the alignment quantity δ_T < 1, supporting the theoretical relevance of alignment in nonconvex SGD.
- Kuzborskij & Abbasi-Yadkori (2025): alignment is a real structural property at stationary points, suggesting it carries information about the solution quality.

**Evidence against the assumption (contrarian view):**
- GWA (arXiv:2510.25480): full-network gradient-weight alignment is dominated by noise in high dimensions; only layer-restricted computation yields a reliable signal.
- Scale-invariance literature (Van Laarhoven 2017; Li & Arora 2019; D'Angelo et al. NeurIPS 2024): in BN/LN networks, ‖w‖ is determined by the effective LR, not by regularization need. The cosine similarity ⟨g, w⟩/(‖g‖‖w‖) becomes a proxy for effective-LR dynamics, not alignment quality.
- SimiGrad (NeurIPS 2021): at batch size ≤256, minibatch gradient cosine similarities are near 0 due to noise. The minibatch alignment proxy δ̂_t suffers the same noise floor.
- Kuzborskij & Abbasi-Yadkori (2025): WD itself *causes* alignment to increase at stationary points, creating a feedback loop: δ̂_t is both a target of WD and a function of WD, making it a potentially circular signal.

### Hypothesis

The information content of δ̂_t for WD adaptation is architecture-dependent and batch-size-dependent:
1. **In BN/LN networks**: δ̂_t is primarily a readout of the current effective LR / weight norm ratio, not an alignment-quality signal. Alignment-aware WD in BN networks gains from gradient-norm adaptation (equivalent to SWD), not from alignment per se.
2. **In non-BN networks** (e.g., networks without normalization): δ̂_t contains genuine alignment information, and alignment-aware WD provides non-trivial improvements over gradient-norm-aware WD.
3. **Layer-restricted alignment** (restricting δ̂_t to the final linear classifier) preserves signal quality by avoiding dimensionality collapse.

### Method

Controlled experimental comparison:

**Primary experiment: Isolating the alignment signal**
- Train ResNet-20 on CIFAR-10 in two conditions: (a) with BN (standard), (b) without BN (same architecture, BN layers removed and replaced with bias-only layers).
- Apply three WD strategies: fixed WD, gradient-norm-aware WD (SWD), continuous alignment-aware WD (the project's δ̂_t formula).
- Measure:
  - Final test accuracy (mean ± std over seeds 42, 123, 456)
  - AIS = rank correlation between δ̂_t trajectory and oracle WD (post-hoc optimal WD schedule estimated by grid search)
  - Variance of δ̂_t over consecutive minibatches (noise estimate)

**Secondary experiment: Batch size effect**
- Train with batch sizes {64, 128, 256, 512, 1024} to verify that alignment signal quality increases with batch size.
- Expected prediction: the gain from alignment-aware over gradient-norm-aware WD should increase with batch size (as noise decreases).

**Tertiary experiment: Layer-restricted alignment**
- Compare full-network δ̂_t vs. last-layer-only δ̂_t as signals for WD modulation.
- Expected prediction: last-layer δ̂_t is more stable (lower variance) and yields better performance.

### Baselines

- Fixed WD (properly tuned: λ ∈ {5e-4, 1e-3, 2e-3, 5e-3} with 3-seed evaluation)
- SWD / gradient-norm-aware WD (NeurIPS 2023 baseline, representing "correct" gradient-norm scheduling)
- CWD sign-alignment (ICLR 2026, representing binary alignment-aware WD)
- Full-cosine continuous alignment-aware WD (the project's δ̂_t formula — the method under examination)
- Last-layer restricted δ̂_t WD (the proposed constructive improvement)

### Experimental Plan

All experiments on local 8x RTX PRO 6000 Blackwell GPUs.

| Experiment | Architecture | Dataset | Time estimate |
|---|---|---|---|
| BN vs. no-BN signal isolation | ResNet-20 (±BN) | CIFAR-10 | 3 seeds × 2 conditions × 5 methods = ~45 min |
| Batch size effect | ResNet-20 with BN | CIFAR-10 | 3 seeds × 5 batch sizes × 3 methods = ~60 min |
| Layer-restricted alignment | ResNet-20 with BN | CIFAR-10, CIFAR-100 | 3 seeds × 2 datasets × 4 methods = ~60 min |
| Scale-up validation | VGG-16-BN | CIFAR-100 | 3 seeds × 4 methods = ~30 min |

Total: ~3 hours. Well within the 1-hour-per-task guideline when parallelized.

### Risk Assessment

**If the mainstream view turns out to be correct:**
- Even if AIS is nonzero in BN networks, the constructive result (layer-restricted alignment is better) survives.
- The batch-size dependency claim provides a useful characterization even if BN doesn't eliminate the signal entirely.
- The paper becomes "When and How Alignment-Aware WD Works" rather than "Alignment-Aware WD Doesn't Work" — still publishable and valuable.

**If the contrarian hypothesis is wrong in detail:**
- We would find that full-network δ̂_t is informative in BN networks. This would require explaining why the GWA (2025) finding about full-network gradient-weight alignment degrading doesn't apply. Possible resolution: WD operates on the steady-state joint distribution of (g, w), not on the instantaneous per-sample distribution, so the noise floor for WD adaptation purposes may be lower than for sample-level generalization tracking.

**Hardest-to-falsify concern:**
- The claim that "gains from alignment-aware WD come from implicit gradient-norm adaptation" requires careful experimental design. Gradient-norm-aware WD (SWD) and alignment-aware WD are not identical — SWD uses ‖g_t‖, while alignment-aware WD uses ⟨g_t, w_t⟩/‖g_t‖‖w_t‖. The correlation between these two signals needs to be measured, not assumed.

### Novelty Claim

The specific insight: **gradient-weight cosine alignment is architecture-conditioned information**. In scale-invariant (BN/LN) architectures, δ̂_t is dominated by effective-LR dynamics rather than genuine alignment between gradient and weight directions. This insight predicts that:
1. Alignment-aware WD's performance will correlate with the degree of scale-invariance in the architecture.
2. Layer-restricted alignment (final-layer only) preserves the genuine alignment signal while discarding the scale-invariant noise.
3. As architectures move toward more norm-invariant designs (e.g., full LN transformers), the practical value of alignment-aware WD over gradient-norm-aware WD should diminish.

This insight is orthogonal to and compatible with the unified framework contribution: it provides the characterization necessary to determine when each branch of the framework applies, turning the framework from a taxonomic exercise into a predictive tool.

---

**Sources consulted:**
- [GWA paper (arXiv:2510.25480)](https://arxiv.org/abs/2510.25480)
- [Cautious Weight Decay ICLR 2026 (arXiv:2510.12402)](https://arxiv.org/abs/2510.12402)
- [CWD OpenReview (with failure report)](https://openreview.net/forum?id=Gwe6gbGng5)
- [SimiGrad NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/abea47ba24142ed16b7d8fbf2c740e0d-Paper.pdf)
- [D'Angelo et al. NeurIPS 2024 (arXiv:2310.04415)](https://arxiv.org/abs/2310.04415)
- [SWD (arXiv:2011.11152)](https://arxiv.org/abs/2011.11152)
- [Optimizer benchmarking needs to account for HP tuning (arXiv:1910.11758)](https://arxiv.org/abs/1910.11758)
- [Jane Street Blog: L2 Regularization and Batch Norm](https://blog.janestreet.com/l2-regularization-and-batch-norm/)
- [Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340)](https://arxiv.org/abs/2502.17340)
- [Kosson et al. (arXiv:2510.19093)](https://arxiv.org/abs/2510.19093)
- [Kosson et al. rotational equilibrium (arXiv:2305.17212)](https://arxiv.org/abs/2305.17212)
- [Galanti et al. low-rank bias (arXiv:2206.05794)](https://arxiv.org/abs/2206.05794)
- [AdamO (arXiv:2602.05136)](https://arxiv.org/abs/2602.05136)
- [CPR NeurIPS 2024 (arXiv:2311.09058)](https://arxiv.org/abs/2311.09058)
- [Power Lines: Scaling Laws for WD (arXiv:2505.13738)](https://arxiv.org/abs/2505.13738)
- [Weight Decay and Batch Normalization (arXiv:2309.04644)](https://arxiv.org/abs/2309.04644)
