# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a fixable training pathology, not an intrinsic limitation of sparse dictionary learning.**
   - Evidence challenging it:
     - Chanin et al. (2024) show absorption increases with both higher sparsity *and* wider dictionaries---the exact scaling regime the field pursues. This suggests absorption is not a transient training artifact but a structural consequence of the sparsity objective under hierarchical feature co-occurrence.
     - Korznikov et al. (2026) demonstrate that **random-decoder SAEs** achieve competitive performance on AutoInterp (0.87 vs. 0.90), sparse probing (0.69 vs. 0.72), and RAVEL (0.73 vs. 0.72) compared to fully trained SAEs. If random dictionaries perform nearly as well on standard metrics, the "fixable pathology" framing loses force---the problem may be geometric, not dynamic.
     - Barin Pacela et al. (2026) in *Stop Probing, Start Coding* prove that SAE-learned dictionaries point in "substantially wrong directions" under OOD compositional shifts. Replacing the encoder with per-sample iterative inference (FISTA) on the *same dictionary* does not close the gap, implicating **dictionary learning itself** rather than amortization or training dynamics.

2. **Assumption: Absorption-mitigation architectures (OrtSAE, Matryoshka SAEs) improve SAEs overall, not just on a single metric.**
   - Evidence challenging it:
     - Chanin et al. (2025) in *Feature Hedging* prove that narrower SAEs reduce absorption but increase **feature hedging**---an opposite failure mode where latents incorrectly mix correlated features. Matryoshka SAEs explicitly trade absorption for hedging because their narrow inner levels must reconstruct independently.
     - Roy et al. (2026) show "catastrophic interpretability collapse" under aggressive sparsification: global metrics (MIG, reconstruction MSE) remain stable while local interpretability collapses systematically. This means methods reporting "65% absorption reduction" may be masking other pathologies with poorly aligned metrics.
     - SAEBench itself refuses to provide a composite score because metrics "operate on different scales and exhibit varying levels of noise" (Karvonen et al., 2025). If the benchmark authors cannot combine metrics, claims of "overall improvement" from any single architecture are epistemically premature.

3. **Assumption: SAEBench and related benchmarks reliably measure interpretability quality.**
   - Evidence challenging it:
     - Kantamneni et al. (2025) find that **SAE probes underperform simple logistic-regression baselines** across over 100 datasets: "SAE probes underperform the baseline of logistic regression in each regime when taking the mean across datasets."
     - Karvonen et al. (2025) note that SAEBench supervised metrics "can only evaluate concepts with reliable ground truth data, representing a small subset of the vast space of concepts encoded in language models." This means architectures can appear equivalent simply because the metrics miss most of the feature space.
     - The auto-interp metric relies on an external LLM judge, introducing "non-determinism, potential biases, and a lack of reproducibility" (Gulko et al., 2025, CE-Bench paper).
     - Korznikov et al. (2026) found that SAEs recover only **9% of true features** in synthetic ground-truth evaluations despite achieving **71% explained variance**---a stark disconnect between reconstruction proxies and actual feature recovery.

4. **Assumption: SAE features are "better" (more canonical, more useful) than simple linear probes or raw neuron activations.**
   - Evidence challenging it:
     - Leask et al. (2025) title their ICLR paper *Sparse Autoencoders Do Not Find Canonical Units of Analysis*---a direct refutation of the field's foundational premise.
     - Kantamneni et al. (2025): "Additionally, we find that baseline methods can provide many of the interpretability benefits typically attributed to SAEs."
     - Barin Pacela et al. (2026): "SAE codes offer no consistent advantage over probing raw activations at any scale."

5. **Assumption: The first-letter absorption metric generalizes to arbitrary semantic hierarchies.**
   - Evidence challenging it:
     - Chanin et al. (2024) explicitly state their metric is **conservative** and likely underestimates true absorption. It cannot capture cases where multiple absorbing latents activate together or where main latents activate only weakly.
     - The metric is defined on a specific spelling task (first-letter detection). No prior work has validated it on non-lexical hierarchies (geography, biology, colors) or shown it correlates with absorption in other domains.
     - Karvonen et al. (2025): "Our supervised metrics examine only a fraction of each SAE's latents."

### Landscape of Doubt

The skeptical turn in SAE research (2024--2026) is not a fringe position---it is increasingly mainstream, with papers at ICLR, ICML, NeurIPS, and EMNLP directly challenging core assumptions. The cracks cluster around three themes:

1. **Metric misalignment**: Reconstruction quality, interpretability proxy scores, and actual ground-truth feature recovery are poorly correlated. Architectures can dominate on one while failing on the others.
2. **Baseline superiority**: Simple logistic regression probes, PCA, and even random-decoder SAEs often match or exceed trained SAEs on downstream tasks, raising the question of what unique value SAEs provide.
3. **Intrinsic tradeoffs**: Absorption, hedging, dead neurons, and polysemanticity appear to form an interconnected pathology network. Fixing one symptom predictably worsens another, suggesting there may be no architectural "cure."

---

## Phase 2: Initial Candidates

### Candidate A: The Impossibility Triangle---Absorption-Mitigation Methods Do Not Pareto-Dominate Standard SAEs

- **Challenged assumption**: OrtSAE, Matryoshka SAEs, and other absorption-mitigation methods are "better" architectures that should replace standard SAEs.
- **Evidence against it**: Chanin et al. (2025) prove Matryoshka SAEs trade absorption for hedging. Roy et al. (2026) show global metrics mask local collapse. SAEBench explicitly refuses to composite-score architectures because metrics are noisy and misaligned.
- **Contrarian hypothesis**: Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Each method occupies a distinct tradeoff region, and the apparent superiority of OrtSAE/Matryoshka on absorption metrics is an artifact of selective single-metric reporting.
- **Exploitation plan**: Conduct the first systematic, training-free, multi-objective Pareto evaluation across existing pretrained SAE checkpoints (Gemma Scope, Llama Scope, SAEBench releases). Normalize metrics within model families, compute empirical Pareto fronts, and test stochastic dominance with Mann-Whitney U tests. If no architecture dominates, reframe the field's agenda from "fixing absorption" to "navigating unavoidable tradeoffs."
- **Novelty estimate**: 8/10. No prior work frames this as a systematic Pareto analysis across the full metric suite using existing checkpoints.

### Candidate B: Feature Absorption Is a Geometric Inevitability, Not a Training Artifact

- **Challenged assumption**: Absorption arises from flawed training dynamics (wrong hyperparameters, insufficient data, bad initialization) and can be eliminated with better training.
- **Evidence against it**: Barin Pacela et al. (2026) show the bottleneck is dictionary learning, not amortization. Korznikov et al. (2026) show random-decoder SAEs match trained SAEs on standard metrics. Chanin et al. (2024) show absorption increases predictably with sparsity and width.
- **Contrarian hypothesis**: Feature absorption is primarily a geometric consequence of sparse dictionary learning on hierarchically structured data, not a pathology of flawed training dynamics. Randomly initialized, frozen-decoder SAEs matched for sparsity will exhibit absorption rates comparable to fully trained SAEs.
- **Exploitation plan**: Compare a trained SAE baseline (`gpt2-small-res-jb`) against a random-decoder SAE (frozen decoder, train only encoder) matched for L0 using TopK. Run the `sae-spelling` absorption metric on both. If random decoders show comparable absorption, the result reframes absorption as intrinsic to sparse coding on hierarchical data.
- **Novelty estimate**: 7/10. Korznikov et al. (2026) introduced random-decoder baselines but did not measure absorption on them.
- **Constraint note**: This candidate **violates the project's training-free constraint** and can only proceed if the constraint is relaxed.

### Candidate C: The First-Letter Absorption Metric Is Unrepresentative of General Absorption

- **Challenged assumption**: Chanin et al.'s first-letter absorption metric generalizes to arbitrary semantic hierarchies and can serve as a universal benchmark.
- **Evidence against it**: The metric is task-specific (spelling), conservative by design, and has never been validated on non-lexical domains. SAEBench authors note supervised metrics examine "only a fraction of each SAE's latents."
- **Contrarian hypothesis**: A task-agnostic absorption metric constructed via automated hierarchy discovery will correlate only weakly (r < 0.4) with the first-letter benchmark, revealing that first-letter absorption is unrepresentative of absorption in broader semantic domains.
- **Exploitation plan**: Use LLM-based automated hierarchy discovery to identify parent-child relationships in 2--3 non-lexical domains (geography, biology, colors). Train logistic regression probes, apply the causal ablation framework, and compute absorption rates. Correlate with the original first-letter benchmark across 20--50 pretrained SAEs. If correlation is weak, the divergence itself is a valuable negative result.
- **Novelty estimate**: 8/10. No prior work has proposed a fully task-agnostic absorption metric.

---

## Phase 3: Self-Critique

### Against Candidate A (Multi-Objective Pareto Evaluation)

- **Steelman**: OrtSAE reports 65% absorption reduction with minimal overhead, and Matryoshka SAEs claim ~10x reduction. These are large, consistent effects across multiple layers and models. If the effect is real and robust, it is plausible that these methods genuinely improve SAE quality overall, even if some secondary metrics degrade slightly.
- **Cherry-picking check**: I am emphasizing hedging, dead neurons, and metric noise while downplaying the magnitude of absorption reduction. However, the counter-evidence is not from a fringe source---it comes from Chanin et al. (2025), the same group that discovered absorption, and from SAEBench itself.
- **Confounding check**: Architecture families differ in dictionary width, L0 target, and training data. A fair Pareto analysis must control for these. The proposed design includes normalization within model family, architecture dummies, and cluster-robust SEs.
- **Actionability check**: Even if no architecture dominates, the paper provides the first rigorous multi-objective validation and motivates future work on task-adaptive SAE selection. This is constructive, not a "gotcha."
- **Verdict**: STRONG

### Against Candidate B (Random-Decoder Baseline)

- **Steelman**: Random decoders may match trained SAEs on AutoInterp and sparse probing because these metrics measure "interpretability of directions," not "recovery of true features." Absorption is a more specific pathology tied to hierarchical feature co-occurrence, which may require the SAE to actually learn the data distribution. A random decoder might fail to capture any hierarchical structure and thus show lower absorption.
- **Cherry-picking check**: I am selectively citing Korznikov et al.'s success on proxy metrics to infer success on absorption, but absorption was explicitly *not* tested.
- **Confounding check**: If random-decoder SAEs show lower absorption, it could mean training dynamics *cause* absorption (the opposite of my hypothesis). The experiment must be designed to distinguish these interpretations.
- **Actionability check**: If random decoders match trained SAEs on absorption, it reframes the problem as geometric and motivates research on data-independent regularization. If they do not, it vindicates training-specific fixes. Either outcome is actionable.
- **Verdict**: MODERATE. The hypothesis is provocative but the experiment is high-risk and violates the training-free constraint.

### Against Candidate C (Task-Agnostic Metric)

- **Steelman**: The first-letter task was chosen precisely because it has clean, verifiable ground truth. Geography and biology hierarchies are fuzzier---a country can belong to multiple continents (transcontinental), and biological taxonomy is constantly revised. Weak correlation with first-letter absorption may reflect noise in the new metric, not unrepresentativeness of the old one.
- **Cherry-picking check**: I am assuming that non-lexical domains are "more general" without evidence that absorption behaves differently across domains. The first-letter task may be perfectly representative.
- **Confounding check**: LLM-generated hierarchies may be noisy or hallucinated, introducing measurement error that attenuates correlation.
- **Actionability check**: Even if correlation is weak, the paper can pivot to analyzing *why*---the divergence itself may reveal domain-specific absorption patterns. This is still constructive.
- **Verdict**: MODERATE. The idea is novel but depends heavily on LLM reliability and domain choice.

---

## Phase 4: Refinement

- **Dropped**: Candidate B is dropped from the front-runner position because it violates the project's training-free constraint. It remains intellectually interesting but is not viable under current project rules.
- **Strengthened**: Candidate A is strengthened by making the critique more precise. The claim is not that absorption-mitigation methods are useless, but that their apparent superiority is an artifact of **selective single-metric reporting**. The constructive proposal is to reframe the research agenda toward **task-adaptive SAE selection** rather than universal architectural superiority.
- **Additional corroboration**: The pilot experiment (`e1_full_gpt2`) confirmed the metric pipeline works end-to-end but revealed that simplified absorption/hedging proxies are too coarse. This validates the need for rigorous, multi-metric evaluation but also shows the methodology must be refined (integrate `sae-spelling`, increase token budget) before full-scale experiments.
- **Selected front-runner**: **Candidate A** --- The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods.

---

## Phase 5: Final Proposal

### Title
**The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods**

(Alternative: **Rethinking Feature Absorption: Why Current Mitigations Trade One Pathology for Another**)

### Challenged Assumption
The field assumes that absorption-mitigation architectures (OrtSAE, Matryoshka SAEs, masked regularization, ATM) are unambiguously "better" than standard SAEs and should replace them. This assumption is supported by selective single-metric evaluations that optimize absorption in isolation.

### Evidence
**For the mainstream view:**
- OrtSAE reports 65% absorption reduction with minimal overhead (Korznikov et al., 2025).
- Matryoshka SAEs claim ~10x absorption reduction while maintaining multi-scale interpretability (Bussmann et al., 2025).
- These effects are replicated across hundreds of SAEs and multiple model families.

**Against the mainstream view:**
- Chanin et al. (2025) prove narrower SAEs reduce absorption but increase **feature hedging**---Matryoshka SAEs trade one pathology for another.
- Roy et al. (2026) show "catastrophic interpretability collapse" under sparsification: global metrics stay stable while local interpretability collapses.
- Kantamneni et al. (2025) find SAE probes underperform simple logistic-regression baselines.
- Korznikov et al. (2026) show random-decoder SAEs match trained SAEs on standard metrics, suggesting much of SAE "quality" is geometric, not learned.
- SAEBench explicitly refuses to composite-score architectures because metrics are noisy, misaligned, and operate on different scales (Karvonen et al., 2025).

### Hypothesis
Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Each method occupies a distinct tradeoff region, and the apparent superiority of OrtSAE/Matryoshka on absorption metrics is an artifact of selective single-metric reporting.

### Method

#### Experiment 1: Multi-Objective Pareto Evaluation (Training-Free)
**Checkpoint corpus:**
- GPT-2 Small SAEs from SAELens (Standard, TopK, Gated, JumpReLU, BatchTopK)
- Pythia-160M SAEs from SAEBench releases
- If accessible: Gemma Scope and Llama Scope variants

**Metrics:**
- **Absorption**: Proper `sae-spelling` first-letter metric (Chanin et al., 2024)
- **Hedging**: Correlated-pair hedging score (Chanin et al., 2025)
- **Reconstruction**: L0, explained variance, CE loss recovered
- **Dead neurons**: Fraction of latents with near-zero activation frequency (computed on >=50k tokens)
- **Downstream probing**: Sparse probing F1 from SAEBench
- **RAVEL**: Cause and Isolation scores for causal disentanglement

**Analysis:**
1. Normalize each metric to [0, 1] within model family.
2. Compute empirical Pareto fronts per architecture family.
3. Test stochastic dominance using Mann-Whitney U tests across the full metric suite.
4. Report pairwise tradeoff curves (absorption vs. hedging, absorption vs. reconstruction, etc.).

**Models:** GPT-2 Small and Pythia-160M (open, accessible, well-studied).

#### Experiment 2: Downstream Causal Cost Meta-Analysis (Training-Free)
Using the 200+ pretrained SAEs from SAEBench:
1. Extract absorption, RAVEL Cause/Isolation, sparse probing F1, TPP, SCR, and L0/loss-recovered.
2. Perform partial correlation and regression with absorption as the predictor, controlling for L0 and reconstruction.
3. Include architecture family dummies and dictionary width as covariates.
4. Use cluster-robust standard errors by architecture family.

**Expected outcome:** Negative partial correlation between absorption and downstream performance, supporting the claim that absorption has unique practical harm.

### Baselines
- **Mainstream method**: Standard SAE and TopK SAE, properly tuned (no strawman).
- **Mitigation methods**: OrtSAE, Matryoshka SAE, JumpReLU, masked regularization, ATM---all evaluated on the same metric suite.

### Experimental Plan & Time Budget
- **Pilot (e1_pilot_v2)**: 5--10 checkpoints, fixed metrics, >=50k tokens. Target: 15--20 minutes.
- **Full Pareto evaluation (e1_full_gpt2)**: 20--30 checkpoints per model family. Target: <=1 hour per independent batch (parallelizable across GPUs).
- **Meta-analysis (e2_meta)**: Purely computational on SAEBench data. Target: 20--30 minutes.

### Risk Assessment
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OrtSAE/Matryoshka actually dominate the Pareto front | Medium | The paper still provides the first rigorous multi-objective validation---a valuable contribution, though less contrarian. |
| SAEBench metrics are too noisy for clean Pareto analysis | Medium | Use existing, well-tested implementations; average across layers/checkpoints; report confidence intervals; be transparent about noise. |
| Downstream correlation is confounded by unobserved architecture differences | Medium | Include architecture dummies, width, and L0 as controls; use cluster-robust SEs; explicitly discuss causal limitations. |
| Proper `sae-spelling` integration is technically blocked | Low | The library is open-source and well-documented; if issues arise, use SAEBench's absorption metric adaptation. |

### Novelty Claim
No prior work has conducted a **systematic multi-objective Pareto evaluation of absorption-mitigation methods** across the full suite of SAE quality metrics using existing checkpoints in a training-free setting. OrtSAE and Matryoshka papers compare architectures on multiple metrics but do not frame this as a Pareto analysis showing no architecture dominates. Switch SAEs, Gated SAEs, and HierarchicalTopK study Pareto improvements in the MSE--L0 tradeoff specifically, not the absorption--hedging--reconstruction impossibility triangle.

### What This Perspective Contributed
The contrarian provided the core thesis: **absorption is not a fixable bug but an intrinsic tradeoff**, and current evaluations are selectively reported on a single objective. The multi-objective evaluation design is the backbone of the front-runner proposal. The contrarian's emphasis on devil's-advocate skepticism prevented the team from accepting "65% absorption reduction" headlines at face value and instead demanded evidence of *overall* improvement.

---

## References

- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507.
- Chanin, D., et al. (2025). *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*. arXiv:2505.11756.
- Karvonen, A., et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033.
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
- Korznikov, A., et al. (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* arXiv:2602.14111.
- Barin Pacela, V., et al. (2026). *Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation*. arXiv:2603.28744.
- Roy, S., et al. (2026). *Fundamental Limits of Neural Network Sparsification: Evidence from Catastrophic Interpretability Collapse*. arXiv:2603.18056.
- Kantamneni, S., et al. (2025). *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. arXiv:2502.16681.
- Leask, P., et al. (2025). *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. arXiv:2502.04878.
- Gulko, L., et al. (2025). *CE-Bench: Towards a Reliable Contrastive Evaluation of Sparse Autoencoders*. BlackboxNLP 2025.
- Ayonrinde, K., Pearce, M. T., & Sharkey, L. (2024). *Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs*. arXiv:2410.11179.
