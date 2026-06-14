# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

#### 1. Assumption: "Feature absorption is a critical problem that undermines SAE interpretability"
**Evidence challenging it:**
- **Google DeepMind MI Team (Smith et al., 2025)**: Explicitly deprioritized SAE research after finding "negative results for sparse autoencoders on downstream tasks" -- suggesting absorption may be a symptom of a deeper issue (SAEs simply don't work well), not the root cause.
- **"Sanity Checks for Sparse Autoencoders" (Galichin et al., 2025, arXiv:2602.14111)**: Random baselines achieve 0.87 interpretability vs 0.90 for trained SAEs, 0.69 sparse probing vs 0.72, and 0.73 causal editing vs 0.72. If random features are almost as "interpretable" and "causal" as trained ones, what exactly is absorption degrading?
- **"Sparse Autoencoders Can Interpret Randomly Initialized Transformers" (Heap et al., 2025, arXiv:2501.17727)**: SAEs achieve similar auto-interpretability scores on random vs trained models. If random models have no "learned computations" to absorb, absorption in trained models may be a red herring -- the SAE is finding statistical patterns in data sparsity, not missing model features.
- **"Are Sparse Autoencoders Useful?" (Kantamneni et al., ICML 2025, arXiv:2502.16681)**: SAEs do not consistently improve downstream probing tasks vs strong baselines. If absorption is so harmful, why can't we detect its impact on practical tasks?

**What this challenges:** The field may be treating absorption as a first-class problem when it is actually a second-order symptom of SAEs' fundamental inability to recover meaningful features.

#### 2. Assumption: "Absorption is caused by sparsity loss + hierarchical co-occurrence"
**Evidence challenging it:**
- **"Decomposing The Dark Matter of Sparse Autoencoders" (Engels et al., TMLR 2024, arXiv:2410.14670)**: ~50% of SAE error is linearly predictable from inputs, and >90% of error norm is linearly predictable. This "linear dark matter" suggests SAEs systematically miss linearly-representable features -- the problem is not just sparsity loss biasing toward specific features, but fundamental capacity/optimization limitations.
- **"Sparse Autoencoders Do Not Find Canonical Units of Analysis" (Leask et al., ICLR 2025, arXiv:2502.04878)**: SAEs are nonidentifiable -- multiple equally valid decompositions exist. If there's no canonical decomposition, "absorption" may be an artifact of comparing against a non-existent ground truth.
- **"Sanity Checks" (Galichin et al., 2025)**: In synthetic experiments with known ground-truth features, SAEs achieved 71% explained variance but recovered only 9% of true features. The reconstruction-sparsity tradeoff is not the issue; SAEs fail even when the statistical model is favorable.

**What this challenges:** The standard causal story (sparsity loss -> absorption) may be incomplete. Absorption could be an inevitable consequence of SAEs' inability to recover the true feature structure, not a tunable artifact of the loss function.

#### 3. Assumption: "Matryoshka SAEs solve absorption (90% reduction)"
**Evidence challenging it:**
- **"Feature Hedging" (Chanin et al., 2025, arXiv:2505.11756)**: Matryoshka SAEs trade absorption for hedging in inner levels. The ~90% absorption reduction is achieved by shifting the problem, not solving it. This is not a solution -- it's a Pareto shift.
- **"Sanity Checks" (Galichin et al., 2025)**: Even "solved" architectures may not beat random baselines on meaningful metrics. The absorption metric itself may be misleading.
- **Alignment Forum discussion on Matryoshka SAEs**: "feature absorption itself is an effective strategy for reducing the L0 at a fixed FVU" -- absorption may be an optimal compression strategy under the constraints SAEs face, not a bug.

**What this challenges:** The celebrated Matryoshka result may be measuring the wrong thing. Reducing absorption while introducing hedging and increasing compute is not clearly a win.

#### 4. Assumption: "Absorption creates interpretability illusions with arbitrary false negatives"
**Evidence challenging it:**
- **"Revising and Falsifying Sparse Autoencoder Feature Explanations" (Ma et al., NeurIPS 2025)**: Current explanation methods have systematic recall bias -- they produce overly broad explanations that ignore polysemanticity. The "false negatives" of absorption may be better understood as the SAE correctly representing polysemantic structure that our explanation methods fail to capture.
- **"SAEs trained on the same data don't learn the same features" (EleutherAI blog, 2024)**: Different seeds share only ~53% of features. If features are this unstable, claims about specific absorption cases may not generalize across training runs.
- **"Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845, 2025)**: SAEs perform worse than linear probes on OOD tasks; zero-ablating features is ineffective; multi-feature interventions produce unwanted side effects. The practical harm of absorption may be dwarfed by other SAE failure modes.

**What this challenges:** The interpretability illusion may not be caused by absorption per se, but by our flawed methods for evaluating what SAE features "mean."

#### 5. Assumption: "SAE features reflect learned model computations"
**Evidence challenging it:**
- **"Sparse Autoencoders Can Interpret Randomly Initialized Transformers" (Heap et al., 2025)**: The most damning evidence. If SAEs find equally interpretable features in untrained models, then absorption in trained models cannot be about "missing learned features" -- there are no learned features in random models. The features SAEs find reflect data statistics and architecture, not training.
- **"Do Sparse Autoencoders Generalize?" (Heindrich et al., 2025, arXiv:2502.19964)**: SAE features for "answerability" generalize inconsistently across domains. Some datasets show near-random performance. If features captured genuine model computations, they should generalize better.
- **"The Reproducibility Problem in Mechanistic Interpretability Studies" (ResearchGate, 2025)**: Documented cases where neuron interpretations failed to replicate across identical architectures. SAE-based interpretations may suffer the same fate.

**What this challenges:** The entire framing of absorption as "missing features" assumes SAEs discover real features in the first place. If they don't, absorption is a non-problem.

#### 6. Assumption: "Cross-architecture absorption comparison is a meaningful research direction"
**Evidence challenging it:**
- **"Sanity Checks" (Galichin et al., 2025)**: If all architectures fail to beat random baselines, comparing their absorption rates is comparing different flavors of failure. The CAAB (Cross-Architecture Absorption Benchmark) proposed in the current project may be benchmarking a symptom, not the disease.
- **"Are Sparse Autoencoders Useful?" (Kantamneni et al., ICML 2025)**: Even when SAEs "work" on individual datasets, ensemble methods combining SAEs with baselines do NOT consistently outperform ensembles using only baselines. The marginal value of SAE-specific research (including absorption) is questionable.

**What this challenges:** The proposed CAAB contribution may be well-executed but answering the wrong question.

### Landscape of Doubt

The emerging critical literature on SAEs reveals a pattern: **absorption is a visible symptom of deeper dysfunction, not an independent problem to solve.** The key cracks are:

1. **Random baseline parity**: SAEs barely beat random features on standard metrics. If random features are "absorbed" too (they must be, since they activate similarly), absorption cannot be the core issue.

2. **Random model parity**: SAEs interpret untrained models. Absorption in trained models cannot reflect missing learned computations if untrained models have none.

3. **Nonidentifiability**: There is no unique ground-truth decomposition. "Absorption" presupposes a canonical feature set that SAEs should recover -- but Leask et al. proved this doesn't exist.

4. **Downstream irrelevance**: Even when absorption is present, its causal impact on practical tasks is weak or undetectable. The field may be optimizing a metric with no external validity.

5. **Explanation bias**: The "interpretability illusions" attributed to absorption may be artifacts of flawed explanation methods, not SAE architecture.

---

## Phase 2: Initial Candidates

### Candidate A: "Absorption is a Symptom, Not a Disease: Rethinking the SAE Feature Recovery Paradigm"

- **Challenged assumption**: Feature absorption is a first-class problem caused by sparsity loss that can be mitigated by architectural changes.
- **Evidence against it**: (1) Random baselines match SAEs on interpretability and causal metrics; (2) SAEs interpret random models; (3) Nonidentifiability means no canonical features exist to be "absorbed"; (4) DeepMind deprioritized SAEs due to downstream failures, not absorption.
- **Contrarian hypothesis**: Absorption is an inevitable consequence of SAEs' inability to recover true features, not a tunable artifact. Matryoshka "solutions" shift the problem to hedging. The field should stop treating absorption as independent and instead question whether SAEs can recover meaningful features at all.
- **Exploitation plan**: Design experiments that directly test whether absorption metrics correlate with any ground-truth feature recovery. Use synthetic data where true features are known. Show that even "low-absorption" architectures recover <10% of true features (replicating Galichin et al.). Publish as "Rethinking Feature Absorption: Evidence from Ground-Truth Feature Recovery."
- **Novelty estimate**: 8/10 -- Directly challenges the framing of the entire absorption research line.

### Candidate B: "The Absorption-Hedging Trade-off is a False Dichotomy: Both are Symptoms of SAE Nonidentifiability"

- **Challenged assumption**: Absorption and hedging are complementary problems on a Pareto frontier that can be balanced via architecture (Matryoshka loss coefficients).
- **Evidence against it**: (1) Leask et al. showed SAEs don't find canonical units -- there is no ground truth to be "absorbed" or "hedged"; (2) Random baselines exhibit similar patterns; (3) The trade-off may be an artifact of evaluation metrics, not a real phenomenon.
- **Contrarian hypothesis**: Both absorption and hedging are epiphenomena of SAE nonidentifiability. Different architectures produce different arbitrary decompositions; calling one "absorption" and another "hedging" imposes a narrative on fundamentally uninterpretable variation.
- **Exploitation plan**: Train multiple SAEs (same architecture, different seeds) on the same data. Show that what one run labels "absorption" another labels "hedging" for the same concept. Demonstrate that the absorption-hedging classification is not stable across random initializations. Publish as "Unstable Boundaries: Absorption, Hedging, and the Illusion of Complementary SAE Failure Modes."
- **Novelty estimate**: 7/10 -- Challenges the most celebrated 2025 insight in SAE research.

### Candidate C: "Feature Absorption is Optimal Compression: Why SAEs Should Absorb"

- **Challenged assumption**: Absorption is always harmful and should be minimized.
- **Evidence against it**: (1) Alignment Forum notes absorption "is an effective strategy for reducing the L0 at a fixed FVU"; (2) Random baselines achieve comparable performance with or without absorption-like patterns; (3) Downstream tasks show weak correlation with absorption rates.
- **Contrarian hypothesis**: Under the constraints SAEs face (fixed dictionary size, sparsity target), absorption is an optimal information compression strategy. The "parent" features that get absorbed are statistically redundant given the "child" features. SAEs are correctly discarding redundant information.
- **Exploitation plan**: Use information theory to formalize absorption as conditional redundancy. Show that absorbed parent features are statistically predictable from child features. Demonstrate that "de-absorbed" SAEs (e.g., via post-hoc compensation) have worse compression efficiency with no downstream benefit. Publish as "Absorption as Optimal Compression: An Information-Theoretic Reinterpretation."
- **Novelty estimate**: 6/10 -- Provocative but may be seen as too contrarian; risk of being dismissed as a "gotcha" paper.

---

## Phase 3: Self-Critique

### Against Candidate A: "Absorption is a Symptom, Not a Disease"

- **Steelman**: Chanin et al.'s absorption metric is well-defined and reproducible. The first-letter feature test cases are concrete. Matryoshka SAEs do reduce absorption on these test cases. Even if SAEs have deeper problems, absorption is a real, measurable phenomenon that affects specific use cases (e.g., sparse circuit analysis where missing parent features break completeness).
- **Cherry-picking check**: Am I selectively citing Galichin et al.'s synthetic experiment (9% feature recovery) while ignoring that it used independent features with no correlations? Real features in LLMs are correlated, making the problem harder but also making absorption more likely to be a genuine issue.
- **Confounding check**: The random baseline results could mean SAE evaluation metrics are flawed, not that absorption is unimportant. Absorption could still harm real applications even if metrics don't capture it.
- **Actionability check**: If absorption is just a symptom, what should researchers do instead? The paper would need to propose an alternative research direction (e.g., abandoning SAEs for transcoders, or focusing on feature consistency rather than absorption). Simply saying "absorption doesn't matter" is not constructive.
- **Verdict**: MODERATE -- The argument is strong but needs a constructive alternative proposal to avoid being a "gotcha" paper.

### Against Candidate B: "The Absorption-Hedging Trade-off is a False Dichotomy"

- **Steelman**: Chanin et al. (2025) provided extensive empirical evidence for the trade-off. Matryoshka SAEs measurably reduce absorption while increasing hedging. The trade-off is not just theoretical -- it's observable in standard metrics.
- **Cherry-picking check**: Leask et al.'s nonidentifiability result is from stitching experiments on specific models. It doesn't directly address whether absorption and hedging are real phenomena or artifacts.
- **Confounding check**: Even if SAEs are nonidentifiable, absorption and hedging could still be real properties of specific decompositions. Nonidentifiability means there's no unique solution, not that all solutions are equivalent.
- **Actionability check**: If the trade-off is false, what explains the empirical pattern? The paper would need an alternative theory for why Matryoshka reduces absorption and increases hedging.
- **Verdict**: WEAK -- The nonidentifiability argument doesn't directly falsify the absorption-hedging trade-off. The trade-off is empirically well-supported.

### Against Candidate C: "Feature Absorption is Optimal Compression"

- **Steelman**: Absorption does reduce L0 at fixed FVU, which is a genuine optimization benefit. If the goal is sparse reconstruction, absorption helps. The Alignment Forum post explicitly notes this.
- **Cherry-picking check**: Am I ignoring cases where absorption clearly causes harm? Chanin et al. showed that absorption creates unreliable classifiers for high-stakes applications (bias detection, deceptive behavior detection).
- **Confounding check**: Even if absorption is optimal for compression, it may be suboptimal for interpretability. The field values interpretability, not just compression.
- **Actionability check**: The paper would need to show that absorption doesn't harm interpretability -- but Chanin et al.'s test cases show it does. This is hard to refute.
- **Verdict**: WEAK -- The evidence for absorption causing concrete harm is too strong. This position is likely wrong.

---

## Phase 4: Refinement

### Dropped
- **Candidate C** (absorption as optimal compression) -- The evidence that absorption causes concrete interpretability failures (Chanin et al.'s test cases) is too strong to dismiss. This position is likely wrong.
- **Candidate B** (false dichotomy) -- The absorption-hedging trade-off has strong empirical support. Nonidentifiability doesn't falsify it.

### Strengthened: Candidate A

**Refined position**: Absorption is a real phenomenon, but its importance has been systematically overstated relative to deeper SAE limitations. The field's focus on absorption distracts from more fundamental problems:

1. **SAEs recover <10% of true features even in favorable synthetic settings** (Galichin et al.)
2. **SAEs interpret random models** (Heap et al.) -- the features don't reflect learned computations
3. **Random baselines match SAEs on standard metrics** (Galichin et al.)
4. **Downstream tasks show no consistent benefit from SAEs** (Kantamneni et al., Smith et al.)

The refined contribution is not "absorption doesn't matter" but rather: **absorption research should be reframed as part of a broader inquiry into whether SAEs can recover meaningful features at all.** The current project (CAAB + causal evaluation) is valuable, but its framing should be adjusted.

**Additional corroboration found**:
- "Do Sparse Autoencoders Generalize?" (Heindrich et al., 2025): SAE features generalize inconsistently -- some datasets near-random. This undermines the assumption that SAE features capture stable, meaningful structure.
- "Decomposing The Dark Matter" (Engels et al., 2024): The linear predictability of SAE error suggests the problem is not absorption but systematic failure to learn linearly-representable features.

### Selected Front-Runner: Candidate A (Refined)

**Title**: "Rethinking Feature Absorption: Evidence that SAEs Fail to Recover Ground-Truth Features"

**Core thesis**: Feature absorption is a measurable phenomenon, but it is a symptom of SAEs' fundamental inability to recover true features, not an independent problem. The field should shift from absorption-specific mitigation to ground-truth-validated feature recovery.

---

## Phase 5: Final Proposal

### Title
"Rethinking Feature Absorption: Ground-Truth Evidence that SAEs Systematically Fail to Recover True Features"

### Challenged Assumption
The mainstream view (Chanin et al., 2024; Bussmann et al., 2025; Chanin et al., 2025) treats feature absorption as a first-class problem caused by sparsity loss and hierarchical co-occurrence, amenable to architectural solutions like Matryoshka SAEs. We challenge this by arguing that absorption is a symptom of a deeper dysfunction: SAEs recover <10% of true features even in controlled settings, interpret random models, and are matched by random baselines on standard metrics.

### Evidence

**For the mainstream view:**
- Chanin et al. (2024): Absorption is reproducible, measurable, and occurs in every LLM SAE tested.
- Matryoshka SAEs reduce absorption by ~90% with measurable metric improvements.
- Absorption creates concrete interpretability illusions (unreliable classifiers, false negatives).

**Against the mainstream view:**
- Galichin et al. (2025): SAEs recover only 9% of true features in synthetic experiments despite 71% explained variance. Random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72).
- Heap et al. (2025): SAEs achieve similar auto-interpretability scores on randomly initialized transformers. If random models have no learned features to absorb, absorption cannot reflect missing learned computations.
- Leask et al. (2025): SAEs do not find canonical units of analysis. Nonidentifiability means there is no unique ground truth for features to be "absorbed" from.
- Engels et al. (2024): ~50% of SAE error is linearly predictable, indicating systematic failure to learn linearly-representable features -- a problem deeper than absorption.
- Kantamneni et al. (2025): SAEs do not consistently improve downstream tasks vs baselines. If absorption is so harmful, its absence should improve tasks -- but it doesn't.
- Smith et al. (2025, DeepMind): Negative downstream results led to deprioritization of SAE research.

### Hypothesis
Feature absorption is a real but secondary phenomenon. The primary issue is that SAEs fail to recover meaningful, ground-truth features. Absorption-specific research (Matryoshka, OrtSAE, etc.) treats a symptom while the disease -- SAEs' inability to faithfully decompose model representations -- remains unaddressed.

### Method

**Experiment 1: Ground-Truth Feature Recovery (synthetic)**
- Replicate Galichin et al.'s synthetic experiment with known ground-truth features.
- Vary feature correlation structure (independent, hierarchical, correlated).
- Measure: (a) explained variance, (b) true feature recovery rate, (c) absorption rate.
- Prediction: Even "low-absorption" configurations recover <20% of true features.
- Time: ~30 minutes (small synthetic model).

**Experiment 2: Random Model Control**
- Train SAEs on randomly initialized GPT-2 Small (no training).
- Measure absorption rates using Chanin et al.'s metric.
- Prediction: Random models show comparable absorption patterns to trained models.
- Time: ~20 minutes.

**Experiment 3: Random Baseline Comparison**
- Implement Galichin et al.'s frozen decoder baseline.
- Compare absorption rates, interpretability scores, and sparse probing accuracy.
- Prediction: Random baselines show absorption-like patterns with comparable metrics.
- Time: ~20 minutes.

**Experiment 4: Absorption-Downstream Correlation (reanalysis)**
- Use existing CAAB data (if available) or train 2-3 SAEs with varying absorption rates.
- Measure downstream task performance (sparse probing, steering).
- Control for reconstruction quality and sparsity.
- Prediction: Partial correlation between absorption and downstream performance is weak (R^2 < 0.1).
- Time: ~30 minutes.

### Experimental Plan Summary
| Experiment | Purpose | Time | Priority |
|------------|---------|------|----------|
| E1: Synthetic recovery | Test ground-truth recovery | 30 min | HIGH |
| E2: Random model control | Test if absorption exists without learning | 20 min | HIGH |
| E3: Random baseline comparison | Test if random features show absorption | 20 min | HIGH |
| E4: Downstream correlation | Test if absorption causally harms tasks | 30 min | MEDIUM |

Total: ~1.7 hours (within project constraints).

### Baselines
- **Mainstream SAE**: Standard JumpReLU or TopK SAE (the "absorbed" baseline).
- **Matryoshka SAE**: The "low-absorption" baseline.
- **Random baseline**: Frozen decoder (Galichin et al.) -- the null control.
- **Random model**: Untrained transformer -- the "no learning" control.

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Synthetic experiments don't replicate Galichin et al. | Low | High | Use their published code; vary parameters |
| Random model shows no absorption | Medium | High | This would actually support the mainstream view; paper pivots to "absorption requires learning" |
| Downstream correlation is strong | Low | High | Would falsify our hypothesis; report honestly and discuss boundary conditions |
| Reviewers dismiss as "gotcha" paper | Medium | Medium | Frame constructively with actionable recommendations |

### Novelty Claim
This paper would be the first to systematically connect the absorption literature with the emerging critical literature on SAE fundamentals (random baselines, random models, nonidentifiability, dark matter). It reframes absorption research from "how do we fix absorption?" to "does absorption matter if SAEs don't recover true features?" -- a question with profound implications for the entire mechanistic interpretability research program.

### Constructive Recommendations
If the hypothesis is supported, we recommend:
1. **Mandatory random baselines**: All SAE papers should include frozen/random baselines.
2. **Ground-truth validation**: SAE research should prioritize synthetic experiments with known features.
3. **Focus shift**: From absorption-specific metrics to feature recovery rates on ground-truth tasks.
4. **Alternative paradigms**: Greater investment in transcoders, crosscoders, and non-SAE interpretability methods.

---

## Sources

1. [Galichin et al., "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?", arXiv:2602.14111, 2025](https://arxiv.org/abs/2602.14111)
2. [Heap et al., "Sparse Autoencoders Can Interpret Randomly Initialized Transformers", arXiv:2501.17727, 2025](https://arxiv.org/abs/2501.17727)
3. [Leask et al., "Sparse Autoencoders Do Not Find Canonical Units of Analysis", ICLR 2025, arXiv:2502.04878](https://arxiv.org/abs/2502.04878)
4. [Engels et al., "Decomposing The Dark Matter of Sparse Autoencoders", TMLR 2024, arXiv:2410.14670](https://arxiv.org/abs/2410.14670)
5. [Kantamneni et al., "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing", ICML 2025, arXiv:2502.16681](https://arxiv.org/abs/2502.16681)
6. [Heindrich et al., "Do Sparse Autoencoders Generalize? A Case Study of Answerability", arXiv:2502.19964, 2025](https://arxiv.org/abs/2502.19964)
7. [Chanin et al., "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders", arXiv:2409.14507, 2024](https://arxiv.org/abs/2409.14507)
8. [Chanin et al., "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders", arXiv:2505.11756, 2025](https://arxiv.org/abs/2505.11756)
9. [Bussmann et al., "Learning Multi-Level Features with Matryoshka Sparse Autoencoders", arXiv:2503.17547, 2025](https://arxiv.org/abs/2503.17547)
10. [Ma et al., "Revising and Falsifying Sparse Autoencoder Feature Explanations", NeurIPS 2025](https://openreview.net/forum?id=OJAW2mHVND)
11. [Smith et al., Google DeepMind MI Team, "Negative Results for Sparse Autoencoders on Downstream Tasks", 2025 (cited in Galichin et al.)](https://arxiv.org/abs/2602.14111)
12. ["The Reproducibility Problem in Mechanistic Interpretability Studies", ResearchGate, 2025](https://www.researchgate.net/publication/392626589_The_Reproducibility_Problem_in_Mechanistic_Interpretability_Studies)
13. ["Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts", arXiv:2506.23845, 2025](https://arxiv.org/abs/2506.23845)
14. [EleutherAI Blog, "SAEs trained on the same data don't learn the same features", 2024](https://blog.eleuther.ai/sae_seed_similarity/)
15. [Alignment Forum, "Matryoshka Sparse Autoencoders", 2024](https://www.alignmentforum.org/posts/zbebxYCqsryPALh8C/matryoshka-sparse-autoencoders)
