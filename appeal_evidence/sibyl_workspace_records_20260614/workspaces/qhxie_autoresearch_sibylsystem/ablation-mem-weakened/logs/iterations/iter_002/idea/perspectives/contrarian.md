# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

#### 1. **Assumption**: Feature absorption is a real, meaningful pathology that undermines SAE interpretability
- **Evidence challenging it**:
  - [Korznikov et al., 2026. "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111](https://arxiv.org/abs/2602.14111): Frozen/random SAE baselines match trained SAEs on AutoInterp (0.87 vs 0.90), Sparse Probing (0.69 vs 0.72), and Causal Editing (0.73 vs 0.72). In synthetic experiments with known ground-truth features, SAEs achieve 71% explained variance but recover only 9% of true features. This suggests that absorption metrics may measure artifacts of massive dictionary sizes rather than genuine interpretability failures.
  - [Heap et al., 2025](https://arxiv.org/abs/2602.14111): SAEs recover "interpretable" features even in randomly initialized transformers with no learned structure whatsoever, proving that SAEs can hallucinate meaningful-looking features from noise.
  - [Paulo & Belrose, 2025. "Sparse Autoencoders Trained on the Same Data Learn Different Features" arXiv:2501.16615](https://arxiv.org/abs/2501.16615): Independently trained SAEs learn substantially different features, suggesting the optimization landscape is underconstrained and feature assignments are arbitrary.
- **What this means for absorption**: If SAE features are largely arbitrary, then "absorption" may be a misnomer for a more fundamental problem: SAEs do not discover ground-truth features at all, so measuring whether parent features are "absorbed" presupposes that parent features exist in the SAE representation in the first place.

#### 2. **Assumption**: Reducing absorption improves downstream interpretability utility
- **Evidence challenging it**:
  - [Wang et al., ICLR 2026. "Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders" arXiv:2510.03659](https://arxiv.org/abs/2510.03659): Weak correlation (tau_b ~ 0.298) between interpretability and steering utility. After feature selection, the correlation vanishes (tau_b ~ 0) or becomes negative. This directly challenges the implicit assumption that absorption reduction (an interpretability metric improvement) translates to better steering/probing (utility).
  - [Kantamneni et al., ICML 2025. "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing"](https://proceedings.mlr.press/v267/kantamneni25a.html): SAEs do not consistently outperform strong non-SAE baselines on downstream probing tasks.
  - [Wu et al., ICML 2025. "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders" arXiv:2501.17148](https://arxiv.org/abs/2501.17148): Prompting and fine-tuning outperform SAE-based steering; simple difference-in-means baselines beat SAEs on concept detection.
- **What this means**: The entire premise of studying absorption---that it harms practical interpretability---may be unfounded. Even if we perfectly eliminated absorption, SAEs might still underperform simple baselines on downstream tasks.

#### 3. **Assumption**: The superposition hypothesis justifies SAEs as the right tool for interpretability
- **Evidence challenging it**:
  - [Elhage et al., 2022. "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html): The foundational theory is based on toy models with simplified feature structures. Real LLM features have complex correlation structures that are not well-captured.
  - [Cui et al., 2025. "On the Limits of Sparse Autoencoders" arXiv:2506.15963](https://arxiv.org/abs/2506.15963): SAEs provably recover ground-truth features only under extreme sparsity conditions that real LLMs do not satisfy.
  - [Tang et al., 2025. "A Unified Theory of Sparse Dictionary Learning" arXiv:2512.05534](https://arxiv.org/abs/2512.05534): Hierarchical structures necessarily induce feature absorption as spurious local minima---this is not a bug to fix but a fundamental mathematical property of the optimization landscape.
- **What this means**: SAEs may be the wrong tool for the job. If superposition in real LLMs differs fundamentally from toy models, and if hierarchical structures necessarily cause absorption, then studying absorption is like studying how water gets wet.

#### 4. **Assumption**: The Chanin et al. absorption metric accurately measures a real phenomenon
- **Evidence challenging it**:
  - [Chanin et al., 2024. "A is for Absorption" arXiv:2409.14507](https://arxiv.org/abs/2409.14507): The authors themselves acknowledge the metric "is not perfect, and is likely an under-estimate of the true level of feature absorption." It cannot apply past layer 17 in Gemma 2 2B due to causal ablation limitations.
  - [SAEBench, Karvonen et al., ICML 2025. arXiv:2503.09532](https://arxiv.org/abs/2503.09532): The benchmark extends the metric but replaces causal ablation with logistic regression probes, trading causal certainty for broader applicability. This suggests the original metric's causal claims were too strong.
  - [Feature Hedging, Chanin et al., 2025. arXiv:2505.11756](https://arxiv.org/abs/2505.11756): Identifies hedging as a distinct failure mode that the absorption metric does not capture, implying the metric is incomplete even within its own framework.
- **What this means**: The foundational metric of the field is acknowledged to be imperfect, limited in scope, and unable to capture the full range of sparsity-induced failures. Building a research program on quantifying this metric may be building on sand.

#### 5. **Assumption**: Training-free analysis of pretrained SAEs (the project's methodology) can yield meaningful insights about absorption
- **Evidence challenging it**:
  - [Paulo et al., 2025. "Transcoders Beat Sparse Autoencoders for Interpretability" arXiv:2501.18823](https://arxiv.org/abs/2501.18823): Transcoders achieve Pareto dominance over SAEs on interpretability metrics. If transcoders are superior, training-free analysis of SAEs may be analyzing a deprecated technology.
  - [Smith et al., 2025. "Negative Results for SAEs On Downstream Tasks"](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9): Google DeepMind's mech interp team officially deprioritized SAE research after finding they fail to outperform simple baselines.
  - [Interpretability Illusions, arXiv:2505.16004](https://arxiv.org/abs/2505.16004): SAE interpretations are vulnerable to minimal input perturbations even when the LLM remains semantically stable. This suggests SAE features are not robust enough to support reliable training-free analysis.
- **What this means**: The project's training-free constraint may preclude the most interesting questions. If SAEs are fundamentally limited, analyzing their pretrained weights is like doing astronomy with a broken telescope.

#### 6. **Assumption**: The field's negative results are exceptions rather than the rule
- **Evidence challenging it**:
  - The 2024-2025 period saw a cascade of negative results: Kantamneni et al. (probing failures), Wu et al. (steering failures), Wang et al. (interpretability-utility disconnect), Korznikov et al. (random baselines match trained SAEs), Smith et al. (DeepMind deprioritization), Heap et al. (spurious features in random networks).
  - ["Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" arXiv:2506.23845](https://arxiv.org/abs/2506.23845): A position paper explicitly repositioning SAEs from action to discovery, acknowledging the negative results.
  - ["Self-Ablating Transformers" arXiv:2505.00509](https://arxiv.org/abs/2505.00509): Proposes an alternative paradigm (built-in interpretability) that challenges the need for external SAE decomposition entirely.
- **What this means**: The negative results are not isolated failures but a systematic pattern. The contrarian view is that the field is in a credibility crisis, and absorption research risks being collateral damage.

### Landscape of Doubt

The SAE field in 2025-2026 resembles the "replication crisis" in psychology: a foundational tool with enthusiastic adoption, followed by systematic negative results that challenge core assumptions. The key cracks are:

1. **Metric validity crisis**: The absorption metric, AutoInterp scores, and L0/Loss Recovered all correlate weakly with practical utility (Wang et al., tau_b ~ 0.3). Random baselines match trained SAEs on these metrics.
2. **Feature reality crisis**: SAEs find "features" in random networks (Heap et al.) and learn different features from the same data (Paulo & Belrose). This undermines the assumption that SAE features correspond to real, stable model-internal structures.
3. **Utility crisis**: SAEs underperform simple baselines on steering, probing, and concept detection---the very tasks they were designed for.
4. **Theoretical crisis**: The superposition hypothesis, while elegant in toy models, lacks validation in real LLMs. The conditions under which SAEs provably recover features (Cui et al.) are not met in practice.

The mainstream response has been defensive repositioning ("SAEs are for discovery, not action") and incremental architectural improvements (Matryoshka, OrtSAE, ATM). The contrarian view is that these responses miss the deeper problem: the entire SAE paradigm may be flawed, and absorption is a symptom, not the disease.

---

## Phase 2: Initial Candidates

### Candidate A: "The Absorption Mirage: Feature Absorption Is an Artifact of Dictionary Size, Not a Real Pathology"

- **Challenged assumption**: Feature absorption is a genuine interpretability failure that harms SAE reliability.
- **Evidence against it**:
  - Korznikov et al. (2026) show random/frozen SAE baselines match trained SAEs on interpretability metrics, suggesting these metrics measure dictionary size artifacts rather than learned features.
  - Heap et al. (2025) show SAEs find "interpretable" features in random networks, proving the SAE training process can hallucinate structure.
  - In synthetic experiments (Korznikov et al.), SAEs achieve 71% explained variance but recover only 9% of true features---high reconstruction does not imply meaningful feature recovery.
- **Contrarian hypothesis**: Feature absorption is not a pathology to fix but an expected statistical artifact of overcomplete dictionary learning. When you have 73K latents for 2K dimensions, some latents will inevitably appear to "absorb" parent features due to random correlations in the decoder matrix. The Chanin metric detects these correlations, not causal absorption.
- **Exploitation plan**: Design experiments that explicitly test whether absorption-like patterns appear in random/frozen SAEs. If random SAEs show comparable "absorption rates" to trained SAEs, this falsifies the claim that absorption is a training-induced pathology. Publish as a "sanity check" paper that reframes absorption research.
- **Novelty estimate**: 7/10 --- Directly challenges the foundational premise of the entire absorption subfield.

### Candidate B: "The Utility Paradox: Reducing Absorption Makes SAEs Worse at Downstream Tasks"

- **Challenged assumption**: Reducing absorption improves SAE interpretability and utility.
- **Evidence against it**:
  - Wang et al. (ICLR 2026) show interpretability-utility correlation is weak (~0.3) and vanishes after feature selection.
  - Matryoshka SAEs reduce absorption from 0.49 to 0.05 but introduce hedging (Chanin et al., 2025) and have reconstruction fidelity costs.
  - OrtSAE reduces absorption by 65% but achieves "slightly lower explained variance" (Korznikov et al., 2025).
- **Contrarian hypothesis**: Absorption is a side effect of the SAE's reconstruction objective. Reducing absorption requires changing the objective (e.g., via orthogonality constraints or hierarchical penalties), which degrades reconstruction and thus downstream utility. The trade-off is not absorption vs. interpretability but absorption vs. reconstruction fidelity, and reconstruction fidelity is what drives utility.
- **Exploitation plan**: Systematically compare absorption-reduced SAEs (Matryoshka, OrtSAE, ATM) against standard SAEs on downstream tasks (steering, probing, circuit discovery) while controlling for reconstruction fidelity. If absorption-reduced SAEs underperform despite better absorption metrics, this demonstrates the paradox. The paper would be the first to explicitly test whether absorption reduction helps or hurts in practice.
- **Novelty estimate**: 8/10 --- Turns the field's implicit assumption on its head with rigorous controlled experiments.

### Candidate C: "Absorption Is Inevitable: The Mathematical Impossibility of Sparsity Without Absorption in Hierarchical Feature Spaces"

- **Challenged assumption**: Absorption can be eliminated or significantly reduced through better SAE architectures or training objectives.
- **Evidence against it**:
  - Tang et al. (2025, Theorem 3.8) prove that hierarchical structures necessarily induce feature absorption as spurious local minima in the SAE optimization landscape.
  - Cui et al. (2025) prove SAEs recover ground-truth features only under extreme sparsity (S -> 1), which is incompatible with the moderate sparsity levels used in practice.
  - Chanin et al. (2024) prove absorption is a "logical consequence of sparsity loss under hierarchical features."
- **Contrarian hypothesis**: Absorption is not a bug to fix but a fundamental mathematical property of sparse dictionary learning in hierarchical feature spaces. All proposed solutions (Matryoshka, OrtSAE, ATM, Balance Matryoshka) merely shift the problem to other failure modes (hedging, reconstruction loss, orthogonality constraints) without addressing the root cause. The field should stop trying to eliminate absorption and instead develop interpretability methods that work *with* absorption.
- **Exploitation plan**: Formalize the impossibility result by proving that any SAE with sparsity constraint L0 < k on a hierarchical feature space with n features must have absorption rate >= f(k, n) > 0. Then survey all proposed solutions and show they each violate at least one desirable property (reconstruction fidelity, computational efficiency, feature completeness). Propose a new framework: "absorption-aware interpretability" that treats absorbed features as distributed representations rather than failures.
- **Novelty estimate**: 9/10 --- A genuine paradigm shift from "fix absorption" to "work with absorption."

---

## Phase 3: Self-Critique

### Against Candidate A (The Absorption Mirage)

- **Steelman**: Chanin et al. validated absorption across hundreds of SAEs with a causal ablation metric. The metric specifically checks that (a) a single latent has a large ablation effect, and (b) the "main" latents for a feature do not activate at all. This is not a correlation-based measure---it directly tests causal mediation. The phenomenon is real and reproducible.
- **Cherry-picking check**: Korznikov et al.'s random baseline results are on a limited set of metrics (AutoInterp, Sparse Probing, Causal Editing) and models (Gemma-2-2B, Llama-3-8B). They do not test whether random SAEs show absorption-like patterns. The 9% true feature recovery in synthetic experiments is concerning but synthetic data may not reflect real LLM feature structures.
- **Confounding check**: Random SAEs matching trained SAEs on some metrics does not imply all SAE phenomena are artifacts. It could mean those specific metrics are flawed, not that absorption is unreal. The Chanin metric uses causal ablation, which is harder to fake with random weights.
- **Actionability check**: Even if absorption is partially artifactual, showing this would be valuable---it would force the field to develop better metrics. But the paper risks being dismissed as a "gotcha" if it doesn't propose alternatives.
- **Verdict**: MODERATE --- The causal ablation component of the Chanin metric is harder to dismiss than AutoInterp scores. But the broader point---that SAE metrics may measure dictionary size artifacts---is well-supported and worth investigating.

### Against Candidate B (The Utility Paradox)

- **Steelman**: Wang et al. (2026) show weak interpretability-utility correlation, but their analysis is correlational, not causal. It does not test whether *manipulating* absorption (the independent variable) affects utility (the dependent variable). Matryoshka SAEs do reduce absorption, and they may improve some downstream tasks---no one has done the controlled experiment.
- **Cherry-picking check**: I am selectively citing negative results (Wang et al., Kantamneni et al., Wu et al.) while ignoring potential positive results. The literature on SAE utility is mixed, not uniformly negative. Some studies show SAEs improve specific tasks (e.g., refusal feature detection, safety steering).
- **Confounding check**: The weak interpretability-utility correlation could be due to poor feature selection methods, not absorption. Wang et al. propose Delta Token Confidence, which improves steering by 52.52%. Maybe the right feature selection makes absorption reduction useful.
- **Actionability check**: This is the strongest candidate because it proposes a direct experimental test. If absorption-reduced SAEs underperform on downstream tasks, the field must rethink its priorities. If they perform well, the paradox is resolved. Either outcome is publishable.
- **Verdict**: STRONG --- The experimental design is clear, the question is important, and either outcome advances the field. The main risk is that controlling for reconstruction fidelity is technically challenging.

### Against Candidate C (Absorption Is Inevitable)

- **Steelman**: Tang et al. (2025, Theorem 3.8) and Chanin et al. (2024) provide strong theoretical evidence that absorption is a mathematical consequence of sparsity + hierarchy. The impossibility result would formalize this intuition and survey existing solutions as special cases that trade absorption for other problems.
- **Cherry-picking check**: I am selectively citing theoretical results that support inevitability while ignoring empirical results that show absorption can be reduced (Matryoshka: 0.49 -> 0.05; OrtSAE: -65%; ATM: ~40% reduction). These are real reductions, not just shifts to other failure modes.
- **Confounding check**: "Inevitable" is too strong. The theorems assume specific conditions (e.g., standard SAE architecture, L1/TopK sparsity). Alternative architectures (transcoders, concept bottleneck SAEs) may escape the impossibility. The claim should be "absorption is inevitable for standard SAEs under standard conditions," not universally.
- **Actionability check**: The "work with absorption" proposal is vague. What does "absorption-aware interpretability" actually mean? Without concrete methods, this risks being a position paper with no technical contribution.
- **Verdict**: MODERATE --- The theoretical core is strong, but the "work with absorption" proposal needs more concrete development. The impossibility proof would be valuable even without a full alternative framework.

---

## Phase 4: Refinement

### Dropped: Candidate A (The Absorption Mirage)

The causal ablation component of the Chanin metric is genuinely harder to fake than AutoInterp scores. While random baselines matching trained SAEs on some metrics is concerning, it does not directly imply that absorption is artifactual. The experiment (testing absorption in random SAEs) is worth doing but is better as a supplementary analysis than a main contribution.

### Strengthened: Candidate B (The Utility Paradox)

This survived the steelman test best. The key refinement:
- **Make it more precise**: Instead of claiming "reducing absorption makes SAEs worse," frame it as "the absorption-utility relationship is mediated by reconstruction fidelity." Test whether absorption-reduced SAEs trade reconstruction for absorption metrics, and whether that trade-off hurts utility.
- **Fair comparisons**: Compare Matryoshka, OrtSAE, ATM, and standard SAEs at matched reconstruction fidelity levels. If absorption-reduced SAEs require lower reconstruction to achieve lower absorption, the trade-off is explicit.
- **Feature selection control**: Test with and without Wang et al.'s Delta Token Confidence feature selection to isolate whether absorption reduction helps when features are well-selected.

### Strengthened: Candidate C (Absorption Is Inevitable)

Refined to a more defensible claim:
- **Weaken "inevitable" to "fundamentally constrained"**: Prove lower bounds on absorption rate given sparsity level and feature hierarchy depth. This is a constraint, not an impossibility.
- **Concrete "absorption-aware" proposal**: Instead of vague repositioning, propose a specific method: "distributed feature steering" where absorbed parent features are steered by activating the ensemble of their child latents, weighted by decoder contributions. This is testable and builds on existing steering literature.
- **Survey existing solutions as Pareto trade-offs**: Show that each solution (Matryoshka, OrtSAE, ATM, Balance Matryoshka) occupies a different point on the absorption-reconstruction-hedging Pareto frontier, and none dominates the others.

### Selected Front-Runner: **Candidate B (The Utility Paradox)**

Reasons:
1. **Direct experimental test**: It asks a clear question with a clear methodology.
2. **High impact either way**: If absorption reduction helps utility, it validates the field's focus. If it doesn't, it forces a major rethink.
3. **Builds on the project's existing work**: The project already has null results connecting absorption to downstream tasks. This reframes those null results as a systematic phenomenon rather than a failure.
4. **Addresses the credibility crisis**: In a field drowning in negative results, a rigorous controlled study that asks "does fixing absorption actually help?" is exactly what is needed.

---

## Phase 5: Final Proposal

### Title
"The Absorption-Utility Paradox: Does Reducing Feature Absorption Actually Improve Sparse Autoencoder Downstream Performance?"

### Challenged Assumption
The SAE field implicitly assumes that reducing feature absorption (as measured by Chanin et al.'s metric) improves the practical utility of SAEs for downstream interpretability tasks (steering, probing, circuit discovery). This assumption underlies the development of Matryoshka SAEs, OrtSAE, ATM, and numerous other architectural innovations. We challenge it directly.

### Evidence

**For the assumption (mainstream view)**:
- Chanin et al. (2024) prove absorption is a logical consequence of sparsity and show it creates "interpretability illusions."
- Matryoshka SAEs reduce absorption from 0.49 to 0.05 (Bussmann et al., ICML 2025).
- OrtSAE reduces absorption by 65% (Korznikov et al., 2025).
- ATM achieves ~40% absorption reduction (Li et al., 2025).
- SAEBench includes absorption as one of 8 core metrics (Karvonen et al., ICML 2025).

**Against the assumption (contrarian evidence)**:
- Wang et al. (ICLR 2026) show weak correlation (tau_b ~ 0.3) between interpretability and steering utility; correlation vanishes after feature selection.
- Kantamneni et al. (ICML 2025) show SAEs underperform simple baselines on probing.
- Wu et al. (ICML 2025) show simple baselines outperform SAEs on steering and concept detection.
- Korznikov et al. (2026) show random SAE baselines match trained SAEs on key metrics, questioning whether metric improvements reflect real progress.
- Absorption-reduced methods have costs: Matryoshka introduces hedging; OrtSAE has "slightly lower explained variance"; ATM is limited to single GPU and Gemma-2-2B.

### Hypothesis
Reducing feature absorption degrades reconstruction fidelity, and the net effect on downstream utility is neutral or negative when controlling for reconstruction fidelity. The field's focus on absorption reduction optimizes the wrong metric.

### Method

**Experimental Design**:
1. **Models**: Gemma-2-2B (primary), GPT-2 small (validation). Use SAELens to load pretrained SAEs.
2. **SAE variants** (all at matched dictionary size and sparsity level where possible):
   - Standard ReLU/TopK SAE (baseline)
   - Matryoshka SAE (reduced absorption)
   - OrtSAE (orthogonality-constrained, reduced absorption)
   - ATM SAE (temporal masking, reduced absorption)
   - Random/frozen decoder SAE (sanity check)
3. **Metrics**:
   - Absorption rate (Chanin et al. metric via SAEBench)
   - Reconstruction fidelity (MSE, explained variance)
   - Downstream utility: steering (perplexity change, target completion rate), sparse probing (F1 on concept detection), circuit discovery accuracy
4. **Control**: Match reconstruction fidelity across variants by tuning hyperparameters. Test whether absorption-utility correlation holds at fixed reconstruction.

**Feature Selection Control**:
- Test with standard top-k feature selection and with Wang et al.'s Delta Token Confidence.
- If absorption reduction only helps with good feature selection, the paradox is nuanced, not absolute.

### Experimental Plan

| Task | Model | Duration | Purpose |
|------|-------|----------|---------|
| Pilot 1: Load & benchmark 4 SAE variants on Gemma-2-2B layer 12 | Gemma-2-2B | 15 min | Verify SAELens loading, compute absorption + reconstruction |
| Pilot 2: Steering comparison (single concept) | Gemma-2-2B | 15 min | Quick utility check on one concept |
| Main 1: Full absorption-reconstruction-utility sweep | Gemma-2-2B | 45 min | Primary experiment: all variants, all metrics |
| Main 2: Validation on GPT-2 small | GPT-2 | 30 min | Cross-model replication |
| Main 3: Feature selection ablation | Gemma-2-2B | 30 min | Delta Token Confidence vs. standard selection |

Total: ~2.5 hours, split into 5 independent tasks (fits within project's time constraints).

### Baselines
- **Standard SAE**: The mainstream method, properly tuned (no strawman).
- **Random/frozen decoder SAE**: Korznikov et al.'s sanity check baseline.
- **Difference-in-means steering vector**: Wu et al.'s simple baseline that outperforms SAEs.

### Risk Assessment

**What if the mainstream view is correct?**
- If absorption-reduced SAEs significantly outperform standard SAEs on downstream tasks at matched reconstruction fidelity, the paradox is resolved. The paper becomes a validation study that confirms the field's focus. This is still publishable as a rigorous confirmation with controlled comparisons.

**What if the effect is mixed?**
- If absorption reduction helps some tasks (e.g., steering) but hurts others (e.g., probing), the paper reveals a nuanced trade-off. This is the most likely outcome and would be the most valuable contribution.

**What if reconstruction fidelity cannot be matched?**
- If absorption-reduced SAEs inherently have lower reconstruction, the paper demonstrates the Pareto frontier and argues for absorption-aware evaluation (considering both metrics). This is also a valuable contribution.

### Novelty Claim
This would be the first paper to systematically test whether absorption reduction improves downstream utility under controlled conditions. The field has assumed the answer is "yes" and built architectures accordingly. We ask: "Is it?"

### Why This Matters
In a field facing a credibility crisis---random baselines matching trained SAEs, weak interpretability-utility correlation, DeepMind deprioritizing SAE research---the most valuable contribution is not another absorption-reduction method. It is a rigorous test of whether absorption reduction matters at all. If it does not, the field can redirect its efforts toward more productive directions (e.g., utility-oriented training objectives, transcoders, or built-in interpretability). If it does, the field gains confidence in its foundational assumption. Either outcome is a service to the community.

---

## Sources

- [Korznikov et al., 2026. Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? arXiv:2602.14111](https://arxiv.org/abs/2602.14111)
- [Wang et al., ICLR 2026. Does Higher Interpretability Imply Better Utility? arXiv:2510.03659](https://arxiv.org/abs/2510.03659)
- [Wu et al., ICML 2025. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. arXiv:2501.17148](https://arxiv.org/abs/2501.17148)
- [Kantamneni et al., ICML 2025. Are Sparse Autoencoders Useful? A Case Study in Sparse Probing](https://proceedings.mlr.press/v267/kantamneni25a.html)
- [Chanin et al., 2024. A is for Absorption. arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- [Tang et al., 2025. A Unified Theory of Sparse Dictionary Learning. arXiv:2512.05534](https://arxiv.org/abs/2512.05534)
- [Cui et al., 2025. On the Limits of Sparse Autoencoders. arXiv:2506.15963](https://arxiv.org/abs/2506.15963)
- [Paulo et al., 2025. Transcoders Beat Sparse Autoencoders. arXiv:2501.18823](https://arxiv.org/abs/2501.18823)
- [Paulo & Belrose, 2025. SAEs Trained on Same Data Learn Different Features. arXiv:2501.16615](https://arxiv.org/abs/2501.16615)
- [Bussmann et al., ICML 2025. Matryoshka Sparse Autoencoders. arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
- [Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders. arXiv:2509.22033](https://arxiv.org/abs/2509.22033)
- [Li et al., 2025. Time-Aware Feature Selection. arXiv:2510.08855](https://arxiv.org/abs/2510.08855)
- [Karvonen et al., ICML 2025. SAEBench. arXiv:2503.09532](https://arxiv.org/abs/2503.09532)
- [Chanin et al., 2025. Feature Hedging. arXiv:2505.11756](https://arxiv.org/abs/2505.11756)
- [Interpretability Illusions, 2025. arXiv:2505.16004](https://arxiv.org/abs/2505.16004)
- [Smith et al., 2025. Negative Results for SAEs](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Elhage et al., 2022. Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- ["Use SAEs to Discover Unknown Concepts", 2025. arXiv:2506.23845](https://arxiv.org/abs/2506.23845)
