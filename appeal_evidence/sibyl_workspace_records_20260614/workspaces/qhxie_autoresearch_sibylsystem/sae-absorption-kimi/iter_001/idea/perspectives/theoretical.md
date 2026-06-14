# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024)** — *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders* (arXiv:2409.14507). **Key mathematical result:** Appendix A.2 proves via $\delta$-absorption that reconstruction loss is flat with respect to absorption while sparsity loss strictly decreases, making absorption an optimal solution under hierarchical feature co-occurrence.

2. **Tang et al. (2025)** — *A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima* (arXiv:2512.05534). **Key mathematical result:** Theorem 3.2 establishes piecewise biconvexity of SDL; Theorem 3.7 proves prevalence of spurious partial minima exhibiting polysemanticity; Theorem 3.10 shows hierarchical structures induce feature absorption patterns as partial minima.

3. **Chanin et al. (2025)** — *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders* (arXiv:2505.11756). **Key mathematical result:** Formal proof that MSE loss causes feature hedging when SAE capacity is narrower than the number of true features and correlations exist; Theorem in Appendix A.4 shows optimal reconstruction weights for orphan features are determined by partial correlations.

4. **Cui et al. (2025)** — *On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond* (arXiv:2506.15963). **Key mathematical result:** Theorems 1–2 give necessary and sufficient conditions for SAE identifiability (extreme sparsity, sparse activation, sufficient hidden dimensions); Theorem 3 decomposes the gap when identifiability fails.

5. **Elhage et al. (2022)** — *Toy Models of Superposition* (arXiv:2209.10652). **Key mathematical result:** Foundational framework showing how neural networks represent more features than dimensions via overlapping directions; establishes the compressed-sensing geometry underlying SAEs.

6. **Ivanov et al. (2026)** — *Spectral Superposition: A Theory of Feature Geometry* (arXiv:2602.02224). **Key mathematical result:** Theorem 3 proves features at capacity saturation organize into tight frames within eigenspaces; provides spectral classification of superposition geometries.

7. **Pacela et al. (2026)** — *Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation* (arXiv:2603.28744). **Key mathematical result:** Proves SAE failure is due to dictionary learning (not amortization); reframes sparse inference under superposition as a compressed sensing problem where dictionary quality is the binding constraint.

8. **Korznikov et al. (2026)** — *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* (arXiv:2602.14111). **Key mathematical result:** Random/frozen-decoder baselines match trained SAEs on standard metrics; supports a "lazy training regime" where decoder vectors remain near random initialization.

9. **Bussmann et al. (2025)** — *Learning Multi-Level Features with Matryoshka Sparse Autoencoders* (arXiv:2503.17547). **Key mathematical result:** Nested dictionaries learn general concepts at small scales and specifics at large scales; empirically reduces absorption with a minor reconstruction trade-off.

10. **Korznikov et al. (2025)** — *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features* (arXiv:2509.22033). **Key mathematical result:** Chunk-wise orthogonality penalty reduces absorption by 65% and composition by 15%; establishes that geometric constraints on decoder weights can mitigate absorption.

### Theoretical Landscape Summary

**What is known:**
- Feature absorption is not an accident of training but a **structural consequence** of optimizing sparsity under hierarchical feature co-occurrence (Chanin et al., 2024; Tang et al., 2025).
- The SAE optimization landscape is piecewise biconvex, with spurious partial minima that exhibit both polysemanticity and absorption (Tang et al., 2025).
- Absorption and hedging are **dual pathologies** caused by different loss terms: sparsity drives absorption in wide SAEs; reconstruction drives hedging in narrow SAEs (Chanin et al., 2024; Chanin et al., 2025).
- SAEs are identifiable only under extreme sparsity and sufficient dictionary size; in realistic regimes, a gap between SAE loss and ground-truth recovery is unavoidable (Cui et al., 2025).
- Random/frozen decoder baselines achieve comparable performance to trained SAEs on many standard metrics, suggesting current SAE training may operate in a "lazy" regime (Korznikov et al., 2026).

**What is conjectured:**
- Orthogonality or hierarchical constraints on the decoder can shift the optimization landscape away from spurious minima (OrtSAE, Matryoshka SAEs, H-SAEs).
- The rate-distortion interpretation of sparse encoding may explain scaling laws in feature usage and sensitivity.

**Where the gaps are:**
- **No unified task-agnostic absorption metric:** The canonical absorption metric is tied to the first-letter spelling task. A generalizable metric for arbitrary hierarchical feature domains is missing.
- **No rigorous multi-objective Pareto characterization:** Prior work compares architectures on multiple metrics but does not systematically evaluate whether absorption-mitigation methods dominate standard SAEs across the full metric suite (absorption, hedging, reconstruction, dead neurons, downstream utility).
- **Weak connection between theory and practical design:** Tang et al.'s piecewise biconvex framework is abstract; few architectural innovations are explicitly derived from it.
- **Random-decoder baselines have not been tested on absorption:** Korznikov et al. (2026) showed random baselines match trained SAEs on AutoInterp, sparse probing, and RAVEL, but did **not** measure absorption. This leaves open whether absorption is a training artifact or a geometric inevitability.

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption-Hedging-Reconstruction Impossibility Triangle

- **Formal claim:** For a sparse autoencoder trained on activations containing hierarchically correlated features, no single architecture family stochastically dominates all others on the Pareto front spanning absorption rate, hedging rate, reconstruction fidelity, dead-neuron fraction, and downstream sparse-probing F1. Each architecture occupies a distinct trade-off region determined by which loss term or constraint it prioritizes.
- **Proof sketch:**
  1. By Chanin et al. (2024), sparsity optimization drives absorption when parent-child co-occurrence probability $p_{11} > 0$.
  2. By Chanin et al. (2025), reconstruction (MSE) optimization drives hedging when SAE width $M < N$ (number of true features) and correlations exist.
  3. By Tang et al. (2025), the SDL landscape contains spurious partial minima corresponding to both absorption and hedging configurations.
  4. Adding orthogonality constraints (OrtSAE) or hierarchical nesting (Matryoshka) changes the feasible set but introduces new trade-offs: orthogonality reduces absorption at the cost of reconstruction flexibility; nested dictionaries increase training complexity and can elevate hedging in inner (narrower) levels.
  5. Therefore, any architecture is a point in a constrained optimization over competing objectives; stochastic dominance across all metrics is impossible because improving one metric tightens constraints that degrade another.
- **Empirical prediction:** When existing pretrained SAE checkpoints (Standard, TopK, JumpReLU, OrtSAE, Matryoshka) are evaluated on all five metrics, each family will occupy a distinct Pareto region. OrtSAE/Matryoshka will show lower absorption but higher hedging or lower reconstruction compared to Standard/TopK.
- **Connection to existing theory:** Extends Chanin et al.'s dual-pathology framework from individual loss terms to a multi-objective landscape; connects Tang et al.'s spurious-minima analysis to empirical architecture comparisons.
- **Novelty estimate:** 8/10. No prior work frames absorption research as a systematic Pareto impossibility triangle using existing checkpoints in a training-free setting.

### Candidate B: A Task-Agnostic Absorption Metric via Automated Hierarchy Discovery

- **Formal claim:** Let $\mathcal{H}$ be a set of parent-child feature hierarchies discovered automatically from SAE latents via LLM labeling and probe validation. Define a task-agnostic absorption score $\alpha_{\text{TA}}$ as the fraction of probe-true parent-feature instances where the primary parent latents fail to activate, and an alternative child latent dominates under causal ablation. Then $\alpha_{\text{TA}}$ correlates moderately-to-strongly ($r > 0.4$) with the original first-letter absorption benchmark while generalizing to arbitrary semantic domains.
- **Proof sketch:**
  1. The original absorption metric (Chanin et al., 2024) is a special case of $\alpha_{\text{TA}}$ where $\mathcal{H}$ is the set of first-letter parent-child pairs.
  2. For any domain with validated hierarchical probes, the causal ablation test (k-sparse probing → false negatives → integrated-gradients ablation → absorption classification) is domain-independent.
  3. If the LLM-discovered hierarchies are semantically coherent (validated by probe success), the structural conditions causing absorption (parent-child co-occurrence + sparsity optimization) are identical across domains.
  4. By the law of large numbers over a diverse set of domains, $\alpha_{\text{TA}}$ converges to a stable estimator of the underlying absorption rate.
- **Empirical prediction:** On 20–50 pretrained SAEs, Pearson correlation between $\alpha_{\text{TA}}$ (averaged over geography, biology, and color hierarchies) and the first-letter absorption score will be $r \in [0.45, 0.70]$.
- **Connection to existing theory:** Builds directly on Chanin et al.'s causal ablation framework; extends it from a task-specific instantiation to a general measurement procedure.
- **Novelty estimate:** 7/10. The automated hierarchy discovery component is new, but the core causal ablation test is borrowed from Chanin et al.

### Candidate C: Feature Absorption as a Geometric Inevitability — A Random-Decoder Baseline Test

- **Formal claim:** Feature absorption in sparse autoencoders is primarily a geometric consequence of sparse dictionary learning on hierarchically structured data, not a pathology of flawed training dynamics. Formally, a randomly initialized, frozen-decoder SAE matched for L0 sparsity will exhibit an absorption rate within 20% (relative) of a fully trained SAE on the same data.
- **Proof sketch:**
  1. In high dimensions, random decoder vectors are approximately uniformly distributed on the sphere (Korznikov et al., 2026).
  2. For hierarchical features with parent-child co-occurrence, the encoder must solve a sparse reconstruction problem with a fixed random dictionary.
  3. By compressed sensing theory (Pacela et al., 2026), the binding constraint is dictionary quality, not encoder architecture. With a random dictionary, the encoder will still prefer sparse solutions that reuse atoms for co-occurring features.
  4. Because the decoder is fixed and random, any systematic alignment with parent vs. child directions is vanishingly small. The encoder therefore faces the same sparsity-reconstruction trade-off that drives absorption in trained decoders.
  5. Consequently, the absorption rate should be comparable to trained SAEs, confirming that absorption is structural rather than training-specific.
- **Empirical prediction:** On GPT-2-small activations, a frozen-decoder SAE trained to matched L0 will show an absorption rate $\geq 80\%$ of the trained SAE baseline.
- **Connection to existing theory:** Connects Korznikov et al.'s random-baseline critique to the absorption literature; tests whether Tang et al.'s spurious-minima analysis applies even without learned decoder dynamics.
- **Novelty estimate:** 6/10. Novel in applying absorption metrics to random-decoder baselines, but the random-baseline idea itself is from Korznikov et al. (2026). **Critical limitation:** This candidate violates the project's training-free constraint.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Proof soundness attack:** The claim relies on the assumption that architectures cannot simultaneously reduce absorption, hedging, and reconstruction loss. Matryoshka SAEs report Pareto improvements on reconstruction-sparsity fronts; it is possible (though not proven) that some hierarchical design could dominate on all three pathology metrics too. The proof sketch is a conceptual argument, not a formal theorem.
- **Tightness attack:** The claim is about stochastic dominance across architectures, not about individual checkpoints. It is possible that one architecture family has a better *mean* performance even if no single checkpoint dominates all others. The claim is carefully phrased to avoid this, but the practical significance depends on effect sizes.
- **Relevance attack:** This is highly relevant to practitioners. If true, it reframes the research agenda from "fixing absorption" to "navigating trade-offs" and motivates task-adaptive SAE selection.
- **Novelty attack:** No prior work explicitly evaluates the absorption-hedging-reconstruction impossibility triangle across existing architectures using existing checkpoints. OrtSAE and Matryoshka compare each other on multiple metrics but do not frame it as a systematic Pareto analysis.
- **Verdict:** STRONG

### Against Candidate B

- **Proof soundness attack:** The correlation claim ($r > 0.4$) is an empirical prediction, not a theorem. The "proof sketch" is really a heuristic argument about structural similarity across domains. LLM-generated hierarchies may be noisy or hallucinated, breaking the validation step.
- **Tightness attack:** If correlation is weak (e.g., $r < 0.3$), the metric fails to generalize. The falsification threshold is pre-registered, which is good, but a weak correlation could mean the first-letter benchmark is unrepresentative rather than the metric being invalid.
- **Relevance attack:** A task-agnostic absorption metric would be valuable for the community, but its practical utility depends on whether it predicts downstream harm better than the existing benchmark. The candidate does not formally prove this predictive relationship.
- **Novelty attack:** No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery. The canonical metric remains tied to the first-letter spelling task.
- **Verdict:** MODERATE

### Against Candidate C

- **Proof soundness attack:** The claim that random decoders will show comparable absorption is plausible but not rigorously proved. The encoder training dynamics on a frozen random dictionary could differ systematically from a jointly trained encoder-decoder. For example, a trained decoder may learn to align with feature directions, which could either increase or decrease absorption relative to a random decoder.
- **Tightness attack:** The 20% threshold is arbitrary. If the result falls in the 50–80% range, interpretation is ambiguous.
- **Relevance attack:** If absorption is indeed geometric, this would be a major reframing. However, the candidate is blocked by the training-free constraint.
- **Novelty attack:** Novel in applying absorption to random-decoder baselines, but Korznikov et al. (2026) already established the random-baseline paradigm.
- **Verdict:** WEAK (due to training-free constraint violation and weaker novelty)

---

## Phase 4: Refinement

**Dropped:** Candidate C is dropped because it explicitly violates the project's training-free constraint and has the weakest novelty among the three.

**Strengthened:** Candidate A is selected as the front-runner. The multi-objective Pareto evaluation is the most theoretically grounded and practically relevant contribution. It directly leverages the dual-pathology theory (absorption from sparsity, hedging from reconstruction) and the piecewise biconvexity framework. The pilot evidence shows the metric pipeline works end-to-end on GPT-2 Small, though proper absorption/hedging metrics and larger token budgets are needed before full-scale experiments.

**Backup retained:** Candidate B remains as a backup. If the open-model anchor for hierarchy discovery becomes available (e.g., via SAEBench releases or Pythia Scope), it can be revisited.

**Selected front-runner:** Candidate A — *The Absorption-Hedging-Reconstruction Impossibility Triangle: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods*

---

## Phase 5: Final Proposal

### Title
**The Hidden Cost of Fixing Feature Absorption: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods**

### Formal Claim
For sparse autoencoders trained on language model activations, absorption-mitigation architectures (OrtSAE, Matryoshka SAE, JumpReLU, masked regularization) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption rate, hedging rate, reconstruction fidelity, dead-neuron rate, and downstream sparse-probing performance. Each architecture family occupies a distinct trade-off region determined by which loss term or structural constraint it prioritizes.

### Proof Sketch
1. **Sparsity drives absorption.** By Chanin et al. (2024), for any parent-child feature pair with co-occurrence probability $p_{11} > 0$, the SAE loss $\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{sp}}$ has zero reconstruction gradient but negative sparsity gradient with respect to the absorption parameter $\delta$. Hence, optimization naturally favors full absorption.
2. **Reconstruction drives hedging.** By Chanin et al. (2025), when SAE width $M < N$ (number of true features), MSE-optimal decoder weights for tracked features load on correlated orphan features with non-zero partial correlations. This creates hedging—latents that mix semantically distinct but correlated concepts.
3. **Architectural constraints shift the feasible set.** OrtSAE adds a decoder orthogonality penalty that reduces absorption by preventing parent-child directions from collapsing, but this tightens the reconstruction feasible set. Matryoshka SAEs enforce nested dictionaries that separate abstraction levels, reducing absorption at the cost of increased hedging in inner (narrower) levels and higher training complexity.
4. **Spurious minima make dominance impossible.** By Tang et al. (2025), the SDL landscape contains spurious partial minima corresponding to both absorption and hedging configurations. Any architecture that pushes the optimizer away from one class of spurious minima risks pushing it toward another, because the global minimum (perfect feature recovery) is only achievable under identifiability conditions that real LLM activations do not satisfy (Cui et al., 2025).
5. **Therefore, no architecture can dominate all metrics.** Improving absorption requires relaxing sparsity or adding geometric constraints, which either degrades reconstruction or elevates hedging. The multi-objective landscape is inherently fractured.

### Assumptions
- The pretrained SAE checkpoints used in the evaluation were trained with standard hyperparameter search and are representative of their architecture families.
- The activation data on which absorption and hedging are measured contains hierarchical and correlated features (validated by the ubiquity of absorption in prior work).
- The metrics (absorption, hedging, reconstruction, dead neurons, sparse probing) jointly capture the practical quality of an SAE for interpretability research.

### Empirical Prediction
When 20–50 pretrained SAE checkpoints per model (GPT-2 Small or Pythia-160M) are evaluated across all five metrics:
- OrtSAE and Matryoshka SAE will show **lower absorption** than Standard/TopK SAEs.
- Matryoshka inner levels and narrower configurations will show **higher hedging**.
- Standard/BatchTopK SAEs will achieve the **best reconstruction fidelity** (explained variance, CE loss recovered).
- No single architecture family will show statistically significant stochastic dominance (Mann-Whitney U test, $p < 0.05$) across $\geq 4$ out of 5 metrics.

### Experimental Plan
**Experiment 1: Multi-Objective Pareto Evaluation (Training-Free, ~45–60 min per batch)**
- **Checkpoint corpus:** Assemble 20–30 pretrained SAEs per model across 5+ architecture families (Standard, TopK, JumpReLU, Gated, BatchTopK, OrtSAE, Matryoshka) using SAELens, SAEBench releases, and open-source scopes.
- **Metrics:**
  - **Absorption:** Rigorous `sae-spelling` implementation or SAEBench adaptation (not the simplified proxy).
  - **Hedging:** Proper hedging metric from Chanin et al. (2025) or SAEBench.
  - **Reconstruction:** L0, explained variance, CE loss recovered on a held-out corpus of $\geq 50\text{K}$ tokens.
  - **Dead neurons:** Fraction of latents with near-zero activation frequency (computed on $\geq 50\text{K}$ tokens).
  - **Downstream probing:** Sparse probing F1 from SAEBench.
- **Analysis:**
  1. Normalize each metric to $[0, 1]$ within model family.
  2. Compute empirical Pareto fronts per architecture family.
  3. Test stochastic dominance using Mann-Whitney U tests across the full metric suite.
  4. Report pairwise trade-off curves and confidence intervals.
- **Models:** GPT-2 Small and/or Pythia-160M (open models; Gemma-2-2B is gated and inaccessible).

**Experiment 2: Downstream Causal Cost Meta-Analysis (Training-Free, ~30 min)**
- Using the 200+ pretrained SAEs from SAEBench:
  1. Extract absorption, sparse probing F1, RAVEL Cause/Isolation, TPP, SCR, and L0/loss-recovered.
  2. Perform partial correlation and regression with absorption as the predictor, controlling for L0 and reconstruction.
  3. Include architecture family dummies and dictionary width as covariates.
  4. Use cluster-robust standard errors by architecture family.
- **Expected outcome:** Negative partial correlation between absorption and downstream performance, supporting the claim that absorption has unique causal harm.

### Baselines
- **Theoretical baselines:** Existing bounds from Chanin et al. (absorption-sparsity relationship), Chanin et al. (2025) (hedging-width relationship), and Tang et al. (spurious minima prevalence).
- **Empirical baselines:** Standard SAE and TopK SAE families serve as the reference architecture against which OrtSAE, Matryoshka, and other mitigations are compared.

### Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OrtSAE/Matryoshka actually dominate the Pareto front | Medium | Medium | The paper still provides the first rigorous multi-objective validation—a valuable contribution, though less contrarian. |
| SAEBench metrics are too noisy for clean Pareto analysis | Medium | Medium | Use well-tested implementations; average across multiple layers/checkpoints; report bootstrap confidence intervals. |
| Gemma-2-2B remains inaccessible | High | Low | Anchor on GPT-2 Small / Pythia-160M, which have abundant open checkpoints. |
| Downstream correlation is confounded by unobserved architecture differences | Medium | Medium | Include architecture dummies, width, and L0 as controls; use cluster-robust SEs; explicitly discuss causal limitations. |

### Novelty Claim
**No prior work has conducted a systematic, training-free, multi-objective Pareto evaluation of absorption-mitigation methods across the full suite of SAE quality metrics using existing pretrained checkpoints.** Prior studies (OrtSAE, Matryoshka, SAEBench) evaluate absorption, reconstruction, and downstream tasks, but they do not frame this as a Pareto impossibility analysis and do not jointly control for hedging, dead neurons, and downstream utility. This work would be the first to rigorously test whether current "fixes" for absorption genuinely improve SAEs overall or merely shift pathologies.
