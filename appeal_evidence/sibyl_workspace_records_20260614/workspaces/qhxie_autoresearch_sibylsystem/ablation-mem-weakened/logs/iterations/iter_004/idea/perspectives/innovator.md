# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Chanin et al., 2024/2025. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507]** — Foundational work defining feature absorption, proving it is a logical consequence of sparsity loss under hierarchical features. Validates across hundreds of SAEs. The primary target phenomenon for our investigation.

2. **[Rozell et al., 2008. "Sparse Coding via Thresholding and Local Competition in Neural Circuits." Neural Computation]** — Seminal neuroscience work introducing the Locally Competitive Algorithm (LCA). The inhibition matrix G_{m,k} = <Phi_m, Phi_k> provides the exact structural correspondence that inspired the current project's front-runner (Local Inhibition Graph).

3. **[Tang et al., 2025. "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534]** — First unified theoretical framework for SDL. Proves feature absorption corresponds to spurious local minima. Introduces "feature anchoring" concept. Critical for understanding absorption from an optimization perspective.

4. **[Li, Michaud, Baek, Engels, Sun & Tegmark, 2025. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." Entropy 27(4):344 / ICLR 2025]** — Multi-scale geometric analysis of SAE features: "crystals" at atomic scale, "lobes" at brain scale, power-law anisotropy at galaxy scale. Suggests rich geometric structure in SAE feature dictionaries that has not been fully exploited.

5. **[Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 / ICML 2025]** — Reduces absorption from 0.49 to 0.05 via nested dictionaries, but introduces hedging trade-off. Shows absorption-hedging as competing failure modes.

6. **[Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033]** — Enforces decoder orthogonality, reducing absorption by 65%. Demonstrates that decoder geometry directly controls absorption rates.

7. **[Marks et al., 2025. "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." ICLR 2025]** — Combines sparse feature circuits with causal interventions. Moves from isolated feature discovery to complete computational graph interpretation.

8. **[Anthropic, 2025. "Circuit Tracing: Revealing Computational Graphs in Language Models." transformer-circuits.pub]** — Cross-Layer Transcoders (CLTs) with attribution graphs. 30 million features across layers. Full end-to-end circuit tracing paradigm.

9. **[Geiger et al., 2025. "A Theoretical Foundation for Mechanistic Interpretability." JMLR Volume 26]** — Generalizes causal abstraction theory to unify MI methods including SAEs, activation patching, causal scrubbing, and circuit analysis.

10. **[Wang et al., 2025. "Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders." arXiv:2510.03659 / ICLR 2026]** — Weak correlation (tau_b ~ 0.3) between interpretability and steering utility. Raises fundamental questions about whether absorption metrics matter practically.

11. **[Gardinazzi et al., ICML 2025. "Persistent Topological Features in LLMs"]** — Applies persistent homology to analyze topological structure of LLM representations. Demonstrates TDA can reveal structure that geometric methods miss.

12. **[iP-VAE / FOND framework, 2024. "Brain-like Variational Inference." arXiv:2410.19315]** — Recurrent spiking neural network with emergent normalization via lateral competition. Combines free energy minimization, predictive coding, lateral inhibition, and sparse representation in a biologically plausible framework.

### Landscape Summary

The SAE field in 2025-2026 is at an inflection point. On one hand, the ecosystem has matured tremendously: SAELens, SAEBench, GemmaScope, and Neuronpedia provide standardized tools and benchmarks. Anthropic's circuit tracing work has pushed the frontier from feature discovery to complete computational graph interpretation. On the other hand, fundamental skepticism is growing: the "Sanity Checks" paper shows frozen/random SAE baselines match trained SAEs on key metrics; Wang et al. show weak interpretability-utility correlation; and the "Interpretability without Actionability" paper (arXiv:2603.18353) found that SAE feature steering produced zero effect on error correction despite thousands of significant features.

Against this backdrop, the current project's front-runner — the Local Inhibition Graph connecting LCA neuroscience to SAE absorption — represents a genuinely novel theoretical contribution. The structural correspondence W_dec^T W_dec = G_LCA is exact, not metaphorical. No prior work has made this connection. The novelty searches conducted in previous rounds confirmed zero matches for "LCA sparse autoencoder," "inhibition graph SAE," and "decoder correlation absorption prediction."

However, the field is moving fast. Anthropic's CLT work has shifted attention from per-layer SAEs to cross-layer architectures. The "Geometry of Concepts" paper reveals rich multi-scale structure in SAE feature dictionaries that remains unexploited. And the theoretical work by Tang et al. provides a rigorous optimization-landscape perspective on absorption that complements (but does not replace) the dynamical-systems perspective offered by LCA.

The key insight from this survey is that while the Local Inhibition Graph is novel and well-motivated, there are adjacent directions with comparable or greater novelty potential that have not been explored: (1) topological analysis of SAE feature geometry, (2) iterative/recurrent feature recovery mechanisms, and (3) information-theoretic bounds on absorption. These form the basis for the three candidates below.

---

## Phase 2: Initial Candidates

### Candidate A: Persistent Homology of SAE Feature Geometry — Topological Signatures of Absorption

- **Hypothesis**: The topological structure (persistent homology) of SAE decoder weight manifolds contains signatures that distinguish absorbed features from non-absorbed features, enabling a training-free diagnostic with theoretical guarantees.
- **Cross-domain insight**: From topological data analysis (TDA): persistent homology captures multi-scale shape features of point clouds that are robust to noise and invariant to continuous deformations. SAE decoder weights form a point cloud in R^{d_model} where the "shape" of neighborhoods around absorbed features differs systematically from non-absorbed features.
- **Why it might work**: Li et al. (2025) showed SAE features exhibit rich geometric structure at three scales (crystals, lobes, galaxy). Absorption occurs when hierarchical features co-occur — this creates specific geometric configurations in decoder space (parent and child directions cluster together). Persistent homology can detect these clusters via Betti numbers and persistence diagrams. The "Geometry of Concepts" paper described but did not exploit this structure.
- **Novelty estimate**: 8/10 — TDA has been applied to LLM representations (Gardinazzi et al., ICML 2025) and loss landscapes, but NEVER to SAE decoder dictionaries for absorption diagnosis. The combination of persistent homology with SAE feature geometry is entirely unexplored.

### Candidate B: Iterative Feature Recovery via Recurrent Refinement — A Predictive Coding Perspective on Absorption Repair

- **Hypothesis**: A recurrent refinement mechanism that iteratively updates SAE activations using top-down prediction errors (inspired by predictive coding) can recover absorbed parent features without modifying SAE weights, achieving better reconstruction-steering trade-offs than single-pass methods.
- **Cross-domain insight**: From predictive coding / free energy principle (Friston 2005, 2010; iP-VAE 2024): the brain uses recurrent processing where higher-level predictions feed back to lower levels, and prediction errors drive representation updates. In SAEs, "absorption" is analogous to a prediction error where the parent feature's expected activation is suppressed by the child's dominant activation. Iterative refinement can progressively "explain away" the child's contribution and restore the parent.
- **Why it might work**: The current project's homeostatic rebalancing (H10) is a single-pass correction: z'_i = z_i + alpha * inh_i. But biological homeostasis operates over time — multiple iterations allow the system to converge. MP-SAE (Matching Pursuit, 2025) shows iterative residual-guided inference monotonically improves reconstruction. The iP-VAE framework demonstrates that recurrent dynamics with lateral competition naturally emerge from variational inference. Combining these: a recurrent SAE activation update that uses both bottom-up input and top-down predictions from higher layers could progressively disentangle absorbed features.
- **Novelty estimate**: 7/10 — Iterative inference in SAEs exists (MP-SAE, DrSAE), and predictive coding is well-known. But combining predictive coding's top-down feedback with SAE absorption repair is novel. No prior work frames absorption repair as iterative prediction-error minimization.

### Candidate C: The Information-Bottleneck Absorption Bound — A Rate-Distortion Theory of Feature Hierarchy

- **Hypothesis**: Feature absorption rate can be bounded below by the rate-distortion function of the feature hierarchy under an information bottleneck constraint, proving that absorption is not merely a training artifact but an information-theoretic necessity for lossy compression of hierarchical features.
- **Cross-domain insight**: From information bottleneck theory (Tishby 1999) and rate-distortion theory (Shannon 1948): when compressing a source with hierarchical structure (where child features imply parent features), the optimal lossy codebook necessarily merges parent information into child representations at sufficiently low rates. This is the information-theoretic analogue of absorption.
- **Why it might work**: Tang et al. (2025) proved absorption corresponds to spurious local minima in the SDL optimization landscape. But this leaves open whether absorption is avoidable in principle. Rate-distortion theory suggests it is not: if the SAE must compress N features into M < N dimensions with bounded distortion, and features have hierarchical dependencies (parent implies child), then the optimal codebook will allocate representation resources to the more specific child features, effectively "absorbing" the parent. The "Geometry of Concepts" paper's power-law eigenvalue structure suggests the feature dictionary itself has a rate-distortion structure.
- **Novelty estimate**: 9/10 — Rate-distortion interpretations of SAEs exist (Peter et al. 2025, Progressive/Matryoshka SAEs), but NO prior work derives an explicit lower bound on absorption rate from rate-distortion theory. This would be the first information-theoretic impossibility result for absorption, transforming it from "a problem to fix" to "a fundamental limit to manage."

---

## Phase 3: Self-Critique

### Against Candidate A: Persistent Homology of SAE Feature Geometry

- **Prior work attack**: Gardinazzi et al. (ICML 2025) applied persistent homology to LLM representations, and Li et al. (2025) analyzed SAE feature geometry. The combination is new, but the components exist. Risk: reviewers may see this as "TDA + SAE = paper" without sufficient depth. Also, "The Geometry of Concepts" paper may have a follow-up using TDA in preparation.
- **Methodological attack**: Persistent homology is computationally expensive for high-dimensional point clouds (d_model = 768 for GPT-2 Small, d_dict = 24K). The Flood Complex (NeurIPS 2025) makes large-scale PH feasible, but implementing it for SAE decoder dictionaries is non-trivial. Risk: computational barriers make the method impractical.
- **Theoretical attack**: Persistent homology captures topological features (connected components, holes, voids) but absorption is a directional, hierarchical phenomenon (parent -> child). The topological structure of decoder weights may not encode directionality. Risk: PH detects clustering but not the parent-child direction of absorption.
- **Scalability attack**: PH works well on small point clouds but may become noisy for 24K+ points in 768 dimensions. The "curse of dimensionality" affects TDA too. Risk: results don't generalize beyond small SAEs.
- **Verdict**: MODERATE — The cross-domain insight is genuine and unexplored. The main weaknesses are computational cost and directional ambiguity. Could be strengthened by focusing on local PH (around individual features) rather than global PH, and by combining with the inhibition graph's directional structure.

### Against Candidate B: Iterative Feature Recovery via Recurrent Refinement

- **Prior work attack**: MP-SAE (2025) already uses iterative residual-guided inference. DrSAE (Rolfe et al., 2013) uses recurrent encoders. The "homeostatic rebalancing" in the current proposal is a single-pass version of the same idea. Risk: reviewers see this as "MP-SAE + biological metaphor."
- **Methodological attack**: Predictive coding's top-down feedback requires a generative model of feature hierarchies. In SAEs, we don't have explicit parent-child relationships — we only have decoder directions. Constructing a predictive model from decoder correlations is speculative. Risk: the predictive model may not capture true feature hierarchies.
- **Theoretical attack**: The free energy principle is notoriously flexible — it can explain almost any neural computation post-hoc. Using it to justify iterative refinement risks appearing as "Friston-washing" rather than genuine theoretical grounding. Risk: the neuroscience connection is metaphorical, not exact (unlike the LCA correspondence).
- **Scalability attack**: Iterative refinement at inference time increases latency. For million-latent SAEs, even 5-10 iterations may be too slow for practical use. Risk: method is theoretically interesting but practically infeasible.
- **Verdict**: MODERATE — The core idea (iterative > single-pass for absorption repair) is sound and supported by MP-SAE's theoretical guarantees. But the predictive coding framing adds conceptual overhead without clear predictive advantages over simpler iterative schemes. Could be strengthened by dropping the Friston framing and focusing on convergence guarantees.

### Against Candidate C: The Information-Bottleneck Absorption Bound

- **Prior work attack**: Tang et al. (2025) already provides a theoretical framework for SDL with provable recovery conditions. Peter et al. (2025) discussed rate-distortion interpretations of sparse encoding. The "Geometry of Concepts" paper notes power-law eigenvalue structure. Risk: reviewers may see this as "IB + SAE = bound" without sufficient technical novelty.
- **Methodological attack**: Deriving a tight lower bound on absorption rate requires knowing the true feature hierarchy distribution, which is unknown for real LLMs. The bound would need to be derived for synthetic data (SynthSAEBench) and then validated empirically on real SAEs. Risk: the bound may be vacuous or not generalize from synthetic to real.
- **Theoretical attack**: Rate-distortion theory applies to source coding with known distributions. SAEs are learned representations, not designed codes. The correspondence between SAE training and rate-distortion optimization is approximate, not exact. Risk: the theoretical bound may not apply to learned SAEs.
- **Scalability attack**: If the bound is derived for toy models (as in Tang et al.), it may not scale to real LLMs. The superposition hypothesis (Elhage et al.) suggests features are approximately orthogonal, but real LLM features have complex correlation structures. Risk: bound is theoretically elegant but empirically irrelevant.
- **Verdict**: STRONG — Despite the attacks, this is the most theoretically ambitious candidate. The key insight — that absorption may be information-theoretically necessary, not just a training artifact — reframes the entire field. Even a loose bound would be valuable. The main risk is technical execution, but the payoff is highest.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Iterative Feature Recovery) dropped** because: The predictive coding framing adds overhead without clear advantage over simpler iterative schemes. MP-SAE already achieves similar goals with cleaner mathematics. The single-pass homeostatic rebalancing in the current front-runner is simpler and more practical. If iterative refinement is needed, it can be incorporated as an extension of H10 rather than a separate research direction.

### Strengthened Ideas

- **Candidate A (Persistent Homology) strengthened**: Instead of global PH on the entire decoder dictionary (computationally expensive and directionally ambiguous), focus on **local persistent homology around absorption pairs**. For each known absorption pair (parent, child), compute the PH of the k-nearest neighbor subgraph in the inhibition graph. Test whether the persistence diagram of absorbed-feature neighborhoods differs systematically from non-absorbed pairs. This combines the inhibition graph's directional structure with TDA's shape analysis, addressing both the computational and directional weaknesses.

- **Candidate C (IB Absorption Bound) strengthened**: Rather than deriving a general bound for arbitrary hierarchies, focus on the **first-letter feature hierarchy** (A-Z, A-SH, etc.) where the ground-truth structure is known. Use SynthSAEBench (Chanin et al., 2026) to generate controlled hierarchical features with known parent-child relationships. Derive the rate-distortion function for this specific hierarchy and compare the theoretical minimum absorption rate with empirical rates from standard SAEs, Matryoshka SAEs, and OrtSAE. This makes the bound concrete and empirically testable.

### Additional Evidence Found

- **Tang et al. (2025)**: Proves feature absorption corresponds to spurious local minima where "a single neuron activates for multiple features while other neurons remain inactive." This is the optimization-landscape counterpart to the rate-distortion argument: spurious local minima exist BECAUSE the global minimum (perfect feature recovery) requires more representational resources than available.
- **Li et al. (2025)**: The "galaxy-scale" power-law eigenvalue structure (W_dec^T W_dec has power-law spectrum) suggests the decoder dictionary itself has a rate-distortion structure. This provides empirical support for the IB bound approach.
- **SynthSAEBench (Chanin et al., 2026)**: Enables controlled experiments with known ground-truth features. The correspondence between synthetic and real absorption rates is unvalidated (Gap 9 in the literature survey) — this is both a risk and an opportunity.
- **"Interpretability without Actionability" (arXiv:2603.18353)**: Found zero effect of SAE feature steering on error correction. This strengthens the case for theoretical/methodological contributions over downstream utility claims. The IB bound reframes absorption as fundamental, not fixable.

### Selected Front-Runner

**Candidate C: The Information-Bottleneck Absorption Bound** is selected as the strongest complementary direction to the existing Local Inhibition Graph front-runner. Here's why:

1. **Theoretical ambition**: It aims to prove absorption is information-theoretically necessary, not merely a training artifact. This is a much stronger claim than "decoder correlations predict absorption" (the current front-runner).
2. **Complementarity**: The inhibition graph provides a mechanistic, local explanation (which features compete with which). The IB bound provides a global, information-theoretic explanation (why competition is unavoidable). Together they form a complete picture.
3. **Field impact**: If absorption is a fundamental limit, the field shifts from "fixing absorption" to "managing absorption." This reframes Matryoshka SAEs, OrtSAE, and all architectural solutions as approximations to the optimal trade-off, not as solutions to a solvable problem.
4. **Testability**: Using SynthSAEBench, the bound can be empirically validated. The theoretical prediction (absorption rate >= R(D) bound) can be compared against measured rates.

However, the IB bound is high-risk/high-reward. If the bound derivation fails or is vacuous, the paper falls back to the inhibition graph alone. Therefore, the recommended strategy is:

- **Primary contribution**: Local Inhibition Graph (current front-runner, cand_f)
- **Secondary/Theoretical contribution**: IB Absorption Bound (Candidate C, refined)
- **Integration**: The inhibition graph explains the mechanism; the IB bound explains why the mechanism is unavoidable.

---

## Phase 5: Final Proposal

### Title

**"The Local Inhibition Graph: A Neuroscience-Inspired Training-Free Diagnostic for Feature Absorption in Sparse Autoencoders"**

*With secondary theoretical contribution:* **"An Information-Theoretic Lower Bound on Feature Absorption in Sparse Autoencoders"**

Alternative unified title: **"Decoder Correlations Reveal Competitive Suppression: Local Inhibition Graphs and Information-Theoretic Limits of SAE Feature Recovery"**

### Hypothesis

**Primary (H6-H10, from existing front-runner):** Edges in the local inhibition graph constructed from decoder correlations (W_dec^T W_dec = G_LCA) predict known absorption pairs with precision significantly above chance (precision@20 >= 0.10 vs. ~0.004 chance), explain precision-recall asymmetry, and enable training-free post-hoc repair via homeostatic rebalancing.

**Secondary/Theoretical (NEW — H11):** For a hierarchical feature source with parent-child dependencies, the feature absorption rate in any sparse autoencoder with dictionary size M and reconstruction distortion D is lower-bounded by the rate-distortion function R(D) of the feature hierarchy. This bound is tight: standard SAEs achieve rates within a constant factor of the bound, while Matryoshka SAEs and OrtSAE approach the bound from above by increasing M or constraining decoder geometry.

### Motivation

The SAE field is at a crossroads. On one hand, absorption has been identified as a fundamental failure mode (Chanin et al., 2024) and dozens of architectural solutions have been proposed (Matryoshka, OrtSAE, ATM, Balance Matryoshka, H-SAE). On the other hand, the "Sanity Checks" paper questions whether SAEs learn meaningful features at all, and Wang et al. (ICLR 2026) show weak interpretability-utility correlation. The field needs:

1. **A mechanistic understanding** of WHY absorption happens (not just THAT it happens).
2. **A principled framework** for determining whether absorption is avoidable or fundamental.
3. **Practical tools** for diagnosing and mitigating absorption without retraining.

The Local Inhibition Graph addresses (1) and (3). The IB bound addresses (2). Together they provide a complete theoretical-practical framework.

### Method

#### Part I: Local Inhibition Graph (Primary Contribution)

*As detailed in the existing proposal (cand_f):*

1. **Graph construction**: For each latent i, compute decoder correlations G_ij = <W_dec[i], W_dec[j]>, keep top-k neighbors.
2. **Validation**: Test precision@k against Chanin absorption pairs on first-letter features.
3. **Precision-recall asymmetry**: Correlate total incoming inhibition with recall loss (predicted: negative) and precision (predicted: none).
4. **Layer-dependent structure**: Compare graph statistics across layers 0/4/8/10.
5. **Homeostatic rebalancing**: z'_i = z_i + alpha * inh_i, with reconstruction constraint.

#### Part II: Information-Bottleneck Absorption Bound (Secondary Contribution)

1. **Synthetic hierarchy setup**: Use SynthSAEBench (Chanin et al., 2026) to generate ground-truth hierarchical features with controlled parent-child dependencies.
2. **Rate-distortion derivation**: For a source where child features imply parent features (e.g., "starts with SH" -> "starts with S"), derive the rate-distortion function R(D) under sparsity constraints. The key insight: at rate R < H(parent), the optimal codebook must merge parent information into child representations.
3. **Bound formulation**: Prove that absorption rate A >= f(R(D), M, sparsity) where M is dictionary size.
4. **Empirical validation**: Train standard SAEs, Matryoshka SAEs, and OrtSAE on the synthetic data. Measure absorption rates and compare against the theoretical bound.
5. **Real-LLM validation**: Test whether the bound predicts the observed absorption ordering: standard SAE > TopK > JumpReLU > Matryoshka ~ OrtSAE.

### Cross-Domain Insight

**From neuroscience (LCA)**: The Locally Competitive Algorithm shows that sparse coding with lateral inhibition naturally produces competitive suppression. The exact correspondence W_dec^T W_dec = G_LCA provides a mechanistic model for absorption.

**From information theory (rate-distortion)**: Shannon's rate-distortion theory shows that lossy compression of dependent sources necessarily merges information. The correspondence: SAE sparsity constraint -> rate limit; hierarchical features -> dependent source; absorption -> information merging.

The two insights are complementary: LCA explains the local mechanism (how features compete); rate-distortion explains the global necessity (why competition is unavoidable).

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time | Part |
|---|---|---|---|---|---|
| E1: Graph construction + validation | GPT-2 Small | gpt2-small-res-jb (24K) | Precision@k, recall@k, Fisher test | ~15 min | I |
| E2: Precision-recall asymmetry | GPT-2 Small | Same | Correlation (inhibition vs recall, precision) | ~15 min | I |
| E3: Layer-dependent structure | GPT-2 Small | Same (layers 0/4/8/10) | Graph stats by layer | ~20 min | I |
| E4: Homeostatic rebalancing | GPT-2 Small | Same | Absorption rate change, recon error | ~30 min | I |
| E5: Cross-model validation | Gemma-2-2B | GemmaScope 16K | All above | ~30 min | I |
| E6: Synthetic hierarchy generation | Toy model | SynthSAEBench | Ground-truth absorption pairs | ~15 min | II |
| E7: Rate-distortion bound fitting | Toy model | Standard/Matryoshka/OrtSAE | Absorption rate vs. theoretical bound | ~30 min | II |
| E8: Real-LLM bound validation | GPT-2 Small | Multiple architectures | Absorption ordering prediction | ~30 min | II |

**Total estimated time:** ~3.5 GPU-hours.

### Resource Estimate

| Item | Estimate |
|------|----------|
| GPU | Single 24GB GPU (RTX 3090/4090 or A10) |
| Graph construction | ~15 min per SAE |
| Validation experiments | ~30 min per SAE |
| Synthetic experiments | ~45 min total |
| Cross-model (Gemma) | ~30 min (if accessible) |
| Total GPU time | ~3.5 hours |
| Wall-clock | 1-2 days |
| Model sizes | GPT-2 Small (primary), Gemma-2-2B (validation), SynthSAEBench (theory validation) |
| Storage | <15GB |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Graph edges don't predict absorption | Medium | High | Fallback to diagnostic-only claims; IB bound stands independently |
| IB bound derivation fails | Medium | High | Fallback to inhibition graph alone; bound becomes conjecture in Discussion |
| IB bound is vacuous (loose) | Medium | Medium | Report as "first step toward information-theoretic understanding"; tighter bounds as future work |
| Synthetic-real gap | Medium | Medium | Validate bound ordering on real SAEs even if absolute values differ |
| Homeostatic rebalancing degrades recon | Medium | Medium | Alpha sweep; report Pareto frontier |
| Gemma-2-2B access issues | High | Low | Primary experiments on GPT-2 Small; Gemma as validation only |

### Novelty Claim

**What is new:**

1. **First LCA-SAE connection** (W_dec^T W_dec = G_LCA) — exact structural correspondence, not metaphorical.
2. **First local inhibition graph for SAE diagnostics** — training-free, scalable to million-latent SAEs.
3. **First mechanistic explanation for precision-recall asymmetry** — competitive suppression explains why absorption affects recall but not precision.
4. **First training-free post-hoc repair for absorption** — homeostatic rebalancing operates on pretrained SAEs.
5. **First information-theoretic lower bound on absorption** — reframes absorption from "fixable problem" to "fundamental limit."

**Prior art check:**
- Rozell et al. (2008): ~2000 citations, zero applications to LLM SAEs.
- Chanin et al. (2024): Identified absorption but did not connect to inhibition or information theory.
- Tang et al. (2025): Optimization-landscape analysis; no rate-distortion bound.
- Peter et al. (2025): Rate-distortion interpretations; no explicit absorption bound.
- Li et al. (2025): Geometric analysis; no TDA or inhibition framework.

**Differentiation from competing directions:**
- vs. Matryoshka/OrtSAE/ATM: These are architectural solutions. We provide theoretical understanding + diagnostic tools.
- vs. Tang et al. (2025): They analyze optimization landscape. We connect to neuroscience (LCA) and information theory (IB).
- vs. "Sanity Checks" (2026): They question SAE validity. We provide a framework that remains meaningful even if SAEs are imperfect.

---

## Synthesis: How This Perspective Integrates with the Project

The existing project has a strong front-runner (Local Inhibition Graph, cand_f) with five testable hypotheses (H6-H10). This Innovator perspective:

1. **Validates the front-runner**: The literature survey confirms no prior work on LCA-SAE connection or inhibition graphs. The novelty claim is solid.
2. **Identifies a complementary theoretical direction**: The IB absorption bound (Candidate C) provides global theoretical grounding that complements the local mechanistic insight of the inhibition graph.
3. **Recommends integration, not replacement**: The IB bound should be pursued as a secondary contribution alongside the inhibition graph, not as a replacement. The two frameworks are complementary.
4. **Flags risks**: The main risks are (a) the IB bound derivation may fail or be vacuous, and (b) the synthetic-real gap may invalidate bound validation. Mitigation: keep the bound as secondary; if it fails, the inhibition graph stands alone.

**Recommended action**: Proceed with cand_f (Local Inhibition Graph) as primary. Add E6-E8 (synthetic hierarchy + bound validation) as exploratory experiments. If the bound works, the paper gains a major theoretical contribution. If not, the inhibition graph contribution is unchanged.
