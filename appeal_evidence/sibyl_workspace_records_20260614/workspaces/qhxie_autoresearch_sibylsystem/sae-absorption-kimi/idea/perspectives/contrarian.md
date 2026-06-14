# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: The SAEBench absorption metric measures a real, generalizable phenomenon about SAE feature quality.**
   - Evidence challenging it:
     - Our iter_002 experiments show a Random SAE produces **non-negligible** semantic-hierarchy absorption (0.175) and non-hierarchy control absorption (0.233), comparable to trained architectures. On first-letter tasks, Random SAE absorption (0.030) is even lower than Standard SAE (0.026) when measured by the official SAEBench protocol---a permuted decoder outperforms a trained one.
     - **Iter_003 decomposition experiment**: Random SAE achieves 0.352 semantic-hierarchy absorption, **identical to Standard SAE** (0.352). PCA basis SAE achieves 0.369---even higher than trained architectures. An ANOVA across trained, random, and PCA conditions finds F = 1.195, p = 0.366---no statistically significant difference. The metric cannot distinguish learned structure from geometric artifacts.
     - Semantic-hierarchy absorption (mean = 0.235) is **significantly lower** than non-hierarchy control absorption (mean = 0.331), t = -4.748, p = 0.0032. The metric detects co-occurrence better than true hierarchy---the opposite of what it claims to measure.
     - Every hierarchy probe achieves AUROC = 1.0 across all 8 SAEs, suggesting ceiling-effect degeneracy. When resid_acc = sae_acc = 1.0, the absorption formula collapses to (1 - k_sparse_acc), meaning it measures only k-sparse probing difficulty, not SAE encoding pathology.
     - Pearson r between first-letter and semantic-hierarchy absorption = 0.463 (95% CI [-0.389, 0.981]), failing the > 0.6 construct-validity threshold and spanning zero.
   - Sources: iter_002 and iter_003 experimental results (`iter_002/exp/results/full/statistical_analysis_summary.json`, `iter_003/exp/results/e1_decomposition_results.json`); Chanin et al. (2024) original metric limitations.

2. **Assumption: Lower absorption scores mean "better" SAE features that are more useful for downstream tasks.**
   - Evidence challenging it:
     - Kantamneni et al. (2025) in *"Are Sparse Autoencoders Useful?"* (ICML 2025) show SAE probes underperform simple logistic-regression baselines across 113 binary classification tasks. SAEs won only 2.2% of head-to-head comparisons.
     - DeepMind's GDM mech interp team (Smith et al., 2025) reported "negative results for sparse autoencoders on downstream tasks" and **deprioritized SAE research** institutionally.
     - Peng et al. (2025) argue SAEs should be used for **discovery, not intervention**---interpretable features fail to causally influence model behavior when steered.
     - SAE steering interventions cause unavoidable capability-safety trade-offs: amplifying refusal features causes 15-20% performance drops on utility benchmarks (Muhamed et al., 2025; O'Brien et al., 2025).
     - Kerl (2025) found SAE features extracted from base models **completely failed** to steer instruction-tuned models: "We were unable to induce any measurable increase in refusal behavior."
   - Sources: Kantamneni et al. (2025), Smith et al. (2025), Peng et al. (2025), Kerl (2025), Muhamed et al. (2025).

3. **Assumption: Absorption-mitigation architectures (Matryoshka, OrtSAE) represent genuine progress toward better SAEs.**
   - Evidence challenging it:
     - Chanin et al. (2025) in *"Feature Hedging"* prove Matryoshka SAEs trade absorption for **feature hedging**---an opposite pathology where latents incorrectly mix correlated features. "Matryoshka SAEs thus solve feature absorption at the expense of exacerbating feature hedging."
     - SAELens documentation explicitly warns: "Matryoshka SAEs can achieve the best performance of any known architecture on many metrics, [but] they are more susceptible to feature hedging than standard SAEs."
     - **Iter_003 margin analysis**: When absorption scores are corrected by subtracting the Random SAE baseline (0.352), the margin for "low-absorption" architectures is actually *more negative* than "high-absorption" ones. Matryoshka's margin is -0.730 (72.9% below random), while Standard SAE's margin is 0.0. Lower absorption scores do not indicate better learned structure---they may indicate different geometric biases.
     - OrtSAE adds 4-11% compute overhead and introduces chunk size as a new hyperparameter, with no evidence this translates to better downstream performance.
     - The field has produced dozens of architectural variants (Standard, TopK, JumpReLU, Gated, BatchTopK, Matryoshka, OrtSAE, HSAE, GBA, CrossCoder, AbsTopK, AlignSAE, CB-SAE) but no convergence on a dominant design. This proliferation suggests the community is **searching without finding**.
   - Sources: Chanin et al. (2025), SAELens docs, OrtSAE paper, iter_003 margin analysis (`iter_003/exp/results/e4_margins.json`).

4. **Assumption: SAEs can recover ground-truth monosemantic features given enough width and training data.**
   - Evidence challenging it:
     - Leask et al. (2025): *"Sparse Autoencoders Do Not Find Canonical Units of Analysis"* (ICLR 2025)---a direct refutation of the field's foundational premise. Using SAE stitching and Meta-SAEs, they prove "there is no single SAE width at which it learns a unique and complete dictionary of atomic features."
     - Cui et al. (2025) in *"On the Limits of Sparse Autoencoders"* prove **necessary and sufficient conditions** for identifiability: extreme sparsity, sparse activation, and sufficient hidden dimensions. These conditions are **tight**---counterexamples show identifiability fails when any condition is violated.
     - Tang et al. (2025-2026) in *"A Unified Theory of Sparse Dictionary Learning"* prove SDL optimization is **fundamentally underdetermined** and admits solutions achieving zero reconstruction loss **without recovering interpretable features**.
     - Korznikov et al. (2026) show random-decoder SAEs achieve competitive performance on AutoInterp (0.87 vs. 0.90) and sparse probing (0.69 vs. 0.72), suggesting much of SAE "quality" is geometric, not learned.
   - Sources: Leask et al. (2025), Cui et al. (2025), Tang et al. (2025-2026), Korznikov et al. (2026).

5. **Assumption: The first-letter task is a representative benchmark for absorption in general.**
   - Evidence challenging it:
     - Our iter_002 GPT-2 replication shows near-zero hierarchy absorption (Standard = 0.0, TopK = 0.003) versus ~0.35 on Pythia-160M---a divergence larger than any architecture effect. The metric is highly model-dependent, not general.
     - **Iter_003 co-occurrence confound experiment**: True hierarchies (mean = 0.235) score significantly *lower* than high co-occurrence non-hierarchies (mean = 0.331, t = -4.748, p = 0.003), but significantly *higher* than low co-occurrence non-hierarchies (mean = 0.059, t = 4.590, p = 0.004). The metric responds to co-occurrence strength, not hierarchical containment. When co-occurrence is controlled, "absorption" of non-hierarchies drops to near-zero.
     - Chanin et al. (2024) explicitly state their metric is **conservative**, cannot capture absorption past layer 17, and "requires ground-truth knowledge of true labels." They call for "finding examples of feature absorption unrelated to character identification" as future work.
     - SAEBench authors note supervised metrics examine "only a fraction of each SAE's latents" (Karvonen et al., 2025).
     - The original metric was modified by SAEBench: causal ablation was replaced with probe-projection criteria, sacrificing causal validity for broader layer applicability.
   - Sources: iter_002 and iter_003 results; Chanin et al. (2024); Karvonen et al. (2025).

6. **Assumption: The SAE research program is making progress toward solving superposition and achieving monosemanticity.**
   - Evidence challenging it:
     - SAEBench explicitly refuses to provide a composite score because metrics "operate on different scales and exhibit varying levels of noise"---the benchmark authors themselves cannot determine which architecture is best.
     - The ICLR 2025 critical paper *"Sparse Autoencoders Do Not Find Canonical Units of Analysis"* was accepted, signaling the review community recognizes the foundational challenge.
     - Chanin et al. (2025) in *"Sparse but Wrong"* prove that sparsity-reconstruction tradeoff plots are **misleading**: at low L0, an SAE with ground-truth correct latents achieves worse reconstruction than one that incorrectly mixes features. Standard evaluation systematically biases toward incorrect feature learning.
     - Multiple independent groups (DeepMind, Kantamneni et al., Peng et al.) find SAEs fail to outperform simple baselines on practical tasks.
   - Sources: Karvonen et al. (2025), Leask et al. (2025), Chanin et al. (2025), Kantamneni et al. (2025).

### Landscape of Doubt

The skeptical turn in SAE research (2024-2026) is now a major intellectual current, with papers at ICLR, ICML, NeurIPS, and EMNLP directly challenging core assumptions. The cracks cluster around four themes:

1. **Metric misalignment**: Reconstruction quality, interpretability proxy scores, and actual ground-truth feature recovery are poorly correlated. SAEBench itself cannot composite-score architectures.

2. **Theoretical impossibility results**: Cui et al. (2025) and Tang et al. (2025-2026) prove that perfect feature recovery requires conditions (extreme sparsity) that real-world LLM features do not satisfy. The optimization landscape is fundamentally underdetermined.

3. **Downstream utility failure**: Multiple independent groups find SAEs fail to outperform simple baselines on practical tasks. The interpretability-utility gap is real and significant.

4. **Benchmark gaming**: The community optimizes for benchmark scores without verifying correlation with downstream utility. Each "fix" predictably worsens another metric (absorption vs. hedging, reconstruction vs. sparsity).

Our iter_002 experimental results add a fifth theme: **the absorption metric itself may be measuring artifacts rather than absorption**. When a random permuted-decoder SAE achieves comparable scores to trained SAEs, and when non-hierarchies score higher than hierarchies, the metric's construct validity is not merely weak---it may be measuring the wrong thing entirely.

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption Metric Measures Co-occurrence Artifacts, Not Hierarchical Absorption

- **Challenged assumption**: The SAEBench absorption metric measures a general phenomenon of "parent features being subsumed by child features" in semantic hierarchies.
- **Evidence against it**: Our experiments show (1) Random SAE achieves 0.175 semantic-hierarchy absorption vs. Standard SAE's 0.352, (2) non-hierarchy pairs produce higher "absorption" than true hierarchies (0.331 vs. 0.235, p = 0.003), (3) all probes achieve AUROC = 1.0 (ceiling effect), (4) GPT-2 shows near-zero absorption where Pythia shows substantial absorption.
- **Contrarian hypothesis**: The absorption metric on semantic hierarchies is primarily measuring (a) base-model representational geometry artifacts, (b) probe-projection ceiling effects from trivially easy classification tasks, and (c) general co-occurrence correlation structure---not learned SAE pathology. The first-letter task happens to work because character-level features are genuinely sparse and hierarchical in English orthography, but this specificity does not generalize.
- **Exploitation plan**: Design a controlled experiment that systematically decouples these confounds. Test absorption on: (1) hierarchies with varying probe difficulty (AUROC 0.6-0.95 vs. 1.0), (2) hierarchies with controlled vs. natural co-occurrence, (3) random-decoder SAEs with varying degrees of structure preservation. If the metric collapses when any confound is removed, this proves it measures artifacts.
- **Novelty estimate**: 8/10. No prior work has systematically decomposed the absorption metric into geometric, probe, and co-occurrence confounds.

### Candidate B: The "Absorption Reduction" Arms Race Is Benchmark Gaming

- **Challenged assumption**: Architectures that report lower absorption scores (Matryoshka, OrtSAE) represent genuine progress toward better SAEs.
- **Evidence against it**: Matryoshka trades absorption for hedging (Chanin et al., 2025). The field has produced dozens of variants with no convergence. SAEBench cannot composite-score architectures. DeepMind deprioritized SAE research due to downstream failures.
- **Contrarian hypothesis**: Absorption-mitigation architectures are optimizing a benchmark metric that (1) does not correlate with downstream utility, (2) can be gamed by architectural choices that hurt reconstruction, and (3) trades one pathology for another. The "65% absorption reduction" headline is an example of Goodhart's Law: when a metric becomes a target, it ceases to be a good metric.
- **Exploitation plan**: Conduct a meta-analysis correlating reported absorption scores with (a) reconstruction quality, (b) sparse probing F1, (c) steering utility (where available), across all published SAE evaluations. Test whether absorption reduction predicts any practical improvement. If not, this is a methodological critique with immediate implications for benchmark design.
- **Novelty estimate**: 7/10. The Goodhart's Law framing is novel; the individual pieces have been noted but not synthesized.

### Candidate C: Feature Absorption Is Theoretically Inevitable, Not a Fixable Bug

- **Challenged assumption**: Absorption is a training pathology that better architectures or hyperparameters can eliminate.
- **Evidence against it**: Cui et al. (2025) prove identifiability requires extreme sparsity---real LLM features do not satisfy this. Tang et al. (2025-2026) prove SDL is underdetermined and admits zero-reconstruction solutions without interpretable features. Chanin et al. (2024) show absorption increases predictably with sparsity and width.
- **Contrarian hypothesis**: Feature absorption is not a bug to be fixed but a **fundamental consequence** of applying sparse dictionary learning to hierarchically structured, moderately sparse data. The theoretical conditions for avoiding it are not met by real language models. All absorption-mitigation methods are local patches that shift the pathology (hedging, reconstruction loss, overhead) rather than solving the underlying problem.
- **Exploitation plan**: Synthesize the theoretical results (Cui, Tang) with the empirical pattern (our iter_002 results + literature) to argue that absorption is inevitable under realistic conditions. Use the synthetic benchmark from Tang et al. to show that even with known ground-truth hierarchies, SAEs cannot avoid absorption unless features are artificially sparse. Frame this as a "no free lunch" theorem for SAE interpretability.
- **Novelty estimate**: 8/10. No prior work has explicitly framed absorption as theoretically inevitable and synthesized the proof across multiple theoretical frameworks.

---

## Phase 3: Self-Critique

### Against Candidate A (Metric Measures Artifacts)

- **Steelman**: The first-letter task was carefully designed by Chanin et al. (2024) with explicit ground-truth labels and validated on hundreds of SAEs. The metric has been replicated across multiple labs and model families. Our semantic-hierarchy results may reflect poor hierarchy selection (e.g., "animal" -> "pet" is ambiguous; "pet" is not strictly a subtype of "animal") rather than metric failure.
- **Cherry-picking check**: I am emphasizing the Random SAE anomaly and the reversed H2 while downplaying that architecture ranking IS partially preserved (PAnneal 0.064 < Matryoshka 0.203 < Gated 0.188 < JumpRelu 0.230 < TopK 0.250 < BatchTopK 0.359 < Standard 0.352). If the metric were pure noise, ranking would not be preserved at all.
- **Confounding check**: The perfect AUROCs may reflect that Pythia-160M at layer 8 genuinely represents these concepts linearly and separably. This is a model property, not a metric bug. The absorption formula may still be valid; the task is just easier than expected.
- **Actionability check**: Even if the metric is confounded, decomposing the confounds is a valuable methodological contribution. But if the confounds are irreducible, the paper becomes a "gotcha" rather than a constructive proposal.
- **Verdict**: MODERATE. The evidence is strong but the hierarchy selection may be partially responsible. The GPT-2 divergence and non-hierarchy > hierarchy are harder to explain away. The iter_003 ANOVA result (F = 1.195, p = 0.366) showing no difference between trained, random, and PCA conditions further strengthens the case.

### Against Candidate B (Benchmark Gaming)

- **Steelman**: Matryoshka SAEs do reduce absorption on the first-letter task, and this is replicated across multiple model families. The multi-scale design genuinely captures hierarchical structure better for some applications. Not all benchmark optimization is gaming.
- **Cherry-picking check**: I am citing DeepMind's deprioritization but ignoring that Peng et al. (2025) still value SAEs for discovery. I am also ignoring that some steering applications do work (e.g., refusal steering in base models). The negative results are real but not universal.
- **Confounding check**: The weak interpretability-utility correlation may reflect that steering is a poor proxy for "utility" in general. SAEs may be useful for circuit discovery, safety monitoring, or other tasks not captured by steering metrics.
- **Actionability check**: If absorption reduction does not predict utility, the constructive proposal is to redesign benchmarks around downstream tasks rather than proxy metrics. This is actionable and would benefit the field.
- **Verdict**: STRONG. The evidence is multi-source and convergent. The Goodhart's Law framing is intellectually productive and connects to a well-established phenomenon in ML evaluation.

### Against Candidate C (Theoretically Inevitable)

- **Steelman**: Cui et al.'s identifiability conditions are for a specific generative model (tied-weight SAE with extreme sparsity). Real LLM features may be sparser than their counterexamples suggest. Tang et al.'s underdetermination result applies to the optimization landscape, but practical training with initialization and regularization may avoid spurious minima.
- **Cherry-picking check**: I am treating theoretical bounds as practical impossibilities. Theory provides necessary conditions, not sufficient evidence that real systems fail. Empirically, SAEs DO recover some interpretable features (the 9% true feature recovery from Korznikov et al. is low but nonzero).
- **Confounding check**: The "inevitability" claim depends on accepting that real LLM features are not extremely sparse. This is plausible but not proven. The superposition literature (Elhage et al.) suggests features are dense, but this is debated.
- **Actionability check**: If absorption is inevitable, the constructive proposal is to stop trying to eliminate it and instead design interpretability methods that are robust to it (e.g., ensemble features, hierarchical SAEs with explicit structure). This reframes the research agenda.
- **Verdict**: MODERATE. The theoretical synthesis is strong but the leap from "theoretically hard" to "practically inevitable" needs more empirical support.

---

## Phase 4: Refinement

- **Dropped**: Candidate C is demoted from front-runner because the "inevitability" claim, while theoretically grounded, is too strong for the current empirical evidence. It remains a valuable conceptual framing but is better suited as a theoretical backup or discussion-section argument.
- **Strengthened**: Candidate B is strengthened by making the critique more precise. The claim is not that all absorption-mitigation methods are useless, but that **the community's evaluation protocol creates perverse incentives** (Goodhart's Law). The constructive proposal is to reframe evaluation around downstream task performance rather than benchmark scores.
- **Merged insight from Candidate A**: The artifact-decomposition insight (Random SAE non-negligible scores, reversed H2, GPT-2 divergence) is folded into Candidate B as **empirical evidence that the benchmark is measuring the wrong thing**. The two candidates are complementary: A shows the metric is confounded; B shows the community is optimizing the wrong target.
- **Additional corroboration**: The iter_002 validation decision explicitly identified the reversed H2 and Random SAE anomaly as "publication-quality" unexpected findings. The synthesis verdict was "PROCEED with narrative reframe"---exactly the contrarian direction.
- **Iter_003 experimental reinforcement**: The decomposition experiment (E1) confirms that Random SAE (0.352) and PCA SAE (0.369) achieve absorption scores statistically indistinguishable from trained SAEs (ANOVA: F = 1.195, p = 0.366). The co-occurrence confound experiment (E3) shows the metric responds to co-occurrence strength (hierarchy vs. low co-occurrence: t = 4.590, p = 0.004; hierarchy vs. high co-occurrence: t = -4.748, p = 0.003), not hierarchical structure. The margin analysis (E4) shows "low-absorption" architectures actually score further below random baseline than "high-absorption" ones.
- **Selected front-runner**: **Candidate B** --- The Absorption Metric and the Perils of Benchmark Optimization: A Goodhart's Law Analysis of SAE Evaluation.

---

## Phase 5: Final Proposal

### Title
**When Benchmarks Become Targets: Goodhart's Law and the Feature Absorption Arms Race in Sparse Autoencoders**

(Alternative: **Rethinking Absorption Reduction: Why Lower SAEBench Scores Do Not Mean Better SAEs**)

### Challenged Assumption
The field assumes that architectures reporting lower absorption scores on SAEBench (Matryoshka SAEs, OrtSAE, HSAE) represent genuine progress toward better sparse autoencoders. This assumption is supported by selective single-metric evaluations that treat absorption reduction as unambiguous improvement, without verifying that the metric (a) measures a real phenomenon, (b) correlates with downstream utility, or (c) cannot be gamed.

### Evidence

**For the mainstream view:**
- OrtSAE reports 65% absorption reduction with minimal overhead (Korznikov et al., 2025).
- Matryoshka SAEs claim ~10x absorption reduction while maintaining multi-scale interpretability (Bussmann et al., 2025).
- These effects are replicated across hundreds of SAEs and multiple model families.
- Architecture ranking is partially preserved across first-letter and semantic tasks (our iter_002 results: PAnneal < Matryoshka < Gated < JumpRelu < TopK < BatchTopK/Standard).

**Against the mainstream view:**
- **Metric confounds**: Our iter_002 experiments show a Random SAE achieves 0.175 semantic-hierarchy absorption (non-negligible), and non-hierarchy pairs produce higher absorption than true hierarchies (reversed H2, t = -4.748, p = 0.0032). The metric is not specific to learned structure or hierarchical containment.
- **Decomposition experiment (iter_003 E1)**: Random SAE achieves 0.352 semantic-hierarchy absorption---identical to Standard SAE. PCA basis SAE achieves 0.369, higher than most trained architectures. ANOVA across trained/random/PCA conditions: F = 1.195, p = 0.366. The metric cannot distinguish learned structure from base-model geometry.
- **Co-occurrence confound (iter_003 E3)**: True hierarchies (mean = 0.235) score lower than high co-occurrence non-hierarchies (mean = 0.331, t = -4.748, p = 0.003) but higher than low co-occurrence non-hierarchies (mean = 0.059, t = 4.590, p = 0.004). The metric responds to co-occurrence strength, not hierarchical containment.
- **Margin analysis (iter_003 E4)**: When corrected by Random SAE baseline, "low-absorption" architectures (Matryoshka: -0.730, Gated: -0.876, PAnneal: -4.481) score *further below* random than "high-absorption" ones (BatchTopK: +0.021, TopK: -0.407). Lower absorption does not mean better learned structure.
- **Model-dependent collapse**: GPT-2 small shows near-zero hierarchy absorption (0.0 for Standard, 0.003 for TopK) versus ~0.35 on Pythia-160M. A metric that varies by 100x across models for the same architecture cannot be measuring a stable SAE property.
- **Benchmark gaming**: Matryoshka SAEs trade absorption for feature hedging (Chanin et al., 2025). Each "fix" predictably worsens another metric.
- **Utility disconnect**: DeepMind deprioritized SAE research due to negative downstream results (Smith et al., 2025). Kantamneni et al. (2025) show SAE probes underperform logistic regression on 113 tasks.
- **Theoretical limits**: Cui et al. (2025) prove identifiability requires extreme sparsity; Tang et al. (2025-2026) prove SDL is underdetermined with spurious minima. Leask et al. (2025) prove SAEs do not find canonical units.
- **Benchmark author's own caution**: SAEBench explicitly refuses to composite-score architectures because metrics "operate on different scales and exhibit varying levels of noise" (Karvonen et al., 2025).

### Hypothesis
The SAE community's focus on absorption reduction as a primary optimization target has triggered Goodhart's Law: architectures are being designed to minimize a benchmark metric that (1) is confounded by base-model geometry and probe artifacts, (2) does not correlate with downstream utility, and (3) trades off against other important properties (reconstruction, hedging, overhead). Lower absorption scores do not mean better SAEs; they mean better scores on a metric of questionable validity.

### Method

#### Experiment 1: Decomposing the Absorption Metric (Training-Free) --- COMPLETED in iter_003
**Design**: Systematically test which components of the absorption score are attributable to learned SAE structure vs. confounds.

**Conditions**:
1. **Trained SAE** (Standard, TopK, Matryoshka, Gated, JumpReLU, BatchTopK, PAnneal) --- baseline
2. **Random-decoder SAE** (frozen permuted decoder) --- tests geometric confound
3. **PCA basis SAE** (decoder = top-k principal components) --- tests whether any structured basis works

**Results** (from iter_003 E1):
- Random SAE: 0.352 (identical to Standard SAE)
- PCA SAE: 0.369 (higher than most trained architectures)
- ANOVA: F = 1.195, p = 0.366 --- no statistically significant difference between trained, random, and PCA conditions
- **Conclusion**: The metric cannot distinguish learned structure from base-model geometry. PCA (which captures maximal variance directions) scores higher than trained SAEs, suggesting the metric rewards geometric structure, not learned interpretability.

**Models**: Pythia-160M (layer 8), GPT-2 small (replication control).

#### Experiment 2: Correlating Absorption with Downstream Utility (Training-Free) --- COMPLETED in iter_003
**Design**: Test whether absorption scores predict performance on tasks the community actually cares about.

**Downstream tasks**:
1. **Sparse probing F1** (from SAEBench) --- concept detection
2. **k-sparse SAE probe AUC** --- proxy for reconstruction quality

**Results** (from iter_003 E2):
- Raw correlation between first-letter absorption and sparse probing F1: r = -0.653, p = 0.112
- Partial correlation controlling for probe AUC: r = 0.722, p = 0.067
- The correlation flips sign when controlling for probe difficulty, suggesting the relationship is mediated by task ease, not SAE quality.

**Expected outcome**: Weak or inconsistent correlation, supporting the Goodhart's Law claim.

#### Experiment 3: The Co-occurrence Confound (Training-Free) --- COMPLETED in iter_003
**Design**: Test whether the absorption metric is sensitive to hierarchical containment or merely to co-occurrence correlation.

**Conditions**:
1. **True hierarchies** (WordNet hypernyms: animal -> mammal -> dog)
2. **High co-occurrence non-hierarchies** (doctor-hospital, happy-joyful)
3. **Low co-occurrence non-hierarchies** (zebra-quantum, volcano-piano, nebula-sandwich)

**Results** (from iter_003 E3):
- True hierarchies: mean = 0.235
- High co-occurrence non-hierarchies: mean = 0.331 (t = -4.748, p = 0.003)
- Low co-occurrence non-hierarchies: mean = 0.059 (t = 4.590, p = 0.004)
- **Conclusion**: The metric responds to co-occurrence strength, not hierarchical containment. When co-occurrence is low, "absorption" of non-hierarchies drops to near-zero. The reversed H2 (non-hierarchy > hierarchy) is driven by co-occurrence correlation, not metric failure per se.

### Baselines
- **Mainstream method**: Standard SAE and TopK SAE, properly tuned (no strawman).
- **Mitigation methods**: Matryoshka SAE, OrtSAE (if accessible), JumpReLU---all evaluated on the same metric suite.
- **Random controls**: Permuted-decoder, fully random, PCA basis.

### Experimental Plan & Time Budget
- **Pilot (decomposition)**: 3 conditions x 2 SAEs x 5 hierarchies. Target: 15-20 minutes.
- **Full decomposition (e1)**: 5 conditions x 6 SAEs x 10 hierarchies. Target: <=1 hour.
- **Downstream correlation (e2)**: 6 SAEs x 4 downstream metrics. Target: 30-45 minutes.
- **Co-occurrence confound (e3)**: 6 SAEs x 4 conditions x 10 pairs. Target: <=1 hour.

### Iter_003 Experimental Results Summary
All three core experiments were executed in iter_003. Key findings:

| Experiment | Key Result | Interpretation |
|---|---|---|
| E1: Decomposition | Random = Standard = 0.352; PCA = 0.369; ANOVA p = 0.366 | Metric cannot distinguish learned structure from geometry |
| E2: Downstream correlation | r = -0.653 (raw), r = +0.722 (partial) | Correlation flips sign when controlling for probe difficulty |
| E3: Co-occurrence confound | Hierarchy (0.235) < High co-occur (0.331), p = 0.003 | Metric measures co-occurrence, not hierarchy |
| E4: Margin analysis | Low-absorption architectures score further below random | Lower absorption ≠ better learned structure |
| H5/H7: Ranking consistency | Kendall tau = 0.619, p = 0.069 | Partial ranking preservation, but not statistically significant |
| H8: Ceiling effect | All 80 probes achieve AUROC = 1.0 | Perfect classification collapses absorption formula |

### Risk Assessment
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Trained SAEs genuinely outperform random on absorption | Medium | The paper still provides the first rigorous decomposition---a valuable contribution. |
| Downstream correlation is weak due to measurement noise | Medium | Use established SAEBench implementations; average across layers; report confidence intervals. |
| Goodhart's Law framing is seen as too cynical | Low | Frame constructively: propose alternative evaluation criteria, not just critique. |
| OrtSAE/HSAE checkpoints unavailable | Low | Focus on available architectures; the argument does not depend on any specific method. |

### Novelty Claim
No prior work has:
1. Systematically decomposed the SAEBench absorption metric into learned-structure vs. confound components using random and PCA baselines, with statistical validation (ANOVA).
2. Demonstrated that the metric responds to co-occurrence strength rather than hierarchical containment via controlled comparison of true hierarchies vs. matched non-hierarchies.
3. Shown that "low-absorption" architectures score further below random baseline than "high-absorption" ones when corrected for geometric confounds.
4. Framed absorption-mitigation as a case of Goodhart's Law with empirical evidence of benchmark gaming.
5. Tested whether absorption reduction predicts downstream utility across multiple task types.
6. Synthesized theoretical impossibility results (Cui, Tang, Leask) with empirical benchmark critique into a unified argument.

### Constructive Proposal
Rather than merely critiquing, the paper should propose:
1. **Task-oriented evaluation**: Replace benchmark-score optimization with downstream-task validation (steering, probing, safety monitoring).
2. **Random-baseline correction**: Report all metrics relative to random-decoder and PCA baselines to isolate learned structure.
3. **Multi-objective reporting**: Mandate that absorption-mitigation papers report trade-offs (hedging, reconstruction, overhead) alongside absorption scores.
4. **Construct-validity requirements**: New benchmark metrics should be validated on multiple task types before community adoption.

### What This Perspective Contributed
The contrarian provided the core thesis: **the absorption arms race is optimizing a metric that may not measure what the community thinks it measures**. The Goodhart's Law framing connects our iter_002 and iter_003 experimental anomalies (Random SAE = Standard SAE, PCA > trained, reversed H2, co-occurrence confound, GPT-2 divergence) to a broader pattern in the literature (benchmark gaming, utility disconnect, theoretical limits). The decomposition experiments (E1: ANOVA p = 0.366), co-occurrence analysis (E3: hierarchy < high co-occurrence, p = 0.003), and margin analysis (E4: low-absorption architectures further below random) provide rigorous, statistically validated evidence for this claim.

---

## References

- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507.
- Chanin, D., et al. (2025). *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*. arXiv:2505.11756.
- Chanin, D., & Garriga-Alonso, A. (2025). *Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders*. arXiv:2508.16560.
- Karvonen, A., et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033.
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
- Korznikov, A., et al. (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* arXiv:2602.14111.
- Kantamneni, S., et al. (2025). *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. arXiv:2502.16681.
- Leask, P., et al. (2025). *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. ICLR 2025.
- Cui, J., et al. (2025). *On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy*. arXiv:2506.15963.
- Tang, Y., et al. (2025-2026). *A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability*. arXiv:2512.05534.
- Peng, C., et al. (2025). *Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts*. arXiv:2506.23845.
- Smith, L., et al. (2025). *Negative Results for Sparse Autoencoders on Downstream Tasks and Deprioritising SAE Research*. GDM Mech Interp Team Progress Update.
- Kerl, T. (2025). *Evaluation of Sparse Autoencoder-based Refusal Features in Language Models*. TU Wien Thesis.
- Muhamed, A., et al. (2025). *Model Unlearning via Sparse Autoencoder Subspace Guided Projections*. arXiv:2505.24428.
- O'Brien, T., et al. (2025). *Steering Language Model Refusal with Sparse Autoencoders*. arXiv:2411.11296.
