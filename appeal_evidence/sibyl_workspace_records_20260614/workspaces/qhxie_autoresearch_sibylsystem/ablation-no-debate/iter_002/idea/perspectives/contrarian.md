# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: Feature absorption is a well-defined, measurable phenomenon that can be systematically quantified across SAEs.
   - Evidence challenging it: The primary absorption rate metric (Chanin et al., 2024/2025) is limited to layers 0-17 in Gemma 2 2B because ablation experiments require verifiable mediation; deep attention layers move information to final token positions, making ablation effects disappear. The metric is a conservative underestimate that misses multi-latent absorption and partial absorption cases. SAEBench adapted it using probe projection criteria, trading causal certainty for broader applicability.
   - Sources: Chanin et al. (arXiv:2409.14507), SAEBench (arXiv:2503.09532), workspace reflection findings on absorption metric limitations.

2. **Assumption**: SAEs find "true" or "canonical" features that correspond to mechanistically meaningful units of analysis in LLMs.
   - Evidence challenging it: Leask et al. (ICLR 2025) directly show that "Sparse Autoencoders Do Not Find Canonical Units of Analysis." Korznikov et al. (2026) find that fully-trained SAEs barely beat random baselines on interpretability (0.90 vs 0.87), sparse probing (0.72 vs 0.69), and causal editing (0.72 vs 0.73, where baseline is actually better). On synthetic data with known ground-truth features, SAEs achieved 71% explained variance but recovered only 9% of true features.
   - Sources: Leask et al. (arXiv:2502.04878), Korznikov et al. (arXiv:2602.14111).

3. **Assumption**: Training-free analysis of pretrained SAEs (like Gemma Scope) provides reliable absorption detection without retraining.
   - Evidence challenging it: The workspace's own EDA (encoder-decoder asymmetry) metric achieved AUROC of only 0.468 at L12-65k and 0.336 (reversed direction) on GPT2-L10. The go/no-go gate was NO_GO (2/6 configurations passed). With n_pos=16 out of 65,536 latents (0.024% prevalence), AUROC is statistically unreliable. The pilot AUROC of 0.853 at enriched prevalence collapsed to 0.468 at full scale -- a predictable consequence of evaluation design, not just label noise.
   - Sources: Workspace reflection findings (iter_001), SAEBench evaluation methodology critiques.

4. **Assumption**: Feature absorption is primarily a "learned" phenomenon that training refines, and therefore studying trained SAEs reveals something about how LLMs represent hierarchical features.
   - Evidence challenging it: The workspace's own pilot found that trained SAE absorption (0.50) was statistically indistinguishable from shuffled features baseline (0.487) and permuted encoder baseline (0.484). Only the random decoder baseline (0.059) differed. The absorption metric showed ZERO variance (std=0.0) across 100 samples, indicating it is effectively a deterministic geometric computation rather than a learned phenomenon. This suggests absorption may be an artifact of decoder geometry, not a discovery about LLM representations.
   - Sources: Workspace critic findings (iter_001), pilot data from multichild_absorption.json.

5. **Assumption**: SAE-based interpretability research has practical downstream utility for safety, steering, and model editing.
   - Evidence challenging it: DeepMind's mechanistic interpretability team formally deprioritized SAE research based on negative downstream task results (Smith et al., 2025). Wu et al. (ICML 2025) found simple difference-in-means baselines achieve 0.942 AUROC on concept detection versus 0.695 for vanilla SAEs. Peng et al. (2025) argue SAEs should be used to "discover unknown concepts, not to act on known concepts." Kantamneni et al. (2025) and Farrell et al. (2024) report negative results on concept detection and unlearning.
   - Sources: Smith et al. (DeepMind, 2025), Wu et al. (arXiv:2501.17148), Peng et al. (arXiv:2506.23845).

6. **Assumption**: The field's proposed solutions to absorption (Matryoshka SAE, HSAE, OrtSAE, AdaptiveK) represent genuine progress that will eventually solve the problem.
   - Evidence challenging it: Marks et al. (2025/2026) provide a unified theoretical framework showing piecewise biconvexity and spurious minima explain why simple linear encoders are hard to outperform -- suggesting architectural innovations may be fighting fundamental optimization landscape properties. The workspace's own H3 steering experiment was broken (baseline exactly equaled steered mean to full double precision), yet the paper drew causal conclusions from it. "Feature invention" (Till, 2024) shows SAEs may increase sparsity by inventing features rather than discovering real ones.
   - Sources: Marks et al. (arXiv:2512.05534), Till (arXiv:2508.09363), workspace critic findings.

7. **Assumption**: Reconstruction quality (MSE, explained variance) is a meaningful proxy for feature quality and interpretability.
   - Evidence challenging it: Gurnee (Alignment Forum, 2024) found SAE reconstructions cause 2-4.5x worse next-token prediction than random points at the same distance -- "the proxy reconstruction objective is behaving pathologically." SAEBench revealed that proxy metrics do not reliably predict practical interpretability. Matching Pursuit SAEs exploit superposition noise to improve reconstruction without learning true features (SynthSAEBench, 2026).
   - Sources: Gurnee (Alignment Forum, 2024), SAEBench (arXiv:2503.09532), SynthSAEBench (arXiv:2602.14687).

8. **Assumption**: The community's move toward quantitative metrics (absorption rate, feature sensitivity, MCC) represents maturation of the field.
   - Evidence challenging it: CE-Bench (arXiv:2509.00691) notes LLM-based evaluations suffer from "prompt sensitivity, generation noise, and resource overhead." Feature sensitivity (Hu et al., 2025) requires LLM-generated similar text, is computationally expensive, and needs human validation. The workspace's own reflection found that AUPRC (not AUROC) is the more honest metric at low prevalence -- yet the field predominantly reports AUROC.
   - Sources: CE-Bench (arXiv:2509.00691), Hu et al. (arXiv:2509.23717), workspace reflection findings.

### Landscape of Doubt

The SAE field in 2025-2026 is experiencing what can only be described as a validation crisis. The foundational assumption -- that sparse autoencoders decompose neural network activations into meaningful, mechanistically relevant features -- is under attack from multiple directions simultaneously:

- **Theoretical**: SAEs do not find canonical units (Leask et al.); the optimization landscape has spurious minima (Marks et al.); some features are inherently non-linear and cannot be captured by linear dictionaries.
- **Empirical**: SAEs barely beat random baselines (Korznikov et al.); reconstruction is pathological (Gurnee); simple linear baselines outperform for steering and detection (Wu et al., Kantamneni et al.).
- **Practical**: Major research teams have deprioritized SAEs (DeepMind); downstream task utility is contested; evaluation metrics are noisy and potentially misleading.
- **Methodological**: The absorption metric itself is limited, conservative, and expensive; training-free proxies fail at scale; prevalence imbalance makes standard metrics unreliable.

The most dangerous assumption in the current project is that we can do "training-free analysis" that yields meaningful, publishable insights about absorption. The evidence suggests that without causal ablation (which only works early layers) or ground-truth labels (which only exist in synthetic settings), we may be measuring geometric artifacts rather than genuine absorption phenomena.

---

## Phase 2: Initial Candidates

### Candidate A: "The Absorption Metric Measures Decoder Geometry, Not Learning"

- **Challenged assumption**: Feature absorption is a learned phenomenon that reveals how SAEs (and by extension, LLMs) represent hierarchical features.
- **Evidence against it**: Workspace pilot data shows trained SAE absorption = 0.50, shuffled features = 0.487, permuted encoder = 0.484 -- all statistically indistinguishable. Only random decoder differs (0.059). Absorption shows ZERO variance (std=0.0) across samples. Korznikov et al. (2026) show SAE decoders remain near random initialization ("lazy training regime").
- **Contrarian hypothesis**: What we call "absorption" is primarily a geometric property of how decoder directions are arranged in high-dimensional space. When parent decoder directions lie near the convex hull of child decoder directions, the encoder naturally routes activation through children. Training barely changes this geometry -- it only fine-tunes encoder alignments to exploit the pre-existing decoder structure.
- **Exploitation plan**: Design experiments that explicitly separate geometric from learned contributions. The workspace already has a factorial design (random/trained encoder x random/trained decoder) that could test this. If random decoder + trained encoder shows absorption rates comparable to fully trained SAEs, the "learned" narrative collapses. Publish as "Absorption in Sparse Autoencoders: A Geometric Artifact, Not a Learned Phenomenon."
- **Novelty estimate**: 8/10. Directly challenges the foundational paper (Chanin et al.) by showing their phenomenon may not be about learning at all.

### Candidate B: "The Field's Training-Free Absorption Detection is Fundamentally Flawed"

- **Challenged assumption**: We can detect and measure absorption in pretrained SAEs without causal ablation or ground-truth labels.
- **Evidence against it**: EDA (the workspace's training-free metric) failed on 4/6 Gemma Scope configurations (AUROC < 0.65) and reversed direction on GPT-2. At 0.024% positive prevalence, AUROC is unreliable. The pilot-to-full AUROC collapse (0.853 to 0.468) was predictable. SAEBench's adapted absorption metric trades causal certainty for applicability.
- **Contrarian hypothesis**: All training-free absorption proxies (encoder-decoder asymmetry, sensitivity-based screening, attention-flow methods) are measuring correlated but causally irrelevant geometric properties. They work when absorption is severe enough to produce large geometric signals, but fail for subtle or partial absorption -- which may be the majority of cases.
- **Exploitation plan**: Systematically evaluate all proposed training-free proxies against a ground-truth synthetic benchmark where absorption is controlled. Show that each proxy has a specific "blind spot" -- e.g., EDA misses partial absorption, sensitivity misses non-activating absorbed features. Quantify the false negative rate. Publish as "The Illusion of Training-Free Absorption Detection: Why Post-Hoc Metrics Fail."
- **Novelty estimate**: 7/10. Addresses a gap the field is actively trying to solve, but argues the gap is unbridgeable rather than solvable.

### Candidate C: "Safety-Critical Feature Absorption is an Overblown Concern"

- **Challenged assumption**: Safety-critical features are disproportionately absorbed, creating hidden risks for SAE-based safety monitoring.
- **Evidence against it**: This was the workspace's highest-novelty candidate (cand_safe, 9/10), but it was never actually tested. The assumption that rare features are more absorbed comes from Chanin et al.'s toy model, but the workspace's own H2 found a POSITIVE correlation between frequency and absorption (rho=+0.171) -- opposite to the expected direction. The "competitive exclusion" framing was deprecated. Feature sensitivity declines with SAE width (Hu et al.), suggesting the problem is architectural, not safety-specific.
- **Contrarian hypothesis**: The concern about safety-critical absorption is theoretically motivated but empirically unsupported. Rare features may actually be LESS absorbed because they have fewer co-occurring child features to steal their direction. The real safety issue is not absorption but the fundamental unreliability of SAE features for any downstream task (as shown by DeepMind's deprioritization and Wu et al.'s baselines).
- **Exploitation plan**: Test whether safety-critical features are actually more absorbed than matched non-safety features. If the null result holds, reframe the paper: "Safety-Critical Features Are Not Uniquely Vulnerable to Absorption: Rethinking the Risk Narrative." This would be provocative but grounded -- it challenges a narrative that has been repeated without evidence.
- **Novelty estimate**: 6/10. High impact if correct, but risks being seen as a "gotcha" paper if not carefully framed.

---

## Phase 3: Self-Critique

### Against Candidate A (Geometric Artifact)

- **Steelman**: Chanin et al. validated absorption on hundreds of LLM SAEs across multiple architectures and sizes. The toy model explicitly proves that hierarchy + sparsity optimization causes absorption. Matryoshka SAEs reduce absorption from 0.49 to 0.05 by explicitly modeling hierarchy -- this would not work if absorption were purely geometric. The workspace pilot used synthetic data with d_model=128, which may not generalize to real LLMs.
- **Cherry-picking check**: I am focusing on the workspace's synthetic pilot where absorption is deterministic. Chanin et al.'s real LLM results show genuine variation. The shuffled/permuted baselines matching trained SAEs could be a property of the specific synthetic hierarchy geometry (cos=0.67) rather than a general phenomenon.
- **Confounding check**: The zero variance could be due to the specific multi-child proportional ablation formula used in the pilot, not absorption itself. Chanin et al.'s original metric shows variation across features and layers.
- **Actionability check**: Even if absorption is geometric, understanding the geometry is still useful -- it could lead to geometry-aware SAE initialization or architecture design. The insight is constructive.
- **Verdict**: MODERATE. The geometric interpretation is compelling for the synthetic setting but may not generalize to real LLMs. The factorial design (random/trained encoder x random/trained decoder) is the critical test.

### Against Candidate B (Training-Free Detection is Flawed)

- **Steelman**: SAEBench successfully uses adapted absorption metrics across 200+ SAEs. Feature sensitivity (Hu et al.) is a genuinely new dimension that captures something absorption rate misses. The field is actively improving training-free methods -- dismissing them entirely would be premature.
- **Cherry-picking check**: I am citing EDA's failures while ignoring potential successes of other methods. The workspace's EDA may have failed due to specific implementation issues (Qwen proxy probes, random QR projection) rather than fundamental limitations of the approach.
- **Confounding check**: The prevalence problem (0.024%) is a methodological issue that could be addressed with AUPRC or stratified sampling, not a fundamental flaw in training-free detection.
- **Actionability check**: If training-free detection is flawed, what should researchers do instead? The answer -- causal ablation or ground-truth labels -- is expensive and limits applicability. A paper that just says "this doesn't work" without offering alternatives has limited value.
- **Verdict**: WEAK. The training-free skepticism is valid but the argument is too negative. A more constructive framing would be "Training-Free Absorption Detection Has Systematic Blind Spots That Can Be Characterized and Partially Addressed."

### Against Candidate C (Safety-Critical Absorption is Overblown)

- **Steelman**: Anthropic's "Scaling Monosemanticity" found genuine safety-relevant features (deception, sycophancy, bias). Templeton et al. (2024) showed these features can be steered to modify model behavior. If safety-relevant features exist and can be manipulated, their absorption would indeed be dangerous.
- **Cherry-picking check**: I am using the workspace's H2 failure (positive frequency correlation) to argue against a claim that was never tested. H2's positive correlation was driven by 1.5% non-zero outliers against 98.5% zeros -- it may not be a meaningful pattern at all.
- **Confounding check**: Even if safety features are not disproportionately absorbed, they may be disproportionately affected by OTHER SAE pathologies (feature hedging, feature invention, poor sensitivity).
- **Actionability check**: A paper showing "safety features are not uniquely absorbed" would need to carefully distinguish absorption from other failure modes. It risks being misread as "SAEs are safe" when the actual message is "the specific absorption risk is overblown, but other risks remain."
- **Verdict**: WEAK. The framing is too easily misinterpreted. The null result would be scientifically valid but politically fraught.

---

## Phase 4: Refinement

**Dropped**: Candidate C (Safety-Critical Absorption is Overblown) -- too easily misinterpreted, politically fraught, and the evidence is genuinely ambiguous.

**Dropped**: Candidate B in its pure form (Training-Free Detection is Fundamentally Flawed) -- too negative, ignores genuine progress in the field. Reframed as a more constructive critique.

**Strengthened**: Candidate A (Geometric Artifact) survives the steelman test but needs real LLM validation, not just synthetic data. The factorial design is the key experiment.

**New candidate emerging from refinement**: The workspace's own experimental history reveals a pattern that none of the candidates directly address: the project has repeatedly found that its most exciting results are methodological artifacts (deterministic absorption, broken steering, AUROC collapse from prevalence mismatch). This meta-pattern itself could be a contribution.

**Selected front-runner**: A hybrid of Candidate A and the meta-pattern insight:

---

## Phase 5: Final Proposal

### Title
"Rethinking Feature Absorption: Evidence That Geometry, Not Learning, Drives the Primary Signal"

### Challenged Assumption
The field treats feature absorption as a learned phenomenon that reveals how SAEs (and LLMs) represent hierarchical features. Chanin et al. (2024/2025) define absorption as a consequence of "hierarchy + sparsity optimization" and validate it on hundreds of trained SAEs. Subsequent work (Matryoshka, HSAE, OrtSAE) proposes architectural solutions that assume absorption is a training-time property that can be mitigated through better optimization.

### Evidence

**For the conventional view**:
- Chanin et al. validated absorption across hundreds of real LLM SAEs with genuine variation.
- Matryoshka SAEs reduce absorption from 0.49 to 0.05 by explicitly modeling hierarchy.
- Toy models prove that hierarchy + sparsity optimization causes absorption.

**Against the conventional view**:
- Our synthetic pilot shows trained SAE absorption (0.50) is statistically indistinguishable from shuffled features (0.487) and permuted encoder (0.484) baselines. Only random decoder (0.059) differs.
- Absorption shows ZERO variance (std=0.0) across 100 samples, indicating deterministic geometric computation.
- Korznikov et al. (2026) show SAE decoders remain near random initialization ("lazy training regime"), achieving 71% explained variance while recovering only 9% of true features.
- The workspace's factorial design (not yet fully executed) can directly test whether random decoder + trained encoder produces absorption comparable to fully trained SAEs.

### Hypothesis
The primary signal in feature absorption measurements comes from the geometric arrangement of decoder directions in high-dimensional space, not from training-induced learning. When parent decoder directions lie within or near the convex hull of child decoder directions, the encoder naturally routes activation through children -- regardless of whether the decoder was trained or randomly initialized. Training primarily fine-tunes encoder alignments to exploit this pre-existing geometry.

### Method

**Experiment 1: Factorial Decomposition (Geometric vs. Learned)**
- Design: 2x2 factorial (random/trained encoder x random/trained decoder)
- Train 4 conditions (5 seeds each) on synthetic hierarchical data
- Measure multi-child proportional absorption rate for each condition
- Prediction: Trained/random decoder conditions will show large differences; trained/random encoder conditions will show small differences
- Time: ~45 minutes (pilot scale)

**Experiment 2: Lazy Training Validation**
- Compare decoder weight movement from initialization during SAE training
- Measure cosine similarity between initial and final decoder weights
- Prediction: High cosine similarity (>0.8) supports lazy training hypothesis
- Time: ~15 minutes

**Experiment 3: Real LLM Sanity Check**
- Reproduce Chanin et al.'s absorption rate on Gemma Scope layers 0-5
- Compare with a random decoder baseline (same architecture, random weights)
- Prediction: If geometric hypothesis holds, random decoder will show non-zero absorption that correlates with trained SAE absorption
- Time: ~30 minutes

### Baselines
- Chanin et al. original absorption rate (the mainstream method)
- Korznikov et al. random baseline framework
- Shuffled features and permuted encoder baselines from workspace pilot

### Risk Assessment
**What if the mainstream view is correct?** If the factorial design shows that trained encoder makes a large difference, the geometric hypothesis is weakened. In this case, the paper pivots to quantifying the geometric contribution as a lower bound: "Even if learning matters, at least X% of absorption signal is geometric." This is still a useful contribution.

**What if real LLMs differ from synthetic data?** The synthetic pilot may be too simple (d_model=128, cos=0.67 hierarchy). Experiment 3 on real LLMs is the critical validation. If real LLMs show different patterns, the paper limits claims to synthetic settings and calls for further investigation.

### Novelty Claim
The specific insight is that feature absorption may be better understood as a geometric property of high-dimensional vector arrangements than as a learned phenomenon. This reframes absorption research from "how do we train better SAEs?" to "what geometric conditions produce absorption, and can we detect them without training?" It also explains why architectural innovations (Matryoshka, HSAE) help -- they change the geometric constraints, not just the optimization.

---

## Sources

- Chanin, D., et al. (2024/2025). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507. NeurIPS 2025.
- Korznikov, A., et al. (2026). "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111.
- Leask, P., et al. (2025). "Sparse Autoencoders Do Not Find Canonical Units of Analysis." ICLR 2025. arXiv:2502.04878.
- Wu, Z., et al. (2025). "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders." ICML 2025. arXiv:2501.17148.
- Peng, K., et al. (2025). "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts." arXiv:2506.23845.
- Marks, S., et al. (2025/2026). "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534.
- Gurnee, W. (2024). "SAE reconstruction errors are (empirically) pathological." Alignment Forum.
- Hu, N., et al. (2025). "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717.
- Karvonen, A., et al. (2025). "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." ICML 2025. arXiv:2503.09532.
- SynthSAEBench (2026). "Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." arXiv:2602.14687.
- Till, D. (2025). "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders." arXiv:2508.09363.
- DeepMind Safety Research (2025). "Negative Results for Sparse Autoencoders on Downstream Tasks and Deprioritising SAE Research."
