# Literature Survey: SAE Absorption, Unsupervised Detection, and Methodological Critique

**Research Topic**: Sparse Autoencoder Feature Absorption -- Unsupervised Detection, Causal Validation, and Methodological Critique
**Survey Date**: 2026-04-28
**Focus Areas**:
1. Unsupervised detection methods for feature absorption
2. Causal inference in feature detection and validation
3. Negative result paper writing paradigms
4. Methodological critique of SAE interpretability

---

## 1. The Absorption Phenomenon: Foundation and State of the Art

### 1.1 Discovery and Definition

Feature absorption was formally identified and named by **Chanin et al. (2024)** in "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" [arXiv:2409.14507](https://arxiv.org/abs/2409.14507). The paper establishes that when SAEs optimize for sparsity over hierarchical data, broad parent features (e.g., "vowel") fail to activate where they should, with their activation being "absorbed" into more specific child features (e.g., "a", "e", "i"). This creates an interpretability illusion: the model appears to have learned the parent concept, but the corresponding latent never actually fires.

The phenomenon is the complement of feature splitting, where a single concept decomposes into multiple finer-grained features. Together, splitting and absorption form a fundamental tension in SAE learning: sparsity pressure drives features toward specificity, while reconstruction pressure demands coverage of all concepts.

### 1.2 Current Solutions and Their Trade-offs

By 2025, several architectural solutions have been proposed:

| Method | Absorption Reduction | Key Innovation | Critical Limitation |
|--------|---------------------|----------------|---------------------|
| **Matryoshka SAE** (Bussmann et al., 2025) [arXiv:2503.17547](https://arxiv.org/abs/2503.17547) | ~90% (0.49 -> 0.05 at L0=40) | Hierarchical nested dictionaries | Introduces **feature hedging** in inner levels (Chanin et al., 2025) |
| **Balanced Matryoshka** (Chanin et al., 2025) [arXiv:2505.11756](https://arxiv.org/abs/2505.11756) | Optimized trade-off | Tuned loss coefficients (beta_m ~ 0.75) | Still requires multi-level training |
| **Orthogonal SAE** (Korznikov et al., 2025) [arXiv:2509.22033](https://arxiv.org/abs/2509.22033) | ~65% | Cosine similarity penalty between latents | Lower explained variance; 4-11% slower inference |
| **ATM** (Li et al., 2025) [arXiv:2510.08855](https://arxiv.org/abs/2510.08855) | ~95% (0.1402 -> 0.0068) | Temporal probabilistic masking with dual EMAs | Very recent; limited community validation |
| **Weighted SAE** (Cui et al., 2025) [arXiv:2506.15963](https://arxiv.org/abs/2506.15963) | Theoretically grounded | Closed-form optimal solution analysis | Limited empirical validation |
| **Group Bias Adaptation** (Chen et al., 2025) [arXiv:2506.14002](https://arxiv.org/abs/2506.14002) | Strong empirical | Adaptive bias adjustment; theoretical guarantees | Requires specific statistical assumptions |

### 1.3 The Absorption-Hedging Trade-off

A critical 2025 insight from Chanin et al. [arXiv:2505.11756](https://arxiv.org/abs/2505.11756): absorption and hedging are **complementary problems** that form a fundamental Pareto frontier:

- **Absorption**: Occurs in wide SAEs where sparsity loss plus hierarchical co-occurrence suppresses parent features
- **Hedging**: Occurs in narrow SAEs where reconstruction (MSE) loss plus feature correlations cause problematic feature assignment

Matryoshka SAEs trade absorption for hedging in inner (narrower) levels. No current architecture eliminates both simultaneously.

---

## 2. Unsupervised Detection Methods: The Critical Gap

### 2.1 The Ground Truth Problem

Chanin et al.'s original absorption metric has a fundamental limitation explicitly acknowledged in their paper:

> *"Our feature absorption metric requires having ground-truth knowledge of true labels to first train a LR probe, whereas many features of interest in a LLM lack such clear-cut ground-truth labels."*

This means their metric is **supervised**: it requires knowing the parent feature a priori, training a logistic regression probe on raw activations, and then measuring whether the SAE's corresponding latent activates correctly. For the vast majority of SAE features -- which lack known semantic labels -- this metric is inapplicable.

### 2.2 Existing Unsupervised or Weakly-Supervised Approaches

Several methods partially address this gap, though none fully solve unsupervised absorption detection:

**Automated Interpretability (Auto-Interp)**
- Uses LLM-as-judge to generate natural language descriptions of SAE features
- Evaluated by another LLM on description quality
- **Limitation**: Descriptions may miss absorption because the feature "looks" correct (the parent concept is represented, just by the wrong latents)
- Used in SAEBench (Karvonen et al., 2025) [arXiv:2503.09532](https://arxiv.org/abs/2503.09532)

**Sparse Probing (k-sparse probing)**
- Tests whether a small set of latents can predict a concept
- **Limitation**: Still requires labeled concepts; measures feature splitting, not absorption directly
- From Chanin et al. (2024) and Gao et al. (2024)

**SAEBench Metrics (2025)**
- SCR (Spurious Correlation Removal): Tests if zero-ablating small latent sets removes unwanted correlations
- TPP (Targeted Probe Perturbation): Measures completeness and isolation of concepts in small latent groups
- RAVEL: Evaluates clean separation of related attributes via interventions
- **Limitation**: These measure disentanglement and interpretability, not absorption specifically

**Feature Sensitivity (Tian et al., 2025)** [arXiv:2509.23717](https://arxiv.org/abs/2509.23717)
- Measures reliability of feature activation on semantically similar texts
- **Limitation**: Complementary to absorption detection but not causal; a feature can be consistently wrong

**Temporal SAEs (Li & Ren, 2025)** [arXiv:2511.05541](https://arxiv.org/abs/2511.05541)
- Uses temporal signal as weak supervision
- **Limitation**: Requires sequential structure; does not directly detect absorption

### 2.3 The Unsupervised Detection Gap

**This is the central research gap identified by this survey**: There is currently **no method to detect feature absorption without ground-truth knowledge of parent features**. The implications are severe:

1. Researchers cannot audit pre-trained SAEs for absorption at scale
2. Every absorption study to date has focused on a handful of hand-selected features (first letters, numbers, colors)
3. The true prevalence of absorption across all SAE features remains unknown
4. Architectural comparisons (Matryoshka vs. TopK vs. JumpReLU) on absorption rely on the same small set of test cases

This gap motivates the development of **Unsupervised Absorption Detection (UAD)** methods that can identify absorbed features using only the SAE's internal structure and activation patterns, without external labels.

---

## 3. Causal Inference in Feature Detection and Validation

### 3.1 The Causal Validation Pipeline

The field has converged on a two-stage approach for validating SAE features:

| Stage | Method | Purpose |
|-------|--------|---------|
| **Discovery** | Sparse Autoencoders | Learn bottom-up feature dictionaries from activations |
| **Validation** | Causal interventions | Test whether features are truly meaningful |

### 3.2 Key Causal Intervention Techniques

**Activation Patching**
- Replace activations at specific positions with activations from a counterfactual input
- Measure change in model output to establish causal connection
- Extended to SAE feature space: patch specific feature activations rather than raw activations
- Used in Chanin et al. (2024) for absorption verification: ablate suspected parent features and verify child features compensate

**Integrated Gradients Ablation**
- From Chanin et al. (2024): used to verify that absorbed features causally mediate model behavior
- Compute gradient-based attribution to confirm the parent feature's causal role
- **Limitation**: Only applicable to early layers; computationally expensive

**Feature Steering**
- Add or subtract feature activations to modify model behavior
- Used to test whether a feature is sufficient for a behavior
- Extended by Crook et al. (2026) [arXiv:2603.04198](https://arxiv.org/abs/2603.04198) with L2 regularization for more stable steering

**SAE-Guided Activation Patching (2025)**
- Patch in SAE feature space rather than raw activation space
- Used in CoT faithfulness studies (arXiv:2507.22928) and gambling behavior analysis (arXiv:2509.22818)
- Introduces patch-curves and random-feature patching baselines for calibration

**Population Mean Patching**
- Apply average feature activations from condition groups across contexts
- Used for causal verification in behavioral studies

### 3.3 Critical Concerns About Causal Interventions

**Representation Divergence**
- A major concern in 2025: causal interventions often create unrealistic model states
- Activation patching sometimes multiplies feature values by up to 15x
- These divergent representations may not reflect the neural network's natural distribution
- Raises fundamental questions about the reliability of causal interventions for claims about natural mechanisms

**Interpretability Illusions**
- Makelov, Lange & Nanda (2024) showed subspace activation patching can create misleading causal interpretations
- Features that appear causally important under intervention may not be so in natural operation

**From Correlation to Causation**
- The field is actively moving from attribution-based methods (linear contributions) to necessity/sufficiency testing
- "Locating Critical Neurons as Sufficient Conditions" (2025) [arXiv:2603.18474](https://arxiv.org/abs/2603.18474) addresses the gap: highly attributed components are often neither necessary nor sufficient

### 3.4 Implications for Absorption Detection

Causal validation is essential for absorption research because:
1. Correlational detection (e.g., low activation of parent features) is insufficient -- the parent might simply not be needed
2. Ablation-based verification (Chanin et al.'s approach) confirms that absorbed features causally mediate the parent's behavior
3. However, causal validation requires knowing what to test -- reinforcing the ground-truth dependency problem

An ideal unsupervised absorption detection method would combine **structural cues** (from the SAE's latent relationships) with **causal verification** (intervention-based confirmation), without requiring pre-specified parent features.

---

## 4. Negative Result Paper Writing Paradigms

### 4.1 The Publication Bias Problem

Machine learning suffers from a systemic bias against negative results. As documented by Professor Jeffrey Bowers and others:

> *"A common feature of these papers is that they highlighted the limitations of specific models and undermined key conclusions that were drawn from their initial successes... editors do not dispute our findings but note that we have not 'solved' the problem we have identified."*

**Actual reviewer quotes from rejections:**
- **ICML**: *"reproduces existing results and failure modes... but does not propose a method to overcome those"*
- **ICLR**: *"would have been a stronger paper if authors had suggested mechanisms or solutions"*
- **NeurIPS**: *"does not suggest a model... or points to a fundamental failure mode"*

### 4.2 The ICBINB Workshop: A Dedicated Venue

The **"I Can't Believe It's Not Better" (ICBINB)** workshop series provides a formal venue for negative results:

| Edition | Venue | Theme |
|---------|-------|-------|
| 2021 | NeurIPS 2021 | Initial edition |
| 2022 | NeurIPS 2022 | "Understanding Deep Learning Through Empirical Falsification" |
| 2023 | NeurIPS 2023 | "Failure Modes in the Age of Foundation Models" |
| 2025 | ICLR 2025 | "Challenges in Applied Deep Learning" |
| 2026 | ICLR 2026 | "Where LLMs Need to Improve" |

**Submission requirements** (ICLR 2025 format):
1. A problem tackled with deep learning
2. A solution proposed in the literature
3. A description of the **(negative) outcome**
4. An investigation into **why it did not work**

**Special awards**:
- "Entropic Award" for the most surprising negative result
- "Dididactic Award" for the most pedagogical paper

### 4.3 Successful Negative Result Papers

**AI Scientist at ICLR 2025 ICBINB**:
- Title: *"Compositional Regularization: Unexpected Obstacles in Enhancing Neural Network Generalization"*
- Score: **6.33** (top 45% of submissions)
- Praised for "valuable negative results"
- First fully AI-generated paper to pass standard peer review

**ICLR 2025 ICBINB accepted examples**:
- "Performance of Zero-Shot Time Series Foundation Models on Cloud Data" -- foundation models underperforming
- "On the Limits of Applying Graph Transformers for Brain Connectome Classification" -- architecture limitations
- "Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges" -- safety evaluation failures

### 4.4 Strategies for Writing Negative Result Papers

| Challenge | Strategy |
|-----------|----------|
| Reviewer bias toward "solutions" | Frame as **"empirical falsification"** or **"diagnostic analysis"** |
| Lack of "novelty" | Emphasize **unexpected findings** and **theoretical implications** |
| Where to submit | Target **ICBINB workshops**, or frame as **"What Can Go Wrong"** |
| Making it appealing | Include **thorough analysis of why** the hypothesis failed |

**Key principles**:
- Be religious about robustness checks -- report failures transparently
- A simple idea with careful analysis (including negative results) may be more scientifically valuable
- Reward clarity, transparency, and intellectual honesty -- not just performance metrics
- Results that are particularly unexpected should be **up-weighted** in evaluation

### 4.5 Relevance to SAE Absorption Research

An SAE absorption study that tests multiple hypotheses (H1-H4) and finds most inconclusive fits the negative result paradigm well:
- The honest reporting of failed hypotheses is scientifically valuable
- The one successful contribution (e.g., UAD method) can be framed as emerging from the systematic investigation
- Limitations should be discussed thoroughly rather than downplayed
- The paper should explain *why* each hypothesis failed, not just that it did

---

## 5. Methodological Critique of SAE Interpretability

### 5.1 Fundamental Theoretical Limitations

**Provable suboptimality**: O'Neill et al. (2025) proved that standard SAEs with linear-nonlinear encoders fail to achieve optimal sparse recovery due to architectural constraints -- the encoder lacks sufficient complexity to recover high-dimensional sparse representations from lower-dimensional projections.

**Amortization gap**: Wright & Sharkey (2024) established theoretical bounds on SAE suboptimality for compute-optimal inference.

**Closed-form analysis**: Cui et al. (2025) [arXiv:2506.15963](https://arxiv.org/abs/2506.15963) provided the first closed-form optimal solution analysis, proving standard SAEs fail unless features are extremely sparse, and identifying feature shrinking/vanishing as fundamental issues.

### 5.2 The Canonical Feature Hypothesis: Challenged

**Leask et al. (2025)** "Sparse Autoencoders Do Not Find Canonical Units of Analysis" [ICLR 2025](https://openreview.net/forum?id=9ca9eHNrdH) presents a devastating critique:

| Technique | Finding | Implication |
|-----------|---------|-------------|
| **SAE Stitching** | Larger SAEs contain "novel latents" that improve performance when added to smaller SAEs | SAEs are **incomplete** |
| **Meta-SAEs** | Latents in larger SAEs decompose into combinations of smaller SAE latents | SAE features are **not atomic** |

Example: A latent representing "Einstein" decomposes into "scientist" + "Germany" + "famous person" when analyzed with meta-SAEs.

**Conclusion**: SAEs do not discover a unique, complete, and atomic set of features. They find useful but non-canonical decompositions.

### 5.3 SAEs Interpret Randomly Initialized Transformers

A 2025 finding [arXiv:2501.17727](https://arxiv.org/abs/2501.17727) showed that SAEs find similar features in trained and untrained models. This raises fundamental questions:
- Do SAE features reflect learned computations or architecture/data statistics?
- Are absorption and dead features consequences of data structure rather than learning?
- Can SAEs distinguish meaningful from random representations?

### 5.4 Superposition as Lossy Compression

"Superposition as Lossy Compression" (2025) [arXiv:2512.13568](https://arxiv.org/abs/2512.13568) identifies critical limitations:

| Issue | Description |
|-------|-------------|
| Linear representation assumption | Features may not correspond to simple directions -- circular/temporal features violate this |
| Feature correlations | Networks exhibit synergistic information and gating mechanisms that SAEs miss |
| Hierarchical representations | Low-level and high-level features aren't substitutable; nonlinear interactions can't be decomposed additively |
| SAE quality dependency | Measurements are proxies, not literal feature counts |

### 5.5 Compositional Reasoning Failures

Research on Othello-GPT found SAEs struggle with compositional tasks where models build algorithms across layers:
- Reconstruction performance lags behind language models
- Sparsity constraint "incurs more cost" for layered computation
- Features are less directly interpretable than in simple prediction tasks

### 5.6 Training Stability: Architecture Comparison

| Architecture | Stability Strength | Stability Weakness | Dead Features | Absorption |
|-------------|-------------------|-------------------|---------------|------------|
| **Standard SAE (L1)** | Simple | Worst absorption; unstable | High | 0.0161 |
| **TopK SAE** | Fast, efficient; fixed sparsity | Absorption (0.1402); dead features | High (without AuxK) | 0.1402 |
| **JumpReLU** | Best reconstruction; minimal dead | Sensitive to initialization | Very low | 0.0114 |
| **BatchTopK** | Adaptive per-sample sparsity | Moderate absorption | Moderate | Moderate |
| **ATM (2025)** | Best absorption score | Newer, less validated | Low | 0.0068 |
| **L2-Reg TopK** | Excellent cross-seed consistency | Aggressive feature death | High (bimodal) | Improved |

### 5.7 The Dead Feature Problem

Fundamental limits were established by Dip Roy et al. (2025) [arXiv:2603.18056](https://arxiv.org/abs/2603.18056):
- Dead neuron recovery is severely limited under extreme sparsification
- Zero recovery on dSprites after 100 epochs
- The problem is intrinsic to the compression process

Current mitigation methods:
- **AuxK Loss** (OpenAI): "Eliminates almost all dead latents"
- **Ghost Gradients** (Anthropic): Often achieves zero dead neurons but ~2x training cost; abandoned due to loss spikes
- **Soft-capping**: 6 vs 491 dead neurons at scale
- **JumpReLU threshold-only gradients**: Minimal dead features but sensitive to initialization

---

## 6. Synthesis: Research Gaps and Opportunities

### 6.1 The Four Focus Areas Converge

The four focus areas of this survey converge on a central challenge: **how do we know what SAEs have actually learned?**

1. **Unsupervised detection** addresses the practical problem of auditing SAEs without labels
2. **Causal inference** provides the methodological rigor to validate findings
3. **Negative result paradigms** create space to report when methods fail
4. **Methodological critique** reminds us that SAEs may not discover "true" features at all

### 6.2 Priority Research Gaps

| Gap | Urgency | Why It Matters |
|-----|---------|----------------|
| **Unsupervised absorption detection** | Critical | No current method works without ground truth; prevents scale |
| **Theoretical characterization of absorption-hedging frontier** | High | Fundamental Pareto frontier not characterized |
| **Absorption-aware training objectives** | High | Current solutions modify architecture; loss-based approaches underexplored |
| **Cross-architecture absorption comparison** | High | Most studies focus on JumpReLU/GemmaScope; systematic comparison lacking |
| **Unified stability metric** | Medium | Many metrics (absorption, dead latents, consistency) but no unified framework |
| **Scale-dependent analysis (7B+)** | Medium | Most research stops at 2B parameters |
| **What do SAEs actually capture?** | High | Randomly initialized transformers finding challenges core assumptions |

### 6.3 Implications for This Project

Based on this survey, the following strategic recommendations emerge:

1. **The UAD contribution is genuinely novel**: No prior work addresses unsupervised absorption detection. This fills Gap #1, the most critical gap in the field.

2. **Honest reporting of negative results is scientifically valuable**: The ICBINB workshop series and successful negative result papers demonstrate that well-analyzed failures are publishable and contribute to the field.

3. **Causal validation is essential but limited**: The project's use of ablation-based verification aligns with best practices, but the ground-truth dependency remains a limitation that should be acknowledged.

4. **Methodological humility is warranted**: Given Leask et al.'s findings that SAEs don't find canonical features, claims about "absorption" should be framed carefully -- the phenomenon is real and measurable, but whether it reflects a failure to discover "true" features or a property of the SAE decomposition itself is an open question.

5. **The absorption-hedging trade-off should be discussed**: If the project compares architectures, the absorption-hedging Pareto frontier (Chanin et al., 2025) provides the appropriate framing.

---

## 7. Bibliography

### Core Absorption Papers
1. Chanin, D., et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507.
2. Chanin, D., et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
3. Bussmann, B., et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547.
4. Korznikov, et al. (2025). Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.

### Causal Inference and Validation
5. Chen, S., et al. (2025). Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.
6. Li, T. E., et al. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training. arXiv:2510.08855.
7. "Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior." arXiv:2603.18474.
8. "How does Chain of Thought Think?" arXiv:2507.22928.

### Methodological Critique
9. Leask, P., et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. ICLR 2025. arXiv:2502.04878.
10. "Sparse Autoencoders Can Interpret Randomly Initialized Transformers." arXiv:2501.17727.
11. "Superposition as Lossy Compression." arXiv:2512.13568.
12. Cui, J., et al. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.
13. Dip Roy, et al. (2025). Fundamental Limits of Neural Network Sparsification. arXiv:2603.18056.

### Benchmarks and Evaluation
14. Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532.
15. Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
16. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.

### Architecture Comparisons
17. Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
18. Bussmann, B., et al. (2024). BatchTopK Sparse Autoencoders. arXiv:2412.06410.
19. "Improving Sparse Autoencoder with Dynamic Attention." arXiv:2604.14925.
20. "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111.

### Negative Results and Scientific Writing
21. Bowers, J. The machine learning community has a fundamental misunderstanding of the role of falsification in science. [Blog post](https://jeffbowers.blogs.bristol.ac.uk/blog/ml-community/).
22. "I Can't Believe It's Not Better" Workshop Series. [NeurIPS/ICLR](https://i-cant-believe-its-not-better.github.io/).
23. "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery." arXiv:2504.08066.
24. Rieck, B. If At First You Don't Succeed -- Counting Paper Rejections. [Blog post](https://bastian.rieck.me/blog/2025/rejections/).

### Training Stability and Dead Features
25. Jermyn, A., & Templeton, A. (2024). Ghost Grads: An improvement on resampling. Transformer Circuits Thread.
26. Crook, O., et al. (2026). Stable and Steerable Sparse Autoencoders with Weight Regularization. arXiv:2603.04198.
27. "Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." arXiv:2602.14687.

### Hierarchical and Alternative Approaches
28. Muchane, M., et al. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.
29. Luo, Y., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.
30. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.

---

*Survey compiled by sibyl-literature agent. All arXiv links verified as of 2026-04-28.*
