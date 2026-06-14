# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Chanin et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507, NeurIPS 2025 Oral).
   - **Key principle**: Feature absorption is a sparsity-driven failure mode where parent features in semantic hierarchies are subsumed by child features. Proved analytically that absorption decreases SAE loss monotonically with absorption parameter delta.
   - **Relevance to construct validity**: The paper's absorption metric relies exclusively on first-letter classification tasks. The authors explicitly note: *"Our metric cannot capture absorption past layer 17... [and] requires ground-truth knowledge of true labels."* They call for *"finding examples of feature absorption unrelated to character identification"* as future work. This is the exact gap our construct-validity study addresses.

2. **Olshausen & Field (1996, 1997)** — Sparse coding in V1 (Nature; Vision Research).
   - **Key principle**: The visual cortex employs overcomplete sparse coding where neurons compete via lateral inhibition. Overcompleteness enables greater flexibility and sparseness.
   - **Relevance**: The Olshausen-Field model does not exhibit absorption because it uses continuous-valued coefficients with soft competition. Hard sparsity constraints in SAEs (TopK, L1) introduce winner-take-all dynamics. This suggests that the *measurement* of absorption depends critically on the sparsity mechanism — a construct-validity concern.

3. **Rosch (1978)** — Prototype theory and basic-level categorization.
   - **Key principle**: Human conceptual hierarchies have a cognitively privileged "basic level" (e.g., "dog") between superordinate ("animal") and subordinate ("Labrador"). Superordinate categories have lower within-category similarity and are harder to form mental images for.
   - **Relevance to construct validity**: First-letter hierarchies ("starts with S" -> "short") are artificial constructs. Real semantic hierarchies ("animal" -> "mammal" -> "dog") have different frequency structures, prototype structure, and representational geometry. If SAE absorption behaves differently on artificial vs. natural hierarchies, the metric lacks ecological validity.

4. **Friston (2010)** — The free-energy principle (Nature Reviews Neuroscience).
   - **Key principle**: The brain minimizes variational free energy through "explaining away" — when one cause adequately explains sensory data, it suppresses alternative explanations.
   - **Relevance**: "Explaining away" in predictive coding is the neuroscience analog of absorption. However, in the brain this suppression is *context-dependent and reversible*; in SAEs, it becomes *structurally frozen* through training. This raises a construct-validity question: does the SAEBench metric measure permanent structural absorption or transient contextual suppression?

5. **Moosavi et al. (2025)** — "Dynamics of energy-efficient coding in visual cortex" (Journal of Neurophysiology).
   - **Key principle**: Coding efficiency (mutual information / metabolic cost) improves over time via dynamic sparse coding. Two-phase response: initial broad activation -> refined sparse representation.
   - **Relevance**: The brain's dynamic optimization suggests that *static* SAE metrics may miss important temporal structure. If absorption is measured at a single point, it may not capture the full phenomenon.

#### Physics / Statistical Mechanics / Information Theory

6. **Tishby, Pereira & Bialek (1999)** — "The Information Bottleneck Method" (arXiv:physics/0004057).
   - **Key principle**: The IB method minimizes I(X;X~) while preserving I(X~;Y). Solutions exhibit hierarchical structure with curves bifurcating at critical beta values through second-order phase transitions.
   - **Relevance to construct validity**: IB phase transitions are *reversible* (tuning beta changes the representation), while SAE absorption is *frozen* by training. The SAEBench metric measures a static snapshot; it cannot distinguish between reversible and frozen absorption. This is a construct-validity gap.

7. **Wu & Fischer (2020)** — "Phase Transitions for the Information Bottleneck in Representation Learning" (ICLR 2020).
   - **Key principle**: IB exhibits sudden jumps in dI(Y;Z)/dbeta when new classes are learned. Each transition finds a component of maximum nonlinear correlation between X and Y, orthogonal to the learned representation.
   - **Relevance**: This formalizes how hierarchical representations emerge through phase transitions. The "absorption" of parent features may correspond to the system being "stuck" in a local minimum before the bifurcation point where the parent would split. The metric's validity depends on whether it captures this phase-transition structure.

8. **Equitz & Cover (1991)** — "Successive Refinement of Information" (IEEE Trans. Info. Theory).
   - **Key principle**: A source is successively refinable if it can be encoded in layers (coarse to fine) such that each layer optimally refines the previous. Gaussian sources under MSE are successively refinable; not all sources are.
   - **Relevance to construct validity**: Matryoshka SAEs implement successive refinement. The theory predicts that *not all feature hierarchies are successively refinable*. If the SAEBench metric is tested only on first-letter hierarchies (which may be successively refinable), it may not generalize to hierarchies that are not.

9. **Zdeborova & Krzakala (2016)** — "Statistical physics of inference: Thresholds and algorithms" (Physica A).
   - **Key principle**: Inference problems exhibit phase transitions with sharp boundaries between recoverable and non-recoverable regimes. The "hard phase" (theoretically possible but computationally hard) is a fundamental statistical-to-computational gap.
   - **Relevance**: Dictionary learning in SAEs is an inference problem. The "absorption" regime may correspond to a phase where parent feature recovery is statistically impossible. The metric's validity depends on whether it correctly identifies this phase boundary.

10. **Lobacheva et al. (2025)** — "SGD as Free Energy Minimization" (arXiv:2505.23489).
    - **Key principle**: SGD implicitly minimizes free energy F = U - TS, where learning rate acts as temperature. At high temperature, entropy dominates (distributed representations); at low temperature, energy dominates (sparse representations).
    - **Relevance**: The sparsity-reconstruction tradeoff can be reframed as energy-entropy tradeoff. If the SAEBench metric is computed at a single "temperature" (sparsity level), it may not capture the full phase diagram of absorption behavior.

#### Measurement Theory / Psychometrics

11. **Cronbach & Meehl (1955)** — "Construct Validity in Psychological Tests" (Psychological Bulletin).
    - **Key principle**: Construct validity is established through a **nomological network** — a theoretical system describing lawful relations between a construct, its observable indicators, and related constructs. Validation is an ongoing, iterative process of theory development and empirical testing.
    - **Relevance to construct validity (direct)**: This is the foundational framework for our study. The SAEBench absorption metric has never been validated within a nomological network. We are conducting the first test of whether the metric's correlations with related constructs (first-letter absorption vs. semantic-hierarchy absorption) match theoretical predictions.

12. **Campbell & Fiske (1959)** — "Convergent and discriminant validation by the multitrait-multimethod matrix" (Psychological Bulletin).
    - **Key principle**: Validity requires both convergent evidence (correlation with theoretically related measures) and discriminant evidence (lack of correlation with theoretically unrelated measures).
    - **Relevance**: Our study tests convergent validity (first-letter vs. semantic-hierarchy absorption should correlate) and discriminant validity (hierarchical vs. non-hierarchical correlated features should show different absorption patterns).

13. **Kane (2013)** — "Validating the interpretations and uses of test scores" (APA).
    - **Key principle**: Validity is not a property of the test itself but of the **interpretations and uses** made of test scores. An argument-based approach links scores to interpretations through chains of reasoning and evidence.
    - **Relevance**: The SAEBench absorption score is used to compare architectures, select hyperparameters, and claim improvements. If the score does not generalize beyond first-letter tasks, these interpretations are invalid.

14. **Borsboom, Mellenbergh & van Heerden (2004)** — "The concept of validity" (Psychological Review).
    - **Key principle**: The causal theory of measurement requires that the attribute being measured (1) exists, (2) is causally related to the measurement outcome, and (3) variation in the attribute causes variation in the outcome.
    - **Relevance**: For the absorption metric to be valid, "absorption" must be a real property of SAEs that causes variation in the score. If the score varies due to task-specific artifacts (e.g., first-letter tokenization) rather than true absorption, the metric fails the causal theory test.

#### Machine Learning / Benchmark Validation

15. **PROXIMA: Proxy Metric Validation Framework** (arXiv:2604.14352, 2026).
    - **Key principle**: Aggregate correlation between proxy metrics and long-term outcomes can mask directional failures. Three-dimensional validation: normalized effect correlation + directional accuracy + segment-level fragility rate.
    - **Key finding**: **Directional accuracy >= 0.95 is more important than high correlation alone**.
    - **Relevance (direct)**: This is the most directly relevant methodological precedent. The SAEBench absorption metric is a proxy for "real" feature absorption. Our study applies the PROXIMA framework's insight: we test not just correlation but directional accuracy (does lower first-letter absorption predict lower semantic-hierarchy absorption?) and robustness across segments (architectures, layers, thresholds).

16. **"Beyond Proxy Metrics: A New Evaluation Framework for LLM Compression"** (OpenReview 2025).
    - **Key principle**: Proxy metrics like perplexity and curated benchmarks "often correlate poorly with real-world generative performance." Significant gap between reported scores and practical utility.
    - **Relevance**: Direct parallel to our concern. The SAEBench absorption metric is a proxy for "real" absorption. If it correlates poorly with semantic-hierarchy absorption, architectures optimized for it may not improve real-world interpretability.

17. **"Establishing Construct Validity in LLM Capability Benchmarks Requires Nomological Networks"** (arXiv:2603.15121, 2026).
    - **Key principle**: LLM benchmarks should be validated using nomological networks — testing whether benchmark scores correlate with theoretically related capabilities and not with unrelated ones.
    - **Relevance (direct)**: This paper explicitly extends Cronbach & Meehl's framework to LLM evaluation. Our study is a direct application: we test whether the SAEBench absorption metric fits within a nomological network of SAE interpretability measures.

#### Biology / Evolution / Ecology

18. **Gause (1934)** — Competitive Exclusion Principle.
    - **Key principle**: Two species competing for identical limited resources cannot coexist indefinitely — one will outcompete and eliminate the other.
    - **Relevance to construct validity**: SAE latents "compete" for activation budget (sparsity constraint). The SAEBench metric measures this competition on first-letter tasks. But ecological theory tells us that competition outcomes depend on resource structure. If first-letter hierarchies have different "resource structure" (frequency, co-occurrence patterns) than semantic hierarchies, the metric may not generalize.

19. **Fetzer et al. (2015)** — "The extent of functional redundancy changes as species' roles shift in different environments" (PNAS).
    - **Key principle**: Functional redundancy is context-dependent. Species that are redundant in one environment become pivotal in another.
    - **Relevance**: Parent features may appear "absorbed" in first-letter tasks but be critical in semantic-hierarchy tasks. The metric's validity depends on whether it captures this context-dependence.

20. **Loreau & de Mazancourt (2013)** — "Biodiversity and ecosystem stability" (Ecology Letters).
    - **Key principle**: What promotes stability is niche complementarity (reduced competition), not competition itself.
    - **Relevance**: If the SAEBench metric measures competition outcomes, it may not predict stability (interpretability) under perturbation (different task types).

### Cross-Disciplinary Gaps

**Where transplants haven't been tried yet:**

1. **No SAE work applies measurement theory (Cronbach & Meehl's nomological network) to validate the absorption metric.** The metric was introduced as a detection tool, not validated as a measurement instrument.

2. **No SAE work applies proxy metric validation frameworks (PROXIMA) to test whether first-letter absorption predicts real-world absorption behavior.** The ML community has developed sophisticated proxy validation methods that have not been applied to SAE benchmarks.

3. **No SAE work tests the causal theory of measurement (Borsboom et al.)** — whether variation in the absorption score is caused by variation in true absorption rather than task-specific artifacts.

4. **No SAE work applies convergent/discriminant validation (Campbell & Fiske)** to test whether the absorption metric correlates with related measures and not with unrelated ones.

5. **No SAE work connects information bottleneck phase transition theory to the question of metric generalization** — whether absorption measured at one "temperature" (sparsity level) generalizes to others.

6. **No SAE work tests whether artificial hierarchies (first-letter) and natural hierarchies (WordNet) produce the same absorption patterns** — an ecological validity question.

---

## Phase 2: Initial Candidates

### Candidate A: Nomological Network Validation of SAEBench Absorption (from Psychometrics / Measurement Theory)

- **Source principle**: Cronbach & Meehl's (1955) nomological network establishes construct validity by testing whether a measure correlates with theoretically related constructs and not with unrelated ones. Campbell & Fiske's (1959) convergent/discriminant validation provides the methodological framework.
- **Structural correspondence**:
  - Psychological construct (e.g., "intelligence") <-> SAE property ("feature absorption")
  - Test score (e.g., IQ test) <-> SAEBench absorption score (first-letter metric)
  - Related construct (e.g., "academic performance") <-> Semantic-hierarchy absorption score
  - Unrelated construct (e.g., "height") <-> Non-hierarchy correlated-feature absorption score
  - Nomological network <-> Network of SAE interpretability metrics (absorption, sparse probing, RAVEL, TPP)
- **Hypothesis**: The SAEBench absorption metric will show convergent validity (correlation with semantic-hierarchy absorption > 0.6) and discriminant validity (semantic-hierarchy absorption > non-hierarchy control absorption). If not, the metric lacks construct validity.
- **Why it's not just a metaphor**: The mathematical structure is identical. Both frameworks test whether a proxy measure (test score / first-letter absorption) correlates with a theoretically related criterion measure (academic performance / semantic-hierarchy absorption) and discriminates from unrelated measures. The validation logic (correlation, bootstrap CI, paired t-test) is the same.
- **Novelty estimate**: 9/10. Psychometric construct validation has never been applied to SAE benchmarks. The PROXIMA framework (2026) extends this to LLM benchmarks generally, but no prior work applies it specifically to SAE absorption.

### Candidate B: Phase-Transition Diagnostic for Absorption Metrics (from Statistical Physics / Information Bottleneck)

- **Source principle**: The Information Bottleneck exhibits phase transitions where representation structure qualitatively changes at critical beta values (Tishby 1999; Wu & Fischer 2020). SGD as free energy minimization (Lobacheva et al. 2025) shows that learning rate acts as temperature, controlling the energy-entropy tradeoff.
- **Structural correspondence**:
  - IB beta parameter <-> SAE sparsity penalty lambda
  - Phase transition point <-> Critical sparsity where absorption abruptly changes
  - Free energy F = U - TS <-> SAE loss = reconstruction + lambda * sparsity
  - Temperature T <-> 1/lambda (inverse sparsity penalty)
  - Order parameter (magnetization) <-> Absorption fraction delta
- **Hypothesis**: Feature absorption measured by SAEBench is not a single number but a phase-dependent quantity. At low sparsity (high temperature), parent and child features coexist. At high sparsity (low temperature), absorption is complete. The metric's validity depends on which "phase" it measures. Testing across sparsity levels reveals whether the metric captures a general phenomenon or a phase-specific artifact.
- **Why it's not just a metaphor**: The mathematical mapping is exact. The SAE objective L = ||x - x_hat||^2 + lambda * S is formally a free energy with T = 1/lambda. Wu & Fischer proved that IB representations undergo phase transitions at critical beta values. The prediction — that absorption is phase-dependent — is a direct consequence of this correspondence.
- **Novelty estimate**: 7/10. Phase transitions in IB have been studied, but not specifically for SAE absorption metric validation. The connection between free energy and SAE training is emerging (Lobacheva 2025) but not applied to construct validity.

### Candidate C: Ecological Validity of Artificial Hierarchies (from Cognitive Science / Ecology)

- **Source principle**: Rosch's (1978) basic-level theory shows that natural categories have a privileged "basic level" with optimal informativeness. Artificial hierarchies (like first-letter classifications) lack this structure. Ecological theory (Gause 1934; Loreau 2013) shows that competition outcomes depend on resource structure.
- **Structural correspondence**:
  - Natural category hierarchy (animal -> mammal -> dog) <-> WordNet semantic hierarchy
  - Artificial category hierarchy (starts with S -> short) <-> First-letter classification hierarchy
  - Basic-level privilege <-> Frequency-matched parent-child representation
  - Resource structure (niche dimensions) <-> Feature co-occurrence and frequency patterns
  - Competition outcome <-> Absorption score
- **Hypothesis**: First-letter hierarchies are "degenerate" ecological systems where all "species" (features) compete for a single resource (sparsity budget) with identical frequency structure. Natural hierarchies have richer resource structure (multiple co-occurrence patterns, frequency variation). The SAEBench metric may measure competition in this degenerate system, which does not generalize to richer systems. Frequency-matching controls for this, but if absorption still differs, the metric captures something deeper than frequency confounds.
- **Why it's not just a metaphor**: The mathematical structure of competition (Lotka-Volterra) and sparse coding with overlap penalties share the same form: da_i/dt = a_i * (r_i - sum_j G_ij a_j). In ecology, G_ij encodes resource overlap; in SAEs, G_ij = <d_i, d_j> encodes decoder overlap. The frequency structure of features (r_i) directly affects competition outcomes in both systems.
- **Novelty estimate**: 8/10. The ecological validity of first-letter hierarchies has been questioned informally but never systematically tested. The connection between Rosch's basic-level theory and SAE feature representation is unexplored.

---

## Phase 3: Self-Critique

### Against Candidate A (Nomological Network Validation)

- **Shallow analogy attack**: Is this just "run correlations and call it psychometrics"? No — the structural correspondence is exact. Cronbach & Meehl's framework was designed precisely for situations where a construct (absorption) cannot be observed directly and must be inferred from proxy measures. The validation logic (convergent + discriminant evidence) is the same. However, a psychometrician might note that our "sample size" (6-8 SAEs) is small for establishing construct validity. The bootstrap CI mitigates this, but the evidence will be suggestive rather than definitive.
- **Scale mismatch attack**: Psychometric validation typically uses hundreds or thousands of subjects. We have 6-8 SAEs. However, each SAE provides multiple measurements (one per hierarchy), and the correlation is across SAEs (not within). The small-n issue is real but addressable with bootstrap methods and effect-size reporting.
- **Prior transplant check**: Has psychometric construct validation been applied to ML benchmarks? Yes — the PROXIMA framework (2026) and "Establishing Construct Validity in LLM Capability Benchmarks" (2026) explicitly do this. However, no prior work applies it to SAE absorption metrics specifically.
- **Testability attack**: Can we distinguish "the metric lacks construct validity" from "the metric has validity but our semantic-hierarchy measurement is noisy"? Yes — the random-SAE control and probe-quality filtering address this. If the random SAE shows non-zero absorption, the metric itself is flawed. If probe AUROC is high (>0.7) but correlation is low, the metric lacks validity.
- **Verdict**: STRONG. The structural correspondence is rigorous, the methodology is well-established, and the diagnostic experiments are clear. The main risk is small sample size.

### Against Candidate B (Phase-Transition Diagnostic)

- **Shallow analogy attack**: Is this just "try different sparsity levels" dressed up in physics language? In part, yes. But the physics framing adds a specific prediction: there should be *critical* sparsity values where absorption qualitatively changes, and annealing across these transitions should reveal the metric's phase-dependence.
- **Scale mismatch attack**: Phase transition theory is derived for asymptotic limits (infinite data, infinite dimensions). SAEs operate at finite scale. Finite-size effects may smooth out transitions. However, Wu & Fischer (2020) showed IB phase transitions empirically in finite neural networks.
- **Prior transplant check**: Has phase-transition analysis been applied to SAE absorption? No. Lobacheva et al. (2025) connected SGD to free energy minimization but did not study absorption. This is genuinely novel.
- **Testability attack**: Can we distinguish "phase-dependent absorption" from "absorption smoothly increases with sparsity"? Yes — the diagnostic experiment tests for discontinuous jumps in absorption vs. sparsity curves. If the curve is smooth, the phase-transition framing is unnecessary. If there are jumps, it is validated.
- **Verdict**: MODERATE. The correspondence is rigorous but the practical benefit over simple sparsity-sweep analysis is uncertain. The diagnostic experiment requires training multiple SAEs at different sparsity levels, which is computationally expensive.

### Against Candidate C (Ecological Validity of Artificial Hierarchies)

- **Shallow analogy attack**: Is "ecological validity" just "test on real data" with fancy branding? No — ecological validity is a specific psychometric concept (Brunswik 1955) referring to whether experimental conditions mirror real-world conditions. The Rosch basic-level theory provides a specific structural prediction: natural hierarchies have a privileged basic level that artificial hierarchies lack.
- **Scale mismatch attack**: Rosch's theory applies to human conceptual systems, not neural network representations. However, if LLMs encode human-like conceptual hierarchies (as suggested by their training on human text), the theory should apply. The test is empirical.
- **Prior transplant check**: Has basic-level theory been applied to neural network representations? Yes — in cognitive modeling and concept learning. But not specifically to SAE absorption metrics.
- **Testability attack**: Can we distinguish "artificial hierarchies produce different absorption" from "frequency differences produce different absorption"? The frequency-matching control addresses this. If absorption still differs after frequency matching, the hierarchy structure (not frequency) is the cause.
- **Verdict**: MODERATE. The idea is novel and theoretically grounded, but the test requires careful control of confounds. The basic-level theory prediction is specific and falsifiable.

---

## Phase 4: Refinement

### Dropped
- **Candidate B (Phase-Transition Diagnostic)** is deferred because it requires training SAEs at multiple sparsity levels, which violates the training-free constraint of the current project. It remains a strong follow-up direction if the construct-validity study supports further investigation.

### Strengthened

**Candidate A (Nomological Network Validation)** is selected as the front-runner because:
1. It directly addresses the construct-validity question that is the core of the current research direction.
2. It is perfectly aligned with the training-free constraint (uses existing pretrained SAEs).
3. The methodology is rigorous and well-established in psychometrics.
4. It can be completed within the 1-hour-per-experiment constraint.
5. It generates actionable implications for the SAE community regardless of outcome.

**Candidate C (Ecological Validity)** is integrated as a secondary hypothesis within the main study:
- H2 (Hierarchy Specificity) tests whether the metric is specific to hierarchical features — this is the discriminant validity test from Campbell & Fiske.
- The frequency-matching control tests the ecological validity of the hierarchy structure.
- The basic-level theory prediction is folded into the interpretation of results.

**Formalized structural correspondence**:

The Cronbach & Meehl nomological network for SAE absorption:

```
Theoretical Construct: "Feature Absorption"
    |
    |-- Observable Indicator 1: SAEBench first-letter absorption score
    |-- Observable Indicator 2: Semantic-hierarchy absorption score (WordNet)
    |-- Observable Indicator 3: Non-hierarchy correlated-feature absorption score
    |
    |-- Related Constructs:
        |-- "Feature Splitting Rate" (should correlate positively)
        |-- "Sparse Probing Performance" (should correlate negatively)
        |-- "Reconstruction Quality" (should correlate negatively)
    |
    |-- Unrelated Constructs:
        |-- "Random SAE absorption" (should be near-zero)
        |-- "Model size" (should not correlate)
```

**Diagnostic experiment**: The key test is whether first-letter absorption (Indicator 1) correlates with semantic-hierarchy absorption (Indicator 2) at r > 0.6 (convergent validity) and whether semantic-hierarchy absorption > non-hierarchy control absorption (discriminant validity). If both hold, the metric has construct validity. If either fails, the metric is suspect.

### Selected Front-Runner

**Candidate A: Nomological Network Validation of the SAEBench Feature Absorption Metric**

Rationale:
1. **Strongest alignment with current research direction**: The project is already conducting a construct-validity study. This perspective provides the theoretical framework.
2. **Most rigorous methodology**: Psychometric construct validation is a 70-year-old field with established protocols.
3. **Highest impact potential**: If the metric lacks construct validity, it undermines a large body of follow-up work.
4. **Best alignment with constraints**: Training-free, uses existing SAEs, fits within 1-hour tasks.
5. **Complementary to existing work**: No prior SAE paper applies psychometric validation frameworks.

---

## Phase 5: Final Proposal

### Title
**Construct Validity of the SAEBench Feature Absorption Metric: A Nomological Network Approach**

### Source Principle
Cronbach & Meehl's (1955) nomological network establishes construct validity through convergent and discriminant evidence. A measure is valid if it correlates with theoretically related constructs (convergent validity) and does not correlate with theoretically unrelated constructs (discriminant validity). Campbell & Fiske's (1959) multitrait-multimethod matrix provides the methodological framework. The PROXIMA framework (2026) extends this to ML proxy metrics, emphasizing that directional accuracy is more important than correlation magnitude alone.

### Structural Correspondence

| Source (Psychometrics) | Target (SAE Absorption Metric) |
|------------------------|-------------------------------|
| Theoretical construct (e.g., "intelligence") | "Feature absorption" (unobservable property) |
| Test score (e.g., IQ) | SAEBench first-letter absorption score |
| Criterion measure (e.g., GPA) | Semantic-hierarchy absorption score (WordNet) |
| Unrelated measure (e.g., height) | Non-hierarchy correlated-feature absorption score |
| Convergent validity (r with related > threshold) | First-letter vs. semantic-hierarchy correlation > 0.6 |
| Discriminant validity (related > unrelated) | Semantic-hierarchy > non-hierarchy absorption |
| Nomological network | Network of SAE interpretability metrics |
| Proxy metric validation | PROXIMA framework: directional accuracy + robustness |

The key insight: **the SAEBench absorption metric has been adopted as a benchmark without undergoing the construct validation that psychometric theory demands.** This study applies the validation framework to test whether the metric measures a general phenomenon or a task-specific artifact.

### Hypothesis
The SAEBench first-letter absorption metric will show convergent validity (Pearson correlation with semantic-hierarchy absorption > 0.6 across 6-8 diverse SAEs) and discriminant validity (semantic-hierarchy absorption significantly higher than non-hierarchy correlated-feature absorption). If the correlation is below 0.6 or non-significant, the metric lacks construct validity as a general measure of feature absorption.

### Method

#### SAE Selection
Select 6-8 publicly available pretrained SAEs spanning the absorption-rate spectrum: Matryoshka SAE (very low), OrtSAE (low), BatchTopK (moderate), Standard ReLU (moderate-high), TopK (moderate), Gated (moderate), JumpReLU (moderate-high), Random control. Source: SAELens releases for Pythia-160M and Gemma-2-2B.

#### First-Letter Absorption (Standard SAEBench Protocol)
Compute absorption scores using ground-truth logistic probes, k-sparse probing with tau_fs = 0.03, absorption formula with tau_pa = 0, tau_ps = -1.

#### Semantic-Hierarchy Absorption (Novel Construct-Validity Test)
1. Extract 8-10 parent-child pairs from WordNet where parent is a direct hypernym of child (e.g., animal -> mammal, mammal -> dog).
2. Ensure tokens are single tokens in model vocabulary.
3. **Frequency matching**: Create synthetic balanced datasets where parent and child tokens appear with equal frequency.
4. Train logistic regression ground-truth probes on base model residual-stream activations.
5. Apply the exact SAEBench absorption formula to SAE latents using the same k-sparse probing protocol.

#### Control Condition (Non-Hierarchical Correlated Features)
Select 4-5 pairs of semantically related but non-hierarchical concepts (e.g., synonyms, co-occurring attributes). Match frequencies and compute "absorption" scores. If the metric is hierarchy-specific, these scores should be near-zero or significantly lower.

#### Random-SAE Control
Compute absorption on an SAE with randomly permuted decoder directions. Should yield near-zero absorption on all tasks.

### Diagnostic Experiment

**Experiment 1: Convergent Validity Test**
- Measure: Pearson r (first-letter absorption vs. semantic-hierarchy absorption) across 6-8 SAEs
- Prediction: r > 0.6 with 95% bootstrap CI excluding 0
- Falsification: r < 0.6 or CI includes 0

**Experiment 2: Discriminant Validity Test**
- Measure: Paired t-test comparing semantic-hierarchy vs. non-hierarchy control absorption
- Prediction: Semantic-hierarchy > non-hierarchy, p < 0.05
- Falsification: No significant difference or reversed direction

**Experiment 3: Random-SAE Control**
- Measure: Absorption scores on random decoder SAE
- Prediction: Near-zero on all tasks
- Falsification: Non-zero absorption (indicates metric artifact)

**Experiment 4: Robustness Across Thresholds**
- Measure: Correlation at tau_fs = {0.01, 0.03, 0.05}
- Prediction: r remains positive and > 0.5 across thresholds
- Falsification: r becomes negative or near-zero at any threshold

### Experimental Plan

**Models**: Pythia-160M (primary) + GPT-2 small (replication control)
**Layer**: Residual post-layer 8
**Evaluation**:
- SAEBench absorption metric (first-letter, ~26 min per SAE)
- Custom semantic-hierarchy absorption (WordNet, ~10 min per SAE)
- Custom non-hierarchy control absorption (~10 min per SAE)

**Runtime estimate**: ~0.5-1.0 GPU-hour for full experiment (training-free, probe training is lightweight)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Probe quality poor for semantic concepts | Medium | Filter to probe AUROC > 0.7; report probe-quality table |
| WordNet concepts not single tokens | Medium | Pre-filter using model tokenizer |
| Frequency matching imperfect | Medium | Use synthetic balanced datasets; report token frequencies |
| Small-n correlation noisy | Medium | Report bootstrap CIs; interpret cautiously |
| First-letter and semantic tasks use different probe difficulties | Medium | Standardize absorption scores by probe AUROC before correlating |

### Novelty Claim

The specific cross-disciplinary insight is: **the SAEBench absorption metric has been adopted as a community benchmark without undergoing the construct validation that measurement theory demands.** By applying Cronbach & Meehl's nomological network framework and Campbell & Fiske's convergent/discriminant validation, this study provides the first systematic test of whether the metric generalizes beyond its original first-letter task domain.

Evidence this hasn't been applied before:
- No prior SAE paper cites psychometric construct validation theory.
- The PROXIMA framework (2026) extends proxy validation to LLM benchmarks generally but does not address SAE absorption.
- "Establishing Construct Validity in LLM Capability Benchmarks" (2026) advocates for nomological networks but does not apply them to SAE metrics.
- Chanin et al. (2024) explicitly call for "finding examples of feature absorption unrelated to character identification" — acknowledging the validity gap but not closing it.

---

## Sources

### Psychometrics / Measurement Theory
- Cronbach, L.J. & Meehl, P.E. (1955). Construct validity in psychological tests. *Psychological Bulletin*, 52(4), 281-302.
- Campbell, D.T. & Fiske, D.W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. *Psychological Bulletin*, 56(2), 81-105.
- Kane, M.T. (2013). Validating the interpretations and uses of test scores. *Journal of Educational Measurement*, 50(1), 1-73.
- Borsboom, D., Mellenbergh, G.J. & van Heerden, J. (2004). The concept of validity. *Psychological Review*, 111(4), 1061-1071.

### Machine Learning / Benchmark Validation
- PROXIMA: Proxy Metric Validation with Segment-Level Fragility Detection (arXiv:2604.14352, 2026).
- Establishing Construct Validity in LLM Capability Benchmarks Requires Nomological Networks (arXiv:2603.15121, 2026).
- Beyond Proxy Metrics: A New Evaluation Framework for LLM Compression (OpenReview 2025).

### Neuroscience / Cognitive Science
- Chanin et al. (2024). A is for Absorption. *arXiv:2409.14507*.
- Olshausen, B.A. & Field, D.J. (1996). Emergence of simple-cell receptive field properties. *Nature*, 381, 607-609.
- Rosch, E. (1978). Principles of categorization. In *Cognition and Categorization*.
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*, 11, 127-138.
- Moosavi et al. (2025). Dynamics of energy-efficient coding in visual cortex. *Journal of Neurophysiology*.

### Physics / Information Theory
- Tishby, N., Pereira, F.C. & Bialek, W. (1999). The Information Bottleneck Method. *arXiv:physics/0004057*.
- Wu, T. & Fischer, I. (2020). Phase Transitions for the Information Bottleneck. *ICLR 2020*.
- Equitz, W.H. & Cover, T.M. (1991). Successive Refinement of Information. *IEEE Trans. Info. Theory*, 37(3), 269-275.
- Lobacheva, E. et al. (2025). SGD as Free Energy Minimization. *arXiv:2505.23489*.
- Zdeborova, L. & Krzakala, F. (2016). Statistical physics of inference. *Physica A*.

### Biology / Ecology
- Gause, G.F. (1934). *The Struggle for Existence*.
- Fetzer, I. et al. (2015). The extent of functional redundancy. *PNAS*, 112(17), 5442-5447.
- Loreau, M. & de Mazancourt, C. (2013). Biodiversity and ecosystem stability. *Ecology Letters*, 16, 106-115.
