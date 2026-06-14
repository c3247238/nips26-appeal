# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Pacela et al., 2026. "Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation." arXiv:2603.28744** — Demonstrates that SAEs fail at compositional generalization because dictionary learning (not amortized inference) is the bottleneck; SAE-learned decoder directions are substantially misaligned with ground truth even when compressed sensing theory guarantees exact recovery. This reframes the entire SAE research agenda.

2. **Li et al., 2025. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." Entropy 27(4), 344** — Examines how SAE features form geometric structures in activation space; builds on the Linear Representation Hypothesis but finds feature manifolds rather than isolated directions.

3. **Li & Ren, 2025. "Adaptive Temporal Masking for Sparse Autoencoders." arXiv:2510.08855** — Dynamic feature selection tracking activation magnitudes and frequencies over time; achieves lower absorption than TopK/JumpReLU while maintaining reconstruction quality.

4. **Gallifant et al., 2025. "Sparse Autoencoder Features for Classifications and Transferability." arXiv:2502.11367** — SAE features transfer cross-model (Gemma 2B -> 9B), cross-lingual (English -> Chinese/French/Spanish/Russian), and cross-modal (text -> vision-language); binarization offers efficient alternative.

5. **Kialy et al., 2026. "A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs." arXiv:2603.26236** — Discovered a language-agnostic informal register subspace in multilingual LLMs; zero-shot steering transfer to unseen languages.

6. **Thasarathan et al., 2025. "SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability." arXiv:2507.06265** — Concept-aligned SAEs enabling cross-model and cross-modal interpretability through shared feature spaces.

7. **OpenReview 2025. "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers"** — SAEs trained on random transformers achieve similar auto-interpretability scores to trained models; high aggregate scores do not guarantee computationally relevant features.

8. **Mueller et al., 2025. "MIB: A Mechanistic Interpretability Benchmark." ICML 2025** — SAE features are not better than raw neurons for causal variable localization; supervised DAS methods performed best.

9. **Pesce et al., 2026. "Phase Transitions in Neural Networks Pruning." arXiv:2602.15224** — Statistical physics framework for pruning; second-order critical behavior with up to 98% edge removal before performance collapse.

10. **Westphal et al., 2025. "A Generalized Information Bottleneck Theory of Deep Learning." arXiv:2509.26327** — Reformulated IB principle through synergy; GIB exhibits compression phases across diverse architectures.

11. **Baker et al., 2025. "Analysis of Variational Sparse Autoencoders." arXiv:2509.22994** — vSAE underperforms standard SAE on core metrics; excessive regularization creates more dead features. Important negative result.

12. **Jiang et al., 2025. "FedCFA: Alleviating Simpson's Paradox in Model Aggregation with Counterfactual Federated Learning." AAAI 2025** — Counterfactual framework for handling aggregation bias; factor decorrelation loss for feature independence.

### Landscape Summary

The SAE field in 2025-2026 is experiencing a methodological reckoning. Three converging lines of critique are reshaping what questions matter:

**First, the validity crisis.** The OpenReview paper showing that SAEs on random transformers score similarly to trained models, combined with MIB's finding that SAE features are no better than raw neurons, and Pacela et al.'s demonstration that SAE dictionaries are systematically misaligned with ground truth---all point to a fundamental construct validity problem. The community has been optimizing proxy metrics (reconstruction, L0, auto-interpretability scores) that may not measure what practitioners assume.

**Second, the geometry turn.** Rather than treating SAE features as isolated "grandmother cells," recent work (Li et al. 2025, SAE-NO, SPARC) emphasizes feature geometry, manifolds, and cross-model alignment. Features are not points but structured subspaces; interpretability requires understanding geometric relationships, not just individual activations.

**Third, the cross-domain transfer opportunity.** Gallifant et al. and Kialy et al. show that SAE features transfer across languages, models, and modalities. This suggests there may be "universal" feature structures that are model-independent---a profound implication that has not been connected to the absorption literature.

The critical gap that emerges from this survey: **no work has connected the absorption problem to the cross-model universality of features.** If absorption is a universal phenomenon (occurring across architectures and models), and if features themselves are universal (transferring across models), then absorption patterns should also transfer. Testing this would simultaneously address the validity crisis (by checking if absorption is a real phenomenon or a metric artifact) and the geometry turn (by examining how absorption structures feature manifolds across model spaces).

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption Manifold Hypothesis---Cross-Model Universality of Absorption Patterns

- **Hypothesis**: Feature absorption patterns are not metric artifacts but reflect a universal geometric property of sparse representations of hierarchical concepts. When the same semantic hierarchy is probed across different base models (e.g., GPT-2, Pythia, Gemma), the absorption scores for that hierarchy will correlate across models, and the absorbing latent clusters will align in a shared "absorption manifold" that is model-independent.

- **Cross-domain insight**: From differential geometry and manifold learning. Just as the same physical object casts different shadows from different light angles (projections), different LLMs may project the same " Platonic" semantic hierarchy onto their activation spaces in different but structurally related ways. The absorption phenomenon is not in the model but in the geometry of hierarchical concepts themselves---it is a property of the semantic hierarchy that manifests differently but predictably across models. This is analogous to how topological invariants (like genus) are preserved under continuous deformations.

- **Evidence for**:
  - Gallifant et al. (2025) showed SAE features transfer cross-model, implying some model-independent structure exists.
  - Kialy et al. (2026) found language-agnostic register subspaces, showing abstract concepts have portable representations.
  - SPARC (2025) aligned concepts across models, establishing that cross-model feature correspondence is measurable.
  - Chanin et al. (2024) proved absorption is incentivized by sparsity loss for hierarchical features---this is a mathematical property independent of any specific model.
  - The "Stop Probing, Start Coding" paper showed SAE dictionaries are misaligned with ground truth, but this does not preclude model-independent *relative* structure (absorption as a comparative, not absolute, phenomenon).

- **Novelty estimate**: 8/10. Cross-model absorption analysis has never been attempted. The manifold framing connects SAE absorption to the emerging geometry turn in interpretability. The topological analogy is unexpected and provides a rigorous mathematical framework.

### Candidate B: The Ecological Fallacy of Feature Absorption---Why Aggregation Misleads

- **Hypothesis**: The SAEBench absorption metric commits an "ecological fallacy" analogous to Simpson's Paradox in statistics. When parent features are "absorbed" by child latents at the aggregate level (averaged across all tokens), the metric misses that absorption is context-dependent: the parent feature may be independently represented on some token subsets (e.g., when the child feature is not contextually relevant) while being absorbed on others. Disaggregating by token context will reveal that "absorption" is not a binary property but a conditional probability that varies dramatically across contexts.

- **Cross-domain insight**: From social statistics and the ecological inference problem (Pavia & Thomsen 2025; McCartan & Kuriwaki 2025). In political science, ecological inference attempts to recover individual-level behavior from aggregate vote counts---and is notoriously prone to error because aggregation destroys subgroup information. Similarly, SAEBench aggregates absorption across all tokens for a parent concept, potentially masking context-dependent substructure. The FedCFA paper (AAAI 2025) showed that counterfactual generation can align local and global representations to prevent paradoxical aggregation---a technique that could be adapted to SAE absorption analysis.

- **Evidence for**:
  - The current project's H2 reversal (semantic-hierarchy absorption lower than non-hierarchy control) is exactly the kind of paradoxical result that ecological fallacy produces: aggregation masks the true relationship.
  - Chanin et al. (2024) noted absorption is context-dependent but their metric averages across contexts.
  - "Stop Probing, Start Coding" showed SAEs fail at compositional generalization---features that appear monosemantic in aggregate may be polysemantic under distributional shift.
  - The OpenReview paper on random transformers suggests aggregate metrics are not trustworthy without disaggregation.
  - Simpson's Paradox is mathematically guaranteed under certain conditions; if absorption metrics aggregate over heterogeneous token populations, paradoxical results are expected.

- **Novelty estimate**: 7/10. The ecological fallacy framing is genuinely new to SAE absorption. Context-dependence of absorption has been noted but never systematically analyzed as a statistical aggregation problem. The connection to FedCFA's counterfactual approach provides a methodological template.

### Candidate C: Adversarial Absorption---Can Absorbed Features Be Weaponized?

- **Hypothesis**: Absorbed parent features create "blind spots" in SAE-based safety monitoring that can be exploited adversarially. An attacker can craft inputs that trigger the child latents (which are monitored) while the parent concept's harmful effect is mediated through the absorbed subspace (which is not monitored because the parent latent appears inactive). This creates a "representation-level jailbreak" that bypasses SAE-based safety filters.

- **Cross-domain insight**: From adversarial machine learning and security. The 2025-2026 literature shows SAEs can be both defenses (SAE insertion reduces jailbreak success 5x) and attack vectors (layer-wise perturbations via SAEs generate adversarial text). The key insight is that absorption creates a *semantic blind spot*: safety systems monitoring for parent concepts (e.g., "harmful instruction") will miss attacks that route through child concepts (e.g., specific harmful subcategories) because the parent latent has been absorbed. This is analogous to how adversarial examples exploit decision boundary geometry---but at the semantic level rather than the pixel level.

- **Evidence for**:
  - "Layer-Wise Perturbations via SAEs for Adversarial Text Generation" (2025) showed SAEs can be exploited to bypass safety filters.
  - "Interpretability Illusions with SAEs" (2025) found SAE interpretations are vulnerable to minimal input perturbations.
  - "Sparse Autoencoders are Capable LLM Jailbreak Mitigators" (2026) showed SAE-based defenses work---implying their absence creates vulnerabilities.
  - Chanin et al. (2024) proved absorption creates "holes" in feature coverage---these holes are exactly blind spots.
  - The current project's finding that Random SAE equals Standard SAE on semantic hierarchies suggests the metric may miss real semantic structure---which attackers could exploit.

- **Novelty estimate**: 7/10. SAE adversarial robustness is an emerging area, but connecting absorption specifically to safety blind spots is new. The "semantic blind spot" concept bridges interpretability failure modes and security vulnerabilities.

---

## Phase 3: Self-Critique

### Against Candidate A: Absorption Manifold Hypothesis

- **Prior work attack**: Cross-model feature transfer has been studied (Gallifant et al. 2025, Kialy et al. 2026, SPARC 2025), but absorption specifically has not. However, the "universal feature" literature focuses on features that *work* across models, not on failure modes. It is possible that absorption is model-specific (dependent on architecture, training data, and hyperparameters) and does not transfer. The topological/manifold framing may be mathematically elegant but empirically vacuous if absorption patterns are idiosyncratic.

- **Methodological attack**: Defining and measuring an "absorption manifold" requires: (1) identifying corresponding hierarchies across models with different tokenizers; (2) computing absorption scores in comparable ways despite different activation dimensions; (3) defining "alignment" of absorbing clusters across models. Each step introduces confounds. Tokenizer differences between GPT-2 (BPE) and Gemma (SentencePiece) mean the same semantic concept may be tokenized differently, making cross-model comparison fraught.

- **Theoretical attack**: The topological invariant analogy may be superficial. In topology, invariants are preserved under homeomorphisms (continuous, bijective, continuous-inverse maps). There is no reason to believe different LLM activation spaces are related by homeomorphisms, or that absorption has any topological meaning. The "Platonic hierarchy" assumption---that there exists a model-independent semantic structure---is itself contested in philosophy of language and cognitive science.

- **Scalability attack**: Testing cross-model absorption requires running the absorption protocol on multiple models, each with their own SAEs. The current project already struggles with pipeline bugs on a single model. Expanding to 3+ models multiplies the debugging surface. The pilot alone would require 2 models x 3 hierarchies = 6 absorption evaluations, each potentially encountering the same bugs that plagued the current iteration.

- **Verdict**: MODERATE. The core insight is strong and novel, but the topological framing may be more poetic than rigorous. The cross-model comparison is confounded by tokenizer differences. However, if reframed as "cross-model consistency of absorption" (a simpler empirical question) rather than "absorption manifold" (a theoretical claim), it becomes more testable.

### Against Candidate B: Ecological Fallacy of Absorption

- **Prior work attack**: Context-dependence of SAE features has been studied. The "Causal Interpretation of SAE Features in Vision" paper (2025) showed features have "hidden context dependencies." SAE-Track (2024) identified context-dependent feature dynamics. However, the specific framing as an ecological fallacy/Simpson's Paradox is new. The FedCFA paper (2025) addressed Simpson's Paradox in federated learning but not in SAE evaluation.

- **Methodological attack**: Disaggregating absorption by token context requires defining "contexts." How? By co-occurring tokens? By sentence topic? By syntactic position? Each choice is arbitrary and may introduce its own biases. The current project's H2 reversal may have a simpler explanation (pipeline bugs) than an ecological fallacy. Attributing buggy results to a sophisticated statistical phenomenon risks over-interpreting noise.

- **Theoretical attack**: The ecological fallacy requires aggregation over heterogeneous subgroups. Are token contexts truly "subgroups" in the statistical sense? In political science, ecological inference aggregates over geographic units with distinct demographics. In SAE absorption, tokens are aggregated over a corpus. But tokens are not independent observations---they are sequentially dependent. The analogy may not hold because the aggregation mechanism is different.

- **Scalability attack**: This is the most feasible of the three candidates. Disaggregation requires: (1) running the existing absorption protocol; (2) stratifying results by token context; (3) comparing aggregate vs. disaggregate scores. This can be done training-free on existing SAEs. The main cost is computing absorption scores per-context rather than globally, which is a modest increase over the current protocol.

- **Verdict**: STRONG. The idea withstands most attacks. The ecological fallacy framing is genuinely new and provides a rigorous statistical lens. The experiment is feasible within constraints. The main risk is over-interpreting the current iteration's buggy results, but this can be addressed by running the disaggregation on a corrected pipeline.

### Against Candidate C: Adversarial Absorption

- **Prior work attack**: SAE adversarial robustness is an active area (2025-2026 papers on SAE-based jailbreak mitigation, interpretability illusions, and layer-wise perturbations). However, no prior work specifically connects *absorption* to adversarial vulnerability. The closest is the "holes in feature coverage" observation from Chanin et al. (2024), but they did not explore security implications.

- **Methodological attack**: Testing whether absorbed features create safety blind spots requires: (1) identifying a harmful parent concept and its child concepts; (2) showing that safety filters monitor the parent latent; (3) crafting inputs that trigger child latents while bypassing parent monitoring; (4) demonstrating harmful output. This is essentially a red-teaming exercise, which is valuable but may be seen as engineering rather than research. The causal chain (absorption -> blind spot -> successful attack) has many links, any of which could fail.

- **Theoretical attack**: The "semantic blind spot" concept assumes that safety monitoring uses SAE features. In practice, most safety systems use input/output classifiers, not internal SAE monitoring. The attack scenario may be unrealistic for current deployment settings. However, as SAE-based interpretability matures, internal monitoring may become more common---making this a forward-looking contribution.

- **Scalability attack**: The experiment requires access to safety-aligned models and harmful content generation, which raises ethical concerns and may violate platform policies. Even if conducted responsibly, the evaluation is subjective (what counts as a "successful" jailbreak?). The training-free constraint is satisfied, but the ethical constraints may be prohibitive.

- **Verdict**: MODERATE. The idea is novel and timely, but the ethical risks and the gap between current safety practices and SAE-based monitoring make it less immediately applicable. Could be reframed as a "vulnerability analysis" rather than an attack demonstration.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Adversarial Absorption)** dropped because: (1) ethical risks of demonstrating jailbreaks are significant; (2) current safety systems do not primarily rely on SAE monitoring, making the attack scenario speculative; (3) the scope is more engineering than science; (4) the project's training-free constraint is satisfied but the ethical constraints are not. Could be revived as a conceptual framing (absorption creates theoretical vulnerabilities) without actual attack demonstrations.

### Strengthened Ideas

- **Candidate A (Absorption Manifold)**: Reframed from a topological claim to an **empirical consistency claim**. Instead of claiming absorption forms a "manifold" (which requires heavy mathematical machinery), we ask a simpler question: Do absorption scores for the same semantic hierarchy correlate across different base models? This is testable with standard correlation analysis. The topological framing becomes a conceptual metaphor rather than a theoretical claim. We also add a control: if absorption scores do NOT correlate across models, this would suggest absorption is a model-specific artifact rather than a universal phenomenon---a valuable negative result.

- **Candidate B (Ecological Fallacy)**: Strengthened by connecting to the current project's specific anomalies. The H2 reversal (semantic-hierarchy absorption lower than control) is exactly what an ecological fallacy would predict if absorption is concentrated in specific token contexts that are diluted by aggregation. We design a concrete disaggregation strategy:
  1. For each parent concept, identify token subsets where the child feature is active vs. inactive (using the SAE's own latents).
  2. Compute absorption scores separately on each subset.
  3. If absorption is concentrated in the child-active subset, the aggregate metric underreports true absorption (because child-inactive tokens dilute the score).
  4. If absorption is uniform across subsets, the ecological fallacy hypothesis is falsified.

### Additional Evidence Found

- **"Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry" (Hindupur et al., 2025, arXiv:2503.01822)**: Establishes a formal duality between SAE structure and geometric concept properties. Supports the geometric framing of Candidate A.

- **"A Survey on Latent Semantic Geometry via AutoEncoder" (2025/2026, arXiv:2506.20083v4)**: Comprehensive survey on how autoencoder architectures shape latent space geometry. Explicitly discusses "geometrically entangled" features in transformers---directly relevant to absorption as an entanglement phenomenon.

- **"Route Sparse Autoencoder" (Shi et al., EMNLP 2025)**: Routing mechanism for multi-layer feature extraction. Suggests that absorption may be layer-dependent in ways that single-layer analysis misses.

- **"Interpretability Illusions with SAEs" (2025, arXiv:2505.16004)**: SAE interpretations are highly vulnerable to minimal input perturbations. Supports the ecological fallacy hypothesis: small context changes (perturbations) can dramatically alter feature interpretations, which aggregate metrics would miss.

### Selected Front-Runner

**Candidate B: The Ecological Fallacy of Feature Absorption** is selected as the front-runner because:

1. It directly explains the current iteration's anomalous results (H2 reversal, Random=Standard) through a well-established statistical lens, rather than treating them as mere bugs.
2. It is perfectly aligned with the training-free constraint and can leverage the existing SAEBench infrastructure.
3. It has clear falsification criteria and a tight experimental protocol.
4. The ecological fallacy framing connects SAE absorption to a century of statistical methodology, providing rigorous theoretical grounding.
5. The results have immediate practical implications: if absorption is context-dependent, the community needs context-aware metrics, not just aggregate scores.
6. It is distinct from all prior perspectives in this project (no one has proposed disaggregating absorption by token context).

**Candidate A is retained as a strong backup** because the cross-model consistency question is important and testable, but it requires a corrected pipeline first.

---

## Phase 5: Final Proposal

### Title
**The Ecological Fallacy of Feature Absorption: Why Aggregate Metrics Mislead and How Context-Aware Evaluation Fixes It**

### Hypothesis

**H1 (Context-Dependence Hypothesis):** Feature absorption is not uniform across token contexts. When disaggregated by child-feature activation status, absorption scores will be significantly higher on tokens where child latents are active compared to tokens where they are inactive. The aggregate SAEBench absorption score is a weighted average that dilutes true absorption by including child-inactive tokens where absorption is minimal.

**H2 (Ecological Fallacy Hypothesis):** The reversal of hierarchy specificity observed in aggregate metrics (semantic-hierarchy absorption < non-hierarchy control) will disappear or invert when computed on the child-active token subset alone. The apparent paradox is an artifact of aggregation over heterogeneous token populations, not a genuine property of the SAE representation.

**H3 (Metric Correction Hypothesis):** A context-aware absorption metric that conditions on child-feature activation will show stronger correlation with first-letter absorption scores than the aggregate metric, because the aggregate metric's noise from child-inactive tokens attenuates the true signal.

### Motivation

The SAE community relies on aggregate absorption scores to compare architectures and guide research. But aggregation can mislead. In social statistics, the ecological fallacy---drawing individual-level conclusions from aggregate data---has been a known pitfall for over a century. Simpson's Paradox, a special case, can cause associations to reverse upon disaggregation. The FedCFA paper (AAAI 2025) recently showed that machine learning is not immune: federated model aggregation can suffer Simpson's Paradox when local and global feature distributions differ.

Our current iteration's results exhibit classic symptoms of ecological fallacy:
- **H2 reversed**: Semantic-hierarchy absorption (0.235) < non-hierarchy control (0.331). In a well-behaved metric, hierarchical features should show more absorption. The reversal suggests aggregation is masking the true relationship.
- **Random = Standard**: Random and Standard SAEs produce identical scores. If absorption were a genuine property of learned structure, random decoders should behave differently. The equality suggests the metric is measuring something other than absorption---possibly a property of the token distribution that is invariant to SAE structure.
- **GPT-2 divergence**: Near-zero absorption on GPT-2 vs. substantial absorption on Pythia. If the metric measured a universal phenomenon, results should be more consistent across models.

These anomalies are individually explainable as bugs, but collectively they match the signature of aggregation bias. If true, the implications are profound: the entire SAEBench absorption metric may need redesign, and architectures optimized for aggregate absorption may not improve behavior on the tokens where absorption actually matters.

### Method

#### SAE Selection
Use the same 8 SAEs from the current iteration (already downloaded and evaluated):
- MatryoshkaBatchTopK, TopK, BatchTopK, Standard, GatedSAE, JumpReLU, PAnneal, Random control
- Model: Pythia-160M-deduped, layer 8 (primary); GPT-2 small (replication)

#### Context-Aware Absorption Protocol

**Step 1: Run standard SAEBench absorption detection**
For each parent concept (first-letter or semantic hierarchy), identify:
- Main latents (k-sparse probing, k=1..10)
- Absorbing latents (SAEBench criteria)
- Absorption score (aggregate)

**Step 2: Stratify tokens by child-feature activation**
For each parent-child pair:
- Compute child-feature activation on each token in the evaluation set.
- Split tokens into two strata:
  - **Child-active**: Tokens where at least one child latent exceeds its median activation.
  - **Child-inactive**: Tokens where no child latent exceeds median activation.

**Step 3: Compute disaggregated absorption scores**
Apply the SAEBench absorption formula separately to each stratum:
- absorption_child_active = absorption score computed only on child-active tokens
- absorption_child_inactive = absorption score computed only on child-inactive tokens

**Step 4: Test the ecological fallacy hypothesis**
- If H1 is true: absorption_child_active >> absorption_child_inactive for hierarchical pairs.
- If H2 is true: On child-active tokens, semantic-hierarchy absorption > non-hierarchy control absorption (reversing the aggregate paradox).
- If H3 is true: Correlation(first-letter, semantic-hierarchy) is stronger when using absorption_child_active than aggregate absorption.

#### Control Conditions
1. **Random stratification**: Split tokens randomly (not by child activation) and recompute absorption. Should show no difference between strata.
2. **Non-hierarchical pairs**: Apply the same disaggregation to non-hierarchy correlated features. Should show minimal difference between strata (no ecological fallacy because there is no true hierarchy).
3. **Median threshold sensitivity**: Test thresholds at 25th, 50th, 75th percentiles to ensure results are not artifacts of threshold choice.

### Experimental Plan

**Primary benchmarks:**
- SAEBench Feature Absorption (first-letter) --- aggregate and disaggregated.
- Custom Semantic-Hierarchy Absorption --- aggregate and disaggregated.
- Custom Non-Hierarchy Control --- aggregate and disaggregated.

**Metrics:**
- Aggregate absorption score (per parent, per SAE)
- Disaggregated absorption score (child-active, child-inactive)
- Difference: delta = absorption_child_active - absorption_child_inactive
- Paired t-test: Is delta > 0 for hierarchical pairs?
- Paired t-test: Is delta ≈ 0 for non-hierarchical pairs?
- Pearson correlation: first-letter vs. semantic-hierarchy (aggregate vs. child-active-only)

**Statistical test plan:**
- Bootstrap 95% CI for all correlations (B = 10,000)
- Paired t-tests with Bonferroni correction for multiple comparisons
- Report all raw scores in appendix for transparency

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| **Varying activation threshold** (25th, 50th, 75th percentile) | Whether threshold choice drives results | Delta should be positive across thresholds; magnitude may vary |
| **Alternative stratification** (by parent-feature activation instead of child) | Whether any activation-based split produces the same effect | Parent-active split should show less dramatic difference (parent is the "absorbed" feature) |
| **Single-token vs. multi-token concepts** | Whether tokenization interacts with stratification | Single-token should be cleaner |
| **GPT-2 replication** | Whether ecological fallacy is model-specific | Pattern should replicate if the phenomenon is general |

### Pilot Design
- **Scope:** 2 SAEs (Matryoshka + Standard) x 3 first-letter parent-child pairs x 2 stratification conditions.
- **Target runtime:** 10-15 minutes on a single GPU.
- **Success criterion:** Pilot shows absorption_child_active > absorption_child_inactive for at least 2/3 hierarchical pairs, with numerical stability. If not, the ecological fallacy hypothesis is falsified at pilot stage.

### Resource Estimate
- **GPU-hours:** ~0.3-0.5 GPU-hour for the full experiment (8 SAEs, disaggregation is lightweight post-processing).
- **Model sizes:** Pythia-160M-deduped (primary), GPT-2 small (replication).
- **All tasks well under the 1-hour limit.**

### Risk Assessment

| Threat | Mitigation |
|--------|------------|
| **Current pipeline bugs persist** | Fix bugs first (as per validation decision); run pilot to verify pipeline integrity |
| **Child-feature activation is noisy** | Use median threshold; test sensitivity; aggregate across multiple child latents |
| **Token subsets are too small** | Ensure minimum 50 tokens per stratum; exclude parent concepts with insufficient data |
| **Disaggregation introduces multiple comparison problems** | Use Bonferroni correction; pre-register primary comparisons |
| **Ecological fallacy framing is seen as overly metaphorical** | Ground in formal statistical definitions; cite established literature; let the data speak |

### Novelty Claim

This would be the **first application of ecological fallacy/Simpson's Paradox analysis to SAE evaluation metrics**. We searched arXiv and Google Scholar for papers disaggregating SAE absorption by token context or applying statistical aggregation-bias frameworks to interpretability metrics. Key findings:

- **Chanin et al. (2024)** noted context-dependence of absorption but did not analyze it as an aggregation problem.
- **SAEBench (2025)** uses aggregate absorption scores exclusively; no disaggregation is performed.
- **"Stop Probing, Start Coding" (2026)** showed SAEs fail at compositional generalization, implying aggregate metrics miss context-dependent structure, but did not connect this to ecological fallacy.
- **FedCFA (2025)** applied Simpson's Paradox analysis to federated learning but not to SAE evaluation.
- **No prior work** has systematically tested whether SAE absorption metrics suffer from aggregation bias or proposed context-aware alternatives.

The novelty is both methodological (importing ecological inference theory to SAE evaluation) and empirical (providing the first disaggregated absorption analysis). If the ecological fallacy hypothesis is supported, it would imply that a widely used benchmark metric has been systematically misleading the field---a high-impact finding with immediate implications for benchmark design.
