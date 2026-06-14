# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al. (2024). "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025)** -- The foundational empirical characterization of feature absorption. Proves absorption is gradient-descent-stable in a 2-feature toy model. Proposes the canonical absorption metric based on integrated-gradients ablation. Finds 15-35% absorption across all tested SAEs. Critical because it establishes the phenomenon but restricts evaluation to the first-letter spelling task and requires known probe directions.

2. **Tang et al. (2025). "A Unified Theory of Sparse Dictionary Learning." arXiv:2512.05534** -- Casts all SDL variants as piecewise biconvex optimization. Theorem 3.10 proves hierarchical features naturally induce absorption as spurious partial minima. Proposes feature anchoring to restore identifiability. Matters because it locates absorption in the optimization landscape structure, not training accidents, but feature anchoring remains validated only on synthetic benchmarks.

3. **Bussmann et al. (2025). "Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** -- Nested dictionary training forces coarser levels to reconstruct alone, breaking the absorption incentive. Best SAEBench absorption scores. Matters because it demonstrates that explicit hierarchy in training can reduce absorption, but inner levels trade absorption for hedging (Chanin & Dulka 2025).

4. **Korznikov et al. (2025). "OrtSAE." arXiv:2509.22033** -- Pairwise cosine similarity penalty on decoder columns reduces absorption by 65%. Matters because it shows structural constraints on the decoder geometry can mitigate absorption without hierarchy-aware training, suggesting the decoder weight space encodes absorption signatures.

5. **Michaud, Baek, Engels, Sun & Tegmark (2025). "The Geometry of Concepts: SAE Feature Structure." Entropy 27(4)** -- Examines SAE decoder weight geometry at three spatial scales: "crystal" (parallelogram/trapezoid structures), "brain" (functional lobes), "galaxy" (global shape). Matters because it establishes that SAE decoder weights have rich geometric structure that can be analyzed without activation data -- the same insight that motivates weight-based absorption detection.

6. **Chanin et al. (2025). "Feature Hedging." arXiv:2505.11756** -- Identifies hedging as absorption's dual failure mode in the opposite capacity regime (narrow SAEs merge correlated features). Shows Matryoshka SAEs trade absorption for hedging. Matters because any absorption study must account for the absorption-hedging tradeoff, and a unified analysis of both is a significant gap.

7. **Jenatton et al. (2011). "Proximal Methods for Hierarchical Sparse Coding." JMLR 12** -- Tree-structured sparsity regularization that enforces sparsity patterns to form connected subtrees. Matters as a cross-domain insight: the exact structure of feature hierarchies that causes absorption in SAEs is precisely the structure that hierarchical sparsity priors were designed to handle in signal processing. This connection has never been drawn.

8. **Klindt et al. (2025). "From Superposition to Sparse Codes." arXiv:2503.01824** -- Connects SAE feature recovery to compressed sensing theory. Proves SAE encoders are suboptimal sparse decoders due to architectural limitations. Matters because it provides the compressed-sensing toolkit for analyzing when hierarchical features are recoverable and when absorption becomes unavoidable.

9. **Bereska et al. (2025). "Superposition as Lossy Compression." arXiv:2512.13568 (ICML 2025)** -- Rate-distortion framework for superposition. Absorption saves bits (reduces effective sparsity) without increasing distortion. Matters because it provides the information-theoretic "reason" absorption occurs: it is compression-optimal for hierarchical features under bandwidth constraints.

10. **Dyer, Rutishauser & Baraniuk (2012). "Group Sparse Coding with Winner-Take-All Networks." BMC Neuroscience** -- Shows how collections of winner-take-all networks solve group sparse coding, finding representations that minimize the number of active functional groups. Matters as a neuroscience analogy: WTA dynamics in the brain produce exactly the kind of competitive suppression between hierarchically related features that SAE absorption mimics.

11. **Narayanaswamy et al. (2026). "Masked Regularization for SAEs." arXiv:2604.06495** -- Disrupts co-occurrence patterns via token masking during training. Reduces absorption across architectures. Matters as the most recent mitigation approach, but is limited to the first-letter evaluation task with no cross-domain validation.

12. **Karvonen et al. (2025). "SAEBench." arXiv:2503.09532** -- Standardized 8-metric evaluation across 200+ SAEs. Finds proxy metrics (CE loss, L0) do not predict practical performance. Matters because it provides the evaluation infrastructure and reveals that absorption measurement itself has confounds (dead features masking absorption rates).

### Landscape Summary

The feature absorption landscape in early 2026 is characterized by a productive but increasingly urgent tension. On one side, absorption has been theoretically grounded: Chanin et al. proved it is gradient-descent-stable, Tang et al. showed it arises from spurious partial minima in the SDL optimization landscape, and Bereska et al. demonstrated it is information-theoretically optimal under compression constraints. On the other side, every proposed mitigation has significant limitations: Matryoshka SAEs trade absorption for hedging, OrtSAE's 65% reduction still leaves substantial residual absorption, masked regularization requires retraining, and feature anchoring has only been validated on synthetic data.

The critical missing piece -- and the gap I will target -- is the disconnect between detection and intervention. Current absorption detection requires supervised probe directions (you must know what features to look for), while current mitigations require retraining the SAE. No method exists that can (a) detect absorption from the SAE weights alone without supervision, AND (b) correct it at inference time without retraining. This gap matters enormously because the majority of SAEs in the ecosystem (400+ Gemma Scope SAEs, all Neuronpedia SAEs) are already trained and deployed. A training-free detection-and-correction pipeline would have immediate impact on the entire community.

A second critical gap is the domain restriction. Every absorption study evaluates on the first-letter spelling task -- a syntactic hierarchy where the parent ("starts with L") is trivially predictable from the child ("token 'lion'"). Whether absorption occurs with the same severity in semantically richer hierarchies (entity types, knowledge taxonomies, safety-relevant features) is completely untested. This matters because the safety motivation for SAE research depends on absorption being a real problem for the features that matter (deception, harmful intent), not just a curiosity in spelling tasks.

The cross-domain insight I bring from signal processing is that hierarchical dictionary learning has already solved a structurally analogous problem: multi-scale wavelet representations naturally encode features at different granularities, and tree-structured sparsity priors (Jenatton et al. 2011) ensure that activating a fine-scale atom implies activating its coarse-scale parent. This "parent-must-fire-when-child-fires" constraint is precisely what absorption violates, and it can be enforced post-hoc on already-trained SAEs.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Spectroscopy -- Diagnosing Absorption from Weight Geometry Alone

- **Hypothesis**: The encoder-decoder angular divergence (EDA) of an SAE latent, defined as EDA(j) = 1 - cos(w_{e,j}, d_j), combined with a directional decomposition of the encoder residual onto the full decoder dictionary, can detect absorbed latents with AUROC >= 0.80 against Chanin et al. ground truth labels, and the residual decomposition pattern reveals the identity of the absorber latent without any activation data.

- **Cross-domain insight**: This is "spectroscopy" in the physics sense -- just as optical spectroscopy identifies chemical composition from the absorption and emission lines of a material without destructive testing, absorption spectroscopy identifies which SAE latents have been absorbed by analyzing the "spectral lines" (directional components) of their encoder weight residuals against the decoder dictionary. The residual r_j = w_{e,j} - proj_{d_j}(w_{e,j}) decomposes into contributions from other decoder columns; peaks in this decomposition are the "absorption lines" that reveal which latents are doing the absorbing.

- **Why it might work**: (a) Chanin et al.'s informal LessWrong post (Oct 2024) observed that "discrepancies between encoder and decoder directions would indicate absorption" -- EDA formalizes this. (b) The Broken Latents result shows tied SAEs (EDA=0 by construction) eliminate absorption in the overcomplete regime, establishing EDA as the absorption-free baseline. (c) Michaud et al. (2025) showed SAE decoder weights have rich geometric structure analyzable without activation data. (d) OrtSAE reduces absorption by constraining decoder column cosine similarity, confirming decoder geometry encodes absorption information.

- **Novelty estimate**: 7/10 -- The scalar EDA metric formalizes an existing informal observation. The directional decomposition (D-EDA) and the "absorption graph" it induces are genuinely novel; no prior work models absorption as a directed graph. The "spectroscopy" framing is new but must demonstrate it adds substantive insight beyond the metaphor.

### Candidate B: Competitive Exclusion Theory of Feature Absorption

- **Hypothesis**: Feature absorption in SAEs is formally equivalent to competitive exclusion in ecology -- when two features share the same "representational niche" (reconstruction variance explained on co-occurring inputs), the sparsity penalty acts as resource limitation, and the more specialized feature drives the general feature to extinction (zero activation). This predicts: (a) absorption severity scales with niche overlap (measured by conditional mutual information), (b) absorption follows a power-law distribution where a few "keystone" child features absorb most of the parent's activation mass, and (c) introducing "niche differentiation" (forcing parent and child features to explain different reconstruction components) eliminates absorption.

- **Cross-domain insight**: Gause's competitive exclusion principle (1934) states that two species competing for the same limiting resource cannot stably coexist. The sparsity penalty in SAEs is the "limiting resource" (each activation slot is a scarce resource under L0 constraint). Parent and child features "compete" for the same activation slots on co-occurring inputs. The child feature "wins" because it is more specialized (lower entropy, better reconstruction per activation), driving the parent to extinction (zero activation = absorption). Niche differentiation in ecology corresponds to forcing features to explain orthogonal reconstruction components.

- **Why it might work**: (a) Tang et al. (2025) prove absorption is a property of the optimization landscape under hierarchy -- competitive exclusion provides the ecological interpretation of WHY this landscape structure exists. (b) Chanin et al.'s result that absorption saves sparsity (dL/ddelta = -p_{11} < 0) is exactly the ecological statement that the specialist outcompetes the generalist for shared resources. (c) The Matryoshka SAE's success can be reinterpreted as "temporal niche partitioning" -- training coarse levels alone creates time windows where the generalist has no competitor. (d) OrtSAE's orthogonality penalty is "spatial niche differentiation" -- forcing decoder columns apart is forcing features into different representational niches.

- **Novelty estimate**: 9/10 -- No one has drawn this connection. The analogy is not superficial: it generates specific quantitative predictions (power-law distribution of absorption, niche overlap as predictor) and reinterprets all existing mitigations through a single unified lens. Risk: the analogy may be a metaphor that doesn't generate testable predictions beyond what the optimization-theory explanation already provides.

### Candidate C: Inference-Time Absorption Correction via Tree-Structured Sparsity Priors

- **Hypothesis**: Post-hoc enforcement of tree-structured sparsity constraints on SAE activations -- requiring that if a child feature fires, its parent must also fire -- can reduce absorption rate by >= 50% on the first-letter task and generalize to entity-type hierarchies, without retraining the SAE and with < 5% increase in reconstruction error.

- **Cross-domain insight**: From hierarchical dictionary learning in signal processing (Jenatton et al. 2011), tree-structured sparsity priors enforce that sparsity patterns form connected rooted subtrees. The proximal operator for this constraint can be computed in O(n) time. The key transplant: if we know (or can infer from decoder weight geometry) which SAE latents form parent-child pairs, we can impose the constraint "child fires implies parent fires" as a post-processing step on already-computed SAE activations.

- **Why it might work**: (a) Absorption's defining symptom is that the child fires but the parent does not. The tree-structured constraint directly forbids this pattern. (b) Jenatton et al. proved that hierarchical sparsity priors improve signal recovery for tree-structured signals -- and hierarchical features are tree-structured by definition. (c) The parent feature's decoder direction exists in the SAE (for "late absorption" cases identified by the D-EDA metric); it just fails to activate because the encoder suppresses it. Re-activating it with the correct magnitude should improve reconstruction at the cost of slightly higher L0. (d) This is entirely training-free: load any SAE, infer the hierarchy from decoder weight cosine structure, apply the tree constraint, done.

- **Novelty estimate**: 8/10 -- Tree-structured sparsity priors are well-established in signal processing but have never been applied to SAE post-hoc correction. The combination with weight-based hierarchy inference (from D-EDA) to make this fully unsupervised is novel. Risk: hierarchy inference from decoder weights may be too noisy to construct a reliable tree.

---

## Phase 3: Self-Critique

### Against Candidate A: Absorption Spectroscopy

- **Prior work attack**: The scalar EDA metric (1 - cos(w_e, d)) is a trivial computation that Chanin et al. themselves informally suggested in their October 2024 LessWrong post. The "Empirical Insights into Feature Geometry" (LessWrong, Jan 2025) directly computed encoder-decoder cosine similarity distributions. The EleutherAI seed similarity study showed average within-SAE encoder-decoder alignment is ~0.72, meaning most latents have EDA ~0.28. The directional decomposition (D-EDA) is more novel but is essentially sparse regression of one weight vector onto a dictionary -- standard in compressed sensing. **Severity: MODERATE.** The scalar EDA is not novel; D-EDA and the absorption graph add genuine novelty.

- **Methodological attack**: The EDA noise floor is a critical risk. If the baseline EDA distribution has high variance (many non-absorbed latents with high EDA due to polysemanticity, training noise, or dead feature recovery), discriminative power will be poor. The pragmatist's analysis correctly identifies that if absorbed latents only shift EDA from 0.28 to 0.35, AUROC will be dismal. Additionally, D-EDA requires decomposing one encoder row onto the full decoder dictionary, which for 65k-wide SAEs means 65k-dimensional sparse regression per latent -- this may be computationally significant even though it uses only weights. **Severity: MODERATE.** Addressable by (a) validating on first-letter ground truth before making broader claims, (b) computing D-EDA only for high-EDA latents (top 10%), (c) using simple top-k projection rather than full sparse regression.

- **Theoretical attack**: The claim that encoder-decoder misalignment "indicates" absorption assumes that absorption is the primary cause of such misalignment. But polysemantic latents (encoding multiple distinct concepts) will also have high EDA, since the encoder must point in a compromise direction between multiple decoder-decodable concepts. Dead features recovering from near-death may have arbitrary encoder-decoder angles. Feature composition (one latent encoding two co-occurring but independent concepts) also causes encoder-decoder divergence. Without controlling for these confounds, EDA may have poor specificity even if it has good sensitivity. **Severity: MODERATE-HIGH.** This is the most serious theoretical concern. D-EDA's residual analysis can partially address it (absorption produces peaked residuals aligned with specific absorber decoder columns; polysemanticity produces diffuse residuals), but this needs empirical validation.

- **Scalability attack**: D-EDA scales well because it requires only weight matrices, not activation data. The absorption graph can be built for any SAE in O(n_latents x d_model) time. No scalability concern.

- **Verdict: MODERATE.** The scalar EDA faces serious specificity concerns and limited novelty. D-EDA and the absorption graph add substantial value if the residual decomposition reliably distinguishes absorption from polysemanticity. Must validate empirically before making strong claims.

### Against Candidate B: Competitive Exclusion Theory

- **Prior work attack**: Searched for "competitive exclusion sparse coding," "ecological niche sparse autoencoder," "niche differentiation feature learning" -- found no prior work making this exact analogy. The "Diversity Through Exclusion" (DTE) paper (Sunehag et al., AAMAS 2023) uses competitive exclusion to motivate multi-agent RL diversity, but in a completely different context. The analogy to SAE absorption is genuinely novel. **Severity: LOW.** No prior work found.

- **Methodological attack**: The competitive exclusion framework is primarily a conceptual reinterpretation, not a computational method. The specific predictions (power-law distribution, niche overlap as predictor) are testable but may not yield insights beyond what the optimization-theoretic explanation already provides. If absorption rates follow a power law, that could also be explained by the Zipfian frequency distribution of features. If niche overlap predicts absorption, that is equivalent to saying co-occurrence probability predicts absorption -- which Chanin et al. already showed informally. The risk is that the ecological framing adds conceptual color but no additional predictive power. **Severity: HIGH.** This is a potentially fatal flaw for a standalone contribution. The framework must generate at least one prediction that the optimization-theoretic account does NOT make.

- **Theoretical attack**: The analogy has structural limits. In ecology, competitive exclusion operates in continuous time with stochastic perturbations and spatial heterogeneity. In SAEs, "competition" happens via gradient descent on a fixed loss function. The dynamics are fundamentally different: SAE training converges to a fixed point, while ecological dynamics may cycle or oscillate. Gause's principle applies to two species on one resource; feature absorption involves many features with complex correlation structure. The analogy may break down precisely when it is most needed (multi-level hierarchies with partially overlapping features). **Severity: MODERATE.** The structural correspondence is imperfect but the core mechanism (sparsity-as-resource-limitation) is sound. The analogy should be presented as an interpretive framework, not a formal equivalence.

- **Scalability attack**: No scalability concern -- this is a theoretical/conceptual contribution.

- **Verdict: MODERATE.** Novel framing with genuine intuitive value, but risks being "just a metaphor" unless it generates specific testable predictions that alternative frameworks miss. Best used as a secondary conceptual contribution, not the paper's core.

### Against Candidate C: Inference-Time Absorption Correction via Tree-Structured Sparsity

- **Prior work attack**: Searched for "tree-structured sparsity autoencoder," "hierarchical constraint sparse autoencoder inference" -- found no work applying tree-structured sparsity priors as post-hoc correction on trained SAEs. Matryoshka SAEs build hierarchy into training, but do not apply tree constraints at inference time on standard SAEs. Feature anchoring (Tang et al. 2025) requires retraining. ITAC (proposed in the previous innovator round) is related but uses a different mechanism (residual-based re-activation rather than tree-constraint enforcement). **Severity: LOW.** The specific combination of tree-structured sparsity priors from signal processing + post-hoc SAE correction is novel.

- **Methodological attack**: The critical challenge is inferring the feature hierarchy from decoder weights alone. If we define parent(j) = argmax_{k != j} cos(d_j, d_k) when cos(d_j, d_k) > threshold, the resulting tree may be highly noisy. Many unrelated features may have moderate cosine similarity by chance in a 65k-dimensional dictionary. False parent-child edges would cause the tree constraint to erroneously activate unrelated latents, increasing false positives and degrading interpretability. Additionally, the tree constraint increases L0 (by forcing parents to fire when children fire), which directly trades off against the sparsity that the SAE was optimized for. If the L0 increase is substantial (e.g., >30%), the correction may degrade other SAE properties (reconstruction error, downstream task performance). **Severity: MODERATE-HIGH.** Hierarchy inference noise is the Achilles heel. Must demonstrate that decoder cosine similarity provides a sufficiently clean signal for parent-child identification. Could mitigate by using a very conservative threshold (only very high cosine similarity pairs) and by validating against known hierarchies (first-letter, RAVEL entities).

- **Theoretical attack**: The tree-structured sparsity prior assumes the features form a tree. But real feature hierarchies in LLMs may form directed acyclic graphs (DAGs) rather than trees -- a feature like "animal" may be parent to both "dog" and "mammal," while "mammal" is also parent to "dog." If the hierarchy is a DAG, tree-structured priors will either miss edges or create incorrect edges. Also, the Jenatton et al. (2011) guarantees assume the dictionary is fixed and the sparsity pattern is the optimization variable; in SAE post-hoc correction, we are modifying activations of a fixed (and potentially poorly conditioned) encoder, which may not satisfy the theoretical conditions for improvement. **Severity: MODERATE.** DAG vs. tree is a real concern but can be addressed by using group sparsity priors (submodular) rather than strict tree priors. The theoretical guarantees may not transfer, but empirical validation on known hierarchies can establish practical effectiveness.

- **Scalability attack**: The tree-structured proximal operator is O(n) per input (Jenatton et al. 2011). For a 65k SAE, this adds negligible overhead. The hierarchy inference step is O(n^2) for pairwise decoder cosine similarity, which for 65k latents means ~4 billion operations -- significant but feasible as a one-time preprocessing step (~30 seconds with batched matrix multiplication on GPU). **Severity: LOW.** Computationally tractable.

- **Verdict: STRONG.** Novel combination of well-established signal processing technique with SAE post-hoc correction. The hierarchy inference noise is the main risk but is manageable with conservative thresholds and empirical validation. The training-free nature is a massive practical advantage. This is the most implementable and highest-impact candidate.

---

## Phase 4: Refinement

### Dropped Ideas

- None dropped entirely, but Candidate B (Competitive Exclusion Theory) is demoted from potential core contribution to a supporting conceptual framework. Reason: the self-critique revealed it risks being "just a metaphor" without generating predictions that alternative frameworks miss. However, its interpretive value remains high -- it provides an intuitive vocabulary for explaining WHY absorption occurs and WHY different mitigations work. It will be integrated as a framing device, not a standalone contribution.

### Strengthened Ideas

**Candidate A (Absorption Spectroscopy / D-EDA) strengthened as follows:**

1. *Addressed specificity concern*: D-EDA explicitly decomposes the encoder residual to distinguish absorption (peaked residual aligned with specific absorber decoder columns) from polysemanticity (diffuse residual across many unrelated columns). This decomposition produces an "absorption spectrum" per latent -- a vector of coefficients showing how much each other decoder column contributes to the encoder's deviation from its own decoder. Absorption produces a sparse spectrum; polysemanticity produces a dense one. This distinction is novel and addresses the empiricist's falsification concern.

2. *Added absorption graph analysis*: The sparse absorption spectra induce a directed graph where edges represent "latent j's encoder borrows from latent k's decoder direction." Graph-level metrics (degree distribution, connected components, cascade depth) provide structural insights unavailable from per-latent metrics.

3. *Operationalized validation*: First validate EDA and D-EDA against Chanin et al. ground truth on the first-letter task before any cross-domain claims. Report AUROC with bootstrap CIs, compare against decoder-decoder cosine similarity baseline, and specifically test D-EDA's ability to distinguish absorption from polysemanticity by checking whether high-EDA latents that are NOT absorbed have diffuse rather than peaked residual spectra.

**Candidate C (Tree-Structured Sparsity Correction) strengthened as follows:**

1. *Addressed hierarchy inference noise*: Instead of building a global tree from pairwise cosine similarity, use a two-stage approach: (i) Identify candidate parent-child pairs using conservative decoder cosine similarity threshold (>0.5). (ii) Validate candidates using D-EDA: a true parent-child pair should have the child's absorption spectrum peak at the parent's decoder column. This cross-validation dramatically reduces false parent-child edges.

2. *Relaxed tree to forest*: Instead of enforcing a single tree, allow a forest (multiple disconnected trees) with DAG-like shortcuts handled by converting to the closest spanning tree. This is more realistic for real feature hierarchies.

3. *Introduced graduated correction*: Instead of binary "parent must fire if child fires," use a soft correction: z_parent_corrected = max(z_parent, alpha * z_child * cos(d_parent, d_child)), where alpha is a single hyperparameter controlling correction strength. alpha = 0 means no correction; alpha = 1 means full correction. This allows the practitioner to trade off absorption reduction against L0 increase.

### Additional Evidence Found

- The "Dense SAE Latents Are Features, Not Bugs" paper (arXiv:2506.15679, 2025) found that SAE latents form "antipodal pairs" with nearly opposite decoder vectors. This confirms that decoder weight geometry contains rich structural information beyond simple cosine similarity -- supporting the premise that weight-based absorption detection is feasible.

- The "Understanding SAE Scaling in the Presence of Feature Manifolds" paper (arXiv:2509.02565, 2025) showed SAEs "tile" feature manifolds with increasingly sparse latents as width increases. This is relevant because manifold tiling creates exactly the kind of parent-child hierarchy (coarse tile = parent, fine tile = child) that triggers absorption. The scaling analysis predicts absorption should increase with width -- consistent with empirical findings.

- The Hierarchical Disentangled Information (HDI) framework (Nature Scientific Reports, Dec 2025) introduces metrics for evaluating hierarchically disentangled representations, including the Hierarchical Disentanglement Score (HDS). This is relevant methodology that could inform our absorption taxonomy.

### Selected Front-Runner

**Candidate C (Inference-Time Absorption Correction via Tree-Structured Sparsity Priors)** is the front-runner, integrated with elements from Candidate A (D-EDA for hierarchy inference and absorption detection) and the conceptual framing from Candidate B (competitive exclusion vocabulary).

Rationale for selection:
1. **Maximum practical impact**: Completely training-free. Applicable to all 400+ Gemma Scope SAEs and every SAELens SAE immediately.
2. **Novel cross-domain transplant**: Tree-structured sparsity priors from signal processing have never been applied to SAE correction. This is not an incremental improvement to existing methods.
3. **Clean falsifiability**: If the correction does not reduce absorption rate by >= 30% on first-letter task, or if it increases reconstruction error by > 10%, it has clearly failed.
4. **Builds on solid foundations**: D-EDA for hierarchy inference + tree-structured correction is a coherent pipeline where each component supports the other.
5. **Addresses the critical gap**: Unlike all prior mitigations (Matryoshka, OrtSAE, masked regularization), this does not require retraining. It is the only proposed method that can fix absorption in already-deployed SAEs.

---

## Phase 5: Final Proposal

### Title

Absorption Spectroscopy and Tree-Structured Correction: Diagnosing and Repairing Feature Absorption in Sparse Autoencoders Without Retraining

### Hypothesis

Feature absorption in SAEs can be (a) detected from SAE weight matrices alone using directional decomposition of encoder-decoder misalignment (D-EDA), with AUROC >= 0.80 against supervised ground truth, and (b) corrected at inference time by enforcing tree-structured activation consistency constraints inferred from the D-EDA absorption graph, reducing absorption rate by >= 30% with < 10% increase in reconstruction error, all without any retraining or fine-tuning of the SAE.

### Motivation

Feature absorption is the most fundamental failure mode of sparse autoencoders for mechanistic interpretability. It creates systematic blind spots where general features silently fail to activate, producing a false sense of interpretability. Every SAE architecture tested -- L1, TopK, JumpReLU, Gated -- exhibits absorption at rates of 15-35% (Chanin et al. 2024). Yet all proposed mitigations require retraining: Matryoshka SAEs (nested training), OrtSAE (orthogonality penalty), masked regularization (modified training objective). This is a critical practical problem because the SAE ecosystem already contains hundreds of pre-trained SAEs (Gemma Scope 400+ SAEs, Neuronpedia thousands of SAEs, community SAELens SAEs) that cannot be economically retrained.

There is an acute need for two capabilities that do not currently exist: (1) an unsupervised method to detect which latents are affected by absorption without requiring known probe directions, and (2) a training-free method to correct absorption in already-deployed SAEs. This paper provides both.

The key insight comes from signal processing, where hierarchical dictionary learning has long dealt with the structural analog of feature absorption: when signals have multi-scale structure (e.g., wavelet decompositions), tree-structured sparsity priors ensure that activating a fine-scale dictionary atom implies activating its coarse-scale parent (Jenatton et al. 2011). This "parent-must-fire-when-child-fires" constraint is precisely what absorption violates, and it can be enforced post-hoc.

### Method

The method has three stages:

**Stage 1: D-EDA Computation and Absorption Spectrum Analysis**

For each SAE latent j:
1. Compute the encoder-decoder angular divergence: EDA(j) = 1 - cos(w_{e,j}, d_j)
2. Decompose the encoder direction into its decoder-aligned and residual components: w_{e,j} = alpha_j * d_j + r_j, where r_j is orthogonal to d_j
3. Project the residual onto the decoder dictionary: r_j = sum_k beta_{j,k} * d_k + epsilon_j (using top-K sparse projection, K=10)
4. The vector beta_j = (beta_{j,1}, ..., beta_{j,n}) is the "absorption spectrum" of latent j
5. Compute spectrum sparsity: S(j) = ||beta_j||_1 / ||beta_j||_inf. Low S(j) (peaked spectrum) indicates absorption; high S(j) (diffuse spectrum) indicates polysemanticity or noise.

**Stage 2: Absorption Graph Construction and Hierarchy Inference**

1. For each latent j with EDA(j) > tau_EDA and S(j) < tau_S (high misalignment AND peaked spectrum):
   - Identify the dominant absorber: k* = argmax_k |beta_{j,k}| where cos(d_j, d_{k*}) > tau_cos
   - Add directed edge k* -> j to the absorption graph with weight |beta_{j,k*}|
2. Extract the hierarchy forest from the absorption graph:
   - Each connected component forms a tree (or DAG, converted to nearest spanning tree)
   - Roots are latents with no outgoing absorption edges (most general features)
   - Leaves are latents with no incoming absorption edges (most specific features)
3. Validate the inferred hierarchy against known hierarchies (first-letter, RAVEL entities) on a held-out set

**Stage 3: Tree-Structured Activation Correction (TSAC)**

For each input token:
1. Compute standard SAE activations z = Enc(x)
2. For each tree in the inferred hierarchy forest, traverse bottom-up:
   - For each child latent j with parent p in the tree:
     - If z_j > 0 (child is active) and z_p = 0 (parent is inactive -- absorption symptom):
       - Compute correction: z_p_corrected = alpha * z_j * cos(d_p, d_j)
       - Set z_p = z_p_corrected
   - alpha in [0, 1] is the single correction strength hyperparameter
3. Output corrected activations z_corrected
4. Reconstruction uses standard decoder: x_hat = Dec(z_corrected)

The tree-structured traversal ensures corrections propagate up the hierarchy: if a grandchild fires and its parent was corrected, the grandparent is also corrected (cascade correction for multi-level absorption).

### Cross-Domain Insight

The core transplanted principle is from hierarchical dictionary learning in signal processing: when the underlying signal has multi-scale structure, the sparsity pattern of the optimal representation must respect the hierarchy (child active implies parent active). Jenatton et al. (2011) proved this formally and developed O(n) proximal operators for tree-structured sparsity. The structural correspondence holds because:

1. SAE features form hierarchies (empirically demonstrated by Chanin et al., theoretically proven by Tang et al.)
2. The SAE loss function penalizes sparsity, creating the exact incentive to violate the hierarchy constraint (absorb parent into child to save one activation)
3. Enforcing the hierarchy constraint post-hoc restores the "correct" sparsity pattern at the cost of slightly higher L0

This is not a superficial analogy. The mathematical framework (tree-structured sparsity priors, proximal operators, hierarchy-consistent sparse coding) transfers directly. The only adaptation needed is inferring the tree structure from the SAE weights rather than specifying it a priori.

### Experimental Plan

| Experiment | Purpose | Metric | Model / SAE | Time |
|-----------|---------|--------|------------|------|
| 1. EDA/D-EDA distribution on Gemma Scope | Characterize baseline EDA distribution and absorption spectrum properties | EDA histogram, spectrum sparsity distribution | Gemma 2 2B, layers 6/12/20, 16k width | 15 min (CPU only) |
| 2. EDA/D-EDA vs. Chanin et al. ground truth | Validate detection accuracy | AUROC, AUPRC, precision@k with bootstrap CIs | Same as above + first-letter absorption labels | 30 min |
| 3. D-EDA specificity test | Verify D-EDA distinguishes absorption from polysemanticity | Spectrum sparsity ratio: absorbed vs. polysemantic vs. random latents | Same SAEs | 15 min |
| 4. Hierarchy inference validation | Test if D-EDA absorption graph recovers known hierarchies | Precision/recall of inferred parent-child pairs vs. first-letter ground truth | Same SAEs | 15 min |
| 5. TSAC on first-letter task | Measure absorption correction on canonical benchmark | Absorption rate before/after; FVU change; L0 change | Gemma Scope 16k and 65k, layers 6/12/20 | 45 min |
| 6. TSAC on RAVEL entity hierarchies | Test cross-domain generalization | Absorption rate on city-country, city-continent; RAVEL disentanglement score | Gemma Scope 16k, layer 12 | 45 min |
| 7. TSAC sensitivity to alpha | Characterize the absorption-sparsity tradeoff curve | Absorption rate vs. L0 vs. FVU as function of alpha in [0.1, 1.0] | Gemma Scope 16k, layer 12 | 30 min |
| 8. GPT-2 replication | Verify model generality | Same as experiments 1-5 on GPT-2-small SAEs | GPT-2, EleutherAI SAEs | 45 min |
| 9. SAEBench integration | Full benchmark evaluation of TSAC-corrected SAEs | All 8 SAEBench metrics before/after TSAC | Gemma Scope 16k, layer 12 | 60 min |
| **Total** | | | | **~5.5 hours** |

Falsification criteria:
- If D-EDA AUROC < 0.65 on first-letter ground truth, the detection method fails
- If TSAC reduces absorption by < 20% at any alpha setting, the correction method fails
- If TSAC increases FVU by > 15%, the correction degrades SAE quality unacceptably
- If hierarchy inference precision < 50% on known first-letter hierarchy, the automatic inference is too noisy

### Resource Estimate

- **Compute**: ~5.5 GPU-hours total. Experiments 1, 3, 4 are CPU-only (pure weight analysis). Experiments 2, 5-9 require 1 GPU for SAE inference and absorption metric computation. No training required.
- **Models**: Gemma 2 2B (via Gemma Scope SAEs, pre-trained, free download), GPT-2 (via EleutherAI SAEs, pre-trained, free download)
- **Data**: First-letter task (constructed from token vocabulary), RAVEL entity datasets (pre-existing, MIT license)
- **Code**: SAELens (SAE loading), sae-spelling (absorption metric), custom implementation for D-EDA and TSAC (~300 lines of PyTorch)
- **Time to first result**: Experiment 1 (EDA distribution) produces publishable results in 15 minutes

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| EDA/D-EDA has poor discriminative power (AUROC < 0.70) due to high baseline variance | HIGH | MEDIUM | (1) Filter polysemantic latents before evaluation; (2) Use D-EDA spectrum sparsity as additional discriminator; (3) If both fail, pivot to combined EDA + activation anticorrelation detector at cost of no longer being activation-free |
| Hierarchy inference from decoder weights produces too many false parent-child edges | HIGH | MEDIUM | (1) Use conservative cosine threshold (>0.5); (2) Require D-EDA spectrum confirmation; (3) For initial validation, use known hierarchies instead of inferred ones; (4) Report both oracle-hierarchy and inferred-hierarchy results |
| TSAC increases L0 substantially (>30%), degrading downstream performance | MEDIUM | MEDIUM | (1) Graduated alpha parameter allows practitioner control; (2) Only correct latents with D-EDA above strict threshold; (3) Report full Pareto frontier: absorption_rate vs. L0 vs. FVU |
| First-letter results do not generalize to entity-type hierarchies | MEDIUM | MEDIUM-LOW | (1) RAVEL datasets are pre-validated with known linear separability; (2) Use multiple hierarchy types (city-country, city-continent, city-language); (3) Non-generalization is an informative result (absorption is domain-dependent) |
| Reviewer objection: "this is just cosine similarity computation + heuristic post-processing" | MEDIUM | MEDIUM | (1) Formal analysis showing D-EDA spectrum sparsity is a sufficient statistic for absorption detection under a linear generative model; (2) The absorption graph is a genuinely new analytical tool; (3) TSAC has a principled basis in hierarchical dictionary learning theory |

### Novelty Claim

The specific novelty claims, supported by evidence that each has NOT been done before:

1. **D-EDA and the absorption spectrum**: No prior work decomposes the encoder-decoder residual onto the decoder dictionary to produce a per-latent "absorption spectrum." Chanin et al. (Oct 2024, LessWrong) informally suggested that encoder-decoder discrepancy indicates absorption; we formalize this into a quantitative metric with directional decomposition that distinguishes absorption from polysemanticity. The "Empirical Insights into Feature Geometry" (Jan 2025, LessWrong) computed scalar encoder-decoder cosine similarity but did not decompose residuals or connect to absorption detection.

2. **The absorption graph**: No prior work models absorption as a directed graph. Chanin et al.'s analysis is pairwise (one parent, one child); our graph reveals cascade structures, hub absorbers, and connected-component hierarchies. This is a fundamentally richer representation of the absorption phenomenon.

3. **Tree-Structured Activation Correction (TSAC)**: No prior work applies tree-structured sparsity priors from signal processing as post-hoc correction on trained SAEs. All existing mitigations (Matryoshka, OrtSAE, masked regularization, feature anchoring) require retraining. TSAC is the first training-free absorption correction method. The transplant of Jenatton et al.'s (2011) proximal methods for hierarchical sparsity to SAE post-hoc correction is a novel cross-domain contribution.

4. **Unified detection-correction pipeline**: The D-EDA -> absorption graph -> TSAC pipeline provides end-to-end absorption management for already-deployed SAEs. No prior work offers both unsupervised detection AND training-free correction in a single framework.
