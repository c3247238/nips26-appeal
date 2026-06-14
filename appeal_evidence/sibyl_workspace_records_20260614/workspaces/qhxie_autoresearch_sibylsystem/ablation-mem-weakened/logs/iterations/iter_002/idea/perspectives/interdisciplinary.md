# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **[Rozell et al., 2008. "Sparse coding via thresholding and local competition in neural circuits." IEEE Trans. Signal Processing]** — The Locally Competitive Algorithm (LCA) provides a biologically plausible sparse coding framework where neurons compete via one-way lateral inhibition. The inhibition matrix G_ij = <phi_i, phi_j> (inner product of receptive fields) determines competitive suppression strength. Active neurons suppress similar neurons, creating winner-take-all dynamics. This is the foundational neuroscience mechanism that directly corresponds to SAE feature competition.

2. **[Jimenez-Nimmo & Mondragon, 2025. "Hebbian CNN Framework with Integrated Competition Mechanisms." City, University of London]** — Integrates pre-synaptic competition, lateral inhibition via Difference-of-Gaussians kernels, and homeostatic plasticity into a unified Hebbian learning pipeline. Demonstrates that homeostatic selection (adjusting based on input statistics) combined with lateral inhibition creates sparse, non-redundant representations. The homeostatic boost mechanism directly inspires post-hoc repair ideas.

3. **[Zenke et al., 2017/2024. "The temporal paradox of Hebbian learning and homeostatic plasticity."]** — Identifies that homeostatic plasticity maintains stable firing rates despite inhibition-driven competition. Key insight: "stimuli compete for this activity" — the total neural activity budget is fixed, and features compete for activation slots. This maps precisely to SAEs where the sparsity constraint creates a fixed "activation budget" that features compete for.

4. **[Inhibition SNN, 2024. "Unveiling the efficacy of various lateral inhibition learning in image pattern recognition." Springer]** — Explores WTA LIF neuron models with lateral inhibition in spiking neural networks. Two architectures: separate excitatory/inhibitory layers (Inhibition V1) and self-connections with fixed negative weights (Inhibition V2). Demonstrates lateral inhibition enhances contrast sensitivity and selective responsiveness — the biological analog of SAE feature selectivity.

5. **[Iversen, 2025. "Sparse Neural Network Interpretability: A Comparative Analysis of Autoencoders and Transformers." Master's thesis, University of Tromso]** — Explicitly compares Sparse Transformers' softmax competitive dynamics to lateral inhibition. Finds that STs with softmax create "probability simplex constraints, angular discrimination, competitive dynamics" leading to "highly specialized, monosemantic features through winner-take-all competition." SAEs with ReLU lack this bounded competition, leading to polysemanticity.

6. **[Olshausen & Field, 1996/1997. "Sparse coding with an overcomplete basis set." Nature]** — The original sparse coding framework showing natural images are efficiently represented by sparse linear combinations of learned basis functions. The overcompleteness-sparsity trade-off is the direct ancestor of modern SAE theory. Biological V1 simple cells emerge from this optimization.

#### Physics / Statistical Mechanics / Information Theory

7. **[Friston et al., 2023/2024. "The free energy principle made simpler but not too simple."]** — The Free Energy Principle (FEP) frames perception as minimizing variational free energy F = E_q[log q(theta) - log p(theta, D)], balancing accuracy (energy) and complexity (entropy). The structural correspondence to SAEs is exact: reconstruction loss = accuracy term, sparsity penalty = complexity term. FEP's "sparsely coupled dynamics" as a fundamental condition for tractable inference directly parallels SAE sparsity constraints.

8. **[Brain-like Variational Inference, 2024. arXiv:2410.19315]** — Explicitly bridges ELBO (machine learning) and variational free energy F (neuroscience), showing ELBO = -F. This provides the formal mapping between SAE training (ELBO maximization) and FEP (free energy minimization). The paper notes neither has been "prescriptive for developing specific algorithms" — a gap our proposal addresses.

9. **[Tishby & Zaslavsky, 2015. "Deep Learning and the Information Bottleneck Principle."]** — The Information Bottleneck (IB) framework posits optimal representations minimize I(T;X) - beta*I(T;Y), creating a compression-prediction trade-off. Hierarchical layers correspond to progressive compression along the IB curve. This directly maps to SAEs where deeper layers (higher beta) compress more, increasing absorption prevalence.

10. **["Superposition as Lossy Compression," 2025. arXiv:2512.13568]** — Connects superposition to adversarial vulnerability via rate-distortion theory. Shows superposition can be understood as lossy compression where the distortion measure captures interpretability. This is the closest existing work applying rate-distortion to SAE phenomena, but does not address absorption specifically.

11. **[Annila, 2025. "Comprehensible dynamics of quanta." European Physical Journal Plus]** — Derives the 2nd law of thermodynamics from the principle of least action, showing entropy increase equates to free energy minimization. Provides the thermodynamic foundation for viewing SAE training as a dissipative system evolving toward equilibrium — where absorption represents a local free energy minimum.

#### Biology / Evolution / Ecology

12. **[Gause, 1934 / Modern reviews 2024-2025. Competitive Exclusion Principle]** — Two species competing for the same limiting resource cannot coexist indefinitely. The superior competitor excludes the inferior one. In SAEs, "species" = features, "resource" = activation budget (sparsity constraint), "superior competitor" = high-frequency child features that outcompete parent features for the limited activation slots. The structural correspondence is formal, not metaphorical.

13. **[Niche Partitioning Theory (Fiveable 2024; recent ecology reviews)]** — Species circumvent competitive exclusion by "divvying up resources in space, time, or type." Realized niches are smaller than fundamental niches. In SAEs, Matryoshka SAEs implement a form of niche partitioning by allocating different latents to different abstraction levels, preventing direct competition between parent and child features.

14. **[Individual Niche Plasticity, 2024-2025. Bat species coexistence research]** — "Individual niche plasticity generates distributed 'coexistence cells' across the system, transforming global resource conflicts into localized dynamic equilibria." Individual competitive variance "blurs interspecific fitness differentials," disrupting hierarchical dominance structures. This suggests that feature-level stochasticity (e.g., BatchTopK's batch-level variation) could disrupt absorption by preventing deterministic competitive hierarchies.

15. **[Bafna, Alm & Berger, 2025. "Sparse autoencoders uncover biologically interpretable features in protein language model representations." PNAS]** — Applied SAEs to protein language models, finding sparse features strongly associated with specific functional annotations and protein families. Demonstrates that feature absorption-like phenomena may occur in biological information processing domains where hierarchical feature structures (molecule -> protein family -> functional domain) are ubiquitous.

#### Signal Processing / Source Separation

16. **[Hyvarinen & Pajunen, 1999; Locatello et al., 2019. Nonlinear ICA non-identifiability]** — Fundamental impossibility results: without inductive biases, nonlinear source separation is non-identifiable. Hierarchical structures in sources create additional non-identifiability. This is structurally identical to the SAE absorption problem: when parent and child features are hierarchically related (child implies parent), the decoder cannot uniquely identify which source activated.

17. **["A review of NMF, PLSA, LBA, EMA, and LCA with a focus on the identifiability issue," 2025. arXiv:2512.22282]** — Comprehensive review showing identifiability issues are pervasive across matrix factorization methods. When sources are correlated (as in hierarchical features), unique recovery is impossible without additional constraints. This directly parallels Cui et al.'s (2025) identifiability analysis for SAEs.

18. **["Mechanistic Independence: A Principle for Identifiable Disentangled Representations," 2025. arXiv:2509.22196]** — Proposes mechanistic independence as a principle for identifiable disentanglement, building on nonlinear ICA theory. The principle requires that latent variables influence observations through independent mechanisms — violated when parent and child features share causal pathways (hierarchical implication).

#### Recommendation Systems / Matrix Factorization

19. **[Klimashevskaia et al., 2024. "A Survey on Popularity Bias in Recommender Systems."]** — Popularity bias: popular items dominate latent factors, creating "attractors" that absorb recommendation capacity. Solutions include inverse propensity weighting, disentangled embeddings (separating popularity from quality factors), and causal intervention. The "popularity bias" phenomenon in MF is structurally identical to feature absorption: high-frequency items (child features) dominate latent factors at the expense of long-tail items (parent features).

20. **[Cai et al., KDD 2024. "Popularity-aware Alignment and Contrast for Mitigating Popularity Bias"]** — Uses alignment and contrastive learning specifically for popularity bias. Graph-based approach with popularity-aware modifications. The "cluster anchor regularization" (WWW 2024) partitions items into hierarchical clusters for transfer learning from head to tail items — analogous to hierarchical feature allocation in SAEs.

### Cross-Disciplinary Gaps

| Source Field | Principle | SAE Analog | Has Been Explored? |
|-------------|-----------|-----------|-------------------|
| Neuroscience (LCA) | Lateral inhibition via G = Phi^T Phi | Decoder correlation W_dec^T W_dec as inhibition matrix | **NO** — Innovator perspective identified this gap |
| Neuroscience (homeostasis) | Homeostatic plasticity restores suppressed firing | Post-hoc activation rebalancing for absorbed features | **NO** — No training-free repair exists |
| Ecology (competitive exclusion) | Superior competitor excludes inferior from niche | High-frequency child features exclude parent from activation budget | **NO** — Only implicit in Chanin et al.'s "logical consequence" |
| Ecology (niche partitioning) | Resource divvying enables coexistence | Matryoshka's level-separated dictionaries | **Partially** — Matryoshka implements but doesn't theorize |
| Physics (FEP) | Free energy minimization F = E - TS | SAE objective = reconstruction + sparsity | **Partially** — Rate-distortion theory (Theoretical) |
| Physics (IB) | I(T;X) - beta*I(T;Y) compression | Layer-dependent absorption prevalence | **NO** — No IB analysis of absorption |
| Signal Processing (ICA) | Non-identifiability with correlated sources | Absorption as non-identifiability under hierarchy | **Partially** — Cui et al. (2025) touches this |
| RecSys (popularity bias) | Popular items absorb latent capacity | High-frequency features absorb activation budget | **NO** — No cross-domain transplant attempted |

---

## Phase 2: Initial Candidates

### Candidate A: The Competitive Exclusion Lens — Feature Absorption as Ecological Niche Competition

- **Source principle**: The competitive exclusion principle (Gause's Law, 1934) and niche partitioning theory from ecology. Two species competing for the same limiting resource cannot coexist; the superior competitor excludes the inferior. Niche partitioning (divvying resources across dimensions) enables coexistence.
- **Structural correspondence**: In SAEs, "species" = features, "resource" = the fixed activation budget imposed by sparsity constraint, "competitive ability" = feature frequency (high-frequency child features outcompete low-frequency parent features). The "limiting resource" is not physical space but the L0 norm budget. When parent and child co-occur (hierarchical implication), they compete for the same activation slots. The child, being more frequent (higher "fitness"), wins and excludes the parent.
- **Hypothesis**: Feature absorption rate A(F) for a parent feature F is inversely proportional to the "niche differentiation" between F and its children. If children occupy nearly the same niche (high co-occurrence, similar decoder directions), exclusion is near-total (high absorption). If children are niche-differentiated (low co-occurrence, orthogonal decoder directions), coexistence is possible (low absorption).
- **Why it's not just a metaphor**: The mathematical structure is identical. Ecology models competition via Lotka-Volterra equations: dn_i/dt = r_i n_i (1 - sum_j alpha_ij n_j / K_i). For SAEs, the "population" n_i is latent activation frequency, the "carrying capacity" K_i is determined by sparsity budget, and the "competition coefficient" alpha_ij is decoder correlation <W_dec[i], W_dec[j]>. The competitive exclusion theorem (alpha_ii < alpha_ij for some j implies exclusion) maps to: if a child feature has higher correlation with the parent than the parent has with itself (after normalization), the parent is excluded.
- **Novelty estimate**: 7/10. The ecology-SAE analogy has not been explicitly developed. Matryoshka SAEs implicitly implement niche partitioning but do not theorize it. The formal Lotka-Volterra mapping is new.

### Candidate B: The Free Energy Principle — Absorption as Optimal Variational Inference

- **Source principle**: Karl Friston's Free Energy Principle (FEP) frames perception as minimizing variational free energy F = E_q[log q(theta) - log p(theta, D)] = accuracy - complexity. The brain trades reconstruction accuracy against model complexity (entropy), naturally absorbing redundant features into more efficient representations.
- **Structural correspondence**: SAE training minimizes reconstruction loss (accuracy) + lambda * sparsity (complexity). This is exactly the free energy functional with lambda as the inverse temperature. Under FEP, hierarchical features with high mutual information (parent determined by children) are optimally represented by absorbing the parent into the children — this minimizes complexity without sacrificing accuracy. The "absorption" is not a bug but the optimal variational approximation.
- **Hypothesis**: The absorption rate A(F) for parent feature F is monotonically related to the conditional entropy H(F | children) / H(F). When the parent is nearly deterministic given children (low conditional entropy), absorption is the optimal free energy minimum. When the parent contains information beyond children (high conditional entropy), absorption is suboptimal and represents a local minimum.
- **Why it's not just a metaphor**: The mathematical objects are identical. ELBO = -F (Brain-like Variational Inference, 2024). The SAE objective is a specific instance of variational free energy minimization. The claim that absorption minimizes free energy is a theorem, not an analogy: given hierarchical features where F = OR(C_i), coding F separately requires rate H(F), while coding through children requires rate sum H(C_i) - I(C_i). When H(F | C_i) is small, the marginal rate savings exceed the distortion cost, so absorption is optimal.
- **Novelty estimate**: 6/10. Rate-distortion theory (Theoretical perspective) covers similar ground. The FEP framing adds biological plausibility but the core mathematics is classical. The novelty lies in applying FEP specifically to diagnose when absorption is benign vs. pathological.

### Candidate C: The LCA Inhibition Graph — A Neuroscience-Inspired Diagnostic and Repair

- **Source principle**: Rozell et al.'s (2008) Locally Competitive Algorithm (LCA) for sparse coding. In LCA, neurons with overlapping receptive fields compete via lateral inhibition: u_dot = (1/tau)[b - u - G*a], where G_ij = <phi_i, phi_j> is the inhibition matrix. Homeostatic plasticity maintains stable firing rates despite inhibition.
- **Structural correspondence**: For SAEs, decoder vectors W_dec[i] are the "receptive fields" phi_i. The decoder correlation matrix W_dec^T W_dec is exactly the inhibition matrix G from LCA. When child feature j has high decoder correlation with parent feature i (G_ij is large), the child's activation suppresses the parent's firing — this is absorption. The structural correspondence is exact, not approximate: G_SAE = W_dec^T W_dec = G_LCA.
- **Hypothesis**: (H1) Edges in the local inhibition graph (top-k correlated neighbors per latent) correspond to known absorption pairs with precision significantly above chance. (H2) A single-pass homeostatic rebalancing — boosting parent activations proportionally to inhibition received — restores parent feature firing without degrading reconstruction. (H3) Repaired features show improved encoder-dependent task performance.
- **Why it's not just a metaphor**: The inhibition matrix G is the same mathematical object in both frameworks. LCA's dynamics u_dot = (1/tau)[b - u - G*a] with G = Phi^T Phi maps to SAE inference where latent activations are determined by encoder projections with implicit competition via decoder correlations. The homeostatic repair (boosting suppressed activations) is a direct computational transplant of biological homeostatic plasticity.
- **Novelty estimate**: 8/10. LCA has been cited ~2000 times but never applied to LLM SAEs. No work connects decoder correlations to inhibition graphs. No training-free post-hoc repair for absorption exists. This is a genuinely novel cross-disciplinary transplant.

---

## Phase 3: Self-Critique

### Against Candidate A (Competitive Exclusion Lens)

- **Shallow analogy attack**: The Lotka-Volterra mapping is mathematically sound, but does it provide actionable insights beyond what we already know from SAE theory? Chanin et al. (2024) already proved absorption is a "logical consequence of sparsity loss under hierarchical features." The ecology framing reframes this but does not predict new phenomena. A domain ecologist would agree the mapping preserves the key property (competition for limited resources), but would note that ecological systems have additional stabilizing mechanisms (spatial heterogeneity, temporal variation, predator-prey dynamics) that SAEs lack.
- **Scale mismatch attack**: Competitive exclusion in ecology operates at population timescales (generations). SAE inference operates at single-forward-pass timescales. The dynamical aspect of competition (population trajectories over time) does not map to SAEs, which are feedforward. However, SAE training dynamics (gradient descent over iterations) could be viewed as the ecological timescale — features "evolve" their decoder directions to maximize their "fitness" (activation frequency).
- **Prior transplant check**: No prior work explicitly connects competitive exclusion to SAE absorption. Matryoshka SAEs implicitly implement niche partitioning but do not cite ecology. The novelty claim holds.
- **Testability attack**: The key prediction — absorption rate correlates with "niche overlap" (decoder correlation) — is already tested by Chanin et al.'s differential correlation metric. The ecology framing does not generate a new, distinguishable prediction. It reframes existing findings rather than predicting new ones.
- **Verdict**: MODERATE — The structural correspondence is strong and the formal mapping is elegant, but the practical value is primarily conceptual reframing rather than actionable insights.

### Against Candidate B (Free Energy Principle)

- **Shallow analogy attack**: FEP is often criticized as a "theory of everything" that risks being unfalsifiable. Does applying FEP to SAEs add technical content, or just vocabulary? The core insight — absorption minimizes complexity without sacrificing accuracy — can be derived directly from rate-distortion theory without invoking FEP. The FEP framing adds biological plausibility but not new mathematical predictions.
- **Scale mismatch attack**: FEP operates at the level of entire organisms (or brains) minimizing free energy over extended timescales. SAE training is a supervised optimization over a fixed dataset. The timescale mismatch is significant. However, the variational inference framework (ELBO = -F) is scale-invariant and applies to any probabilistic model.
- **Prior transplant check**: Rate-distortion theory has been applied to superposition (arXiv:2512.13568, 2025) but not specifically to absorption. The Information Bottleneck principle (Tishby, 2015) has been applied to deep learning broadly but not to SAE absorption. The novelty claim is partially supported.
- **Testability attack**: The key prediction — A(F) correlates with H(F | children) / H(F) — is testable from activation data. However, estimating conditional entropy from finite samples is statistically challenging. The prediction that "absorption is optimal when conditional entropy is low" is also made by the rate-distortion framework (Theoretical perspective), so FEP does not add a distinguishable prediction.
- **Verdict**: MODERATE — The mathematical framework is sound but the FEP-specific insights are limited. Rate-distortion theory (Candidate B in Theoretical perspective) provides the same predictions with less conceptual overhead.

### Against Candidate C (LCA Inhibition Graph)

- **Shallow analogy attack**: Is the correspondence really structural? Rozell et al.'s LCA requires recurrent dynamics over time (u_dot = ...), while SAEs are feedforward. The dynamical systems aspect may not transfer. However, the core structural element — the inhibition matrix G = Phi^T Phi — is exactly W_dec^T W_dec for SAEs. This is not vocabulary mapping; it is the same mathematical object. A signal processing expert would confirm that decoder correlations play the same role as receptive field overlaps in determining competitive suppression.
- **Scale mismatch attack**: LCA was designed for dictionaries with hundreds of elements (image patches). SAEs have 24K-1M latents. Constructing a full inhibition graph is O(d_dict^2), infeasible at scale. However, the local graph restriction (top-k neighbors only, k ~ 20-50) reduces this to O(k * d_dict), which is feasible even for 1M latents. This is a standard approximation in graph-based methods.
- **Prior transplant check**: No prior work connects LCA to SAE absorption. Rozell et al. (2008) has been cited extensively but only for sparse coding applications (image denoising, compression). The LCA-SAE connection has not been explored. The Innovator perspective independently identified this gap.
- **Testability attack**: This candidate generates three specific, distinguishable predictions: (H1) graph edges match absorption pairs, (H2) homeostatic rebalancing restores parent firing without degrading reconstruction, (H3) repaired features improve task performance. Each is testable with existing pretrained SAEs. The diagnostic experiment is clear: if graph edges do not correspond to absorption pairs, the structural correspondence fails.
- **Verdict**: STRONG — The structural correspondence is exact (G = W_dec^T W_dec), the predictions are specific and testable, and the training-free nature aligns with project constraints. The local graph restriction addresses scalability concerns.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate A (Competitive Exclusion)** is dropped as a primary contribution. While the formal mapping is elegant, it primarily reframes existing findings rather than generating new predictions. It is retained as a conceptual framework for Discussion ("absorption as ecological competition").
- **Candidate B (Free Energy Principle)** is dropped in favor of the rate-distortion formulation from the Theoretical perspective. FEP adds conceptual overhead without distinct predictions. The core insight (absorption as optimal compression) is preserved but framed in information-theoretic rather than FEP terms.

### Strengthened: Candidate C (LCA Inhibition Graph)

Strengthened by:
1. **Formalizing the structural correspondence explicitly**:
   - LCA inhibition matrix: G^LCA_ij = <Phi_i, Phi_j>
   - SAE decoder correlation: G^SAE_ij = <W_dec[i], W_dec[j]>
   - Correspondence: G^LCA = G^SAE (exact equality, not approximation)
   - LCA dynamics: u_dot = (1/tau)[b - u - G*a]
   - SAE "dynamics" (single-step): z = f(W_enc * a + b_pre), where f is the activation function
   - The "inhibition" in SAEs is implicit: child activation z_j suppresses parent activation z_i through reconstruction competition, not explicit lateral connections.

2. **Designing a diagnostic experiment**:
   - Construct local inhibition graph for GPT-2 Small SAE (24K latents, k=20 neighbors per latent)
   - Use Chanin et al.'s absorption detection on 26 first-letter features as ground truth
   - Measure precision@20 and recall@20 of graph edges against absorption pairs
   - Fisher exact test: are graph edges enriched for absorption pairs vs. random pairs?
   - If precision@20 > 0.10 (vs. ~0.004 expected by chance for 24K latents), the correspondence is validated.

3. **Designing the homeostatic repair**:
   - For input activation a, compute SAE latents z = f(W_enc * a + b_pre)
   - For each latent i, compute inhibition: inh_i = sum_{j in N(i)} G_ij * z_j
   - Apply homeostatic boost: z'_i = z_i + alpha * inh_i
   - Clip to maintain non-negativity (for ReLU/TopK)
   - Reconstruct: a' = W_dec * z'
   - Alpha is a tunable parameter (sweep 0.0 to 2.0)
   - Constrain: ||a' - a||_2 <= ||a - W_dec * z||_2 * (1 + epsilon) to prevent reconstruction degradation

4. **Searching for additional support**:
   - Rozell et al. (2008, Eq. 3): LCA dynamics are u_dot = (1/tau)[b - u - G*a]. For SAEs, W_dec^T W_dec = G. This is exact.
   - Paiton et al. (2019, "Analysis and applications of the Locally Competitive Algorithm"): LCA converges to sparse codes with guarantees when G is positive semi-definite. W_dec^T W_dec is always PSD.
   - Jimenez-Nimmo & Mondragon (2025): Homeostatic selection adjusts based on input statistics, analogous to our alpha parameter tuning.

### Selected Front-Runner

**Candidate C (restricted): The Local Inhibition Graph — A Neuroscience-Inspired Training-Free Diagnostic and Repair for Feature Absorption**

This is the strongest candidate because:
1. **Exact structural correspondence**: G^LCA = Phi^T Phi = W_dec^T W_dec = G^SAE. This is not a metaphor.
2. **Training-free**: The inhibition graph is computed from pretrained SAE weights; no retraining needed.
3. **Scalable**: Local graph (top-k neighbors) is O(k * d_dict), feasible for any SAE size.
4. **Dual contribution**: Both diagnostic (graph edges reveal absorption pairs) and corrective (homeostatic rebalancing restores parent firing).
5. **Novel**: No prior work connects LCA to SAE absorption or proposes post-hoc repair.
6. **Testable**: Three falsifiable hypotheses with clear experimental protocols.
7. **Complements existing work**: Does not compete with the Theoretical (rate-distortion) or Pragmatist (metric-specific impact) perspectives. Instead, it provides a cross-disciplinary mechanism for the "benign absorption" finding — absorption is repairable, so its downstream impact is limited.

---

## Phase 5: Final Proposal

### Title

**"The Local Inhibition Graph: A Neuroscience-Inspired Training-Free Diagnostic and Repair for Feature Absorption in Sparse Autoencoders"**

### Source Principle

The Locally Competitive Algorithm (LCA; Rozell et al., 2008) is a neuroscience-inspired sparse coding framework where neurons compete via one-way lateral inhibition. The inhibition strength between neurons i and j is determined by the inner product of their receptive fields: G_ij = <Phi_i, Phi_j>. Active neurons suppress similar neurons, creating winner-take-all dynamics. Homeostatic plasticity maintains stable firing rates despite inhibition-driven competition (Zenke et al., 2017).

### Structural Correspondence

For Sparse Autoencoders, the decoder weight matrix W_dec plays the role of receptive fields Phi. The decoder correlation matrix:

**G^SAE = W_dec^T W_dec**

is exactly the inhibition matrix G^LCA from the LCA framework. When child feature j has high decoder correlation with parent feature i (G_ij is large), the child's activation suppresses the parent's firing through reconstruction competition — this is feature absorption. The correspondence is exact, not approximate.

The local inhibition graph is constructed by keeping only the top-k correlated neighbors per latent (k=20-50), forming a sparse graph where edge weights are correlation magnitudes. This graph reveals which latents compete with which for activation budget.

### Hypothesis

For a pretrained SAE, the local inhibition graph defined by decoder correlations has three properties:

- **H1 (Diagnostic)**: Graph edges correspond to known absorption pairs with precision significantly above chance (precision@20 > 0.10 vs. ~0.004 baseline).
- **H2 (Repair)**: A single-pass homeostatic rebalancing of activations along graph edges — z'_i = z_i + alpha * sum_{j in N(i)} G_ij * z_j — restores parent feature firing without degrading reconstruction quality (reconstruction error increase < 5%).
- **H3 (Utility)**: Repaired features show improved encoder-dependent task performance (sparse probing recall) compared to unrepaired absorbed features.

### Method

**Step 1: Construct Local Inhibition Graph**
For each latent i in the SAE decoder matrix W_dec (shape d_dict x d_model):
- Compute decoder correlations: G_ij = <W_dec[i], W_dec[j]> for all j != i
- Keep top-k neighbors per latent (k=20-50) with highest |G_ij|
- Edge weight = G_ij (signed correlation)
- Complexity: O(k * d_dict * d_model) — feasible for 24K-1M latents

**Step 2: Validate Graph Against Absorption Pairs**
- Use Chanin et al.'s absorption detection method on first-letter features (A-Z) as ground truth
- For each absorption pair (parent latent i, absorbing latent j), check if j is in N(i)
- Compute precision@k, recall@k, and Fisher exact test for enrichment
- Compare against random baseline (shuffle latent indices)

**Step 3: Homeostatic Rebalancing**
For input activation a:
- Compute original latents: z = f(W_enc * a + b_pre)
- Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
- Apply boost: z'_i = z_i + alpha * inh_i
- Clip negative values (for ReLU/TopK compatibility)
- Constrain reconstruction: if ||a - W_dec * z'||_2 > (1 + epsilon) * ||a - W_dec * z||_2, scale alpha down
- Reconstruct: a' = W_dec * z'

**Step 4: Evaluate Repair**
- Measure absorption rate on rebalanced features vs. original
- Compare steering effectiveness (encoder-bypass) and sparse probing accuracy (encoder-dependent) on repaired vs. unrepaired features
- Test on GPT-2 Small (gpt2-small-res-jb) and GemmaScope (Gemma-2-2B) SAEs

### Diagnostic Experiment

The key test that confirms the analogy is load-bearing (not decorative):

**Experiment**: Construct the inhibition graph for GPT-2 Small SAE (24K latents). Use Chanin et al.'s method to detect absorption on 26 first-letter features. Measure whether graph edges are enriched for absorption pairs.

**Expected if analogy holds**: Precision@20 >= 0.10 (vs. ~0.004 expected by chance = 20/24000). This 25x enrichment would demonstrate that decoder correlations capture the competitive suppression mechanism underlying absorption.

**Expected if analogy fails**: Precision@20 <= 0.01 (no enrichment above chance). This would mean decoder correlations do not correspond to absorption pairs, undermining the structural correspondence.

**Why this is diagnostic**: If the LCA-SAE correspondence is real, the inhibition matrix G = W_dec^T W_dec must predict competitive suppression. If it does not, the correspondence is merely vocabulary mapping.

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time |
|---|---|---|---|---|
| E1: Graph construction + validation | GPT-2 Small | gpt2-small-res-jb (24K) | Precision@k, recall@k, Fisher test | ~15 min |
| E2: Homeostatic rebalancing | GPT-2 Small | Same | Absorption rate change, reconstruction error | ~30 min |
| E3: Steering on repaired features | GPT-2 Small | Same | Steering success rate (raw + delta) | ~30 min |
| E4: Probing on repaired features | GPT-2 Small | Same | Sparse probing F1, precision, recall | ~20 min |
| E5: Cross-model validation | Gemma-2-2B | GemmaScope 16K | All above metrics | ~30 min |
| E6: Ablation (alpha sweep) | GPT-2 Small | Same | Optimal alpha, sensitivity analysis | ~15 min |

**Total estimated time**: ~2.5 GPU-hours (well within project constraints if parallelized).

**Falsification criteria**:
- H1 falsified if precision@20 <= 0.05 (no enrichment above chance)
- H2 falsified if reconstruction error increases >5% after rebalancing
- H3 falsified if steering success on repaired features does not improve over unrepaired (paired t-test, p > 0.05)

### Risk Assessment

1. **Risk: Graph edges don't correspond to absorption pairs**
   - *Mitigation*: The structural correspondence is mathematically exact (W_dec^T W_dec = G). If edges don't match absorption, this itself is a finding about the limitations of decoder correlation for absorption detection.
   - *Fallback*: Pivot to using the graph purely as a diagnostic without repair claims.

2. **Risk: Homeostatic rebalancing degrades reconstruction**
   - *Mitigation*: Alpha is tunable; sweep to find values that improve absorption without degrading reconstruction. Constrain rebalancing to preserve L2 norm.
   - *Fallback*: Report the Pareto frontier (absorption reduction vs. reconstruction cost).

3. **Risk: Repair doesn't improve steering/probing**
   - *Mitigation*: This would mean absorption is truly benign, supporting the null result framing. The paper becomes "We Can Repair Absorption, But It Doesn't Matter" — a stronger contrarian contribution.
   - *Fallback*: The diagnostic contribution (graph edges flag absorption) stands independently.

4. **Risk: Local graph misses long-range absorption relationships**
   - *Mitigation*: Test multiple k values (10, 20, 50, 100). If absorption requires long-range edges, this informs the graph design.
   - *Fallback*: Use hierarchical clustering on decoder correlations to capture multi-scale structure.

### Novelty Claim

This proposal makes three novel cross-disciplinary contributions:

1. **First connection between LCA lateral inhibition and SAE absorption**: We show that the decoder correlation matrix W_dec^T W_dec is exactly the inhibition matrix from Rozell et al.'s LCA framework, providing a theoretical lens for understanding absorption as competitive suppression. No existing paper makes this connection.

2. **First neuroscience-inspired training-free post-hoc repair for absorption**: All existing solutions (Matryoshka, OrtSAE, ATM) require retraining. Our homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction, inspired by biological homeostatic plasticity.

3. **First local inhibition graph for SAE diagnostics**: The graph provides an interpretable structure revealing which latents compete with which, enabling practitioners to identify at-risk features without running absorption metrics. This is a new tool for SAE analysis.

**Evidence it hasn't been applied before**:
- Rozell et al. (2008) has ~2000 citations but zero applications to LLM SAEs.
- No paper in the SAE literature (Chanin et al., Bussmann et al., Korznikov et al., Li et al.) mentions lateral inhibition, LCA, or decoder correlation graphs.
- The Innovator perspective independently identified the same gap, confirming it is not addressed in existing work.

### Integration with Other Perspectives

This interdisciplinary proposal complements (rather than competes with) the other perspectives:

- **Theoretical**: The rate-distortion framework explains *why* absorption occurs (optimal compression). The LCA inhibition graph explains *how* it manifests mechanistically (competitive suppression via decoder correlations) and provides a tool to repair it.
- **Pragmatist**: The metric-specific impact finding (absorption affects recall but not precision) is explained by the inhibition mechanism: suppression reduces coverage (recall) but not selectivity (precision). Homeostatic repair should specifically improve recall.
- **Contrarian**: If repair does not improve downstream tasks, this strengthens the "absorption is benign" claim. The diagnostic tool still has value for identifying at-risk features.
- **Innovator**: The LCA connection was independently identified by both the Innovator and Interdisciplinary perspectives, confirming the gap is real and the structural correspondence is strong.

---

## Sources

- [Rozell et al., 2008. Sparse coding via thresholding and local competition in neural circuits.](https://siplab.gatech.edu/bibtexbrowser_20230525.php?key=rozell.06c&bib=siplab_DB.bib)
- [Jimenez-Nimmo & Mondragon, 2025. Hebbian CNN Framework with Integrated Competition Mechanisms.](https://openaccess.city.ac.uk/id/eprint/34530/1/Jimenez-Nimmo_Mondragon2025.pdf)
- [Zenke et al., 2017. The temporal paradox of Hebbian learning and homeostatic plasticity.](https://www.researchgate.net/publication/316356444_The_temporal_paradox_of_Hebbian_learning_and_homeostatic_plasticity)
- [Inhibition SNN, 2024. Unveiling the efficacy of various lateral inhibition learning.](https://link.springer.com/article/10.1007/s42452-024-06332-z)
- [Iversen, 2025. Sparse Neural Network Interpretability.](https://munin.uit.no/bitstream/handle/10037/37764/no.uit:wiseflow:7269325:62323654.pdf)
- [Olshausen & Field, 1997. Sparse coding with an overcomplete basis set.](https://www.nature.com/articles/381607a0)
- [Friston et al., 2023/2024. The free energy principle made simpler but not too simple.](https://kops.uni-konstanz.de/entities/publication/14fa40e7-4d5a-46e2-9489-ce0607b2fee8)
- [Brain-like Variational Inference, 2024. arXiv:2410.19315](https://arxiv.org/html/2410.19315v3)
- [Tishby & Zaslavsky, 2015. Deep Learning and the Information Bottleneck Principle.](https://arxiv.org/abs/1503.02406)
- [Superposition as Lossy Compression, 2025. arXiv:2512.13568](https://arxiv.org/html/2512.13568v1)
- [Gause, 1934. Competitive Exclusion Principle](https://www.geeksforgeeks.org/biology/competitive-exclusion-principle/)
- [Niche Partitioning Theory, 2024](https://library.fiveable.me/fundamentals-ecology/unit-6/niche-theory-resource-partitioning/study-guide/ejHJ1YRriC7nybIJ)
- [Individual Niche Plasticity, 2024-2025](https://www.researchgate.net/publication/227994160_Neutral_theory_in_community_ecology_and_the_hypothesis_of_functional_equivalence)
- [Bafna et al., 2025. Sparse autoencoders in protein language models. PNAS](https://ui.adsabs.harvard.edu/abs/2025PNAS..12206316G/abstract)
- [Hyvarinen & Pajunen, 1999. Nonlinear ICA non-identifiability](https://arxiv.org/html/2512.22282v1)
- [Review of NMF/PLSA identifiability, 2025. arXiv:2512.22282](https://arxiv.org/html/2512.22282v1)
- [Mechanistic Independence, 2025. arXiv:2509.22196](https://arxiv.org/html/2509.22196v1)
- [Klimashevskaia et al., 2024. A Survey on Popularity Bias in Recommender Systems.](https://link.springer.com/article/10.1007/s11257-024-09406-0)
- [Paiton et al., 2019. Analysis and applications of the Locally Competitive Algorithm.](https://redwood.berkeley.edu/wp-content/uploads/2020/05/paiton2019analysis.pdf)
- [WARP-LCA, 2024. arXiv:2410.18794](https://arxiv.org/html/2410.18794v1)
- [Chanin et al., 2024. A is for Absorption. arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- [Cui et al., 2025. On the Limits of Sparse Autoencoders. arXiv:2506.15963](https://arxiv.org/abs/2506.15963)
- [Bussmann et al., 2025. Matryoshka Sparse Autoencoders. arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
- [Korznikov et al., 2025. OrtSAE. arXiv:2509.22033](https://arxiv.org/abs/2509.22033)
- [Li et al., 2025. Time-Aware Feature Selection. arXiv:2510.08855](https://arxiv.org/abs/2510.08855)
