# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1996, 2004)** — *Sparse coding of sensory inputs* — Established that V1 simple cells learn sparse, parts-based representations of natural images through competitive coding. The learned basis vectors decompose complex scenes into elementary features, providing the foundational analogy for how SAEs might decompose neural network activations.

2. **Spratling (2008); bioRxiv (2024)** — *Unifying sparse coding, predictive coding, and divisive normalization* — Winner-take-all (WTA) mechanisms via lateral inhibition can be viewed as "extreme sparse coding." Predictive coding frameworks propose that cortical areas learn generative models where higher-level predictions compete to explain away lower-level sensory evidence, suppressing alternative explanations.

3. **Rozell et al. (2008); Shapero et al. (2012)** — *Locally Competitive Algorithms (LCA) and group sparse coding with WTA networks* — LCA uses lateral inhibition to enforce sparsity in neural circuits. Grouped WTA microcircuits require fewer long-range connections than global LCA and achieve faster convergence, suggesting biologically plausible sparse coding architectures.

4. **Rosch (1973, 1978); Rosch & Mervis (1975)** — *Prototype theory and basic-level categories* — Human categorization operates around a three-level hierarchy: superordinate (abstract, e.g., "furniture"), basic (most salient, e.g., "chair"), and subordinate (specific, e.g., "kitchen chair"). The basic level is cognitively privileged, suggesting that hierarchical feature decomposition has natural "anchor" levels.

5. **Hampton (various); Smith & Osherson (1988)** — *Conceptual combination and selective modification* — When humans combine concepts (e.g., "pet fish"), they selectively modify prototypes rather than averaging feature sets. This implies that hierarchical concepts are not simply additive; parent and child features interact through competitive, context-dependent selection mechanisms.

6. **PMC4685756 (Nature Neuroscience)** — *Hierarchical competitions subserving multi-attribute choice* — Competition via mutual inhibition occurs at multiple representation levels simultaneously (within attributes, between attributes, and at integrated value levels), not just at a final output stage.

#### Physics / Information Theory

7. **Friston (2005, 2010); Aguilera et al. (2022)** — *The free energy principle* — Variational free energy minimization unifies sparse coding, predictive coding, and VAEs. Under a Laplace prior, sparse coding emerges naturally from iterative inference within the FEP. The complexity-accuracy tradeoff in free energy directly maps to the reconstruction-sparsity tradeoff in SAEs.

8. **Kabashima, Krzakala, Mézard, Sakata, & Zdeborová (2016)** — *Phase transitions and sample complexity in Bayes-optimal matrix factorization* — Applied statistical physics methods (replica method, cavity/mean-field theory) to matrix factorization, the foundational problem underlying dictionary learning. Derives precise phase transitions for sparse recovery and factorization.

9. **Barbier & Macris (2022)** — *Statistical limits of dictionary learning: random matrix theory and the spectral replica method* — Directly addresses dictionary learning using the replica method, providing a statistical physics lens on when dictionaries can be recovered and when spurious minima emerge.

10. **Maillard, Krzakala, Mézard, & Zdeborová (2022)** — *Perturbative construction of mean-field equations in extensive-rank matrix factorization* — Develops perturbative mean-field theory for extensive-rank factorization, relevant to overcomplete dictionaries where the number of latent features exceeds the input dimension.

11. **Li & Wang (2018); Koch-Janusz & Ringel (2018); Liaw (2024)** — *Neural network renormalization group* — Deep network layers can be viewed as renormalization group transformations, with each layer implementing coarse-graining that extracts increasingly abstract features. Fixed points in RG flow correspond to scale-invariant representations.

12. **Ngampruetikorn et al. (2025)** — *Data coarse graining can improve model performance* — RG-inspired data coarse graining reveals a nonmonotonic dependence of prediction risk on the degree of feature integration, with "high-pass" schemes (filtering less relevant features) improving generalization.

#### Biology / Evolution

13. **Espinosa-Soto & Wagner (2010); Espinosa-Soto (2018)** — *On the role of sparseness in the evolution of modularity in gene regulatory networks* — Modularity in GRNs evolves because of specialization in gene activity. Sparseness amplifies modularity: selection for multiple gene activity phenotypes produces greater modularity increases in sparser networks. However, sparseness alone is insufficient.

14. **Erwin & Davidson (2009)** — *The evolution of hierarchical gene regulatory networks* — GRNs have a hierarchical architecture: conserved network kernels at the core, plug-in functional subcircuits, and rapidly evolving switch nodes. This hierarchy enables evolvability by localizing changes to specific modules without pleiotropic disruption.

15. **Gause (1932); Hutchinson (1957); MacArthur & Levins (1967)** — *Competitive exclusion principle and niche partitioning* — The competitive exclusion principle states that complete competitors cannot coexist. Niche partitioning (resource differentiation, spatial segregation, temporal separation) is the mechanism that enables species coexistence in ecosystems.

16. **Eisen & Rajewsky (various); Victora & Nussenzweig (various)** — *Clonal selection and affinity maturation in the immune system* — B-cell repertoires implement sparse, competitive feature learning: a vast repertoire is generated, but only a small subset is activated and refined based on antigen-driven competition. Higher clone abundance can paradoxically inhibit affinity maturation by suppressing competing variants.

### Cross-Disciplinary Gaps

Despite extensive work on SAE feature absorption, several cross-disciplinary transplants remain unexplored:

- **Predictive coding / explaining away**: While lateral inhibition and LCA have been applied to sparse coding (and recently to SAEs by Rajamanoharan et al., 2024), the specific "explaining away" mechanism from predictive coding has not been framed as a diagnostic or intervention for feature absorption.
- **Renormalization group / coarse graining**: RG-inspired methods have been applied to deep learning broadly, but not specifically to analyze or mitigate feature absorption in SAEs. The concept of "relevant" vs. "irrelevant" operators in RG could map to parent vs. child features in absorption.
- **Ecological niche partitioning**: The competitive exclusion principle and niche partitioning have not been explicitly mapped to SAE dictionary learning, despite the obvious structural correspondence between species competing for resources and features competing for activation budget under sparsity constraints.
- **Immune system affinity maturation**: The dynamics of clonal competition, where high-abundance clones suppress competitors, has a striking structural similarity to feature absorption (high-frequency child features suppressing parent features), but this analogy has not been formalized for SAEs.

---

## Phase 2: Initial Candidates

### Candidate A: "Explaining Away" as a Diagnostic for Feature Absorption (from Neuroscience / Predictive Coding)

- **Source principle**: In predictive coding and Bayesian networks, "explaining away" occurs when two possible causes compete to explain the same evidence. If cause A is activated, it suppresses cause B because the evidence is already "explained" — even if B is also a valid explanation.

- **Structural correspondence**: In SAEs, a parent feature (e.g., "starts with S") and a child feature (e.g., "short") co-occur on the same token. The child feature "explains away" the evidence that should also activate the parent, causing the parent to be absorbed. The SAE's sparsity penalty acts like a Bayesian prior favoring the most specific (highest-confidence) explanation.

- **Hypothesis**: Feature absorption rates will correlate with the strength of explaining-away dynamics between parent-child feature pairs. If we explicitly model SAE inference as competitive explaining-away (e.g., via a modified inference procedure that subtracts child contributions before testing parent activation), we can recover absorbed parent features and measure absorption more accurately.

- **Why not just a metaphor**: The correspondence is formal. Predictive coding's explaining-away is mathematically equivalent to sparse inference with a Laplace prior under the free energy principle. SAE training explicitly minimizes a reconstruction + sparsity objective that is isomorphic to variational free energy. The suppression of parent latents when child latents fire is not merely analogous to explaining away — it is the same competitive inference dynamic.

- **Novelty estimate**: 7/10. The explaining-away concept is well-known in neuroscience and has been applied to sparse coding, but framing feature absorption specifically as an explaining-away pathology and using it to design a diagnostic metric or intervention is underexplored in the SAE literature.

### Candidate B: Renormalization Group Coarse-Graining for Multi-Scale SAE Hierarchies (from Statistical Physics)

- **Source principle**: In the renormalization group (RG), a system's degrees of freedom are iteratively coarse-grained. At each step, "relevant" operators (those that grow under scaling) are retained, while "irrelevant" operators (those that decay) are integrated out. The result is a hierarchy of effective theories valid at different length scales.

- **Structural correspondence**: SAE features can be viewed as operators in an RG flow. Parent features ("animal") are relevant at coarse scales, while child features ("mammal," "dog") become relevant at finer scales. Feature absorption occurs because the SAE tries to represent all scales with a single dictionary — like trying to describe both atomic and macroscopic physics with the same Hamiltonian. A multi-scale SAE should apply RG-inspired coarse-graining to separate scales.

- **Hypothesis**: If we design an SAE training objective inspired by RG coarse-graining — where features are explicitly organized by "scale" and reconstruction is performed hierarchically (coarse features first, residuals decoded by finer features) — feature absorption will decrease because parent and child features operate at different scales and do not compete for the same sparsity budget.

- **Why not just a metaphor**: The correspondence is mathematical. Both RG and SAEs involve linear transformations that map microscopic variables to coarse-grained representations. The concept of "relevant" operators in RG has a direct mapping to "high-variance, frequently activating" features in SAEs. The scale separation in RG is analogous to the abstraction-level separation needed to prevent absorption.

- **Novelty estimate**: 5/10. Matryoshka SAEs and hierarchical SAEs already explore multi-scale architectures. The RG framing could provide theoretical justification and novel algorithmic ideas, but the core architectural direction is already active in the SAE literature.

### Candidate C: Niche Partitioning as a Mechanism to Prevent Feature Absorption (from Ecology)

- **Source principle**: The competitive exclusion principle states that species competing for identical resources cannot stably coexist. Niche partitioning enables coexistence by differentiating resource use (e.g., one species eats seeds, another eats insects; or they forage at different times/depths).

- **Structural correspondence**: SAE latents are like species competing for the limited "activation budget" imposed by the sparsity penalty. When a parent feature and a child feature compete for the same tokens (the same "resource"), the more specific child feature wins (competitive exclusion), and the parent is absorbed. Niche partitioning would require parent and child features to specialize on different "resources" — different aspects of the reconstruction task.

- **Hypothesis**: If we modify the SAE objective to enforce "niche partitioning" between hierarchically related features — for example, by penalizing parent-child cosine similarity in the decoder (OrtSAE does this weakly) or by assigning parent features to reconstruct coarse-grained residuals while child features reconstruct fine-grained details — absorption will decrease because the features no longer compete for exactly the same reconstruction tokens.

- **Why not just a metaphor**: The correspondence is game-theoretic and dynamical. Both ecosystems and SAEs are systems of competing agents (species/latents) subject to a resource constraint (carrying capacity/sparsity budget). The Lotka-Volterra competition equations and the dynamics of sparse coding under lateral inhibition share the same mathematical structure: winner-take-all competition with coexistence only when niches are differentiated.

- **Novelty estimate**: 8/10. While orthogonality penalties (OrtSAE) and hierarchical architectures (Matryoshka) address absorption, the explicit framing in terms of ecological competitive exclusion and niche partitioning is novel. The ecological literature provides richer concepts (resource differentiation, character displacement, limiting similarity) that have not been mapped to SAE training.

---

## Phase 3: Self-Critique

### Against Candidate A: Explaining Away

- **Shallow analogy attack**: Is this really explaining away, or is it just sparse coding? Critics could argue that SAEs already implement sparse coding, and "explaining away" adds nothing but vocabulary. However, the key distinction is directional: standard sparse coding treats all features as symmetric competitors, while explaining away is explicitly hierarchical (higher-level predictions explain away lower-level alternatives). This directional asymmetry maps directly onto the parent-child asymmetry in absorption.

- **Scale mismatch attack**: Explaining away in the brain operates at the level of perceptual inference over milliseconds. SAE absorption is a static property of trained dictionaries. However, we can bridge this by treating SAE inference (the forward pass that computes latents given an input) as the dynamical process where explaining away occurs.

- **Prior transplant check**: Rajamanoharan et al. (2024) applied LCA (lateral inhibition) to GPT-2 SAEs, which is related but different. LCA is a mechanism for enforcing sparsity; explaining away is a diagnostic principle for why certain features fail to activate. A search for "explaining away sparse autoencoder feature absorption" found no direct prior work. The concept appears in predictive coding and sparse coding literature but not as an absorption-specific framework.

- **Testability attack**: Can we distinguish "this works because of explaining away" from "this works because of standard sparse coding"? Yes. We can design a diagnostic experiment: (1) measure parent activation when the child is absent vs. present; (2) if parent activation drops significantly when the child is present (controlling for reconstruction need), this is evidence of explaining away; (3) compare this drop across SAE architectures. A training-free version can analyze existing pretrained SAEs.

- **Verdict**: STRONG

### Against Candidate B: Renormalization Group

- **Shallow analogy attack**: Is RG anything more than a fancy name for multi-scale architectures? Matryoshka SAEs already nest dictionaries at different scales. The RG framing risks being decorative unless it generates genuinely new algorithms.

- **Scale mismatch attack**: RG in physics operates on spatial degrees of freedom with well-defined symmetries and scaling dimensions. SAE features live in a high-dimensional semantic space without clear metric structure or symmetries. Defining "scale" for semantic features is non-trivial.

- **Prior transplant check**: Multiple papers (Li & Wang 2018, Koch-Janusz & Ringel 2018, Liaw 2024) have applied RG to deep learning. Matryoshka SAEs (2025) and HSAE (2026) explicitly explore hierarchical multi-scale dictionaries. The RG framing is unlikely to be novel as a pure concept.

- **Testability attack**: It is difficult to design an experiment that would succeed *if and only if* the RG principle is the active ingredient, because any multi-scale architecture improvement could be attributed to simpler factors (reduced competition, better residual modeling).

- **Verdict**: MODERATE (interesting theoretical framing, but weak novelty due to existing hierarchical SAE work)

### Against Candidate C: Niche Partitioning

- **Shallow analogy attack**: Is niche partitioning anything more than orthogonality constraints? OrtSAE already penalizes feature similarity. However, niche partitioning is richer than orthogonality: it includes resource differentiation (different features reconstruct different *aspects* of the input), spatial/temporal segregation (features active on different token subsets), and limiting similarity (there is a minimum viable distance between coexisting features). These concepts go beyond simple cosine-similarity penalties.

- **Scale mismatch attack**: Ecological dynamics unfold over generations with mutation and selection. SAE training uses gradient descent over hours. However, both are optimization processes subject to competitive exclusion dynamics. The timescale difference does not invalidate the structural correspondence.

- **Prior transplant check**: No prior work explicitly frames SAE feature absorption in terms of competitive exclusion or niche partitioning. OrtSAE uses orthogonality but does not invoke ecological theory. Matryoshka SAEs use scale separation but not resource partitioning.

- **Testability attack**: Yes. We can define "niche overlap" between two SAE features as the correlation of their activation patterns or the cosine similarity of their decoder directions. We can then test: (1) Does high niche overlap between parent-child pairs predict absorption? (2) If we artificially reduce overlap (e.g., by projecting child features onto the orthogonal complement of parent features), does absorption decrease? Both experiments are training-free and testable on existing SAEs.

- **Verdict**: STRONG

---

## Phase 4: Refinement

### Dropped
- **Candidate B (Renormalization Group)** is dropped as a front-runner. While the RG-multi-scale analogy is intellectually compelling, Matryoshka SAEs and HSAE already occupy this space, making the pure framing contribution weak. It could be revived as a theoretical lens for a future paper but is not the strongest interdisciplinary transplant for the current project.

### Strengthened

**Candidate A (Explaining Away)** is strengthened by formalizing the structural correspondence:

In predictive coding, the posterior over causes given data is approximated by minimizing variational free energy:
\[ F = \mathbb{E}_q[\log q(z) - \log p(x|z)p(z)] \]

Under a Laplace prior on causes, this reduces to sparse coding. The "explaining away" effect arises because causes are not independent a posteriori: activating cause \(z_i\) reduces the need for cause \(z_j\) if both explain the same residual.

In an SAE, the training objective is:
\[ \mathcal{L} = ||x - \hat{x}||^2 + \lambda ||z||_1 \]

where \(z = \text{encode}(x)\) and \(\hat{x} = D z\). For hierarchical features where parent \(p\) and child \(c\) have correlated decoder directions (\(D_p \approx D_c + \epsilon\)), activating \(c\) reduces the residual that \(p\) would explain. Under the sparsity penalty, the SAE prefers the single most specific cause — the child — and suppresses the parent.

**Diagnostic experiment for Candidate A**: 
1. Select parent-child feature pairs from an existing SAE (e.g., GemmaScope or GPT-2 SAEs).
2. For tokens where the child fires, measure the parent activation.
3. For a matched set of tokens where the child does NOT fire but the parent concept is present, measure the parent activation.
4. If parent activation is significantly lower when the child is present (controlling for input similarity), this is direct evidence of explaining-away dynamics causing absorption.
5. Compute an "explaining-away score" for each pair and correlate it with the absorption metric.

**Candidate C (Niche Partitioning)** is strengthened by formalizing the mapping:

In ecology, the Lotka-Volterra competition equation for two species is:
\[ \frac{dN_i}{dt} = r_i N_i \left(1 - \frac{N_i + \alpha_{ij} N_j}{K_i}\right) \]

where \(\alpha_{ij}\) is the competition coefficient. Coexistence requires \(\alpha_{ij} \alpha_{ji} < 1\).

In SAE training, two features compete through the shared sparsity penalty. Let \(a_i, a_j\) be mean activation frequencies and \(s_{ij} = \cos(D_i, D_j)\) be decoder similarity. The effective "competition coefficient" is proportional to \(s_{ij}\). Feature absorption occurs when \(s_{ij} \approx 1\) (parent and child are nearly collinear), making \(\alpha \gg 1\) and competitive exclusion inevitable.

**Diagnostic experiment for Candidate C**:
1. For each parent-child pair in an SAE, compute "niche overlap" as the cosine similarity of decoder directions weighted by activation co-occurrence.
2. Test whether pairs with higher overlap show higher absorption rates.
3. Perform an intervention: project child decoder directions onto the orthogonal complement of parent directions (a training-free geometric modification).
4. Measure whether this "niche partitioning" intervention reduces absorption while preserving reconstruction.

### Selected Front-Runner

**Candidate C: Niche Partitioning from Ecology** is selected as the front-runner.

Rationale:
1. **Highest novelty**: No prior SAE work explicitly uses ecological competitive exclusion theory.
2. **Strong structural correspondence**: The Lotka-Volterra competition equations and SAE sparse coding dynamics are formally related.
3. **Training-free diagnostic**: The diagnostic experiment (measuring niche overlap and testing a geometric intervention) can be performed entirely on existing pretrained SAEs, strictly respecting the project constraint.
4. **Rich conceptual vocabulary**: Ecology provides concepts beyond orthogonality — resource differentiation, limiting similarity, character displacement — that could inspire future architectural innovations.
5. **Directly addresses absorption**: The core mechanism of absorption (parent and child competing for the same tokens because their decoder directions are too similar) is exactly the competitive exclusion scenario.

Candidate A remains a strong backup. The explaining-away framework could be integrated as a complementary diagnostic (measuring the dynamical suppression of parents by children), while Candidate C provides the structural/geometric explanation (why they compete in the first place).

---

## Phase 5: Final Proposal

### Title
**Niche Partitioning in Sparse Autoencoders: An Ecological Framework for Understanding and Mitigating Feature Absorption**

### Source Principle
The **competitive exclusion principle** from ecology states that species competing for identical resources cannot stably coexist. **Niche partitioning** — the differentiation of resource use, habitat, or temporal activity — is the primary mechanism enabling coexistence. In mathematical ecology, the Lotka-Volterra competition model formalizes this: two species can coexist only if their interspecific competition coefficients are smaller than their intraspecific competition coefficients.

### Structural Correspondence

| Ecology | Sparse Autoencoder |
|---------|-------------------|
| Species | SAE latent features |
| Resource (food, territory) | Reconstruction tokens / activation budget |
| Carrying capacity | Sparsity budget (L0 or L1 penalty) |
| Competition coefficient \(\alpha_{ij}\) | Decoder direction similarity + activation co-occurrence |
| Competitive exclusion | Feature absorption (parent suppressed by child) |
| Niche partitioning | Orthogonality / differentiation of decoder directions |
| Limiting similarity | Minimum viable distance between coexisting features |

In an SAE, the sparsity penalty creates a hard activation budget. When a parent feature (e.g., "starts with S") and a child feature (e.g., "short") have highly similar decoder directions, they compete for the same tokens. Because the child feature is more specific and thus more efficient at reconstructing those tokens, it wins the competition, and the parent feature is excluded — this is feature absorption. The ecological insight is that absorption is not merely a "bug" in sparsity optimization; it is the inevitable consequence of **insufficient niche partitioning** between hierarchically related features.

### Hypothesis
**H1 (Structural)**: Parent-child feature pairs in SAEs with higher "niche overlap" (decoder direction similarity weighted by activation co-occurrence) will exhibit higher rates of feature absorption.

**H2 (Interventional)**: A training-free geometric intervention that enforces niche partitioning (projecting child decoder directions onto the orthogonal complement of parent directions) will reduce feature absorption without significantly degrading reconstruction quality.

### Method

**Step 1: Niche Overlap Quantification**
For a corpus of pretrained SAEs (GemmaScope, GPT-2 SAEs, SAEBench checkpoints):
- Identify parent-child hierarchical feature pairs using existing absorption benchmarks (Chanin et al.'s first-letter task) or LLM-based hierarchy discovery.
- For each pair \((p, c)\), compute:
  - Decoder similarity: \(s_{pc} = \cos(D_p, D_c)\)
  - Activation co-occurrence: \(o_{pc} = \mathbb{E}[\mathbb{1}(z_p > 0) \cdot \mathbb{1}(z_c > 0)]\)
  - Niche overlap: \(N_{pc} = s_{pc} \cdot o_{pc}\)

**Step 2: Correlation with Absorption**
- Using the sae-spelling absorption metric or a simplified proxy, measure absorption rate for each parent-child pair.
- Test the correlation between \(N_{pc}\) and absorption rate (Pearson/Spearman).

**Step 3: Niche Partitioning Intervention**
- For high-absorption pairs, modify the child decoder direction:
  \[ D_c' = D_c - \frac{D_c \cdot D_p}{||D_p||^2} D_p \]
  (project \(D_c\) onto the orthogonal complement of \(D_p\)).
- Re-normalize \(D_c'\).
- Re-run SAE inference with the modified decoder (keeping the encoder fixed).
- Measure the change in absorption rate and reconstruction MSE.

**Step 4: Ecological Metrics**
- Compute "species richness" analogs: effective number of active features per token.
- Compute "Shannon diversity" of feature activations to test whether niche partitioning increases feature diversity.

### Diagnostic Experiment
The key diagnostic is the **orthogonal projection intervention (Step 3)**. If absorption decreases after this intervention, it confirms that the similarity of decoder directions (niche overlap) is the active ingredient causing absorption — not merely a correlated property. If absorption does not decrease, it falsifies the niche-partitioning hypothesis and suggests that absorption is driven by encoder dynamics or training artifacts rather than geometric overlap.

Pre-registered falsification criterion: If the mean absorption rate does not decrease by at least 15% relative to baseline for high-overlap pairs after orthogonal projection, H2 is falsified.

### Experimental Plan
- **Models**: GPT-2 Small (open, accessible), Gemma-2-2B if HF token becomes available.
- **SAEs**: Existing pretrained checkpoints from SAELens / SAEBench (training-free).
- **Compute**: Single GPU (RTX 4090 or A100). Each step is lightweight: loading SAEs, computing similarities, running inference on held-out tokens.
- **Time budget**: 
  - Pilot (10–15 min): 1 SAE, 5–10 parent-child pairs, Steps 1–3.
  - Full experiment (30–45 min): 10–20 SAEs, 50–100 pairs, all steps.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Niche overlap does not correlate with absorption | Medium | High | If correlation is weak, pivot to analyzing *why* — the null result itself informs whether absorption is geometric or dynamic. |
| Orthogonal projection degrades reconstruction severely | Medium | Medium | Measure MSE change; if degradation is >5%, try partial projection (soft niche partitioning) or weighted orthogonalization. |
| Parent-child pair identification is noisy | Medium | Medium | Use multiple hierarchy sources (sae-spelling benchmark + LLM labeling) and report robustness checks. |
| The ecological framing is seen as a metaphor | Low | Low | Emphasize the formal Lotka-Volterra → sparse coding mapping in the paper; provide mathematical derivations in the appendix. |

### Novelty Claim
The specific cross-disciplinary insight is that **feature absorption in SAEs is a competitive exclusion phenomenon**. While prior work treats absorption as a training artifact or a sparsity pathology, the ecological framing reveals it as a structural consequence of insufficient niche partitioning between hierarchically related features. The training-free orthogonal projection intervention is a direct, testable consequence of this insight. No prior work has:
1. Framed SAE feature absorption in terms of ecological competitive exclusion.
2. Proposed a "niche overlap" metric for SAE feature pairs.
3. Tested a training-free geometric niche-partitioning intervention on existing pretrained SAEs.

### Integration with Front-Runner Proposal
The current project front-runner (Candidate A: multi-objective Pareto evaluation) focuses on *quantifying* absorption and its tradeoffs across architectures. This interdisciplinary perspective provides a *theoretical mechanism* for *why* absorption occurs and *why* certain architectures (OrtSAE with orthogonality penalties, Matryoshka with scale separation) succeed or fail. The niche-partitioning framework predicts:
- **OrtSAE** reduces absorption by directly decreasing niche overlap (competition coefficients).
- **Matryoshka SAE** reduces absorption by scale separation (spatial/temporal niche differentiation).
- **Standard SAEs** show high absorption because they lack niche partitioning mechanisms.

These predictions can be tested within the existing multi-objective evaluation framework, strengthening the theoretical contribution of the front-runner without requiring additional training.

---

## References

- Olshausen, B. A., & Field, D. J. (1996). *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*. Nature, 381(6583), 607–609.
- Rozell, C. J., Johnson, D. H., Baraniuk, R. G., & Olshausen, B. A. (2008). *Sparse coding via thresholding and local competition in neural circuits*. Neural Computation, 20(10), 2526–2563.
- Spratling, M. W. (2008). *Predictive coding as a model of biased competition in visual attention*. Vision Research, 48(12), 1391–1408.
- Rosch, E. (1978). *Principles of categorization*. In Cognition and Categorization (pp. 27–48). Lawrence Erlbaum.
- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127–138.
- Kabashima, Y., Krzakala, F., Mézard, M., Sakata, A., & Zdeborová, L. (2016). *Phase transitions and sample complexity in Bayes-optimal matrix factorization*. IEEE Transactions on Information Theory, 62(7), 4228–4265.
- Barbier, J., & Macris, N. (2022). *Statistical limits of dictionary learning: Random matrix theory and the spectral replica method*. Physical Review E, 106(2), 024136.
- Koch-Janusz, M., & Ringel, Z. (2018). *Mutual information, neural networks and the renormalization group*. Nature Physics, 14(6), 578–582.
- Espinosa-Soto, C. (2018). *On the role of sparseness in the evolution of modularity in gene regulatory networks*. PLOS Computational Biology, 14(5), e1006172.
- Erwin, D. H., & Davidson, E. H. (2009). *The evolution of hierarchical gene regulatory networks*. Nature Reviews Genetics, 10(2), 141–148.
- MacArthur, R., & Levins, R. (1967). *The limiting similarity, convergence, and divergence of coexisting species*. The American Naturalist, 101(921), 377–385.
- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507.
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033.
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
- Rajamanoharan, S., et al. (2024). *Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders*. arXiv:2411.13117.
