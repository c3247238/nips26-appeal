# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507** — Coins feature absorption; introduces the canonical metric; validates across hundreds of LLM SAEs. Essential baseline for any absorption work.

2. **Chanin et al., 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756** — Discovers the dual pathology to absorption: narrow SAEs merge correlated features. Shows Matryoshka SAEs trade absorption for hedging.

3. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Reduces absorption by 65% via chunk-wise cosine-similarity penalty on decoder weights. Demonstrates that geometric constraints on decoder weights can mitigate absorption with minimal overhead.

4. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Uses nested dictionaries to separate abstraction levels; achieves ~10x reduction in absorption (0.49 -> 0.05) at cost of ~2pp explained variance and ~50% training time increase.

5. **Michaud et al., 2025. "Understanding sparse autoencoder scaling in the presence of feature manifolds." arXiv:2509.02565** — Identifies a phase-transition threshold (alpha vs. beta) in SAE scaling. In the pathological regime (beta < alpha), common manifolds absorb the majority of latents, severely limiting feature discovery.

6. **Yao et al., 2025. "AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations." arXiv:2508.17320** — Shows text complexity is linearly encoded in LLM representations and proposes input-dependent sparsity. Outperforms fixed-sparsity baselines on SAEBench including absorption.

7. **Girrbach & Akata, 2025. "Sparse Autoencoders are Topic Models." arXiv:2511.16309** — Reframes SAEs as topic models via a continuous-space extension of LDA. Derives the SAE objective as a MAP estimator under this generative model.

8. **Zheng et al., 2025. "Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders." arXiv:2507.23220** — Defines topics over SAE-learned features rather than words, enabling topic-based steering. Bridges interpretability and topic modeling.

9. **Olshausen & Field, 1997. "Sparse coding with an overcomplete basis set: A strategy employed by V1?" Vision Research** — Foundational neuroscience result: sparse coding in visual cortex learns parts-based representations via competitive dynamics.

10. **Hosoya & Hyvarinen, 2015. "A mixture of sparse coding models for holistic and parts-based face processing in IT cortex."** — Shows cortex uses separate sparse coding submodels for different abstraction levels (faces vs. non-faces), analogous to Matryoshka nesting.

11. **Karklin & Lewicki, 2005. "Hierarchical nonlinear generative sparse coding models."** — Proposes hierarchical Bayesian sparse coding where each level builds a sparse model of the level below, capturing progressive abstraction.

12. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability." ICML 2025 / arXiv:2503.09532** — The dominant benchmark suite. Essential for validating any new metric or architecture against 200+ existing SAEs.

### Landscape Summary

The SAE field has rapidly matured from "can we find interpretable features?" to "why do our features systematically fail?" Feature absorption—where general parent features are swallowed by specific child features due to sparsity pressure—has emerged as a central pathology. The community has responded with architectural fixes (OrtSAE, Matryoshka, masked regularization, AdaptiveK) and theoretical frameworks (piecewise biconvexity, scaling phase transitions, topic-model reinterpretations).

However, three gaps remain particularly striking from an innovator's perspective:

**Gap 1: No one has treated absorption as a dynamical, competitive process.** Current work frames absorption as a static property of trained SAEs or a training-dynamics artifact. But SAEs are fundamentally competitive systems: latents compete to reconstruct each input under a sparsity budget. The neuroscience literature on visual cortex offers rich tools for analyzing such competition—winner-take-all dynamics, lateral inhibition, and hierarchical sparse coding—yet these have barely been applied to diagnose or quantify absorption in LLM SAEs.

**Gap 2: The topic-modeling reinterpretation of SAEs is underexploited for absorption.** If SAEs are topic models, then absorption is equivalent to "sub-topic collapse" in LDA, where specific topics absorb general ones. Hierarchical topic models (like hLDA) solve this by explicitly modeling topic hierarchies. No work has yet used the topic-model lens to design a training-free diagnostic for absorption severity or to predict which features will absorb which others.

**Gap 3: Context-dependent absorption is almost entirely unexplored.** Absorption is measured as a scalar property of an SAE checkpoint. But real features are hierarchical only in some contexts: "mammal" absorbs "dog" when the input is about animals, but not when it is about music genres. The recent AdaptiveK work shows that input complexity modulates sparsity needs; it is natural to ask whether input semantic domain modulates absorption patterns.

These gaps suggest three unconventional, cross-domain research directions.

---

## Phase 2: Initial Candidates

### Candidate A: Competitive Dynamics Diagnostic — "Lateral Inhibition Reveals Absorption Hotspots"

- **Hypothesis**: Feature absorption in SAEs corresponds to pathological winner-take-all dynamics where high-frequency child features suppress parent features via effective lateral inhibition. By quantifying the "inhibition field" around each SAE latent (how strongly its presence suppresses correlated latents), we can predict absorption rates *without* task-specific probes or causal ablations.

- **Cross-domain insight**: In visual cortex, lateral inhibition ensures that when an edge detector fires, nearby orientation detectors are suppressed. This prevents feature merging. In SAEs, the sparsity penalty creates an *implicit* lateral inhibition, but it is not uniform: high-frequency features develop stronger inhibitory fields, causing them to absorb low-frequency parents. We transplant the neuroscience concept of "inhibition fields" into SAE analysis.

- **Evidence for**:
  - OrtSAE reduces absorption by penalizing decoder cosine similarity—this is essentially reducing the spatial overlap of inhibitory fields (Korznikov et al., 2025).
  - Matryoshka SAEs separate abstraction levels into nested dictionaries, which mimics the cortical hierarchy (V1->IT) where separate subpopulations encode different scales (Hosoya & Hyvarinen, 2015; Bussmann et al., 2025).
  - The scaling phase-transition work shows that common manifolds absorb latents when alpha > beta, which is exactly what would happen if high-frequency features had stronger competitive advantage (Michaud et al., 2025).

- **Novelty estimate**: 8/10. The connection between SAE sparsity and lateral inhibition is intellectually natural but has not been operationalized as a predictive diagnostic for absorption. Winner-take-all dynamics are mentioned in passing in SAE tutorials but never used to quantify absorption severity.

### Candidate B: Topic-Model Absorption Spectrum — "SAEs as Hierarchical Topic Models"

- **Hypothesis**: If SAEs are topic models (Girrbach & Akata, 2025), then feature absorption is equivalent to "sub-topic collapse" in hierarchical topic models. We can repurpose hierarchical topic modeling tools—specifically, the nested Chinese restaurant process (nCRP) and topic-tree coherence metrics—to construct a training-free "absorption spectrum" that reveals how general-to-specific feature hierarchies are distributed across the SAE dictionary.

- **Cross-domain insight**: In hierarchical LDA (hLDA), topics are organized into a tree where general topics are parents of specific topics. A well-fit hLDA model has clear tree structure; a collapsed model has specific topics directly under the root, bypassing general topics. We transplant the nCRP tree-reconstruction algorithm: treat SAE decoder directions as "topic vectors," use LLM-based feature descriptions to estimate semantic similarity, and fit a topic hierarchy. The degree of "skipping" (specific topics attached near the root) quantifies absorption.

- **Evidence for**:
  - Girrbach & Akata (2025) prove that SAEs are MAP estimators of a continuous LDA extension, giving theoretical legitimacy to the topic-model analogy.
  - Zheng et al. (2025) successfully use SAE features as topic atoms, showing the semantic coherence of SAE-learned "topic" directions.
  - Chanin et al. (2024) show absorption is hierarchical (parent->child); hLDA is the canonical framework for modeling such hierarchies.

- **Novelty estimate**: 7/10. The SAE-as-topic-model paper is very recent (Nov 2025), and no follow-up work has applied it to absorption. The nCRP tree-reconstruction idea is well-known in topic modeling but would be novel in the SAE literature. The main risk is that the analogy might be too loose to yield predictive power.

### Candidate C: Context-Dependent Absorption Atlas — "Absorption is Not a Scalar"

- **Hypothesis**: Feature absorption varies systematically with input semantic domain and complexity. By constructing a "context-dependent absorption atlas"—measuring absorption rates separately for different input clusters (e.g., scientific text, fiction, code, dialogue)—we will find that some domains exhibit dramatically higher absorption than others, and that current scalar absorption metrics systematically underestimate the problem for specialized domains.

- **Cross-domain insight**: AdaptiveK SAEs (Yao et al., 2025) show that sparsity needs scale with input complexity. In neuroscience, hierarchical visual processing is *task-dependent*: object recognition recruits IT more strongly than simple detection. We transplant the idea of "domain-adaptive representation" into SAE evaluation: absorption is not a fixed property of the SAE, but an emergent property of the SAE-input interaction.

- **Evidence for**:
  - Yao et al. (2025) show that complex inputs need more active features (higher K). If the SAE has a fixed sparsity budget, complex domains should force more absorption as the SAE "sacrifices" general features to fit specifics.
  - Chanin et al. (2024) measure absorption on a spelling task (first-letter detection), which is a very specific domain. It is unknown whether the same SAEs show similar absorption on, say, biological taxonomies or geographical hierarchies.
  - The SAEBench benchmark uses mixed corpora, but its absorption metric is aggregated across all inputs, masking domain-specific variation.

- **Novelty estimate**: 8/10. Context-dependent absorption is explicitly identified as a gap in the literature survey (Gap 4), and almost no work has studied it. The idea of an "absorption atlas" is visually and conceptually compelling. The main risk is that domain differences might be smaller than hypothesized, making the result a null finding.

---

## Phase 3: Self-Critique

### Against Candidate A (Competitive Dynamics Diagnostic)

- **Prior work attack**: OrtSAE already operationalizes decoder orthogonality as a geometric fix for absorption. Is "inhibition field" just a neuroscience-flavored rebranding of decoder cosine similarity? The key distinction would be that inhibition fields are *asymmetric* (child suppresses parent more than parent suppresses child) and *input-dependent*, whereas OrtSAE's orthogonality penalty is symmetric and static. But has anyone already measured asymmetric suppression in SAEs? A search for "asymmetric feature suppression sparse autoencoder" and "causal ablation directionality SAE" finds no direct hits. The idea appears unstudied.

- **Methodological attack**: How do we measure an "inhibition field" in a trained SAE without causal ablations? One approach: for each latent j, measure the average activation of all other latents when j is active vs. when j is inactive, conditioned on input similarity. But this is correlational. The Chanin et al. (2024) metric requires causal verification for a reason. Our competitive-dynamics metric might pick up correlation patterns that are not causal.

- **Theoretical attack**: Does the lateral inhibition analogy actually hold? In cortex, inhibition is implemented by dedicated interneurons. In SAEs, there are no explicit inhibitory connections—suppression is mediated entirely by the encoder matrix and sparsity penalty. The "effective inhibition" is a post-hoc statistical construct, not a mechanistic circuit. This weakens the structural correspondence.

- **Scalability attack**: Computing pairwise inhibition fields for a 65K-feature SAE is O(d_sae^2) and expensive. Chunk-wise approximations (like OrtSAE) might be needed, introducing approximation error.

- **Verdict**: MODERATE. The asymmetric suppression angle is genuinely novel, but the measurement challenge is significant. The neuroscience analogy is more metaphorical than mechanistic. Worth pursuing if we can define a computationally tractable, falsifiable proxy for asymmetric inhibition.

### Against Candidate B (Topic-Model Absorption Spectrum)

- **Prior work attack**: The SAE-as-topic-model paper (Girrbach & Akata, 2025) is extremely recent. No one has applied it to absorption yet. However, hierarchical topic models have been applied to neural representations before. A search for "hierarchical topic model neural network activations" finds work from 2018-2022 on interpreting CNNs and RNNs with hLDA. The specific combination (SAE + nCRP + absorption) appears novel, but the component ideas are not.

- **Methodological attack**: The nCRP requires a document-feature matrix. For SAEs, this means constructing a corpus x latent activation matrix, then running hLDA. But hLDA is notoriously slow and finicky. More critically, hLDA assumes a tree-structured hierarchy, while real semantic feature relationships are DAGs (a dog is both a mammal and a pet). Forcing a tree structure may create artifacts that look like absorption when they are actually just polysemy.

- **Theoretical attack**: The SAE-as-topic-model derivation assumes a specific generative model with Gaussian activations and exponential priors. Real LLM activations are not Gaussian. The MAP equivalence is elegant but may not be robust enough to support precise predictions about absorption. The topic-model analogy could be a superficial rebranding of dictionary learning.

- **Scalability attack**: hLDA does not scale well to 65K "topics" (SAE latents). Approximate inference (variational or neural) would be needed, reintroducing the black-box problem that interpretability research tries to avoid.

- **Verdict**: WEAK. The tree-structure assumption is too restrictive for real semantic hierarchies, and hLDA scaling is poor. The topic-model analogy is intellectually rich but empirically fragile. Drop this candidate.

### Against Candidate C (Context-Dependent Absorption Atlas)

- **Prior work attack**: Is anyone already measuring absorption by domain? The Chanin et al. (2024) metric is defined on a spelling task, which is implicitly domain-specific. But they do not compare across domains. SAEBench aggregates absorption across all inputs. A search for "domain-specific absorption sparse autoencoder" or "input-dependent feature absorption" finds no direct hits. The idea appears genuinely unexplored.

- **Methodological attack**: How do we define "domains"? Topic modeling (LDA) on the activation corpus is one approach, but introduces model-dependence. Manual domain labels (scientific, fiction, code, dialogue) are cleaner but require dataset curation. The pilot would need to show that domain differences are large enough to be detectable with reasonable sample sizes.

- **Theoretical attack**: Is it obvious that absorption should vary by domain? If absorption is driven by the SAE's fixed decoder/encoder weights, then for any given parent-child feature pair, absorption is a structural property of the SAE. However, the *prevalence* of absorption—the fraction of inputs on which absorption occurs for that pair—absolutely depends on how often the domain triggers the hierarchy. A "mammal->dog" absorption will only manifest in animal-related text. So the theoretical objection is weak: domain-dependence is a natural consequence of hierarchical feature sparsity.

- **Scalability attack**: Running the full Chanin et al. absorption pipeline separately per domain is expensive. But we can use a simplified proxy: for each domain, train logistic regression probes for a fixed set of hierarchical concepts, then measure the false-negative rate where probes succeed but SAE latents fail. This is much cheaper than full causal ablation per domain and can be done training-free.

- **Verdict**: STRONG. The idea is novel, theoretically grounded, and practically feasible. The main challenge is defining domains and ensuring the per-domain measurement is reliable, but this is solvable.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Topic-Model Absorption Spectrum)** dropped because: The nCRP tree-structure assumption is too restrictive for DAG-like semantic hierarchies, and hLDA inference does not scale to 65K latents. The topic-model analogy is elegant but empirically fragile for this specific application.

### Strengthened Ideas

- **Candidate A (Competitive Dynamics Diagnostic)**: To address the measurement challenge, we refine the idea from a full "inhibition field" to a **training-free, computable proxy: the Asymmetric Decoder Suppression Index (ADSI)**. For each pair of latents (parent candidate p, child candidate c), ADSI measures the cosine similarity of c's decoder direction with the *residual reconstruction error* when p is ablated. If c's decoder aligns with p's missing reconstruction, c is effectively suppressing p. This is correlational but can be validated against the causal Chanin metric on a subset. The key change: we do not claim to measure biological lateral inhibition; we measure *effective suppression* in the decoder geometry, which is a well-defined linear-algebraic quantity.

- **Candidate C (Context-Dependent Absorption Atlas)**: To make this tractable, we scope the pilot tightly: use **3 domains** (biology taxonomies, geographical hierarchies, color shades) on a single SAE (GemmaScope 16K or GPT-2 equivalent), with **10 parent-child pairs per domain**. We use a simplified absorption proxy based on probe success + latent activation threshold, rather than full causal ablation. The full atlas would require scaling to more domains and SAEs, but the pilot is designed to fit within 15 minutes.

### Additional Evidence Found

- **AdaptiveK SAE results** (Yao et al., 2025): On SAEBench's Feature Absorption metric, AdaptiveK outperforms fixed-sparsity baselines despite being trained on 2000x less data. This supports the claim that input-dependent sparsity allocation affects absorption, and by extension, that input domain/complexity likely modulates absorption patterns.
- **Michaud et al. (2025) phase transition**: The alpha~0.5-0.7 estimate for Gemma Scope activation frequencies suggests real LLM SAEs may operate near the pathological regime boundary, making context-dependent competitive effects (Candidate A) especially relevant.
- **Chanin et al. (2025) hedging paper**: Explicitly notes that Matryoshka SAEs trade absorption for hedging. This reinforces the multi-objective tradeoff framing of the front-runner and suggests that any single-metric diagnostic (including our ADSI) must be validated against the full metric suite.

### Selected Front-Runner

**Candidate C: Context-Dependent Absorption Atlas** is the strongest idea. It is genuinely novel (no prior work studies domain-dependent absorption), theoretically grounded (hierarchical features only absorb when triggered), and practically feasible within the project's training-free constraints. It also naturally integrates with the existing front-runner (multi-objective Pareto evaluation): the atlas adds a *contextual* dimension to the tradeoff analysis, asking not just "which SAE is best?" but "which SAE is best for which inputs?"

**Candidate A is retained as a backup** because the ADSI proxy is computationally cheap and could be run in parallel with the atlas pilot. If ADSI correlates with Chanin's causal absorption metric, it becomes a valuable training-free screening tool.

---

## Phase 5: Final Proposal

### Title
**The Absorption Atlas: Context-Dependent Feature Absorption in Sparse Autoencoders**

(Alternative: **Feature Absorption is Not a Scalar: A Domain-Resolved Analysis of SAE Hierarchical Failures**)

### Hypothesis
Feature absorption in sparse autoencoders varies systematically with input semantic domain and complexity. Current scalar absorption metrics—aggregated across diverse corpora—systematically underestimate absorption severity for specialized domains and mask cross-domain tradeoffs between SAE architectures.

### Motivation
Feature absorption is the dominant failure mode preventing SAEs from achieving reliable monosemanticity. All existing absorption metrics (Chanin et al., 2024; SAEBench) report a single scalar per SAE checkpoint. But hierarchical features are not universal: a "mammal->dog" absorption only matters when the input is about animals; a "continent->country" absorption only matters for geography. If absorption is domain-dependent, then a single scalar is not merely incomplete—it may be misleading. An SAE with low average absorption could have catastrophic absorption in scientific text, while an SAE with higher average absorption could be safer for general-domain use. Understanding this domain structure is essential for task-adaptive SAE selection and for designing architectures that mitigate absorption where it actually occurs.

### Method
We propose a training-free, domain-resolved absorption analysis pipeline:

**Step 1: Domain Discovery.**
Partition a large activation corpus (e.g., C4, The Pile) into semantic domains using a lightweight topic model or off-the-shelf classifier. For the pilot, we use three manually curated domains with clear hierarchical structure:
- Biology: taxonomic ranks (kingdom -> phylum -> class -> order -> family -> genus -> species)
- Geography: spatial containment (continent -> country -> city)
- Colors: shade hierarchies (color -> shade -> specific pigment name)

**Step 2: Hierarchy Probe Training.**
For each domain, define 10 parent-child concept pairs (e.g., "mammal"->"dog", "Europe"->"France", "red"->"crimson"). Train logistic regression probes on the LLM's residual-stream activations to detect these concepts. A probe with high F1 confirms the concept is represented in the model.

**Step 3: Absorption Detection (Simplified Proxy).**
For each probe-positive input token:
1. Identify the top-k SAE latents by activation magnitude.
2. Check whether any latent matching the parent concept (via automated interpretation or cosine similarity to probe direction) fires above threshold.
3. If the parent probe succeeds but no parent-matching latent fires, classify the token as absorbed.
4. Measure the absorption rate per domain as: `absorbed_tokens / probe_true_positives`.

**Step 4: Cross-Architecture Comparison.**
Run the atlas on 10-20 pretrained SAE checkpoints across 3-4 architecture families (Standard, TopK, JumpReLU, Gated) on GPT-2 Small or Pythia-160M. Test whether architectures with low *average* absorption hide high *domain-specific* absorption, and whether any architecture dominates across all domains.

**Step 5: Validation.**
On a random sample of 50 absorbed tokens per domain, run the full Chanin et al. causal ablation pipeline to validate that the simplified proxy agrees with the gold-standard metric. Report proxy accuracy and calibration.

### Cross-Domain Insight
The core transplanted principle is **domain-adaptive representation** from both neuroscience and dynamic sparsity research. In the visual cortex, hierarchical sparse coding is task-dependent: object recognition recruits high-level IT features, while texture discrimination recruits V1/V2. In SAEs, Yao et al. (2025) showed that input complexity modulates optimal sparsity. Our proposal extends this to **input semantics**: the SAE's fixed dictionary must serve all domains, but the competitive pressure that creates absorption is domain-local. The structural correspondence holds because SAEs, like cortex, use a shared overcomplete dictionary where feature competition is resolved input-by-input.

### Experimental Plan

| Experiment | Description | Expected Duration |
|---|---|---|
| **e_atlas_pilot** | Run Steps 1-4 on 1 SAE x 3 domains x 10 pairs each (GPT-2 Small). | ~15 min |
| **e_atlas_validation** | Validate 50 absorbed tokens with causal ablation. | ~15 min |
| **e_atlas_full** | Scale to 20 SAEs x 3 domains, compute domain-specific Pareto fronts. | ~45-60 min |
| **e_adsi_backup** | Compute Asymmetric Decoder Suppression Index on the same SAEs; correlate with domain absorption rates. | ~10 min |

**Falsification criteria:**
- If domain-specific absorption rates have coefficient of variation < 0.2 (i.e., all domains are within 20% of the mean), the context-dependence hypothesis is falsified.
- If the simplified proxy disagrees with causal ablation on >30% of samples, the measurement method is invalid and the pilot fails.
- If one architecture family dominates across all domains (lower absorption everywhere), the "hidden tradeoff" claim is weakened.

### Resource Estimate
- **Models**: GPT-2 Small (117M) or Pythia-160M. Gemma-2-2B is gated and inaccessible per pilot evidence.
- **SAEs**: Existing pretrained checkpoints from SAELens / SAEBench (training-free).
- **Compute**: Single RTX 4090 or equivalent. Each experiment batch is <=1 hour. Total pilot + validation + full run ~ 2 hours, easily parallelizable.
- **LLM calls**: ~500-1,000 calls for automated feature labeling and domain classification (negligible cost).

### Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Domain-specific absorption differences are too small to be meaningful | Medium | Pre-register the CV threshold (0.2). If the effect is weak, pivot to analyzing *why* absorption is domain-invariant—a valuable negative result. |
| Simplified proxy fails validation against causal ablation | Medium | If failure rate >30%, integrate the full sae-spelling metric per domain (more expensive but still training-free). |
| Domain labels are noisy or confounded | Low-Medium | Use simple, unambiguous domains (biology, geography, colors) where hierarchy is culturally stable. |
| Pilot reveals no novel architectural insights | Medium | The contribution is descriptive (the atlas itself) as much as prescriptive. Even without a new architecture, a validated domain-resolved absorption benchmark is a new artifact for the community. |

### Novelty Claim
**No prior work has systematically measured feature absorption as a function of input semantic domain.** The canonical absorption metric (Chanin et al., 2024) and its SAEBench adaptation (Karvonen et al., 2025) both report a single scalar per SAE. We introduce the first domain-resolved absorption analysis, showing that absorption is an emergent property of the SAE-input interaction rather than a fixed SAE pathology. This reframes absorption research from "which SAE is best?" to "which SAE is best for which inputs?" and provides a practical foundation for task-adaptive SAE selection.

---

## Sources

- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507. [https://arxiv.org/abs/2409.14507](https://arxiv.org/abs/2409.14507)
- Chanin, D., et al. (2025). *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*. arXiv:2505.11756. [https://arxiv.org/abs/2505.11756](https://arxiv.org/abs/2505.11756)
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033. [https://arxiv.org/abs/2509.22033](https://arxiv.org/abs/2509.22033)
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547. [https://arxiv.org/abs/2503.17547](https://arxiv.org/abs/2503.17547)
- Michaud, E. J., et al. (2025). *Understanding sparse autoencoder scaling in the presence of feature manifolds*. arXiv:2509.02565. [https://arxiv.org/abs/2509.02565](https://arxiv.org/abs/2509.02565)
- Yao, Y., et al. (2025). *AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations*. arXiv:2508.17320. [https://arxiv.org/abs/2508.17320](https://arxiv.org/abs/2508.17320)
- Girrbach, L., & Akata, Z. (2025). *Sparse Autoencoders are Topic Models*. arXiv:2511.16309. [https://arxiv.org/abs/2511.16309](https://arxiv.org/abs/2511.16309)
- Zheng, C., et al. (2025). *Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders*. arXiv:2507.23220. [https://arxiv.org/abs/2507.23220](https://arxiv.org/abs/2507.23220)
- Karvonen, A., et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532. [https://arxiv.org/abs/2503.09532](https://arxiv.org/abs/2503.09532)
- Olshausen, B. A., & Field, D. J. (1997). *Sparse coding with an overcomplete basis set: A strategy employed by V1?* Vision Research.
- Hosoya, H., & Hyvarinen, A. (2015). *A mixture of sparse coding models for holistic and parts-based face processing in IT cortex*.
- Karklin, Y., & Lewicki, M. S. (2005). *Hierarchical nonlinear generative sparse coding models*.
