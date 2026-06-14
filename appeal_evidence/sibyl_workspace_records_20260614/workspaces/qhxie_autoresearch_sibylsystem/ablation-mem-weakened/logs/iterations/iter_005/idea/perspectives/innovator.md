# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

Based on the comprehensive literature survey provided in `context/literature.md` and `context/idea_context.md`, the following key papers are most relevant:

1. **Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507** — First systematic study of feature absorption; establishes the foundational detection metric and proves absorption is caused by hierarchical feature co-occurrence under sparsity optimization.

2. **Chanin et al., 2025a. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756** — Identifies feature hedging as the complement failure mode in narrow SAEs; reveals Matryoshka SAEs trade absorption for hedging; proposes balanced loss coefficients (~0.75).

3. **Chanin & Garriga-Alonso, 2025b. Sparse but Wrong: Incorrect L0 Leads to Incorrect Features. arXiv:2508.16560** — Proves L0 is not a neutral hyperparameter; introduces decoder pairwise cosine similarity (c_dec) metric for identifying optimal L0.

4. **Karvonen et al., 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532 (ICML 2025)** — Standardized 8-metric evaluation suite including absorption; projection-based absorption metric works across all layers.

5. **Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547** — Nested dictionaries reducing absorption but trading for hedging at inner levels.

6. **Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033** — Enforces orthogonality via cosine similarity penalty; reduces absorption by 65%.

7. **Costa et al., 2025. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit. arXiv:2506.03093 (NeurIPS 2025)** — MP-SAE uses residual-guided greedy selection; conditional orthogonality reduces absorption.

8. **Cui et al., 2026. On the Limits of Sparse Autoencoders. arXiv:2506.15963 (ICLR 2026)** — First closed-form theoretical analysis; proves standard SAEs generally fail to recover ground-truth features.

9. **Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353** — Critical negative result: 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation.

10. **Peng et al., 2025. CE-Bench: Contrastive Evaluation Benchmark. arXiv:2509.00691** — LLM-free contrastive evaluation; 5,000 story pairs; 77% alignment with SAEBench.

### Landscape Summary

The SAE field has rapidly matured from showing that SAEs extract interpretable features (Cunningham et al., 2023) to critically examining their failure modes. Feature absorption — where general features are subsumed by specific ones during sparse optimization — was first systematically studied by Chanin et al. (2024), who showed this creates an "interpretability illusion" where latents appear monosemantic but have systematic false negatives.

The 2025-2026 period has seen an explosion of critical work. Three complementary failure modes are now identified: absorption (wide SAEs, hierarchical features), hedging (narrow SAEs, correlated features), and L0 sensitivity (Chanin & Garriga-Alonso, 2025b). Architectures like Matryoshka SAE, OrtSAE, ATM, and MP-SAE have proposed partial solutions, but all involve trade-offs or require retraining.

A critical tension has emerged: Basu et al. (2026) demonstrate that even near-perfect feature detection (98.2% AUROC) translates to zero output change via steering due to residual stream compensation. This raises the fundamental question: does quantifying absorption matter if we cannot act on that knowledge? The field risks producing increasingly sophisticated diagnostics for a phenomenon that may be practically intractable to address.

Key gaps remain: systematic cross-model absorption quantification (Gap 1), training-free detection methods that work on existing pretrained SAEs without retraining (Gap 4), and — most critically — bridging absorption research to practical intervention strategies (Gap 9).

## Phase 2: Initial Candidates

### Candidate A: Cross-Architecture Absorption Scaling Laws

- **Hypothesis**: Absorption rates follow predictable scaling laws across model families, layer depths, and SAE configurations, enabling principled configuration selection to minimize absorption.
- **Cross-domain insight**: Drawing from neural scaling laws (Kaplan et al., 2020), where loss decreases predictably with model size and data, I hypothesize absorption rates similarly follow systematic patterns that can be modeled and predicted.
- **Evidence for**: Chanin et al. (2024) only studied layers 0-17 of Gemma-2-2B; the systematic variation across architectures, widths (16k/65k/131k in GemmaScope), and layer depths (early vs. middle vs. late) remains uncharacterized. SAEBench includes absorption as one metric but does not systematically analyze scaling patterns.
- **Novelty estimate**: 5/10 — This is a natural extension of Chanin et al. and SAEBench work, but the comprehensive cross-architecture study has not been done. The scaling law framing is novel.

### Candidate B: Training-Free Absorption Detection via Encoder-Decoder Asymmetry

- **Hypothesis**: Absorption can be detected in pretrained SAEs without ground-truth probes by measuring encoder-decoder weight asymmetry — absorbing latents have encoder representations that more closely match their decoder representations than non-absorbing latents.
- **Cross-domain insight**: From compressed sensing theory (Donoho & Tanner, 2009), the geometry of sparse representations encodes redundancy patterns. In absorption, the hierarchical co-occurrence structure should leave geometric signatures in the SAE's learned representations.
- **Evidence for**: Chanin et al. (2024) toy models suggest encoder-decoder asymmetry is a telltale sign. SAEBench's projection-based absorption metric (Appendix A.13) is a step in this direction but still requires probe construction. No fully training-free method exists.
- **Novelty estimate**: 7/10 — No prior work explicitly proposes encoder-decoder asymmetry as an absorption detection signal for pretrained SAEs. The connection to compressed sensing geometry is novel.

### Candidate C: Absorption-Aware Steering: From Quantification to Intervention

- **Hypothesis**: Standard SAE steering fails not because features are misidentified but because steering a single latent ignores its absorption relationships — an "absorption-aware" steering strategy that accounts for the hierarchical structure of absorbed features will produce measurable downstream behavioral changes.
- **Cross-domain insight**: From causal mediation analysis (Imai et al., 2010), when a treatment (steering) affects an outcome through a mediator (absorbed feature), proper intervention requires accounting for the mediator's full causal pathway. Absorption is fundamentally a mediation phenomenon.
- **Evidence for**: Basu et al. (2026) show steering fails (45.1% output sensitivity despite 98.2% probe AUROC) but do not propose a solution. No prior work connects absorption structure to steering methodology. This directly addresses Gap 9.
- **Novelty estimate**: 8/10 — This is the first work to connect absorption quantification to intervention strategy. The causal mediation framing is novel in the SAE literature.

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: SAEBench (Karvonen et al., 2025) already measures absorption across 200+ SAEs. While not as systematic as a dedicated study, the data exists. Gurnee et al.'s k-sparse probing implicitly detects absorption-like effects across layers. This direction is more of an incremental extension than a novel contribution.
- **Methodological attack**: Computing absorption across many SAEs is expensive (~26 min per SAE with ablation-based metric). The projection-based alternative may be faster but has limitations (probe quality dependence). Cross-architecture comparison adds complexity due to different SAE architectures (JumpReLU for Gemma, TopK for Llama).
- **Theoretical attack**: Absorption may not follow clean scaling laws. The relationship between absorption and configuration (width, depth, sparsity) may be non-monotonic and architecture-dependent, making "scaling laws" an overly strong claim.
- **Scalability attack**: A truly comprehensive study would require computing absorption for hundreds of SAEs across multiple models. This exceeds the 1-hour per experiment guideline unless using the faster projection-based method.
- **Verdict**: MODERATE — Valuable systematic measurement but incremental over existing work. Would need a strong theoretical grounding for the "scaling law" framing to differentiate from a dataset paper.

### Against Candidate B

- **Prior work attack**: SAEBench's projection-based metric (Appendix A.13) already does something similar — measuring the proportion of feature representation accounted for by absorbing latents rather than main latents. The encoder-decoder asymmetry framing may be too similar to existing work.
- **Methodological attack**: The encoder-decoder asymmetry is a post-hoc geometric observation, not a causal mechanism. It may correlate with absorption but not reliably detect it across different SAE architectures and training configurations.
- **Theoretical attack**: The compressed sensing analogy may be superficial. SAEs are trained with gradient descent on specific objectives, not random measurement matrices. The geometric structure of absorption may not match the theoretical predictions from random sensing.
- **Scalability attack**: Computing encoder-decoder asymmetry for all latents in a large SAE (e.g., 131k latents in GemmaScope) is computationally intensive. Would need to identify a subset of features most likely to exhibit absorption for efficient detection.
- **Verdict**: WEAK — The prior work attack is serious. SAEBench already implements a projection-based absorption metric. Without a clear theoretical justification for why encoder-decoder asymmetry specifically indicates absorption, this risks being incremental over existing methods.

### Against Candidate C

- **Prior work attack**: Basu et al. (2026) identify the problem (steering fails despite good detection) but do not propose absorption-aware solutions. No prior work connects absorption structure to steering methodology. However, the causal mediation analysis framing is well-established in statistics, so the novelty is in the application to SAEs, not the framework itself.
- **Methodological attack**: This requires defining what "absorption-aware" steering means operationally. Possible approaches: (1) steering parent and child features together, (2) steering the residual after accounting for absorption relationships, (3) identifying "pure" unabsorbed features for steering. All require non-trivial implementation.
- **Theoretical attack**: Basu et al. argue the failure is due to residual stream compensation — the model route around interventions. Absorption-aware steering may not overcome this fundamental limitation. The proposed method could still fail even with perfect absorption modeling.
- **Scalability attack**: Defining the absorption hierarchy requires understanding which features absorb which — this is exactly what the absorption metric measures. Without an efficient way to compute absorption structure for arbitrary pretrained SAEs, this approach is not scalable.
- **Verdict**: STRONG — This directly addresses the most critical gap in the field (Gap 9: actionability). Even if absorption-aware steering partially succeeds, it would be a meaningful contribution. The main risk is whether residual stream compensation can be overcome.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate A (Cross-Architecture Absorption Scaling Laws)** dropped because: This is fundamentally a measurement study rather than a solution to a problem. While systematic measurement is valuable, it does not directly address any of the identified gaps in the literature. The "scaling law" framing requires strong theoretical justification that may not be achievable.

- **Candidate B (Training-Free Absorption Detection via Encoder-Decoder Asymmetry)** dropped because: SAEBench already implements a projection-based absorption metric that does not require ground-truth probes. The encoder-decoder asymmetry framing is likely too similar to existing work to be a novel contribution.

### Strengthened Ideas

- **Candidate C (Absorption-Aware Steering)**: Strengthened by specifying a concrete operationalization:
  1. Use the SAEBench projection-based absorption metric to identify absorbing/absorbed latent pairs
  2. For steering experiments, compare standard single-latent steering vs. multi-latent steering that includes both parent (absorbed) and child (absorbing) features
  3. Measure both output sensitivity (following Basu et al.) and downstream task performance (e.g., circuit discovery reliability, concept erasure effectiveness)

### Additional Evidence Found

The literature review provides strong evidence for the feasibility of this approach:
- SAEBench's projection-based absorption metric (Karvonen et al., 2025) provides a computationally tractable way to identify absorption relationships across all layers (unlike ablation which is limited to layers 0-17)
- CE-Bench (Peng et al., 2025) provides LLM-free evaluation that could validate absorption-aware steering without relying on the steered model's own outputs
- The Matryoshka SAE work (Bussmann et al., 2025) suggests that hierarchical feature structure can be partially recovered, implying that accounting for hierarchy is possible
- Chanin et al. (2025a) show that absorption and hedging are two sides of the same coin, suggesting that a unified approach accounting for both failure modes may be more effective

### Selected Front-Runner

**Candidate C: Absorption-Aware Steering** is selected as the front-runner because:

1. It directly addresses the most critical gap in the field (Gap 9: actionability), identified by Basu et al. as a fundamental limitation
2. Even partial success would be a meaningful contribution — if absorption-aware steering achieves even modest improvements over standard steering, it validates that absorption research has practical implications
3. The approach is testable: compare standard steering vs. absorption-aware steering on well-established downstream tasks (circuit discovery, concept erasure, model steering)
4. The computational approach is feasible within the project constraints: SAEBench provides the absorption metric, CE-Bench provides LLM-free evaluation, and GPT-2/Gemma-2-2B SAEs are readily available

## Phase 5: Final Proposal

### Title

**Absorption-Aware Steering: Bridging Feature Quantification to Intervention in Sparse Autoencoders**

### Hypothesis

Standard SAE steering fails (as documented by Basu et al., 2026) not because features are misidentified but because steering a single latent ignores its absorption relationships. An "absorption-aware" steering strategy that accounts for the hierarchical structure of absorbed features — by steering both parent (absorbed) and child (absorbing) features together — will produce measurable downstream behavioral changes where standard steering fails.

### Motivation

Basu et al. (2026) demonstrate a troubling disconnect: SAE features can be detected with 98.2% AUROC yet steering produces zero output change (45.1% sensitivity). They conclude "mechanistic interpretability cannot correct LLM errors." However, their analysis does not account for absorption structure. If absorbed features systematically fail to respond to steering due to hierarchical masking, then accounting for this structure may restore steering efficacy.

This matters for the entire SAE research program: if we can demonstrate that understanding absorption enables better interventions, it validates the entire research direction of analyzing and quantifying SAE failure modes. If not, we have a clear negative result that bounds the practical utility of absorption research.

### Method

1. **Absorption Structure Identification**: For each SAE under study, compute the absorption metric (using SAEBench's projection-based method) to identify absorbing/absorbed latent pairs. This produces a directed graph where edge (i → j) indicates latent i is absorbed into latent j.

2. **Standard Steering Baseline**: Replicate Basu et al. steering experiments on a set of well-understood features (e.g., features from Neuronpedia with clear human interpretations). Measure output sensitivity and downstream task performance.

3. **Absorption-Aware Steering**: For features with absorption relationships:
   - **Multi-latent steering**: Instead of steering a single latent, steering both the absorbed parent and the absorbing child together with equal magnitude but opposite sign (to preserve overall activation magnitude)
   - **Residual-aware steering**: Project the steering intervention onto the residual space orthogonal to absorbing latents
   - **Hierarchy-following steering**: Trace the full absorption chain and apply interventions at the deepest level of the hierarchy

4. **Downstream Task Validation**: Test absorption-aware steering on:
   - Circuit discovery reliability (does steering produce more consistent circuit findings?)
   - Concept erasure effectiveness (can we more reliably erase concepts with absorption-aware steering?)
   - Model behavior modification (does steering produce more reliable downstream behavioral changes?)

5. **Comparison with Architecture Variants**: Compare absorption-aware steering effectiveness across SAE architectures (TopK, JumpReLU, Matryoshka) to understand how architectural differences in absorption structure affect intervention efficacy.

### Cross-Domain Insight

The key transplanted principle is **causal mediation analysis** (Imai et al., 2010). In causal mediation, when a treatment T affects outcome Y through a mediator M, the total effect decomposes into direct effect (T → Y) and indirect effect through M (T → M → Y). Steering a single feature is like intervening on T while ignoring the mediator M — it may not produce the intended effect if the causal pathway goes through absorbed features.

Absorption creates a mediation structure: the "true" feature (parent) is mediated through the more specific feature (child). Proper intervention requires understanding this mediation structure. This is not a superficial metaphor — the mathematical structure is identical: absorption means the parent's effect on model behavior is routed through the child latent.

### Experimental Plan

**Pilot (10-15 minutes)**:
- Load GPT-2 residual stream SAE (from SAELens)
- Implement SAEBench projection-based absorption metric on a subset of features (n=100)
- Identify 5-10 features with clear absorption relationships
- Run standard steering vs. absorption-aware steering comparison on these features
- Measure output sensitivity using cosine similarity between steering direction and model logits

**Main Experiments (≤1 hour each)**:
- Experiment 1: Systematic comparison of standard vs. absorption-aware steering on GPT-2 SAE (all features with identified absorption relationships, n≈50-100)
- Experiment 2: Replicate on Gemma-2-2B SAEs (layers 0-17 where ablation-based metric is reliable, plus projection-based for deeper layers)
- Experiment 3: Test on downstream tasks: indirect object identification (IOI circuit), concept erasure tasks
- Experiment 4: Compare across SAE architectures (TopK vs. JumpReLU) to understand architectural effects

**Falsification Criteria**:
- If absorption-aware steering shows no improvement over standard steering on any task, the hypothesis is falsified
- If absorption-aware steering improves output sensitivity but not downstream task performance, the hypothesis is partially confirmed but limits on practical utility remain
- If absorption-aware steering improves both sensitivity and downstream performance, the hypothesis is strongly confirmed

### Resource Estimate

- **Models**: GPT-2 small (80M params) for rapid prototyping; Gemma-2-2B for main experiments
- **SAEs**: GPT-2 residual stream SAEs (SAELens, ~10K latents), GemmaScope 16k width for layers 0-17
- **Compute**: Each steering experiment ~15-30 minutes on single GPU
- **Total**: ~4-6 hours for full experimental suite (within project constraints with parallelization)

### Risk Assessment

1. **Residual Stream Compensation (High Risk)**: Basu et al. argue that residual stream compensation is the fundamental reason steering fails. Absorption-aware steering may not overcome this even with perfect absorption modeling. Mitigation: If this is the dominant effect, the negative result itself is publishable as a bound on absorption research utility.

2. **Absorption Metric Reliability (Medium Risk)**: The projection-based absorption metric may not accurately identify absorption relationships, especially for deeper layers. Mitigation: Validate against ablation-based metric on layers 0-17 where ablation is reliable.

3. **Downstream Task Sensitivity (Medium Risk)**: Even successful steering may not produce measurable changes on complex downstream tasks if other features compensate. Mitigation: Use well-validated tasks (IOI, concept erasure) with established evaluation protocols.

### Novelty Claim

This is the first work to:
1. Connect absorption quantification to intervention methodology (Gap 9 in the literature)
2. Propose and empirically validate "absorption-aware" steering as an alternative to standard single-latent steering
3. Test whether understanding SAE failure modes enables better practical interventions

The causal mediation framing of absorption is novel in the SAE literature. While mediation analysis is well-established in statistics, its application to SAE steering has not been explored. If successful, this provides a principled framework for thinking about SAE interventions that accounts for their hierarchical structure.

---

*Output generated by Innovator agent based on comprehensive literature survey in `context/literature.md` and `context/idea_context.md`.*