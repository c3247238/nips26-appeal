# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. [Chanin et al., 2024/2025. A is for Absorption: Studying Feature Splitting and Absorption in SAEs. arXiv:2409.14507] — First systematic study of feature absorption; establishes detection metric; proves absorption is caused by hierarchical feature co-occurrence under sparsity. Limitation: metric only reliable for early layers (0-17), uses only first-letter spelling task.

2. [Cui et al., ICLR 2026. On the Limits of Sparse Autoencoders. arXiv:2506.15963] — First closed-form theoretical analysis proving standard SAEs cannot fully recover ground-truth monosemantic features due to intrinsic representational interference. Establishes that full disentanglement is mathematically impossible under realistic sparsity.

3. [Costa et al., NeurIPS 2025. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit. arXiv:2506.03093] — MP-SAE uses residual-guided greedy selection to extract hierarchical features; promotes conditional orthogonality; reduces absorption vs Vanilla/BatchTopK.

4. [Karvonen et al., ICML 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532] — 8-metric evaluation suite including absorption measurement via probe projection approach that works across all layers (unlike ablation-based metric limited to early layers).

5. [Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033] — Enforces orthogonality via cosine similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features.

6. [Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353] — Critical negative result: 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation. Raises fundamental questions about SAE practical utility.

7. [Gao et al., 2024. Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093] — Proposes k-sparse autoencoders; scales to 16M latents on GPT-4; establishes scaling laws. Does not address absorption.

8. [Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka SAEs. arXiv:2503.17547] — Nested dictionaries of increasing size to organize features hierarchically; reduces feature absorption; superior on sparse probing and concept erasure.

### Landscape Summary

Feature absorption in SAEs is now established as a central open problem in mechanistic interpretability. The phenomenon — where hierarchical features cause general features to be subsumed by more specific ones during sparse optimization — creates an "interpretability illusion" where latents appear monosemantic but have systematic false negatives. Key developments in 2025-2026 include: (1) theoretical limits established by Cui et al., proving full disentanglement is mathematically impossible under realistic sparsity; (2) MP-SAE demonstrating that greedy residual-guided selection can recover hierarchical structure; (3) OrtSAE showing 65% absorption reduction via orthogonality enforcement; (4) a critical negative result (Basu et al.) questioning whether any absorption quantification matters if SAE steering produces zero output change.

Three major gaps remain underexplored: (1) systematic cross-model/layer quantification of absorption rates; (2) training-free detection in existing pretrained SAEs without retraining; (3) quantitative impact of absorption on downstream interpretability tasks (circuit discovery, steering, concept erasure).

---

## Phase 2: Initial Candidates

### Candidate A: Absorption-Aware Circuit Discovery

- **Hypothesis**: Feature absorption systematically biases circuit discovery toward child-feature circuits, causing parent-feature circuits to be rediscovered as "new" circuits at deeper layers.
- **Cross-domain insight**: From epidemiology: confounding bias where an exposure's effect is mediated through a collider, causing spurious circuit attribution.
- **Why it might work**: If absorption causes parent features to be represented by child latents, then circuit analysis that ablates child circuits will incorrectly attribute parent-feature effects to the ablated circuit.
- **Novelty estimate**: 7/10 — Circuit discovery (Marks et al., 2024) does not account for absorption; no prior work maps absorption bias to circuit misattribution.

### Candidate B: Causal Mediation Framework for Absorption Detection

- **Hypothesis**: Absorption can be reframed as a causal mediation problem: the parent's effect on the output is mediated through the absorbing child latent. Standard causal mediation analysis can quantify the proportion of effect mediated.
- **Cross-domain insight**: From causal inference: the mediation formula (Pearl, 2001) allows decomposition of total effect into direct and indirect components even when ground-truth mechanisms are unknown.
- **Why it might work**: The projection-based absorption metric in SAEBench implicitly measures mediation. Formalizing it as causal mediation provides a principled, theoretically grounded quantification framework.
- **Novelty estimate**: 6/10 — SAEBench already has a probe projection metric; formalizing it as causal mediation is novel but the connection may have been implicitly recognized.

### Candidate C: Cross-Model Absorption Fingerprinting

- **Hypothesis**: Absorption patterns (which feature hierarchies are absorbed, at which layers) constitute a "fingerprint" unique to each model family, revealing architecture-specific representational compression strategies.
- **Cross-domain insight**: From bioinformatics: conserved pathways across species reveal functional constraints; divergent pathways reveal adaptation.
- **Why it might work**: If absorption arises from hierarchical co-occurrence patterns in training data, different model architectures should show different absorption fingerprints, reflecting their different compression strategies.
- **Novelty estimate**: 8/10 — No prior work has studied cross-model absorption fingerprints; cross-model SAE analysis exists (GemmaScope vs LlamaScope) but not focused on absorption patterns as architectural fingerprints.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: Marks et al. (2024) on circuit discovery does not account for absorption. However, the absorption literature (Chanin et al.) already acknowledges that ablation-based circuit analysis may be affected. Searching specifically for "circuit discovery SAE absorption" — no prior paper explicitly studies absorption bias in circuits.
- **Methodological attack**: How do we isolate absorption bias from confounding by model depth? If parent features are simply weaker at earlier layers (independently of absorption), circuit discovery would naturally find them later. Need a control condition with non-hierarchical features.
- **Theoretical attack**: The causal model assumes absorption is the sole source of circuit misattribution. But multiple failure modes (feature splitting, representation holes) could also bias circuit discovery.
- **Scalability attack**: Full circuit reconstruction is expensive. Even for a single model-layer pair, enumerating circuits for all absorbed pairs is computationally prohibitive.
- **Verdict**: MODERATE — The core insight is valid (absorption biases circuit analysis) but needs careful experimental design to isolate from confounds. The approach is feasible with existing circuit analysis tools.

### Against Candidate B

- **Prior work attack**: The SAEBench probe projection metric (Karvonen et al., 2025) already implements a form of mediation quantification. Formalizing it as causal mediation may be adding theoretical machinery without new empirical predictions.
- **Methodological attack**: Causal mediation analysis requires strong assumptions (no unmeasured confounding, correct mediation structure). These assumptions are likely violated in LLM activation spaces where thousands of features interact simultaneously.
- **Theoretical attack**: The "mediator" in standard causal mediation is a well-defined variable. In SAE absorption, the "mediator" is a latent in a learned autoencoder — the causal status is unclear.
- **Scalability attack**: Full causal mediation analysis requires interventions (do-interventions) which are computationally expensive for LLMs. The projection-based metric is a practical alternative but loses the theoretical guarantees.
- **Verdict**: WEAK — The causal mediation framing is theoretically elegant but the assumptions are violated in practice. SAEBench already covers this space more directly.

### Against Candidate C

- **Prior work attack**: Cross-model SAE comparison exists (GemmaScope papers, LlamaScope release), but none focus on absorption patterns as model fingerprints. Searching for "SAE cross-model absorption fingerprint" — no prior work found.
- **Methodological attack**: What makes an absorption pattern "unique"? Need a principled metric for fingerprint distance. Is it absorption rate, absorption graph topology, specific absorbed hierarchies, or all of the above?
- **Theoretical attack**: The analogy to conserved pathways assumes absorption reflects training data statistics. But if absorption is an inevitable consequence of the SAE objective (as Cui et al. suggest), it may be similar across models rather than unique.
- **Scalability attack**: Systematic cross-model comparison requires SAEs for multiple model families (GPT-2, Pythia, Gemma, Llama) at comparable configurations. Availability is uneven (GemmaScope has JumpReLU, LlamaScope has TopK).
- **Verdict**: MODERATE — The insight is novel but requires careful definition of "fingerprint" metrics. The approach is feasible with available pretrained SAEs if we focus on GemmaScope and GPT-2 SAEs where both are well-represented.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Causal Mediation Framework)** dropped because: SAEBench already implements a probe projection metric that effectively measures absorption without requiring explicit causal mediation assumptions. The theoretical machinery adds complexity without new empirical predictions. The approach would be "theoretically elegant but empirically redundant."

### Strengthened Ideas

- **Candidate A (Absorption-Aware Circuit Discovery)**: Strengthened by noting that the E5 downstream impact results show absorbed features have significantly lower coefficient of variation (CV: 1.07 vs 1.46, p=0.005). This suggests absorbed features are more "stable" — if circuit analysis relies on ablating features and measuring output change, stable absorbed features may systematically escape detection, being discovered only through indirect child-feature effects.

- **Candidate C (Cross-Model Absorption Fingerprinting)**: Strengthened by the E2 cross-layer results showing layer 6 is a consistent "absorption hotspot" across layers (highest mean edge weight 0.559, most fragmented graph with 9 components). This suggests absorption has a consistent layer-specific structure that could be compared across models. The refinement focuses on graph-theoretic fingerprinting (absorption graph topology) rather than aggregate rates.

### Additional Evidence Found

From the E4 causal factor analysis: co-occurrence correlates negatively with absorption score (r = -0.52), contradicting the theoretical model. This suggests the current scoring formula conflates two distinct phenomena: (1) co-occurrence-driven absorption (parent encoded via child), and (2) co-occurrence-driven activation correlation (parent and child both active due to shared input). This distinction is critical for circuit analysis — case (1) causes circuit misattribution, case (2) does not.

### Selected Front-Runner

**Candidate A: Absorption-Aware Circuit Discovery** is selected as the front-runner because:

1. It addresses Gap 3 (impact on downstream interpretability tasks) identified in the literature survey — this gap is acknowledged but unquantified in prior work.
2. The E5 CV difference provides direct empirical evidence that absorbed features have systematically different activation properties, which would affect circuit analysis reliability.
3. The E4 finding (negative co-occurrence correlation) provides a critical refinement: the scoring formula should distinguish absorption (parent encoded via child) from mere activation correlation (shared input). This distinction is operationally important for circuit analysis.
4. It produces actionable outputs: a method for bias-corrected circuit discovery that could be adopted by the circuit discovery community.
5. Resource efficient: uses existing circuit analysis tools ( TransformerLens hooks), requires no new SAE training, experiments can be completed within the 1-hour budget.

---

## Phase 5: Final Proposal

### Title

**Absorption-Adjusted Circuit Discovery: Quantifying and Correcting Feature Absorption Bias in Mechanistic Interpretability**

### Hypothesis

When circuit discovery methods ablate a feature that has been absorbed by a child feature in the SAE, the resulting output change is spuriously attributed to the child feature's circuit rather than the parent feature's circuit. This causes parent-feature circuits to be systematically rediscovered at deeper layers as "new" circuits. We can detect and correct this bias by (1) identifying absorption pairs using the revised scoring formula (decoder_cosine × freq_ratio × (1 - cooccurrence)), and (2) running circuit analysis on both the absorbed and absorbing features to map the absorption graph.

### Motivation

Circuit discovery (Marks et al., 2024) is a cornerstone of mechanistic interpretability, identifying which components (attention heads, MLP neurons, features) are causally responsible for model behavior. However, all circuit discovery methods assume that features are independently represented — an assumption violated by feature absorption. If absorption systematically biases circuit discovery, then existing circuit analyses may be structurally incorrect, with profound implications for the reliability of interpretability research.

The problem is quantified in the literature: Chanin et al. (2024) showed absorption is widespread; Basu et al. (2026) showed that even near-perfect feature detection (98.2% AUROC) produces zero output change via steering, raising questions about the actionability of interpretability. Yet no prior work has studied how absorption specifically biases circuit discovery. This is a critical gap because circuit analysis is used to identify safety-relevant circuits (e.g., sycophancy, deception).

### Method

1. **Absorption Pair Detection**: For each layer, use the revised scoring formula on candidate pairs (high decoder cosine similarity, frequency ratio > 5, cooccurrence > 0.5) to identify absorption candidates. The revised formula: `score = decoder_cosine × log(freq_ratio) × (1 - cooccurrence_score)` replaces the degenerate ablation component with geometric signal only.

2. **Absorption Graph Construction**: Build a directed graph where nodes are features and edges indicate absorption (parent → child). Assign edge weights as the absorption score. Identify "absorption chains" where parent → child → grandchild patterns exist.

3. **Circuit Analysis with Absorption Adjustment**:
   - For each absorption pair (parent P, child C), run standard circuit discovery (ablating P, measuring output change) and record the affected components.
   - Run circuit discovery on C and record its circuit.
   - Compare: if the P-circuit is a subgraph of the C-circuit and P is absorbed (score > threshold), flag as absorption-biased circuit.
   - Correct: attribute the overlapping components to P rather than C.

4. **Bias Quantification**: For each circuit analysis result, compute the "absorption bias ratio" = (number of absorption-biased components) / (total components). Compare bias ratios across layers and models.

### Experimental Plan

| Experiment | Target | Metric | Time |
|------------|--------|--------|------|
| E1: Absorption pair detection (revised formula) | GPT-2 layer 6, top 100 pairs | absorption score distribution | 15 min |
| E2: Circuit analysis comparison | Absorbed vs non-absorbed pairs (n=20 each) | bias ratio, circuit overlap | 30 min |
| E3: Cross-layer bias quantification | Layers 0, 3, 6, 9, 11 | absorption bias ratio per layer | 20 min |
| E4: Cross-model validation | GPT-2 vs Gemma-2-2B | fingerprint similarity | 25 min |
| E5: Ground-truth validation (SynthSAEBench) | Synthetic features with known hierarchy | precision, recall of absorption detection | 20 min |

**Baselines**: Standard circuit discovery without absorption adjustment (Marks et al., 2024 procedure)

**Success criterion**: Absorption-adjusted circuit analysis identifies at least 30% more parent-feature circuits than baseline, with the identified circuits showing different intervention profiles (measured by output logit change).

### Resource Estimate

- **Models**: GPT-2-small (85M parameters, rapid prototyping), Gemma-2-2B (for cross-model validation)
- **SAEs**: GPT-2 residual stream SAEs via SAELens (all layers); GemmaScope JumpReLU SAEs (layers 0-17)
- **Compute**: All experiments fit within 1-hour budget per task; total ~110 min across 5 experiments
- **Code**: SAELens for SAE loading, TransformerLens for circuit analysis hooks, custom absorption graph construction

### Risk Assessment

1. **Risk: Absorption scoring formula still discriminates poorly**. Even with the revised formula, absorption scores may not exceed the 0.7 threshold. *Mitigation*: Use decoder_cosine as primary signal (continuous 0-1), report continuous distributions rather than binary detection.

2. **Risk: Circuit analysis is too expensive for large-scale validation**. Full circuit reconstruction per absorption pair is computationally intensive. *Mitigation*: Use an established circuit analysis dataset (e.g., indirect object identification from prior work) rather than running full circuit discovery from scratch.

3. **Risk: Basu et al. (2026) negative result makes circuit analysis irrelevant**. If absorbed features produce no output change via steering, circuit analysis may be moot. *Mitigation*: This risk is acknowledged but does not eliminate the need for the analysis — understanding WHY absorbed features resist intervention is itself a publishable finding.

### Novelty Claim

This is the **first work to systematically quantify how feature absorption biases circuit discovery in mechanistic interpretability**. Prior work has: (1) measured absorption rates (Chanin et al., 2024), (2) proposed architectural remedies (Matryoshka SAE, OrtSAE), (3) evaluated absorption in SAEBench (Karvonen et al., 2025), and (4) questioned actionability (Basu et al., 2026). None has studied the **downstream impact on circuit discovery as an interpretability methodology**. The absorption graph construction and bias-adjusted circuit analysis are novel contributions that produce actionable outputs for the interpretability community.

The research directly addresses **Gap 3** (impact on downstream interpretability tasks) identified in the literature survey, which the survey notes is "largely unquantified." The proposed experimental plan produces quantitative evidence connecting absorption to circuit misattribution, filling a critical gap in the field.
