# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507** — Foundational paper that identified and named feature absorption; linked cause to sparsity loss + hierarchical co-occurrence; established the first absorption detection methodology using k-sparse probing + integrated gradients ablation.

2. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Proposed hierarchical nested SAEs achieving ~90% absorption reduction (0.49 -> 0.05 at L0=40); introduced the absorption-hedging trade-off as a central tension.

3. **Chanin et al., 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756** — Identified hedging as the complement to absorption; showed Matryoshka trades absorption for hedging; proposed balanced loss coefficients (beta_m ~ 0.75).

4. **Korznikov et al., 2025. "Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Alternative solution via orthogonality constraints (cosine similarity penalty); ~65% absorption reduction with ~50% less compute than Matryoshka.

5. **Li et al., 2025. "Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders." arXiv:2505.16004 (EACL 2026)** — Revealed SAE concept representations are highly vulnerable to adversarial perturbations; a single token replacement can nearly double concept overlap with unrelated targets; ~60% of selected latents are manipulable at individual level.

6. **Saiyed et al., 2026. "Towards Understanding the Robustness of Sparse Autoencoders." arXiv:2604.18756** — Showed SAE insertion can serve as inference-time defense against jailbreak attacks (up to 5x reduction in attack success); monotonic relationship between L0 sparsity and robustness.

7. **Luo et al., 2026. "From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders." arXiv:2602.11881** — HSAE jointly learns SAEs and parent-child relationships; recovers meaningful semantic hierarchies; reframes feature splitting as evidence of underlying hierarchy.

8. **Farnik et al., 2025 (ICML). "Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations." arXiv:2502.18147** — Proposes JSAEs that sparsify the Jacobian matrix (input-to-output computation), not just activations; finds computational sparsity is learned during training (distinguishes pre-trained from random models).

9. **Kissane et al., 2024. "Interpreting Attention Layer Outputs with Sparse Autoencoders." arXiv:2406.17759** — First major application of SAEs to attention layer outputs; found >=90% of attention heads in GPT-2 Small are polysemantic; introduced Recursive Direct Feature Attribution (RDFA).

10. **He et al., 2025. "Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition." arXiv:2504.20938** — Coined "attention superposition"; proposed Lorsa to disentangle attention into individually comprehensible components; discovered arithmetic-specific heads in Llama-3.1-8B.

11. **Lawson et al., 2025 (ICLR). "Residual Stream Analysis with Multi-Layer SAEs." arXiv:2409.04185** — Introduced MLSAE: single SAE trained on residual stream activations from every layer simultaneously with tied parameters; found latents are often active at a single layer for a given token.

12. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532** — Standardized evaluation with 8+ metrics across interpretability, disentanglement, and applications; 200+ SAEs evaluated.

### Landscape Summary

The SAE field has matured rapidly from a single technique (standard SAE on MLP activations) to a rich ecosystem of architectures, evaluation frameworks, and application domains. Feature absorption, identified in 2024, has become a central concern driving architectural innovation. By 2025-2026, we see three major solution paradigms: (1) hierarchical approaches (Matryoshka, HSAE) that embrace multi-scale structure, (2) constraint-based approaches (OrtSAE) that enforce feature independence, and (3) training-objective modifications (WSAE, GBA) that theoretically ground the learning process.

However, a critical tension has emerged: the absorption-hedging trade-off. Matryoshka SAEs reduce absorption but introduce hedging in inner (narrow) levels. No architecture simultaneously minimizes both. This suggests the problem may be deeper than architecture — perhaps fundamental to the sparse coding objective itself.

Simultaneously, the field is expanding beyond the traditional "residual stream + MLP" focus. Attention SAEs (Kissane et al.), Jacobian SAEs (Farnik et al.), and multi-layer SAEs (Lawson et al.) are opening new analytical dimensions. The adversarial robustness work (Li et al., Saiyed et al.) reveals that SAE features are both vulnerable (can be manipulated) and potentially protective (can disrupt attacks) — a duality that remains unexplored.

Most importantly, the finding that SAEs interpret randomly initialized transformers (arXiv:2501.17727) raises a fundamental question: how much of what SAEs discover reflects learned computation versus architectural/data artifacts? Jacobian SAEs partially address this by showing computational sparsity is learned, but the broader question remains open.

## Phase 2: Initial Candidates

### Candidate A: Adversarial Absorption Probing (AAP)

- **Hypothesis**: Feature absorption creates systematic vulnerabilities to adversarial perturbations. Specifically, absorbed features (where broad parent concepts are suppressed by specific child concepts) can be selectively activated or deactivated with small input perturbations, because the "absorption" represents an unstable equilibrium in the sparse coding landscape.

- **Cross-domain insight**: From adversarial robustness research — models with "shortcut" features (superficial patterns that correlate with labels) are more vulnerable to adversarial examples. Absorption is analogous: the SAE learns a "shortcut" where a specific token pattern suppresses a broader concept. The "Adversarial Examples Are Not Bugs, They Are Superposition" paper (arXiv:2508.17456) suggests superposition itself creates adversarial vulnerability. Absorption is a specific form of superposition failure.

- **Evidence for**: Li et al. (2025) showed 60% of SAE latents are individually manipulable. Saiyed et al. (2026) showed SAE insertion disrupts adversarial optimization. The superposition-adversarial connection (arXiv:2508.17456) provides theoretical grounding. If absorption represents an unstable sparse coding solution, it should be more vulnerable than non-absorbed features.

- **Novelty estimate**: 8/10 — While adversarial robustness of SAEs and absorption have been studied separately, their intersection is unexplored. The idea that absorption creates a "vulnerability signature" is new.

### Candidate B: Computational Absorption in Jacobian SAEs (CAJ)

- **Hypothesis**: Feature absorption in standard SAEs (activation-space) has a computational counterpart in Jacobian SAEs: certain input-output computational paths are "absorbed" into dominant pathways, meaning the Jacobian matrix exhibits block-diagonal structure where off-diagonal blocks (representing cross-feature computation) are systematically suppressed.

- **Cross-domain insight**: From control theory and dynamical systems — the Jacobian matrix of a system encodes how perturbations propagate. In neural networks, if feature A's computation should depend on feature B but the SAE suppresses this dependency (analogous to absorption), the Jacobian will show spurious zeros. This connects to the "information bottleneck" idea (Tishby et al.) where compression discards certain information pathways.

- **Evidence for**: Farnik et al. (2025) showed JSAEs reveal computational sparsity that distinguishes pre-trained from random models. If standard SAEs absorb features (suppress parent activations when children fire), JSAEs should reveal whether the corresponding computational dependencies are also suppressed. This would distinguish "representation absorption" from "computation absorption."

- **Novelty estimate**: 9/10 — Jacobian SAEs are very recent (ICML 2025). No work has connected JSAEs to absorption phenomena. The idea of "computational absorption" as distinct from "representational absorption" is genuinely novel.

### Candidate C: Attention Absorption — Cross-Layer Feature Migration

- **Hypothesis**: Features that are "absorbed" in residual-stream SAEs at one layer are not lost but migrate to attention-head SAEs at the same or adjacent layers. The absorption phenomenon may be a layer-local artifact: what looks like absorption in the residual stream is actually feature delegation to attention mechanisms.

- **Cross-domain insight**: From organizational theory ("division of labor") and neuroscience ("functional specialization across brain regions") — complex systems distribute functions across specialized subsystems. In transformers, attention and MLP layers have different computational roles. If a feature is "absorbed" in the residual stream SAE, it may be because that feature is primarily computed and represented in the attention mechanism, not because the SAE failed.

- **Evidence for**: Kissane et al. (2024) showed >=90% of attention heads are polysemantic, suggesting attention outputs contain rich feature structure. He et al. (2025) discovered arithmetic-specific attention heads via Lorsa. Lawson et al. (2025) found MLSAE latents activate at specific layers. If absorption varies systematically between residual-stream SAEs and attention-output SAEs, this would suggest cross-layer feature migration.

- **Novelty estimate**: 7/10 — Cross-layer SAE comparison is emerging (MLSAE, cross-layer transcoders) but no work has specifically linked absorption patterns across layer types. The "feature migration" framing is new.

## Phase 3: Self-Critique

### Against Candidate A: Adversarial Absorption Probing

- **Prior work attack**: Li et al. (2025) already evaluated adversarial robustness of SAE concept representations broadly. Saiyed et al. (2026) used SAEs as defense. The specific connection to absorption has not been made, but the general adversarial-SAE space is active. A more targeted search: "adversarial feature absorption sparse autoencoder" returns no direct hits — the intersection is indeed unexplored.

- **Methodological attack**: How do we distinguish "absorbed features are more vulnerable" from "all features are somewhat vulnerable and absorbed features happen to be the ones we test"? We need a control: compare vulnerability of absorbed vs. non-absorbed features within the same SAE, holding feature frequency and semantic category constant. This requires careful matching.

- **Theoretical attack**: The superposition-adversarial connection (arXiv:2508.17456) argues adversarial examples arise from superposition generally, not specifically from absorption. Absorption is one manifestation of superposition failure, but adversarial vulnerability may stem from other superposition properties (e.g., overlapping feature directions). We need to show absorption has a *unique* vulnerability signature.

- **Scalability attack**: If absorption is more common in early layers (as Chanin et al. suggest), and adversarial vulnerability is more pronounced in late layers (where classification decisions are made), the effect may be weak or absent in practice.

- **Verdict**: MODERATE — The idea is novel and grounded, but the methodological challenge of isolating absorption-specific vulnerability is significant. The theoretical link to superposition-adversarial papers is promising but needs sharpening.

### Against Candidate B: Computational Absorption in Jacobian SAEs

- **Prior work attack**: Farnik et al. (2025) introduced JSAEs but did not study absorption. No prior work connects JSAEs to absorption. However, the broader idea of "computational vs. representational sparsity" is central to their paper. We would be extending their framework rather than introducing a wholly new concept.

- **Methodological attack**: Computing Jacobians for absorption analysis is computationally expensive. For each absorbed feature pair (parent, child), we need to compute how perturbing the parent affects the child's computation. With thousands of features, this is O(n^2) in feature space. We need an efficient approximation.

- **Theoretical attack**: The analogy between "activation absorption" and "Jacobian absorption" may be superficial. In standard SAEs, absorption means the parent latent fails to fire when the child fires. In JSAEs, "computational absorption" would mean perturbing the parent input doesn't affect the child output. But these are different phenomena — one is about representation, the other about computation. The structural correspondence needs rigorous justification.

- **Scalability attack**: JSAEs themselves are computationally expensive and have only been demonstrated on small models (Pythia 70M-410M). Extending to Gemma-2-2B scale may be infeasible within our time budget.

- **Verdict**: STRONG — The idea is genuinely novel with strong theoretical grounding. The computational cost is a real concern but can be addressed with approximations and smaller models. The distinction between representational and computational absorption could be a significant conceptual contribution.

### Against Candidate C: Attention Absorption — Cross-Layer Feature Migration

- **Prior work attack**: Kissane et al. (2024) applied SAEs to attention outputs but did not study absorption. Lawson et al. (2025) trained multi-layer SAEs but did not compare absorption patterns across layer types. The specific "feature migration" hypothesis is new, but the general cross-layer comparison is becoming active.

- **Methodological attack**: Detecting absorption requires ground-truth parent features (Chanin et al. methodology). If we need to compare absorption across residual-stream and attention-output SAEs, we need the same ground-truth concepts to be detectable in both. But attention-output SAEs may not have the same interpretable features, making comparison difficult.

- **Theoretical attack**: The "feature migration" metaphor may be misleading. In transformers, information flows through both attention and MLP pathways simultaneously. A feature being "absorbed" in the residual stream doesn't mean it "migrated" to attention — it may simply not be well-represented in the residual stream at that layer. The organizational theory analogy may be a stretch.

- **Scalability attack**: Training attention-output SAEs requires careful hook placement and may have different optimal hyperparameters than residual-stream SAEs. Comparing absorption rates across different SAE types with different hyperparameters introduces confounds.

- **Verdict**: MODERATE — The idea is intuitively appealing but the "migration" framing may be a superficial metaphor. The methodological challenges of fair cross-layer comparison are significant. However, if the hypothesis is falsified (i.e., absorption patterns are similar across layer types), that negative result would itself be valuable.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate A (Adversarial Absorption Probing)** dropped because: While novel, the methodological challenge of isolating absorption-specific vulnerability from general superposition vulnerability is severe. The effect, if it exists, may be small and hard to detect. The adversarial robustness space is also becoming crowded (Li et al., Saiyed et al., masked regularization papers).

### Strengthened Ideas

- **Candidate B (Computational Absorption in Jacobian SAEs)** strengthened by:
  1. **Narrowing scope**: Instead of full Jacobian analysis, focus on a specific computational pathway: MLP layers. MLPs are approximately linear in the JSAE basis (Farnik et al.), making Jacobians tractable.
  2. **Leveraging existing infrastructure**: Farnik et al. released code built on SAELens. We can extend their implementation rather than building from scratch.
  3. **Clear falsifiable prediction**: If "computational absorption" exists, then for absorbed feature pairs (parent, child), the Jacobian block connecting parent input to child output should be suppressed compared to non-absorbed pairs. This is precisely measurable.
  4. **Conceptual contribution**: Distinguishing "representational absorption" (standard SAE) from "computational absorption" (JSAE) could resolve the "randomly initialized transformers" puzzle — perhaps standard SAEs find similar features in trained and random models (representation), but JSAEs would show different computational absorption patterns (computation).

- **Candidate C (Attention Absorption)** partially integrated: Rather than a standalone proposal, the cross-layer comparison becomes a validation mechanism for Candidate B. If computational absorption is found in MLP JSAEs, we can check whether attention JSAEs show different patterns, providing context.

### Additional Evidence Found

- **"Sparse Autoencoders Do Not Find Canonical Features" (ICLR 2025)**: This paper challenges whether SAEs identify "true" features. It introduces meta-SAEs to decompose SAE latents into compositional structures. This supports our framing: if SAE features are not canonical, then "absorption" may be an artifact of the SAE's inductive bias rather than a property of the underlying model. JSAEs offer a way to test this — computational sparsity being learned (not present in random models) suggests JSAEs capture something more fundamental.

- **"Attention Superposition" (He et al., 2025)**: The Lorsa decomposition reveals that attention heads contain interpretable sparse structure. This means attention-output SAEs are not just analyzing noise — they reveal genuine computational structure. If absorption is found in attention SAEs, it would confirm absorption is a general sparse coding phenomenon, not specific to residual streams.

### Selected Front-Runner

**Candidate B: Computational Absorption in Jacobian SAEs (CAJ)**

This is the strongest idea because:
1. **Genuine novelty**: No prior work connects JSAEs to absorption. The concept of "computational absorption" is new.
2. **Strong theoretical grounding**: The information bottleneck / Jacobian framework provides rigorous mathematical structure.
3. **Addresses a fundamental puzzle**: The "randomly initialized transformers" finding challenges what SAEs actually capture. CAJ offers a path to distinguish learned computation from architectural artifacts.
4. **Falsifiable with clear predictions**: The Jacobian block suppression hypothesis is directly testable.
5. **Feasible within constraints**: Can leverage existing JSAE code; focus on MLP layers where Jacobians are tractable; use small models (Pythia 410M, GPT-2 Small).

## Phase 5: Final Proposal

### Title
"Beyond Representational Absorption: Discovering Computational Absorption in Sparse Autoencoders via Jacobian Analysis"

### Hypothesis
Feature absorption in standard SAEs (representational absorption) has a computational counterpart in Jacobian SAEs: for absorbed feature pairs, the Jacobian matrix exhibits suppressed off-diagonal blocks, indicating that the computational dependency between parent and child features is attenuated. Furthermore, computational absorption distinguishes pre-trained from randomly initialized models, whereas representational absorption does not.

### Motivation
The finding that SAEs interpret randomly initialized transformers (arXiv:2501.17727) raises a fundamental question: what do SAEs actually capture? Standard SAEs sparsify activations, but activation patterns may reflect architectural constraints and data statistics as much as learned computation. Farnik et al. (2025) showed that JSAEs reveal computational sparsity that is genuinely learned (distinguishing pre-trained from random models). If absorption is a property of learned computation, not just representation, then JSAEs should reveal "computational absorption" that is absent or different in random models. This would (a) deepen our understanding of what absorption fundamentally is, (b) provide a tool to distinguish meaningful from artifactual absorption, and (c) connect the absorption literature to the broader question of what SAEs capture.

### Method

**Step 1: Establish representational absorption baseline**
- Train standard SAEs on MLP activations from GPT-2 Small (layers 0, 4, 8, 12) using SAELens
- Use Chanin et al. methodology (k-sparse probing + integrated gradients) to identify absorbed feature pairs
- Focus on first-letter features (established test case) and top-activated features

**Step 2: Train Jacobian SAEs on same layers**
- Extend Farnik et al. JSAE implementation to target MLP layers
- Ensure matched hyperparameters (dictionary size, sparsity level) with standard SAEs
- Verify JSAE achieves comparable reconstruction quality and latent interpretability

**Step 3: Measure computational absorption**
- For each identified absorbed pair (parent, child) from Step 1, compute the Jacobian block J_parent->child
- Compare with non-absorbed feature pairs (matched by activation frequency and semantic category)
- Test hypothesis: J_parent->child is suppressed for absorbed pairs vs. non-absorbed pairs
- Quantify effect size (Cohen's d) and statistical significance

**Step 4: Random model control**
- Repeat Steps 1-3 on randomly initialized GPT-2 Small (same architecture, untrained weights)
- Compare: (a) representational absorption rates, (b) computational absorption patterns
- Test hypothesis: representational absorption is similar between trained and random, but computational absorption differs

**Step 5: Cross-architecture validation**
- Repeat on Pythia-410M (1-2 layers) to verify generalization
- If time permits, compare MLP JSAEs vs. attention-output JSAEs on same layers

### Experimental Plan

| Experiment | Model | Layers | SAE Type | Metric | Time Budget |
|-----------|-------|--------|----------|--------|-------------|
| E1: Rep. absorption baseline | GPT-2 Small | 0,4,8,12 | Standard TopK | Chanin absorption rate | 20 min |
| E2: JSAE training | GPT-2 Small | 0,4,8,12 | Jacobian SAE | FVU, L0, interpretability | 30 min |
| E3: Comp. absorption measurement | GPT-2 Small | 0,4,8,12 | Jacobian SAE | Jacobian block norms, Cohen's d | 15 min |
| E4: Random model control | GPT-2 Small (random) | 0,4,8,12 | Both | Absorption rate comparison | 25 min |
| E5: Cross-architecture | Pythia-410M | 8,16 | Both | Generalization check | 20 min |

**Total estimated time**: ~1.5 hours (within project constraints)

**Falsification criteria**:
- If computational absorption (suppressed Jacobian blocks) is not significantly different between absorbed and non-absorbed pairs (p > 0.05, Cohen's d < 0.5), H1 is falsified.
- If computational absorption patterns are similar between trained and random models, H2 is falsified.

### Resource Estimate

- **Models**: GPT-2 Small (124M params), Pythia-410M — both loadable on single GPU
- **Training**: JSAE training is ~2-3x slower than standard SAE due to Jacobian computation; expect 5-10 min per layer
- **Compute**: Single GPU (RTX PRO 6000 or equivalent) sufficient
- **Storage**: Minimal — activations computed on-the-fly
- **Software**: SAELens (standard SAEs), extend Farnik et al. JSAE code, TransformerLens for activation extraction

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| JSAE implementation too complex | Medium | High | Start with Farnik et al. released code; if integration fails, pivot to analyzing their pre-computed JSAE outputs |
| Computational absorption effect too small | Medium | High | Pre-register analysis plan; if effect size < 0.5, report as negative result and discuss implications |
| Random model shows same computational absorption | Low | High | This would be a significant finding — SAEs capture architectural, not learned, computation. Paper pivots to discussing implications. |
| Jacobian computation OOM | Low | Medium | Use gradient checkpointing; compute Jacobians in batches; reduce dictionary size if needed |
| JSAE training instability | Medium | Medium | Use same hyperparameters as Farnik et al.; reduce learning rate if needed |

### Novelty Claim

This work makes three novel contributions:

1. **Conceptual**: Introduces "computational absorption" as distinct from "representational absorption," extending the absorption framework from activation space to computation space.

2. **Methodological**: Applies Jacobian SAEs (Farnik et al., 2025) to absorption analysis, providing the first computational (not just representational) characterization of absorption.

3. **Empirical**: Tests whether absorption reflects learned computation or architectural artifacts, directly addressing the "randomly initialized transformers" puzzle. If computational absorption distinguishes pre-trained from random models, it validates that JSAEs capture meaningful learned structure that standard SAEs miss.

No prior paper has: (a) connected JSAEs to absorption, (b) distinguished representational from computational absorption, or (c) used absorption patterns to test what SAEs capture. The closest work is Farnik et al. (JSAEs) and Chanin et al. (absorption), but their intersection is entirely unexplored.
