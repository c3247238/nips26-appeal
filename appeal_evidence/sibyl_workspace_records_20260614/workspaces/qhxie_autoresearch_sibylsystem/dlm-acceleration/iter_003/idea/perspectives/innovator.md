# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Stancevic et al., 2025. Entropic Time Schedulers for Generative Diffusion Models. arXiv:2504.13612** -- First principled information-theoretic framework connecting entropy rates to optimal sampling schedules in diffusion models. Directly relevant to deriving denoising schedules from first principles rather than heuristics.

2. **Chen et al., 2025. Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty. arXiv:2512.21336** -- Introduces "Denoising Entropy" as a computable metric for path-level uncertainty in MDMs. Demonstrates strong correlation between Path Entropy and output quality. Key insight: uncertainty quantification can be turned into an optimization signal.

3. **Lu et al., 2025. AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size. arXiv:2509.26432 (ICLR 2026)** -- Identifies the "volatility band" in confidence dynamics during DLM decoding, showing semantic structure can guide adaptive scheduling. Training-free, plug-and-play, up to 5.3% accuracy improvement.

4. **Yang et al., 2026. Improving Sampling for Masked Diffusion Models via Information Gain. arXiv:2602.18176** -- Information-gain-based sampler that optimizes both *what* and *where* to decode in MDMs, explicitly treating the expanded decision space.

5. **Learn2PD, 2025. Learning to Parallel: Accelerating Diffusion LLMs. arXiv:2509.25188** -- Demonstrates that learnable parallel decoding + KV caching compounds to 57.51x speedup, confirming strong orthogonality between these two acceleration families.

6. **dLLM-Serve, 2026. Taming the Memory Footprint Crisis for Production Diffusion LLM Serving. arXiv:2512.17077** -- Identifies the Refresh/Reuse phase oscillation in DLM inference that creates alternating compute-bound and memory-bound regimes. First production serving system for DLMs.

7. **I-DLM, 2026. Introspective Diffusion Language Models. arXiv:2604.11035** -- First DLM to match AR quality; introduces introspective strided decoding that verifies tokens while advancing new ones. Represents the hybrid AR-diffusion architecture direction.

8. **ES-dLLM, 2026. Efficient Inference for Diffusion LLMs by Early-Skipping. arXiv:2603.10088** -- Training-free layer-skipping via tensor variation + confidence scores. Achieves 5.6x-16.8x speedup by exploiting the insight that early layers are redundant for already-decided tokens.

9. **JoT, 2026. Just on Time: Token-Level Early Stopping for Diffusion Language Models. arXiv:2602.11133** -- Per-token early stopping. Claims orthogonality to KV caching but does not empirically validate composition. 7.2x speedup at 98.3% quality retention.

10. **Salvatori et al., 2026. Brain-inspired Computational Intelligence via Predictive Coding. Neural Networks** -- Comprehensive survey connecting predictive coding (PC) to efficient neural computation. Key insight: attention = precision weighting of prediction errors. PC frameworks achieve efficient inference by only propagating residuals (prediction errors).

11. **Two-Stage LDPC-CRC Decoding, 2025. An Efficient Two-Stage Decoding Scheme. PMC:12468592** -- Iterative belief propagation with early termination when codeword is valid; demonstrates that >90% of the benefit comes from first 20 iterations out of 100, with diminishing returns. Direct structural analog to DLM denoising.

12. **Dominant Balance AMR, 2024. arXiv:2411.02677** -- Adaptive mesh refinement using Gaussian mixture models to classify grid cells into active/passive regions, reducing computational costs by up to 70% without heuristic sensors. The "active region" concept maps directly to "uncertain tokens" in DLMs.

### Landscape Summary

The DLM acceleration field has reached a critical inflection point as of April 2026. Individual acceleration techniques are well-established across at least six families: KV caching (8+ variants), parallel/confidence-aware decoding, speculative methods, step reduction/early stopping, sparse attention, and layer skipping. The frontier questions have shifted from "can individual methods work?" to "how should methods be composed?" and "what is the theoretical limit of acceleration?"

Two developments fundamentally change the landscape for iteration 3. First, the emergence of **information-theoretic tools** for MDM inference (Denoising Entropy, entropic time schedulers, information-gain sampling) provides a principled framework for understanding *why* certain acceleration decisions work and *when* they fail -- this was entirely absent in iterations 1-2. Second, the **binary composability discovery** from ComposeAccel's iter_001/iter_002 experiments (M1+CD-SSD Ortho=1.385 synergy vs. all other pairs showing destructive interference) creates a concrete mechanistic puzzle: why does exactly one pair synergize?

The gap that remains most open is **not** another individual method (the space is crowded), but rather a **principled, information-theoretic framework that predicts composability from method-level properties** -- converting the composability atlas from an empirical catalog into a predictive theory. This is where cross-domain insights (from coding theory, predictive coding, adaptive mesh refinement) become powerful.

---

## Phase 2: Initial Candidates

### Candidate A: Denoising Entropy Budget Allocation (DEBA) -- An Information-Theoretic Composability Predictor

- **Hypothesis**: The composability of two MDM acceleration methods can be predicted from their joint effect on Denoising Entropy (as defined by Chen et al., 2025): two methods are orthogonal if and only if they reduce entropy in non-overlapping regions of the (token position, denoising step) plane, and they destructively interfere when they attempt to reduce entropy in the same region, creating entropy "vacuums" that destabilize the denoising trajectory.
- **Cross-domain insight**: From LDPC decoding, belief propagation converges when messages along different edges of the factor graph reduce uncertainty in complementary variable nodes. When two updates target the same node, they create overconfident beliefs that cause oscillation. This is structurally identical to two acceleration methods that both try to "decide" the same uncertain tokens.
- **Evidence for**: The M1+CD-SSD synergy (Ortho=1.385) can be explained: KV caching (M1) reduces per-step computational cost (entropy reduction per FLOP), while CD-SSD's frozen-token mechanism reduces the *number* of tokens that need entropy reduction. These target different axes (computational cost vs. target set). M1+M3 interference (Ortho=0.301) can be explained: both M1 (cache reuse) and M3 (AR guidance) modify the attention distribution over the same token set, creating conflicting entropy gradients.
- **Novelty estimate**: 8/10 -- No existing work connects Denoising Entropy to composability prediction. Chen et al. use Denoising Entropy for quality optimization (E-BoN, E-SMC), not for predicting method interactions. The LDPC analogy is entirely new.

### Candidate B: Adaptive Denoising Mesh (ADM) -- Hierarchical Token-Step Resource Allocation

- **Hypothesis**: DLM denoising can be formulated as a 2D resource allocation problem over the (token position, denoising step) grid, where each cell requires a "resolution" of computation proportional to its local uncertainty gradient. A hierarchical refinement strategy -- analogous to adaptive mesh refinement in CFD -- can allocate computation dynamically: coarse resolution (cached KV, skipped layers) for low-uncertainty cells, fine resolution (full attention, full model) for high-uncertainty cells, with the mesh adapting at each step based on uncertainty feedback.
- **Cross-domain insight**: In CFD, adaptive mesh refinement reduces computation by 70% by concentrating grid points where gradients are steep (shocks, boundary layers) while using coarse grids in smooth regions. The same principle applies to DLM denoising: most token-step cells are "smooth" (token already decided or far from decision boundary), while a small fraction are "sharp" (token actively being resolved). The key structural correspondence is that both problems have spatially varying resolution requirements that change dynamically over time.
- **Evidence for**: ES-dLLM's finding that early layers are redundant for stable tokens (up to 16.8x speedup) proves the "coarse mesh" idea works in the layer dimension. JoT's per-token early stopping proves it works in the step dimension. AdaBlock-dLLM's volatility band proves it works in the block-size dimension. But no one has unified these into a single hierarchical refinement framework that simultaneously adapts across token position, denoising step, layer depth, AND attention scope.
- **Novelty estimate**: 7/10 -- Individual dimensions are explored (ES-dLLM for layers, JoT for steps, DyLLM for tokens, AdaBlock for blocks). The unified hierarchical framework with explicit AMR analogy is new, but the "allocate compute where needed" insight is implicit in many existing works.

### Candidate C: Predictive Coding Denoiser (PCD) -- Brain-Inspired Residual-Only DLM Inference

- **Hypothesis**: DLM denoising can be reformulated as hierarchical predictive coding, where each denoising step generates a "prediction" of the final sequence, and only the "prediction error" (tokens that changed from the previous step) is propagated through the full model. Tokens whose predictions are stable across steps carry zero prediction error and can be entirely skipped -- not just cached, but completely removed from the computation graph.
- **Cross-domain insight**: From neuroscience, the brain's predictive coding framework achieves efficient perception by only processing surprise (prediction errors), not the full sensory input. A stable visual scene requires minimal neural computation; only unexpected changes trigger full processing. In DLMs, the iterative denoising process creates an analogous situation: after the first few steps, most tokens are stable (the "expected" prediction), and only a shrinking set of uncertain tokens generate "surprise" (prediction errors) that need full-model computation.
- **Evidence for**: DyLLM's saliency-based approach (up to 9.6x) already identifies "salient" tokens (our "prediction error" tokens) and computes only those. ES-dLLM's tensor variation signal is essentially measuring the magnitude of prediction errors across layers. The key evidence is quantitative: DyLLM reports that only 10-30% of tokens are "salient" at any given step, meaning 70-90% of tokens carry near-zero prediction error and could be fully skipped. If we formalize this as predictive coding and propagate ONLY the residuals (not just reduce computation for non-salient tokens but eliminate them entirely from the forward pass), the theoretical speedup is 3-10x on top of existing caching methods.
- **Novelty estimate**: 7/10 -- DyLLM and ES-dLLM already do partial versions of this. The full predictive coding formulation (with formal error propagation, precision weighting for attention, and hierarchical residual passing) is new. But the "only compute changing tokens" insight is well-explored in the sparse attention literature.

---

## Phase 3: Self-Critique

### Against Candidate A (DEBA -- Entropy Budget Composability Predictor)

- **Prior work attack**: Searched for "composability prediction information theory" and "entropy-based method interaction". Chen et al.'s Denoising Entropy paper (arXiv:2512.21336) does not address composability at all -- it uses entropy for quality optimization. The entropic time schedulers paper (arXiv:2504.13612) derives optimal schedules but for single-method settings. The information-gain sampler (arXiv:2602.18176) optimizes sampling decisions but does not predict method interactions. **No prior work connects entropy metrics to composability prediction.** The LDPC belief propagation analogy is genuinely novel in this context.

- **Methodological attack**: The key risk is that Denoising Entropy may not be decomposable into method-specific contributions. If Method A and Method B both affect the entropy of the same tokens in correlated ways, the joint entropy reduction is not separable. Testing this requires computing Denoising Entropy under each method independently AND jointly, then checking whether H(A+B) can be predicted from H(A) and H(B) alone. This is an empirical question -- the theory predicts separability for orthogonal methods, but confounders (implementation coupling, shared buffers) may break separability even for theoretically orthogonal methods.

- **Theoretical attack**: The LDPC analogy is suggestive but not rigorous. LDPC decoding operates on a known code structure (parity check matrix), while DLM methods interact through the model's learned representations, which are not structured in a known way. The factor graph structure that enables belief propagation analysis in LDPC does not have an obvious DLM analog. The analogy may be illuminating but not predictive.

- **Scalability attack**: Computing full Denoising Entropy for all method pairs requires running each pair on a substantial evaluation set. For K methods, that is O(K^2) evaluations. With current compute budgets (1-hour experiments), this is feasible for 4-6 methods but becomes expensive for 10+ methods. The "predictive" aspect -- predicting composability without running the pair -- would be the real breakthrough, but it requires validating the entropy decomposition hypothesis first.

- **Verdict**: MODERATE. The core idea is novel and the entropy decomposition hypothesis is testable. The main weakness is that it may be an elegant description of the composability landscape without being a useful predictor. The LDPC analogy needs more formal grounding. However, even as a descriptive framework (explaining WHY M1+CD-SSD synergizes while M1+M3 does not), this adds significant value to the ComposeAccel analysis paper.

### Against Candidate B (ADM -- Adaptive Denoising Mesh)

- **Prior work attack**: Searched for "hierarchical resource allocation diffusion language model" and "adaptive computation token step". Window-Diffusion (arXiv:2601.20332) already implements a sliding window with active/buffer/pruned tokens -- this IS a 1D adaptive mesh in the token dimension. ES-dLLM's layer skipping IS a 1D adaptive mesh in the layer dimension. JoT's per-token early stopping IS a 1D adaptive mesh in the step dimension. The multi-dimensional unification is partially done by dLLM-Serve's phase-multiplexed scheduler, which adapts across batch dimension and denoising phase. The gap is: no one has explicitly formulated the 2D (token x step) grid with hierarchical refinement, but the components exist.

- **Methodological attack**: Implementing a true 2D adaptive mesh for DLM inference is an engineering challenge. Current DLM architectures process all tokens simultaneously through transformer layers -- you cannot easily "refine" a subset of tokens at a finer resolution while keeping others coarse, because the attention mechanism couples all tokens. You would need masked attention patterns that vary spatially AND temporally, which is essentially what sparse attention methods (Sparse-dLLM, Focus-dLLM) already do in a less principled way.

- **Theoretical attack**: The CFD analogy breaks down in one key way: in CFD, the mesh is defined on a continuous spatial domain where the PDE solution varies smoothly. In DLMs, the "mesh" is over discrete token positions where the uncertainty can change abruptly from one position to the next. The hierarchical multigrid structure that makes AMR efficient in CFD (coarse grids capture long-wavelength modes, fine grids capture short-wavelength modes) does not have a clean analog in discrete token space. The "resolution" metaphor is more of a conceptual framing than a mathematically grounded framework.

- **Scalability attack**: A 2D adaptive mesh with dynamic refinement at every step would require per-step bookkeeping of which tokens need full computation and which can be cached/skipped. This overhead could negate the savings for short sequences. The approach is most promising for long sequences (>4K tokens) where the fraction of "active" cells is small, but this is exactly where current DLM evaluation is weakest.

- **Verdict**: MODERATE. The unifying framing is attractive, but the implementation challenges are real, and the individual 1D versions already capture most of the benefit. The 2D unification may not provide enough additional speedup over the best 1D methods to justify the complexity. The main value would be if the framework reveals non-obvious interactions between dimensions (e.g., "tokens that need fine step resolution also need fine layer resolution, so combining JoT + ES-dLLM is better than either alone").

### Against Candidate C (PCD -- Predictive Coding Denoiser)

- **Prior work attack**: Searched for "predictive coding neural network inference efficiency" and "residual-only computation transformer". DyLLM's saliency-based approach is essentially a heuristic version of predictive coding -- it identifies tokens whose "prediction" changed between steps and only recomputes those. ES-dLLM's tensor variation metric is literally measuring the prediction error across iterations. The conceptual connection to neuroscience's predictive coding is novel in this context, but the algorithmic mechanism (skip stable tokens, recompute changing ones) is exactly what these papers already do. Framing it as "predictive coding" adds theoretical elegance but may not add algorithmic novelty.

- **Methodological attack**: "Completely removing" stable tokens from the computation graph (not just caching their KV) is non-trivial. In a transformer, every token attends to every other token. If you remove token j from the forward pass, you lose its contribution to the attention weights of active tokens. You would need to store and reuse its KV representation (which is just KV caching) or approximate its attention contribution (which is what sparse attention methods do). True "removal" is only possible if the removed token's attention contribution is provably negligible, which is the premise of Window-Diffusion's pruning. The predictive coding framing does not add a new implementation mechanism beyond what existing sparse/caching methods already provide.

- **Theoretical attack**: The predictive coding framework from neuroscience assumes a hierarchical generative model where top-down predictions and bottom-up errors propagate through layers. DLM transformers are not hierarchically organized in this way -- all layers see the same input (with residual connections). The structural correspondence between PC's hierarchical Bayesian inference and DLM's iterative denoising is suggestive but loose. A rigorous formulation would require showing that each DLM denoising step corresponds to a "layer" in a PC hierarchy, which is not obviously true.

- **Scalability attack**: The approach would work well for long sequences (many stable tokens) but offer diminishing returns for short sequences. Since most current DLM benchmarks use relatively short outputs (GSM8K: ~200 tokens, HumanEval: ~300 tokens), the empirical benefit may be modest in the evaluation settings where the field currently operates.

- **Verdict**: WEAK-to-MODERATE. The framing is intellectually appealing but adds limited algorithmic novelty over DyLLM/ES-dLLM. The implementation would essentially re-derive existing sparse attention methods from a different theoretical perspective. Unless the predictive coding formulation yields a specific, non-obvious algorithmic improvement (e.g., a new proxy signal that is cheaper to compute than DyLLM's cosine similarity or ES-dLLM's tensor variation), this risks being a re-branding rather than a genuine advance.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (PCD)** dropped because: The algorithmic mechanism (skip stable tokens, recompute only changing ones) is already well-explored by DyLLM and ES-dLLM. The predictive coding framing adds theoretical elegance but not algorithmic novelty. Without a concrete, non-obvious improvement over existing proxy signals, this would be perceived as repackaging.

### Strengthened Ideas

**Candidate A (DEBA) strengthened into: Entropy-Decomposed Composability Theory (EDCT)**

Key refinements based on Phase 3 critique:

1. **Weakened the LDPC analogy to "motivating analogy" rather than formal equivalence.** The factor-graph structure of LDPC does not directly transfer. Instead, the analogy motivates the hypothesis that composability depends on whether methods target different "dimensions" of uncertainty, without requiring the full LDPC mathematical apparatus.

2. **Made the entropy decomposition hypothesis concrete and testable.** Define three quantities for each acceleration method M_i:
   - Delta_H_token(M_i): reduction in per-token entropy (averaged over steps)
   - Delta_H_step(M_i): reduction in per-step entropy (averaged over tokens)
   - Delta_H_cross(M_i): change in the token-step cross-entropy term
   
   Hypothesis: Two methods compose well (Ortho >= 1.0) when they primarily reduce entropy on different axes (one reduces Delta_H_token, the other reduces Delta_H_step) and compose badly (Ortho < 0.5) when they both target the same axis AND create conflicting cross-entropy effects.

3. **Tied the framework directly to ComposeAccel's existing data.** The existing Ortho scores (M1+CD-SSD=1.385, M3+CD-SSD=0.493, M1+M3=0.301) provide exactly the empirical ground truth needed to validate or falsify the entropy decomposition hypothesis. This converts the EDCT from a standalone proposal into a natural theoretical extension of ComposeAccel.

4. **Reduced experimental scope to be feasible within ComposeAccel's existing infrastructure.** Computing Denoising Entropy for the existing method combinations requires only adding entropy logging to the existing evaluation pipeline -- no new methods need to be implemented.

**Candidate B (ADM) partially absorbed into Candidate A.** The "2D resource allocation" framing from ADM becomes the visual/intuitive explanation of EDCT's entropy decomposition -- methods are "orthogonal" because they operate on different dimensions of the token-step grid. The AMR analogy provides a concrete way to explain why M1 (reduces column-wise cost in the grid) and CD-SSD (reduces row-count in the grid) are orthogonal, while M1+M3 (both modify column-wise attention patterns) are not.

### Additional Evidence Found

- **AdaBlock-dLLM's "volatility band"** (arXiv:2509.26432) provides empirical support for the spatial structure of uncertainty in the token dimension. The VB partitions tokens into three regimes (high-confidence plateau, volatile band, low-confidence floor), which maps directly onto EDCT's Delta_H_token decomposition.

- **Learn2PD's 57.51x compound speedup** (arXiv:2509.25188) when combining learned parallel decoding with KV caching confirms that methods targeting different axes (token parallelism vs. step-level caching) achieve multiplicative composition -- consistent with EDCT's prediction.

- **Entropic Time Schedulers** (arXiv:2504.13612) provide the mathematical tools (conditional entropy rate estimation from trained model losses) needed to compute Delta_H_step without expensive Monte Carlo sampling.

### Selected Front-Runner

**Candidate A: Entropy-Decomposed Composability Theory (EDCT)** is the front-runner because:

1. It directly addresses the most novel finding from ComposeAccel (binary composability) and upgrades it from an empirical observation to a theoretical explanation.
2. It is immediately testable with ComposeAccel's existing experimental infrastructure -- no new methods to implement, only entropy measurements to add.
3. It provides a predictive framework (not just descriptive): given entropy decomposition of a new method, EDCT predicts whether it will compose well with existing methods. This is genuinely useful and has no competitor in the literature.
4. It absorbs the best parts of Candidates B and C (2D resource allocation view, prediction-error interpretation) as supporting explanations without requiring their full implementation.
5. It transforms the ComposeAccel paper from a "benchmark" paper (here is a table of Ortho scores) into a "science" paper (here is WHY the scores are what they are, and here is how to predict future scores).

---

## Phase 5: Final Proposal

### Title

Entropy-Decomposed Composability Theory: Predicting When Training-Free MDM Acceleration Methods Synergize, Interfere, or Destroy

### Hypothesis

The composability (Ortho score) of two training-free MDM acceleration methods is determined by the degree of overlap in their Denoising Entropy reduction profiles across the (token position, denoising step) plane. Specifically: (1) methods that reduce uncertainty on orthogonal axes (token-axis vs. step-axis) achieve super-multiplicative composition (Ortho > 1.0); (2) methods that reduce uncertainty on the same axis but in non-overlapping regions achieve near-multiplicative composition (Ortho in [0.8, 1.0]); (3) methods that reduce uncertainty on the same axis in overlapping regions with conflicting mechanisms achieve destructive interference (Ortho < 0.5). This hypothesis is falsifiable: if the entropy decomposition fails to predict the sign or magnitude of Ortho for any tested method pair, the theory is wrong.

### Motivation

ComposeAccel's iteration 1-2 experiments revealed a striking binary composability pattern: exactly one method pair (M1+CD-SSD) achieves super-multiplicative synergy (Ortho=1.385) while all others show destructive interference (Ortho <= 0.50). This pattern has no existing theoretical explanation. The current literature treats each acceleration method independently, with no framework for predicting method interactions. Practitioners deploying DLMs face a combinatorial explosion of possible method combinations with no guidance beyond exhaustive empirical testing.

EDCT provides the missing theoretical layer. By decomposing each method's effect into token-axis and step-axis entropy reduction components, it explains:
- **Why M1+CD-SSD synergizes**: M1 (EntropyCache) reduces per-step computational cost without changing WHICH tokens are resolved (step-axis reduction). CD-SSD's frozen-token mechanism reduces WHICH tokens enter the REFINE phase (token-axis reduction). These operate on orthogonal dimensions, and the frozen tokens amplify M1's cache hit rate (cross-term synergy).
- **Why M1+M3 destroys**: Both M1 (cache reuse changes attention weights) and M3 (AR guidance changes token selection probabilities) modify the token-axis uncertainty profile. They create conflicting gradient signals -- cached attention patterns encode one denoising trajectory while AR guidance pushes toward a different trajectory.
- **Why M2 is structurally broken**: Adaptive step scheduling creates discontinuities in the step-axis entropy profile (entropy gaps from skipped steps) that cascade into token-axis instabilities through the bidirectional attention coupling.

### Method

1. **Denoising Entropy Profiling (DEP)**: For each acceleration method M_i (including no-acceleration baseline), run the evaluation pipeline while logging:
   - Per-token, per-step logit distributions: p_theta(x_j | x_{-j}, t) for all positions j and steps t
   - Token-level state entropy: H_j(t) = -sum_v p(x_j=v|t) log p(x_j=v|t)
   - Step-level aggregate entropy: H(t) = (1/N) sum_j H_j(t)
   - Path entropy: PE = sum_t H(t) (Chen et al.'s Denoising Entropy)
   
   Compute these under: baseline, M1 alone, CD-SSD alone, M3 alone, M1+CD-SSD, M1+M3, M3+CD-SSD.

2. **Entropy Decomposition Analysis (EDA)**: For each method M_i, compute:
   - Delta_H_token(M_i) = change in token-position variance of entropy (does the method make some tokens much more certain than others?)
   - Delta_H_step(M_i) = change in step-level entropy gradient (does the method make certain steps more or less decisive?)
   - Spatial overlap coefficient: O(M_i, M_j) = measure of how much the low-entropy regions of M_i overlap with the low-entropy regions of M_j in the 2D (token, step) plane
   
   Test the prediction: Ortho(M_i, M_j) is inversely correlated with O(M_i, M_j).

3. **Composability Predictor Validation**: Using the single-method DEP profiles, predict the joint Ortho score for each pair. Compare predicted vs. observed Ortho. If correlation > 0.8 across all tested pairs, the theory has predictive power.

4. **New Pair Prediction (Zero-Shot Composability)**: Extend the framework to predict composability for untested pairs:
   - JoT + M1: JoT reduces step-axis cost (early stop per token), M1 reduces per-step cost. EDCT predicts: near-orthogonal (Ortho >= 0.8).
   - ES-dLLM + M1: ES-dLLM reduces layer-axis cost (skip early layers), M1 reduces step-axis cost via caching. EDCT predicts: orthogonal (Ortho >= 0.9).
   - JoT + CD-SSD: Both reduce effective step count through different mechanisms. EDCT predicts: partially overlapping (Ortho in [0.5, 0.8]).
   
   Run these pairs to validate predictions.

### Cross-Domain Insight

The structural correspondence is with iterative decoding in error-correcting codes (LDPC/turbo codes). In belief propagation decoding:
- Messages along different edges of the factor graph reduce uncertainty about different variable nodes -- this is analogous to orthogonal acceleration methods targeting different dimensions.
- When two message schedules try to update the same variable node simultaneously with conflicting beliefs, the decoder oscillates and fails to converge -- this is analogous to destructive interference between methods targeting the same tokens.
- Early termination (stopping iterations when syndrome = 0) is optimal when all variable nodes have converged -- this is analogous to token-level early stopping (JoT) being optimal when all tokens have low entropy.

The analogy is not formal (DLM attention coupling is fundamentally different from LDPC factor graph structure), but it motivates the correct hypothesis: composability depends on dimensional separation of uncertainty reduction.

The adaptive mesh refinement analogy from CFD provides the visual framework: the (token, step) plane is like a 2D computational domain, each method is like a mesh refinement strategy, and composability depends on whether methods refine different regions of the domain.

### Experimental Plan

**Phase 1: Entropy Profiling (Pilot, ~30 min)**
- Run LLaDA-8B-Instruct baseline (64 steps) on 100 GSM8K samples with full logit logging
- Compute token-step entropy maps H_j(t) for each sample
- Verify: does entropy decrease monotonically? Are there "hotspot" tokens with persistently high entropy? What fraction of (token, step) cells are effectively decided (H < threshold)?
- Expected: >60% of cells decided by step 16 (consistent with CD-SSD's ~52% frozen-token rate)

**Phase 2: Single-Method Entropy Signatures (~45 min)**
- Run M1 (EntropyCache, t=2.0), M3 (AR-guided, gw=0.3), CD-SSD (tau=0.9, T_draft=16) on same 100 samples with logit logging
- Compute entropy decomposition (Delta_H_token, Delta_H_step, spatial overlap) for each method
- Verify: M1 primarily reduces H(t) (step-axis), CD-SSD primarily changes the set of high-H_j tokens (token-axis), M3 changes both (mixed)

**Phase 3: Composability Prediction (~45 min)**
- Run M1+CD-SSD, M1+M3, M3+CD-SSD on same 100 samples
- Compute actual Ortho scores (should approximately match iter_001 values on this subset)
- Compute predicted Ortho from single-method entropy signatures using overlap coefficient O(M_i, M_j)
- Measure prediction accuracy: correlation between predicted and actual Ortho

**Phase 4: Zero-Shot Prediction Validation (~60 min)**
- Implement JoT (per-token early stopping) and ES-dLLM (layer skipping) -- both training-free, should be straightforward
- Compute their entropy signatures from Phase 2-style runs
- Predict Ortho for JoT+M1 and ES-dLLM+M1 from single-method signatures
- Run these pairs and compare predicted vs. actual Ortho
- This is the decisive test: if EDCT successfully predicts composability for new, untested pairs, it has genuine predictive power

**Baselines**: M1 alone, CD-SSD alone, M3 alone, naive-T16, M1+CD-SSD from ComposeAccel iter_001/02

**Metrics**: Ortho score (existing ComposeAccel metric), Denoising Entropy (PE), entropy spatial overlap O(M_i, M_j), QAS

**What would falsify the hypothesis**: If predicted and actual Ortho have correlation < 0.5, or if two methods with high spatial overlap achieve Ortho > 1.0, or if two methods with zero spatial overlap achieve Ortho < 0.5.

### Resource Estimate

- **Compute**: All experiments on a single GPU (LLaDA-8B-Instruct, ~16 GB VRAM). Total estimated GPU time: ~3 hours for entropy profiling + ~3 hours for composability validation = ~6 hours.
- **Implementation**: Entropy logging requires adding ~50 lines to the existing evaluation harness (log softmax outputs, compute token-level entropy). No new model implementations needed. JoT and ES-dLLM are both described in enough detail to implement in ~2 hours each.
- **Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (validation on second architecture).
- **Evaluation**: GSM8K (100 for pilot, 500 for full), MATH500 (100 for pilot, full 500 for full).

### Risk Assessment

**Risk 1: Entropy decomposition is descriptive but not predictive.**
The entropy profiles may perfectly explain the existing Ortho scores (M1+CD-SSD=1.385, M1+M3=0.301) but fail to predict new pairs (JoT+M1, ES-dLLM+M1). This would reduce the contribution from "predictive theory" to "explanatory framework," which is still valuable but less impactful.
*Mitigation*: Test prediction on held-out pairs (Phase 4) before claiming predictive power. If prediction fails, present EDCT as an explanatory analysis of the existing ComposeAccel data.

**Risk 2: Entropy computation is too expensive to be practical.**
Full logit logging and per-token entropy computation at every step could 2-3x the inference time, making EDCT impractical for routine composability testing.
*Mitigation*: Develop a lightweight proxy (e.g., sample entropy from top-k logits instead of full distribution, or use the entropic time scheduler's loss-based entropy estimate from Stancevic et al.). Even a noisy proxy that correctly predicts the SIGN of composability (synergy vs. interference) would be useful.

**Risk 3: The binary composability pattern is an artifact of the specific methods tested, not a fundamental property.**
If JoT+M1 achieves Ortho in [0.5, 0.8] (neither synergy nor destruction), the "binary" narrative breaks. This would not invalidate EDCT but would require reframing from "predicting binary outcomes" to "predicting continuous Ortho scores from entropy overlap."
*Mitigation*: Frame EDCT as a continuous predictor from the start, with the binary pattern in ComposeAccel being an instance, not the general rule.

### Novelty Claim

**What is new:**
1. The concept of Entropy-Decomposed Composability -- predicting whether acceleration methods compose from their individual entropy reduction profiles -- has no predecessor in the DLM acceleration literature or in the broader neural network acceleration literature.
2. The application of Denoising Entropy (Chen et al., 2025) to method-level composability analysis (rather than quality optimization) is novel.
3. The cross-domain connection to belief propagation convergence conditions in LDPC decoding is novel in this context.
4. The 2D (token, step) entropy map visualization and spatial overlap metric provide a new analytical tool for understanding DLM inference dynamics.

**What is NOT new:**
- Denoising Entropy itself (Chen et al., 2025).
- The individual acceleration methods (M1, M3, CD-SSD, JoT, ES-dLLM).
- The composability atlas framework (ComposeAccel iteration 1-2).
- The observation that some methods compose and others interfere (ComposeAccel, Learn2PD).

**Evidence of novelty:**
- No paper in the arXiv search results connects entropy metrics to composability prediction.
- No paper in the LDPC/turbo coding literature has drawn the analogy to neural network acceleration composability.
- The existing ComposeAccel proposal does not include any theoretical framework for predicting or explaining composability -- EDCT fills this gap.
