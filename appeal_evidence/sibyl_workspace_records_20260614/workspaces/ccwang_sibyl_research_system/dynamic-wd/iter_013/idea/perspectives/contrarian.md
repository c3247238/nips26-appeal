# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

The existing research context makes four major assumptions that deserve rigorous interrogation:

1. **Assumption: Alignment signal (δ_t) from minibatch gradients is a reliable proxy for population-level alignment**
   - Evidence challenging it: GWA paper (arXiv:2510.25480) acknowledges that "quantifying variance is the key signal for generalization" and that per-sample gradient alignment is highly variable in stochastic optimization. Van Laarhoven (2017, arXiv:1706.05350) demonstrated that for scale-invariant layers (those with BatchNorm/LayerNorm), weight decay loses its regularization effect entirely — the entire network output becomes approximately invariant to weight scaling. If alignment is computed between gradient and parameter vectors in BN layers, the alignment signal is measuring a quantity that does not govern generalization outcomes in those layers.
   - Sources: [GWA paper](https://arxiv.org/abs/2510.25480), [L2 vs BN paper](https://arxiv.org/abs/1706.05350)

2. **Assumption: A unified theoretical framework that encompasses all four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) is scientifically meaningful — that is, these approaches have a common mathematical core worth unifying**
   - Evidence challenging it: The optimizer benchmarking literature (Choi et al. 2019, arXiv:1910.05446; Sivaprasad et al. 2020) shows that the *apparent* performance gaps between optimizer variants are largely artifacts of hyperparameter tuning budgets and search spaces, not inherent algorithmic differences. If CWD, SWD, AlphaDecay, and AdamWN each depend on different hyperparameter choices and are sensitive to architecture/dataset combinations, "unifying" them into a single lambda(t, w, g) = f(...) framework may be mathematically elegant but practically vacuous — each method may work for different reasons on different tasks, not as special cases of a single principle.
   - Sources: [Choi et al.](https://arxiv.org/abs/1910.05446), [Sivaprasad et al.](https://openreview.net/pdf?id=s6M0gjo0rL0)

3. **Assumption: Dynamic/alignment-aware WD provides a benefit beyond simply adding one more tunable hyperparameter**
   - Evidence challenging it: The NOVAK paper (arXiv:2601.07876) explicitly shows that "coupling WD with α_eff (not α) degrades generalization 4-8pp on CIFAR-100." The SWD paper itself was motivated by the fact that *standard* WD can hurt performance by producing large gradient norms at the end of training — i.e., naive dynamic WD can make things worse. CWD's own ablation shows that a random mask achieves substantially worse results than the sign-aligned mask, suggesting the signal has real content, but the margin over a well-tuned fixed WD baseline is typically small (fractions of a percent on language modeling). D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) found that WD's role is primarily dynamics-modification and numerical precision stabilization, not generalization — suggesting the entire alignment-aware WD direction may be solving the wrong problem.
   - Sources: [NOVAK](https://arxiv.org/abs/2601.07876), [D'Angelo et al.](https://arxiv.org/abs/2310.04415), [CWD](https://arxiv.org/abs/2510.12402)

4. **Assumption: The Budget Equivalence Metric, Coupling Stability Index, and Alignment Informativeness Score are standardizable — that they will produce consistent rankings across architectures, datasets, and optimizers**
   - Evidence challenging it: The standardized metrics literature shows that attempts to create universal evaluation metrics for optimizer comparison have consistently failed to produce stable rankings. Choi et al. showed that "the results can be contradicted when hyperparameter search spaces are changed." The OUI metric (arXiv:2504.17160) is architecture-specific and has only been validated on DenseNet/EfficientNet/ResNet — it is unknown whether it transfers to Transformers or LLMs. If our three proposed metrics are validated primarily on CIFAR + ResNet settings, they may not generalize.
   - Sources: [Choi et al.](https://arxiv.org/abs/1910.05446), [OUI](https://arxiv.org/abs/2504.17160)

### Supporting Resources for Phase 1 (Additional)

5. **Assumption: Gradient-weight cosine alignment (δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖)) is interpretable for all layers**
   - Evidence: ResearchGate paper "The Hidden Pitfalls of the Cosine Similarity Loss" (June 2024) shows that gradient of cosine similarity goes to zero when a point has large magnitude — making the alignment signal degenerate precisely when weight norms are large, which is when we most want to apply WD. This is a circular problem: the alignment signal degrades when the regularization is most needed.

6. **Assumption: WD scheduling and alignment-aware WD are improvements over fixed WD because they're more adaptive**
   - Evidence: NeurIPS 2024 paper on normalization and effective learning rates shows that for scale-invariant (normalized) layers, weight decay merely acts as an effective learning rate scheduler anyway. So "adaptive WD" in normalized networks may be functionally indistinguishable from "adaptive learning rate" — a crowded and well-studied space.

7. **Assumption: The Sun et al. CVPR 2025 paper's alignment quantity δ_T is a meaningful discriminator in practice**
   - Evidence: Sun et al. prove that WD improves generalization when δ_T < 1 (gradient not collinear with parameters). But this is a worst-case quantity over all T timesteps. In practice, δ̂_t from minibatches is both noisy (high variance) and architecture-dependent (BN layers are scale-invariant, making cosine alignment between gradient and weights ill-defined in those layers). The theory may not translate to practical algorithms because the preconditions are rarely checkable.

8. **Assumption: Unifying all WD methods is novel — that no prior work achieves sufficient synthesis**
   - Evidence: Ye (arXiv:2410.00232) "Preconditioning for Optimization and Regularization" already provides a unified framework where AdamW selects intrinsic parameters for regularization and derives L1-regularization analogues. Newhouse MIT thesis (2025) unifies SGD, Adam, Shampoo through duality maps. The gap our paper claims to fill may already be partially filled by these works.

### Landscape of Doubt (Synthesis)

Four structural problems make the project vulnerable:

**Problem A (Signal quality):** The alignment signal δ̂_t is simultaneously (1) noisy from minibatch sampling, (2) degenerate for scale-invariant layers (BN/LN), (3) potentially ill-defined when weight norms are large (cosine similarity degrades). These are not incidental issues but fundamental limitations of the gradient-weight angle as an actionable signal for WD modulation.

**Problem B (Hyperparameter confounding):** The entire literature on dynamic WD operates in a regime where the methods have more tunable hyperparameters than standard fixed WD. Fair comparison requires matching hyperparameter tuning budgets. The standard practice of reporting best-found results conflates "better algorithm" with "more hyperparameters to tune." Our proposed BEM metric does not address this.

**Problem C (Architecture specificity):** Benefits of alignment-aware WD may be confined to architectures without heavy normalization (i.e., old-style CNNs without BatchNorm). Modern Transformers (ViTs, GPT-class models) have extensive LayerNorm, making the per-layer gradient-weight alignment signal theoretically uninformative for those layers. This could explain why CWD's Muon results are sensitive to the interpretation of "optimizer update" vs "gradient."

**Problem D (Survey risk):** A paper whose main contribution is a unifying framework + standardized metrics (rather than a new algorithm + strong empirical results) is hard to get accepted at ICML/NeurIPS unless the framework reveals genuinely novel mathematical insights. The risk is producing a paper that is "interesting but not surprising" — the methods are already known, the connections are plausible, and reviewers may ask "what does this explain that wasn't already understood?"

---

## Phase 2: Initial Candidates

### Candidate A: The Alignment Signal Is a Red Herring for Normalized Networks

- **Challenged assumption**: That gradient-weight alignment (δ̂_t) is a meaningful signal for modulating WD in modern deep learning architectures.
- **Evidence against it**:
  - Van Laarhoven (2017) proved that L2 regularization (of which WD is a variant) has no regularization effect on scale-invariant layers (those with BatchNorm). The network output is invariant to weight scaling in those layers, so the cosine angle between gradient and weight vector is measuring a quantity that does not influence the model's predictions.
  - Kobayashi et al. (arXiv:2410.23819) showed that WD on multiplicative (attention) layers is equivalent to nuclear norm regularization — a fundamentally different structure from L2/alignment. Treating alignment uniformly across layer types is architecturally unsound.
  - The "Hidden Pitfalls of Cosine Similarity" paper shows gradient signal of cosine similarity degrades when norms are large — creating a systematic bias: alignment appears high (making CWD-style methods not decay) precisely when weights are large and most in need of regularization.
- **Contrarian hypothesis**: The alignment signal is only meaningful for non-normalized layers in old-style architectures (pre-BN CNNs). For modern architectures with extensive normalization, alignment-aware WD is equivalent to random WD modulation with extra hyperparameters. This explains why CWD's improvements are most clearly demonstrated on language model pre-training (where the benefit comes from preventing gradient divergence, per D'Angelo et al.) rather than from genuine geometric alignment.
- **Exploitation plan**: Run controlled experiments on (a) ResNet-20 without BN vs with BN, (b) ViT with and without disabling LN, showing alignment-aware WD advantage collapses when normalization is present. Quantify alignment signal variance across BN vs non-BN layers.
- **Novelty estimate**: 8/10 — directly challenges the core mechanism assumed by multiple papers (CWD, the proposed alignment-aware WD in the project spec)

### Candidate B: Dynamic WD Improvements Are Explained by Implicit Learning Rate Rescaling, Not Alignment

- **Challenged assumption**: That alignment-aware WD improves outcomes through a distinct mechanism (geometric alignment) rather than through implicit learning rate adjustment.
- **Evidence against it**:
  - NeurIPS 2024 normalization paper proves that in scale-invariant networks, WD is equivalent to an effective learning rate modifier: reducing WD on certain steps is functionally equivalent to increasing LR on those steps.
  - Defazio (arXiv:2506.02285) shows WD drives ‖g‖/‖w‖ to a steady state — "layer balancing." If alignment-aware WD modulates λ based on geometric signal, it changes ‖g‖/‖w‖ in a particular way, which is equivalent to a structured LR schedule.
  - GALA (OpenReview 2025) uses consecutive gradient alignment to adapt LR — functionally identical to using alignment signal for WD in normalized networks.
  - CWD's own ablation shows a random mask degrades performance — but this doesn't prove the *alignment* is the cause; it could be that any structured sparsity in WD application that avoids "opposing the update" works, which is a simpler mechanism than cosine alignment.
- **Contrarian hypothesis**: In normalized networks, every alignment-aware WD method is a confounded version of alignment-aware LR scheduling. Proving this equivalence could kill two birds with one stone: (1) explain why CWD works (it's an implicit LR adapter), and (2) suggest that the entire WD-alignment direction should be reframed as "effective LR scheduling with a weight shrinkage constraint."
- **Exploitation plan**: For each aligned-WD variant (CWD, the proposed δ̂_t-based rule), derive the equivalent LR modification in a normalized 2-layer network. Run ablation comparing: CWD vs equivalent LR schedule (same effective update magnitude). If performance is equivalent, the alignment signal adds nothing beyond implicit LR rescaling.
- **Novelty estimate**: 7/10 — requires careful theoretical work; potential for high-impact insight if equivalence holds

### Candidate C: The Benchmark Fragmentation Problem Is Unsolvable — New "Standardized Metrics" Will Also Fragment

- **Challenged assumption**: That proposing Budget Equivalence Metric, Coupling Stability Index, and Alignment Informativeness Score will actually standardize evaluation in the WD research community.
- **Evidence against it**:
  - Choi et al. (arXiv:1910.05446) showed optimizer rankings are determined by hyperparameter search spaces, not by algorithm. Introducing new evaluation metrics without fixing the underlying hyperparameter confounding only adds another fragmentation dimension.
  - The OUI metric (arXiv:2504.17160) is a recent standardization attempt for WD selection that the literature has largely ignored (it's not referenced in any of the 47 papers in the existing survey except as #19 in our own literature review). New metrics don't become standards just by being proposed.
  - The BEM metric (normalizing by compute) faces a structural problem: methods with more hyperparameters require more tuning compute, making BEM-normalization systematically favor methods with fewer hyperparameters regardless of their intrinsic quality.
  - History of ML standardization efforts: AIS (Alignment Informativeness Score) may not be stable across random seeds in minibatch training, making it unreliable for comparison purposes.
- **Contrarian hypothesis**: The WD research community's evaluation fragmentation is not due to missing standardized metrics — it's due to the fact that WD's effects are context-specific (architecture, dataset, optimizer, initialization, training duration). No single set of metrics can capture this context-dependence. A more honest contribution would be a *taxonomy of contexts* under which each WD method is beneficial, rather than universal metrics that paper over this heterogeneity.
- **Exploitation plan**: Meta-analysis: collect results from 10+ WD papers, normalize by hyperparameter budget, and show that performance advantages shrink or disappear under budget-matched tuning. Demonstrate that existing metrics (test accuracy, validation perplexity) already provide sufficient comparison information when combined with rigorous tuning protocols — new metrics are not needed.
- **Novelty estimate**: 6/10 — important corrective but harder to make into a positive contribution

---

## Phase 3: Self-Critique

### Against Candidate A: Alignment Signal as Red Herring

**Steelman of conventional view**: CWD (ICLR 2026) shows consistent improvements across LLMs (338M to 2B parameters) where LayerNorm is present throughout. If the alignment signal were truly meaningless for normalized layers, CWD would not work on LLMs. The ICLR acceptance and robust empirical evidence across many architectures suggests the alignment signal retains some meaning even in normalized networks.

Additionally, Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340) prove that at stationary points of L2-regularized objectives, there is parameter-gradient alignment — suggesting alignment is not random but actually structured by the optimization process.

**Cherry-picking check**: The evidence against alignment (Van Laarhoven 2017) is theoretical and applies to the *regularization* effect, not the *dynamics* effect. D'Angelo et al. distinguish between WD's regularization role (which BN eliminates) and WD's training dynamics role (which BN does not eliminate). The alignment signal might still be meaningful for the dynamics role even if the regularization interpretation fails.

**Confounding check**: The scale-invariance argument applies most strongly to intermediate layers. Final linear layers and embedding layers are typically not scale-invariant. CWD improvements in LLMs may be concentrated in these non-scale-invariant layers, making the finding architecturally nuanced rather than globally applicable.

**Actionability check**: Even if proven, "alignment signal is degraded by BN" suggests we should use layer-type-aware alignment computation, not abandon alignment-aware WD entirely. This could lead to a refined algorithm rather than a purely negative result.

**Verdict**: MODERATE — The steel-manned conventional view holds for normalized networks via dynamics effects. The critique should be refined: alignment signal is degraded specifically for the *regularization* interpretation in normalized layers, but may remain valid for *dynamics control* purposes. The challenge is to develop experiments that isolate these two mechanisms.

### Against Candidate B: Dynamic WD = Implicit LR Rescaling

**Steelman of conventional view**: If alignment-aware WD were simply implicit LR rescaling, then a well-tuned learning rate schedule should match its performance. In practice, CWD consistently outperforms standard AdamW with equivalent LR schedules (as reported in the paper). The ablation shows random masks fail — implying structure beyond pure LR rescaling.

Moreover, CWD's bilevel interpretation (searching for Pareto-optimal stationary points while preserving the original loss's stationary points) is mathematically distinct from LR adaptation. LR adaptation doesn't change the set of reachable stationary points — CWD does.

**Cherry-picking check**: The LR-equivalence argument (Defazio's steady-state analysis) holds at equilibrium but not during the transient training dynamics. During early training, the dynamics of alignment-aware WD and LR-scheduling differ. The improvement from CWD may come specifically from the transient phase.

**Confounding check**: LR schedules are typically monotonically decreasing, while alignment-based WD modulation is non-monotonic (follows the geometric trajectory). These have different qualitative properties. The equivalence may break in networks with complex optimization landscapes.

**Actionability check**: Even if partial equivalence holds, proving it would be a genuine theoretical contribution explaining *why* CWD works. This can be framed constructively as "understanding the mechanism of alignment-aware WD through the lens of effective LR dynamics."

**Verdict**: MODERATE — Partial equivalence likely holds in normalized networks at quasi-steady states. The claim should be weakened to: "In normalized networks, alignment-aware WD functions partly as an implicit LR adapter; this equivalence is asymptotically exact and helps explain observed gains." This is still publishable and insightful without claiming the strong form that "alignment adds nothing."

### Against Candidate C: Metrics Won't Standardize

**Steelman of conventional view**: The optimizer benchmarking literature (Choi et al. 2019) is old and predates modern understanding of WD. The specific issue with WD benchmarking is not just hyperparameter sensitivity but the absence of any framework to compare methods that optimize fundamentally different objectives (e.g., CWD preserves the original loss; AdamW optimizes a regularized loss). Standardized metrics designed specifically for this structural difference would be a genuine contribution.

**Cherry-picking check**: OUI being ignored is not strong evidence against new metrics — it may simply be an inferior metric design. A better-designed metric could gain adoption.

**Confounding check**: The BEM metric confounds "compute for training" with "compute for tuning." Separating these explicitly would address the concern raised.

**Actionability check**: Even a negative result ("here's why WD methods cannot be fairly compared") would be valuable if it redirects the community toward more honest reporting practices.

**Verdict**: WEAK — The critique is valid but the alternative (taxonomy of contexts) is essentially what all survey papers do. This candidate does not provide sufficient basis for a strong research contribution on its own.

---

## Phase 4: Refinement

### Dropped
**Candidate C** (metrics standardization critique) is dropped as a standalone idea. The critique is valid but provides insufficient material for a positive research program. It can be incorporated as a "limitations of our metrics" discussion in the main paper.

### Strengthened — Candidate A becomes the front-runner

The core insight from Candidate A survives the steelman test in a refined form:

**Refined claim**: The alignment signal δ̂_t captures two distinct phenomena depending on layer type:
- **Non-normalized layers**: δ̂_t reflects genuine parameter-gradient geometric alignment that influences the regularization landscape. Alignment-aware WD is informative here.
- **Normalized layers** (BN, LN, GroupNorm): δ̂_t captures a gradient-weight angle in the pre-normalization representation space, but the network output is invariant to weight scaling in these layers. For these layers, WD's effect is purely a learning rate modifier, and the alignment signal modulates the effective LR — not the regularization.

**Additional corroboration found**:
- PyTorch forums discussion "Weight decay in the optimizers is a bad idea (especially with BatchNorm)" confirms practitioners have long observed this
- Kobayashi et al. (2024) showing WD induces nuclear norm (not L2) regularization on attention layers further supports that uniform alignment computation across layer types is architecturally unsound
- NeurIPS 2024 paper on normalization and effective learning rates formally extends Van Laarhoven's result to modern architectures

**Key strengthening**: Merge Candidate A and B: prove that in normalized layers, alignment-aware WD collapses to LR adaptation (Candidate B), while in non-normalized layers, alignment retains independent value (Candidate A survives there). This gives a unified theory explaining when alignment-aware WD helps and when it is redundant.

**Additional search findings**:
- Found that CWD reports one case of failure in Conda optimizer integration (practitioner report), suggesting architecture-dependence is not fully characterized
- The WD = effective LR argument is confirmed by Jane Street blog (L2 regularization and batch norm analysis) and NeurIPS 2024 normalization paper
- Sun et al. CVPR 2025 alignment quantity δ_T is defined for the entire network, obscuring the layer-wise heterogeneity

### Selected Front-Runner

**Candidate A/B Hybrid**: "Alignment-Aware Weight Decay Through a Layer-Type Lens: When Gradient-Weight Geometry Is Informative vs. Redundant"

---

## Phase 5: Final Proposal

### Title
**Rethinking Alignment-Aware Weight Decay: Layer Heterogeneity, Signal Reliability, and the Effective Learning Rate Duality**

### Challenged Assumption (Precise)

The current alignment-aware WD literature (CWD, the proposed δ̂_t-based dynamic WD in the project spec, GWA) implicitly assumes that the gradient-weight cosine similarity δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε) is a universally informative signal for deciding when to apply weight decay. This assumption is violated for scale-invariant layers (those preceded by BatchNorm, LayerNorm, GroupNorm, or followed by softmax normalization), because:

1. The network output is invariant to weight scaling in these layers (Van Laarhoven 2017; NeurIPS 2024)
2. Therefore, WD's only effect in these layers is to modify the effective learning rate: λ_t → effective LR reduction ∝ λ_t / ‖w_t‖²
3. The alignment signal δ̂_t, for normalized layers, is therefore a proxy for "how much we want to reduce the effective LR" — not for "how much geometric alignment justifies regularization"
4. This means alignment-aware WD in normalized networks is equivalent to alignment-aware LR scheduling (GALA) — with the added complexity of extra hyperparameters but no distinct benefit

### Evidence: For and Against the Assumption

**Evidence supporting conventional alignment-aware WD view**:
- CWD achieves consistent improvements across LLMs (338M-2B) with LayerNorm throughout (ICLR 2026)
- Kuzborskij & Abbasi-Yadkori (2025): at stationary points, L2 regularization induces parameter-gradient alignment — alignment is not accidental but structured
- Sun et al. (CVPR 2025): δ_T < 1 is the theoretical condition for WD to improve generalization; this is architecture-independent in the theoretical framework
- CWD's random-mask ablation fails, showing alignment structure matters, not just random sparsity

**Evidence challenging the assumption**:
- Van Laarhoven (2017): L2 regularization has no regularization effect in scale-invariant layers; WD acts as effective LR modifier
- NeurIPS 2024 (normalization + ELR paper): formally extends this to modern RL + vision architectures
- Kobayashi et al. (2024): WD on attention (multiplicative) layers is equivalent to nuclear norm, not L2; computing cosine alignment between gradient and weight vector in these layers uses the wrong metric
- Defazio (2025): WD drives ‖g‖/‖w‖ to a steady state; in normalized networks, this is equivalent to adjusting the effective LR steady state — confirming the LR-duality view
- PyTorch community + Jane Street blog: practitioner consensus that WD on BN layers is "a bad idea" or at least has unexpected effects
- CWD failure in Conda optimizer: demonstrates architecture-specific failure modes exist but are not characterized in the paper

### Hypothesis

**Dual-track alignment hypothesis**: The information content of gradient-weight alignment for WD decisions is layer-type-dependent:

- **Track 1 (Non-normalized layers)**: δ̂_t captures genuine regularization-relevant information. In these layers, alignment-aware WD provides incremental benefit beyond fixed WD because it modulates the objective geometry rather than just the effective LR. This explains why alignment-aware WD works on fully-connected networks and pre-BN CNNs.

- **Track 2 (Normalized layers)**: δ̂_t is a proxy for effective LR dynamics. Alignment-aware WD in these layers is equivalent to alignment-aware LR scheduling. The benefit seen in LLMs (where CWD succeeds) may come from non-normalized layers (embedding, final projection) or from preventing gradient norm divergence (D'Angelo et al.'s "loss stabilization" mechanism), not from genuine geometric regularization.

**Implication**: A layer-type-aware alignment mask (apply alignment-aware WD only to non-normalized layers; use fixed WD or alignment-aware LR for normalized layers) should outperform uniform alignment-aware WD across all layers.

### Method

**Experiment 1: Layer-type ablation of CWD**
- Architecture: VGG-16-BN (has BN) and VGG-16 without BN (non-normalized)
- Method: Apply CWD to (a) all layers, (b) only BN layers, (c) only non-BN layers, (d) no layers (fixed WD)
- Dataset: CIFAR-100 (3 seeds: 42, 123, 456)
- Expected finding: CWD benefit collapses when applied only to BN layers; full benefit is recovered when applied only to non-BN layers

**Experiment 2: δ̂_t signal variance analysis by layer type**
- Track δ̂_t trajectory separately for BN layers vs non-BN layers throughout training
- Compute temporal autocorrelation and SNR of δ̂_t signal in each layer type
- Expected finding: BN layers show high variance and low autocorrelation in δ̂_t; non-BN layers show structured, reliable δ̂_t dynamics

**Experiment 3: WD-LR duality test in normalized networks**
- Architecture: ViT-Tiny (all LN, representative of modern transformers)
- Comparison: (a) CWD with λ=5e-4, (b) fixed WD λ=5e-4 + GALA-style LR adaptation calibrated to match CWD's effective update magnitudes
- Dataset: CIFAR-10, ImageNet-100 (small subset; target <1 hour per run)
- Expected finding: GALA-matched LR scheduling achieves equivalent or better performance than CWD — confirming the LR duality for normalized networks

**Experiment 4: Alignment-aware WD restricted to embedding + projection layers**
- Architecture: GPT-2 small (NanoGPT codebase)
- Method: Apply CWD only to embedding and final projection layers (non-normalized); use fixed WD for all LN-preceded layers
- Expected finding: Selective layer-type CWD matches or exceeds full-network CWD, with lower compute overhead (fewer alignment checks)

### Baselines (No Strawmen)

All baselines are properly tuned using matched hyperparameter search budgets (50 random trials each):
- Standard AdamW with fixed WD (well-tuned λ)
- CWD (full network, as published)
- SWD / AdamS (gradient-norm-aware scheduling)
- GALA (alignment-aware LR adaptation — the "LR duality" control)
- Normalize-and-Project (NaP) — alternative to WD for norm control

### Experimental Plan: Small-Scale Priority

All experiments target ≤1 hour per run on local GPUs (8x RTX PRO 6000). Override: ImageNet experiments are permitted up to 4 hours per the project constraints.

| Experiment | Architecture | Dataset | Est. Time | Seeds |
|---|---|---|---|---|
| 1a: CWD layer ablation | VGG-16-BN | CIFAR-100 | 30 min | 3 |
| 1b: VGG-16 no-BN ablation | VGG-16 | CIFAR-100 | 25 min | 3 |
| 2: δ̂_t signal analysis | ResNet-20 | CIFAR-10 | 20 min | 1 (diagnostic) |
| 3: WD-LR duality test | ViT-Tiny | ImageNet-100 | 60 min | 3 |
| 4: GPT-2 selective CWD | GPT-2 small | OpenWebText | 90 min | 2 |

### Risk Assessment

**If the conventional view is correct after all**:
- δ̂_t may be informative in normalized layers through the dynamics channel (not regularization channel)
- In this case, our layer-type ablation will show CWD benefits are uniform across layer types
- Salvage path: Reframe the contribution as "explaining why CWD works via LR duality — a constructive understanding, not a debunking"
- The WD-LR duality itself is a theoretically novel insight even if it doesn't invalidate CWD

**Potential confounders to control**:
- BatchNorm's own trainable parameters (gamma, scale) should exclude WD (standard practice); ensure all baselines handle this consistently
- Training duration effects: alignment-aware WD benefits may be more pronounced in early training vs late training — need full training curves
- BN layers during inference use population statistics — any training-phase alignment signal computed for BN layers may not correspond to inference behavior

### Novelty Claim

The specific insight this research contributes, regardless of experimental outcome:

1. **Theoretical**: For scale-invariant (normalized) layers, alignment-aware WD is provably equivalent to alignment-aware effective LR adaptation. This provides the first mechanistic explanation for why CWD works in LLMs (LR duality + loss stabilization) while also explaining why it may not generalize to all architectures uniformly.

2. **Practical**: Layer-type-aware WD masking (apply alignment reasoning only to non-normalized layers) is a principled and computationally cheaper variant of CWD. If confirmed, it suggests the unified dynamic WD framework should be structured around layer-type ontologies, not a single global alignment function.

3. **Critical/corrective**: The project's proposed δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε) alignment proxy is architecturally naive. For normalized layers, this quantity has high variance and low information content for regularization decisions. The "worst-case alignment" quantity δ_T from Sun et al. (CVPR 2025) aggregates this signal across all layer types, conflating a useful signal (non-normalized layers) with a noisy one (normalized layers). Accounting for this heterogeneity is a prerequisite for any principled unified framework.

4. **Constructive for the unified framework**: The "unifying" formula λ(t, w, g) = f(alignment(w,g), norm(w), schedule(t), target_norm(tau)) should be augmented with a layer-type indicator: λ(t, w, g, type) = f_type(alignment(w,g), norm(w), schedule(t), target_norm(tau)), where f_BN collapses alignment's role and f_nonBN preserves it. This is a meaningful refinement that makes the unified framework more truthful.

---

*Contrarian perspective completed: 2026-03-19*
*Agent: sibyl-contrarian (sibyl-standard tier)*
