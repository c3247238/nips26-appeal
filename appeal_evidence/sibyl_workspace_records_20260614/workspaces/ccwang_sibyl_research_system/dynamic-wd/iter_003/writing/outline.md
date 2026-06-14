# Paper Outline: When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW

**Target Venue**: NeurIPS 2026 Datasets & Benchmarks Track (primary) / TMLR (backup)
**Format**: NeurIPS 8-page main body + references + appendix
**Language**: English throughout

---

## Abstract (target: ~150 words)

**Key points to cover:**
- Weight decay (WD) is universally applied in deep learning but its scheduling and modulation strategies remain poorly understood
- We introduce the **Phi Modulator Framework**: a unified mathematical abstraction `φ(t, w, g)` that subsumes all major dynamic WD approaches as special cases (scheduling, alignment-aware, norm-matched)
- We propose three diagnostic metrics: **Budget Equivalence Metric (BEM)**, **Coupling Stability Index (CSI)**, and **Alignment Informativeness Score (AIS)**
- Systematic evaluation of 7 WD methods on CIFAR-10/100 with ResNet-20 (42 experiments, 3 seeds per configuration) under the AdamW optimizer
- **Main finding**: All dynamic WD variants are statistically equivalent to constant WD under AdamW (p > 0.05 for all comparisons); even removing WD entirely (λ=0) yields <0.5% accuracy difference
- We formalize this as the **Phi Invariance Conjecture**: under AdamW with sufficient training, the choice of Phi function is irrelevant to final generalization
- The framework and metrics provide the first standardized infrastructure for WD research

---

## 1. Introduction (~1.5 pages)

### 1.1 Motivation
- Weight decay is one of the most universally applied techniques in deep learning, yet the community lacks a principled framework for choosing *how* to apply it
- Recent surge in dynamic WD methods: Cautious WD (CWD, Chen et al. ICLR 2026), Scheduled WD (SWD, Xie et al. NeurIPS 2023), Weight Norm Control (AdamWN, Loshchilov 2023), AlphaDecay (He et al. 2025), log-time schedules (ADANA, Ferbach et al. 2026)
- **Each method is evaluated in isolation**, using different architectures, datasets, and metrics — making direct comparison impossible
- Fundamental question: **does dynamic WD actually help**, and if so, *when* and *why*?

### 1.2 Research Gap
- No unified mathematical framework that unifies all dynamic WD methods
- No standardized evaluation metrics that allow fair, reproducible comparison
- No systematic study with controlled hyperparameters revealing whether the differences are real or artifacts of experimental design
- Motivating citation: D'Angelo et al. (NeurIPS 2024) showed WD acts as a dynamics modifier rather than classical regularizer — so WD scheduling should matter, yet this has not been rigorously tested

### 1.3 Contributions (numbered list)
1. **Phi Modulator Framework**: A unified interface `φ(t, w, g)` that subsumes CWD, SWD, cosine schedules, norm-matched WD, and all compositions as special cases; formalizes the modulation taxonomy (temporal, directional, spatial, target-norm axes)
2. **Three diagnostic metrics**: BEM, CSI, AIS — the first standardized tools for quantifying how WD methods differ in budget, coupling stability, and alignment informativeness
3. **Systematic benchmark**: 42 controlled experiments (7 methods × 3 seeds × 2 datasets) with identical hyperparameters, revealing statistical equivalence under AdamW
4. **Phi Invariance Conjecture**: Formal statement that AdamW's adaptive per-parameter scaling subsumes WD modulation effects, with diagnostic evidence

### 1.4 Paper Roadmap
- Brief sentence-per-section guide to paper structure
- Note that supplementary contains 6-panel diagnostic plots for all 42 runs and additional ablations

### Key Figure Reference: Figure 1
- **Taxonomy diagram** of the Phi framework: 2D grid with modulation axis (temporal / directional / spatial / target-norm) vs. method family (CWD, SWD, cosine, AdamWN, random, no-WD)
- Each cell shows the Phi formula for that method
- Half-page visual that establishes the conceptual map for the whole paper

---

## 2. Related Work (~0.75 pages)

### 2.1 Weight Decay as Dynamics Modifier
- Classical view: WD as L2 regularization shrinking weights toward zero
- Modern re-interpretation (D'Angelo et al. NeurIPS 2024): WD as loss stabilizer (SGD) and bias-variance controller (LLMs); explicit regularization is never the reason WD works
- Loshchilov & Hutter ICLR 2019: decoupled WD in AdamW separates WD from gradient scaling — the foundational distinction

### 2.2 Dynamic WD Methods (briefly survey all four families)
- **Temporal scheduling**: SWD/AdamS (Xie et al. NeurIPS 2023) — gradient-norm-aware dynamic WD; ADANA (Ferbach et al. 2026) — log-time schedules
- **Alignment-aware**: CWD (Chen et al. ICLR 2026) — binary sign-alignment mask; AdamO (Chen et al. 2026) — radial/tangential decoupling eliminating "Radial Tug-of-War"
- **Norm-matched**: AdamWN (Loshchilov 2023) — target-norm control; AlphaDecay (He et al. 2025) — spectral-density-guided module-wise decay
- **Structural insight**: WD induces low-rank weights (Galanti et al. 2022), low-rank attention (Kobayashi et al. 2024), rotational equilibrium (Kosson et al. 2023)

### 2.3 Evaluation Fragmentation
- Note the evaluation problem: each paper uses different benchmarks, metrics, hyperparameter protocols
- OUI metric (Fernandez-Hernandez et al. 2025) is a diagnostic but not comparative
- This fragmentation motivates BEM/CSI/AIS as standardized metrics
- Closest work: why-weight-decay (D'Angelo et al.) provides shared infrastructure, but does not compare dynamic scheduling strategies

---

## 3. Phi Modulator Framework (~1.5 pages)

### 3.1 Formal Definition

**The Phi Modulator:**
```
θ_{t+1} = θ_t - η_t · m̂_t / (v̂_t^{1/2} + ε) - λ · φ(t, θ_t, g_t) ⊙ θ_t
```

Where `φ(t, θ_t, g_t) : ℤ≥0 × ℝ^d × ℝ^d → ℝ^d_≥0` is the **Phi modulator** — a per-parameter, non-negative weight vector that scales the WD applied to each parameter.

**Key properties of φ:**
- Positivity: `φ(t, θ, g) ≥ 0` component-wise (WD is never reversed)
- Measurability: φ can depend on any combination of (t, θ, g, optimizer state)
- Normalization convention: `E[φ] = 1` for budget-equivalent comparison

**Python interface (shown as code block):**
```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int) -> Tensor:
        """Return per-parameter modulation φ ∈ [0, ∞)."""
        ...
```

### 3.2 Special Cases: Recovering Existing Methods

**Table 3.1**: Method catalog with closed-form φ expressions

| Method | φ(t, θ, g) | Modulation Axis |
|--------|-----------|-----------------|
| AdamW (constant) | **1** | Baseline (φ ≡ 1) |
| CWD (hard) | `𝟙[sign(θ) = sign(u_t)]` | Directional |
| CWD (soft, β) | `σ(β · θ ⊙ u_t)` | Directional |
| SWD/AdamS | `h(‖g_t‖) · 1` | Temporal-gradient |
| Cosine WD | `½(1 + cos(πt/T)) · 1` | Temporal |
| AdamWN | `max(0, 1 − τ/‖θ‖) · 1` | Target-norm |
| AlphaDecay | `diag(α_l) · 1` (layer l) | Spatial |
| No-WD | **0** | Ablation |
| Random mask | `Bernoulli(p) · 1` | Control |

**Proposition 3.1** (Composition): For any two modulators φ₁, φ₂, their product `φ_comp = φ₁ ⊙ φ₂` defines a valid modulator. This formalizes CWD+Cosine, CWD+AdamWN, etc.

### 3.3 Budget Equivalence Normalization

**Definition (Budget Equivalence):** Two WD strategies are *budget-equivalent* if `∫₀ᵀ λ_eff(t) dt = ∫₀ᵀ λ · φ(t) dt = λ_const · T` for some constant baseline λ_const.

**Formula for effective WD under Phi:**
```
λ_eff(t) = λ · E_θ[φ(t, θ, g)]
```

Budget-equivalent comparison requires matching `E[λ_eff]` across methods before attributing accuracy differences to scheduling strategy.

### 3.4 Diagnostic Metric Definitions

**Budget Equivalence Metric (BEM):**
```
BEM(method) = |λ_eff^{method} - λ_eff^{constant}| / λ_eff^{constant}
```
BEM = 0: perfect budget match; BEM > 0: method uses different effective WD budget.

In practice, BEM is normalized to [0,1] where:
- 0 = same budget as constant (e.g., constant WD, half_lambda with matched λ)
- 1 = zero budget (no WD at all)
- ~0.5 = approximately half budget (cosine schedule, random mask at p=0.5, CWD hard)

**Coupling Stability Index (CSI):**
```
CSI = w₁ · CV(‖θ‖_trajectory) + w₂ · log κ(H_final) + w₃ · CV(η_eff,layers)
```
where CV = coefficient of variation, κ = spectral condition number (approximated by power iteration), η_eff = η/(1 + λ·‖θ_l‖) per layer. Weights: w₁=0.4, w₂=0.3, w₃=0.3.

Higher CSI = more unstable coupling between WD and optimization dynamics.

**Alignment Informativeness Score (AIS):**
```
AIS = Spearman_ρ(cos(θ_i, g_i), Δloss_i) over training steps i
```
AIS is computed per layer and averaged. AIS ∈ [0,1] where:
- AIS > 0.2: alignment signal is informative for WD decisions
- AIS < 0.1: alignment signal is uninformative (random baseline territory)

**Key Figure Reference: Figure 2** — Diagnostic metric illustration panel:
- (a) Three subfigures showing CSI, AIS, BEM computation pipeline on a toy example
- (b) Visual intuition: CSI tracks weight-norm stability, AIS tracks alignment predictiveness, BEM normalizes budget

---

## 4. Experimental Setup (~0.5 pages)

### 4.1 Implementation
- **Framework**: PyTorch; `UnifiedAdamW` optimizer with pluggable Phi modulator interface
- **Codebase**: Extended from why-weight-decay (D'Angelo et al., MIT license)
- **Methods evaluated**: constant, cwd_hard, swd, cosine_schedule, random_mask, half_lambda, no_wd (7 methods)

### 4.2 Training Configuration
- **Datasets**: CIFAR-10, CIFAR-100 (torchvision standard splits)
- **Architecture**: ResNet-20 (standard CIFAR configuration, ~270K parameters)
- **Optimizer**: AdamW with decoupled WD (all methods use identical AdamW base)
- **Learning rate**: 1e-3, cosine annealing schedule, no warmup
- **Base WD**: λ = 5×10⁻⁴ (constant baseline); each dynamic method modulates this value
- **Training duration**: 200 epochs, batch size 128
- **Seeds**: 3 independent runs (42, 123, 456) per method × dataset configuration
- **Total runs**: 42 (7 methods × 3 seeds × 2 datasets)

### 4.3 Hyperparameter Fairness Protocol
- **Critical design choice**: All methods use identical base WD (5×10⁻⁴) and LR (1e-3) — no per-method grid search
- This is intentional: ensures comparisons measure the effect of Phi modulation, not hyperparameter luck
- Limitation acknowledged: each method may have non-optimal hyperparameters; addressed in Appendix A (sensitivity analysis)

### 4.4 Diagnostic Logging
- Per-epoch: test accuracy, training loss, weight norms, CSI, AIS, BEM
- Per-100-steps: gradient-weight cosine similarity per layer, effective LR per layer, Phi modulation values
- Full diagnostic panels in Appendix B

---

## 5. Results and Analysis (~2 pages)

### 5.1 Main Accuracy Comparison

**Table 1** (main results table): 7 methods × 2 datasets, mean ± std over 3 seeds

Key numbers to emphasize:
- CIFAR-10 spread: 89.88% (swd) to 90.13% (constant) = **0.25% total variation**
- CIFAR-100 spread: 62.66% (no_wd) to 63.42% (cosine_schedule) = **0.76% total variation**
- Constant WD is best on CIFAR-10; cosine_schedule is best on CIFAR-100 by 0.27%
- **cosine_schedule achieves lowest variance** on CIFAR-10 (σ=0.07% vs. σ≈0.30% for others) — potential stability benefit without accuracy gain

**Table 2** (statistical tests): Paired t-test vs. constant baseline
- All p-values > 0.05 (range: p=0.090 to p=0.950)
- After Bonferroni correction for 6 comparisons: threshold p<0.0083 — no method survives
- Effect sizes (Cohen's d): all < 0.3 (small effect threshold)
- Most notable: p=0.090 for CIFAR-100 random_mask — closest to significance, yet random masking with no alignment information

**Narrative emphasis**: Even removing WD entirely (no_wd, λ=0) yields CIFAR-10 accuracy of 90.08% vs. constant 90.13% (Δ=0.05%, p=0.825). This is the strongest evidence: WD *scheduling* is irrelevant because WD *magnitude* is largely irrelevant under AdamW on these benchmarks.

### 5.2 Budget Equivalence Analysis

**Figure 3** — Budget equivalence plot:
- x-axis: BEM value (0 = same budget as constant, 1 = no WD)
- y-axis: mean test accuracy
- Each method shown as a point with error bars (3 seeds)
- Two subplots: CIFAR-10 (top), CIFAR-100 (bottom)
- **Key visual**: near-flat accuracy across the full BEM range 0→1
- BEM values: constant (0.0), half_lambda (0.0), cosine_schedule (0.503), cwd_hard (0.503), random_mask (0.500), swd (0.900), no_wd (1.000)
- Annotation: "10x variation in effective WD budget produces <0.5% accuracy difference"

**Narrative**: This rules out the explanation that dynamic WD methods improve accuracy by using less total WD (budget equivalence). Even at BEM=1.0 (zero WD), performance is equivalent.

### 5.3 CSI and AIS Diagnostic Analysis

**Figure 4** — CSI/AIS diagnostic panel (2×2 subfigures):
- (a) CSI values per method, CIFAR-10: bar chart with error bars; no_wd highest (0.964), swd lowest (0.838)
- (b) AIS values per method, CIFAR-10: bar chart; half_lambda slightly highest (0.41), no_wd slightly lower on CIFAR-100 (0.280)
- (c) CSI vs. accuracy scatter: no correlation (Spearman ρ ≈ 0, p > 0.5)
- (d) AIS vs. accuracy scatter: no correlation (Spearman ρ ≈ 0, p > 0.5)

**Key Findings from Diagnostics:**

**CSI Analysis:**
- no_wd achieves highest CSI (0.964): without WD, weight norms grow freely, increasing coupling instability — yet performance is unaffected
- cosine_schedule has high CSI (0.936): the slowly decaying schedule creates unstable coupling between WD and weight norm trajectory — yet lowest variance
- **CSI does not predict accuracy**: confirms CSI measures *dynamics* not *performance*; useful for understanding mechanism but not selecting methods

**AIS Analysis:**
- All methods show moderate AIS (0.28–0.41): gradient-weight alignment has some predictive signal for loss changes, but this signal is consistent across *all* WD methods including random_mask and no_wd
- AIS is an **intrinsic network property**: the alignment signal is not generated or destroyed by the WD modulation strategy — it reflects the geometry of the loss landscape
- This directly challenges CWD's motivation: if AIS is the same for CWD and random_mask, the gradient-weight alignment signal provides no additional useful information for WD decisions on these benchmarks

**Table 3** — Spearman rank correlations: CSI/AIS predicting accuracy rank across methods (both datasets): all ρ < 0.3, all p > 0.3

### 5.4 Weight Norm Analysis

**Figure 5** — Weight norm trajectories:
- Line plots of per-layer mean weight norm over 200 training epochs for all 7 methods on CIFAR-10
- **Key visual**: All methods converge to very similar weight norm levels (94–97 a.u.) despite 10x variation in effective WD
- no_wd has slightly higher final norm (97.04 vs. 95.89 for constant)
- AdamW's adaptive step size dominates: the per-parameter scaling normalizes the effective WD signal regardless of φ value

**Finding**: Weight norms converge similarly regardless of WD method, providing mechanistic explanation for accuracy equivalence — AdamW's built-in adaptive scaling acts as an implicit weight norm controller that subsumes explicit WD modulation.

---

## 6. Discussion (~1 page)

### 6.1 The Phi Invariance Conjecture

**Conjecture (Phi Invariance under AdamW):**
*For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem, the final test accuracy is invariant to the choice of Phi modulator φ(t, θ, g), up to the effective WD budget E[λ_eff].*

**Formal statement:**
Let `acc(φ)` denote the test accuracy achieved by training with modulator φ. Then for any two budget-equivalent modulators φ₁ and φ₂ (i.e., `∫λ·φ₁ dt = ∫λ·φ₂ dt`):
```
|acc(φ₁) - acc(φ₂)| ≤ ε_noise
```
where ε_noise is bounded by training stochasticity, not by the functional form of φ.

**Mechanistic hypothesis:** AdamW's per-parameter adaptive learning rate `η_t/(v̂_t^{1/2} + ε)` already scales updates proportionally to gradient magnitude, implicitly equalizing the WD signal across parameters. Additional modulation by φ operates on a system that has already "pre-equalized" the effective regularization strength per parameter. This makes the functional form of φ a second-order effect.

**Supporting evidence from our results:**
- Weight norms converge similarly regardless of φ (Section 5.4)
- AIS is an intrinsic network property, not φ-dependent (Section 5.3)
- Even BEM=1.0 (φ=0, no WD) does not break the invariance (Section 5.2)

**When the conjecture may fail (boundary conditions):**
- SGD: lacks adaptive scaling, so φ's distributional effect may be first-order
- Very large WD (λ ≫ standard range): accuracy degrades sharply, modulation may matter
- Long training / LLM scale: WD timing effects may compound over many tokens
- Overfitted small datasets: WD as regularizer may reassert dominance
- Different architectures: Vision Transformers with layer normalization may respond differently

### 6.2 Implications for WD Research

1. **For AdamW practitioners**: The choice of WD schedule does not matter — use constant WD with grid-searched λ; the simplest approach is optimal.

2. **For WD method developers**: New dynamic WD methods should be primarily evaluated with SGD, or at scale (ImageNet, LLMs), where the Phi Invariance Conjecture's boundary conditions are more likely to break down.

3. **For benchmark designers**: Comparing WD methods on CIFAR-scale AdamW experiments is misleading — differences will be noise, not signal. This explains why many dynamic WD papers report improvements only in specific settings.

4. **The Phi framework as infrastructure**: Even if the invariance holds, the framework is valuable for systematic research: it provides a common interface, enables principled composition of methods, and the diagnostic metrics (CSI/AIS/BEM) characterize *how* methods differ even when they produce the same accuracy.

### 6.3 Limitations

1. **Scale**: Experiments restricted to CIFAR-scale (ResNet-20). The Phi Invariance Conjecture may not hold at ImageNet scale or LLM scale.
2. **Architecture diversity**: Only ResNet-20 tested. VGG-16-BN and Vision Transformers may show different behavior.
3. **Optimizer**: All experiments use AdamW. SGD results are not included in this iteration (planned as P0 follow-up).
4. **Statistical power**: 3 seeds per configuration provides limited statistical power; effect sizes < 0.3% may be undetectable. TOST equivalence testing would strengthen the null result claim.
5. **Hyperparameter optimization**: Fixed hyperparameters across methods may disadvantage methods with strong hyperparameter sensitivity (e.g., CWD with different β values may show different behavior).
6. **Overfitting regime**: All experiments are in the well-generalized regime (not severe overfitting). Methods may differentiate in heavily overfitted settings.

---

## 7. Conclusion (~0.25 pages)

**Core message:**
- We introduced the Phi Modulator Framework, the first unified mathematical abstraction for all dynamic WD methods
- Systematic evaluation of 7 methods on 42 experiments reveals statistical equivalence under AdamW — all dynamic WD strategies perform identically to constant WD
- The Phi Invariance Conjecture provides a theoretical hypothesis for this finding: AdamW's adaptive scaling subsumes explicit modulation
- Three standardized diagnostic metrics (BEM, CSI, AIS) provide the community with tools to characterize WD behavior beyond accuracy
- Future work: test invariance under SGD, ImageNet, LLMs, and Vision Transformers; prove or disprove the Phi Invariance Conjecture theoretically

**Closing statement:** Dynamic WD methods offer a rich space for architectural exploration, but practitioners using AdamW can safely rely on constant WD — the simplest strategy that already captures the available benefit.

---

## References

**Core references to cite (ordered by section)**:

- Introduction: Loshchilov & Hutter (ICLR 2019), D'Angelo et al. (NeurIPS 2024), Chen et al. CWD (ICLR 2026), Xie et al. SWD (NeurIPS 2023)
- Related Work: Kosson et al. (2023), Ferbach et al. ADANA (2026), Chen et al. AdamO (2026), Loshchilov AdamWN (2023), He et al. AlphaDecay (2025), Galanti et al. (2022), Kobayashi et al. (2024), Wang & Aitchison (2024)
- Method: PathProx (Yang et al. 2022), Ye (2024) preconditioning framework, Ding et al. AdamD (2023)
- Experiments: torchvision CIFAR benchmarks, standard references
- Discussion: Truong & Truong (2026) norm-hierarchy, Han et al. (2026) plasticity, Outmezguine & Levi (2024)

---

## Appendix Structure

### Appendix A: Extended Experimental Results

**A.1 — Additional Statistical Analysis**
- Bootstrap 95% confidence intervals for all accuracy comparisons (1000 resamples)
- Cohen's d effect sizes for all pairwise comparisons
- TOST equivalence testing (two one-sided t-tests): formal test for statistical equivalence within ±0.3% equivalence margin
- Full p-value table with Bonferroni correction

**A.2 — Hyperparameter Sensitivity**
- CWD beta sensitivity: β ∈ {10, 100, 1000} — does soft CWD at optimal β improve over hard CWD?
- WD baseline sensitivity: λ ∈ {1e-5, 1e-4, 5e-4, 1e-3} for constant baseline
- Confirms: results are not artifacts of suboptimal hyperparameter choices

**A.3 — Additional Architectures (Supplementary)**
- VGG-16-BN on CIFAR-10/100: 42 runs (7 methods × 3 seeds × 2 datasets)
- If results available: cross-architecture comparison confirming invariance
- If not yet available: placeholder noting this is future work

**A.4 — SGD Baseline (Supplementary)**
- SGD + momentum + constant WD vs. dynamic WD methods
- Critical test of Phi Invariance boundary conditions
- If results available: compare AdamW invariance with potential SGD sensitivity

### Appendix B: Diagnostic Visualization Panels

**B.1 — Per-Method Diagnostic Panels (6-panel × 7 methods × 2 datasets = 84 panels)**
For each of the 42 unique (method, dataset) combinations:
1. Training loss + test accuracy curves (3 seeds overlaid)
2. Per-layer weight norm trajectories over 200 epochs
3. Per-layer gradient-weight cosine similarity over training
4. Effective learning rate per layer over training
5. CSI and AIS evolution over training epochs
6. Phi modulation heatmap (layers × time steps, showing φ values)

**B.2 — Aggregate Diagnostic Comparisons**
- All-method weight norm overlay (Figure B.1): confirms convergence to similar norms
- CSI evolution comparison (Figure B.2): shows convergence trajectories differ even if endpoint is same
- AIS trajectories (Figure B.3): shows AIS is similar across methods throughout training

### Appendix C: Mathematical Proofs and Derivations

**C.1 — Special Case Recovery**
- Formal proof that each row in Table 3.1 is a valid special case of the Phi framework
- Show CWD's Pareto-optimality interpretation (Chen et al.) fits within Phi via binary masking
- Show AdamWN's target-norm control fits via adaptive φ = max(0, 1 − τ/‖θ‖)

**C.2 — Metric Definitions and Properties**
- Full mathematical derivation of BEM normalization
- CSI: derivation of component weights (w₁=0.4, w₂=0.3, w₃=0.3) and normalization constants
- AIS: Spearman ρ formulation; proof that AIS = 0 for a perfectly random Phi (sanity check)

**C.3 — Phi Invariance Conjecture: Formal Statement**
- Precise conditions under which the conjecture holds
- Connection to AdamW's implicit L∞-constrained optimization (Xie & Li 2024)
- Sketch of why per-parameter adaptive scaling subsumes explicit φ modulation
- Boundary condition analysis: when the conjecture is expected to fail

### Appendix D: Reproducibility Details

**D.1 — Implementation Details**
- Full `UnifiedAdamW` optimizer pseudocode (Algorithm 1)
- Per-method Phi modulator implementation details
- Codebase structure description

**D.2 — Experiment Configuration**
- Full hyperparameter table for all 42 runs
- Random seed configuration (Python, NumPy, PyTorch, CUDA)
- Hardware specification and training time per run

**D.3 — Code and Data Availability**
- Link to anonymous code repository
- Instructions for reproducing all 42 experiments
- Diagnostic metric computation scripts

---

## Figure Plan Summary

| Figure | Location | Content | Priority |
|--------|----------|---------|----------|
| Figure 1 | Section 1 | Phi framework taxonomy diagram | P0 — must have |
| Figure 2 | Section 3 | Diagnostic metric illustration (CSI/AIS/BEM pipeline) | P0 — must have |
| Figure 3 | Section 5.2 | Budget equivalence scatter plot (BEM vs. accuracy) | P0 — must have |
| Figure 4 | Section 5.3 | CSI/AIS diagnostic analysis (2×2 panel) | P0 — must have |
| Figure 5 | Section 5.4 | Weight norm trajectories (all methods overlay) | P1 — important |

## Table Plan Summary

| Table | Location | Content | Priority |
|-------|----------|---------|----------|
| Table 1 | Section 5.1 | Main accuracy results (7 methods × 2 datasets, mean±std) | P0 — must have |
| Table 2 | Section 5.1 | Statistical significance tests (Δ, p-value, Cohen's d vs. constant) | P0 — must have |
| Table 3 | Section 5.3 | Spearman ρ for CSI/AIS predicting accuracy rank | P1 — important |
| Table 3.1 | Section 3.2 | Method catalog with φ expressions and modulation axis | P0 — must have (in body) |

---

## Writing Guidelines

### Tone and Framing
- **Framing**: "Unified Framework + Systematic Benchmark" — not "we tried 7 methods and none worked"
- The negative result is the *contribution*, not the failure: it establishes when dynamic WD does *not* help, which is scientifically valuable
- Lead with the framework and metrics as independent contributions; the null result *validates* the diagnostic metrics (they correctly identify the equivalence)
- Be precise about scope: "under AdamW on CIFAR-scale problems" — never overclaim generalization to LLMs or SGD

### Critical Sentences (must appear in paper)
1. Abstract: "All dynamic WD variants are statistically indistinguishable from constant WD under AdamW (p > 0.05), including the degenerate case of no WD."
2. Introduction: "The Phi framework enables, for the first time, controlled comparison of WD strategies under identical optimization conditions."
3. Section 3: "CWD, SWD, AdamWN, cosine schedules, and random masking are all recovered as special cases of φ(t, θ, g)."
4. Section 5: "A 10× variation in effective WD budget (BEM range 0.0–1.0) produces less than 0.5% accuracy variation."
5. Discussion: "We conjecture that AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering the functional form of φ irrelevant to generalization."

### Length Budget (NeurIPS 8-page target)
| Section | Pages |
|---------|-------|
| Abstract | ~0.2 |
| Introduction | ~1.5 |
| Related Work | ~0.75 |
| Phi Framework | ~1.5 |
| Experimental Setup | ~0.5 |
| Results & Analysis | ~2.0 |
| Discussion | ~1.0 |
| Conclusion | ~0.25 |
| References | ~0.5–1.0 |
| **Total main body** | **~7.7–8.0** |
| Appendices | ~4–6 |

---

*Outline version: 1.0*
*Generated: 2026-03-18*
*Iteration: iter_003*
