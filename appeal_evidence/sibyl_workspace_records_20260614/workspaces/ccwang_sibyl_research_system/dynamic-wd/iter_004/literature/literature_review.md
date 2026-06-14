# Literature Survey Report — Iteration 4

**Research Topic**: Unified Dynamic Weight Decay Framework — Unifying WD Scheduling, Alignment-Aware WD, Decoupled WD, and Norm-Matched WD into a Theoretical Framework with Standardized Evaluation Metrics (BEM, CSI, AIS)
**Survey Date**: 2026-03-18 (Iteration 4 Update)
**Focus**: Gaps identified in Iteration 3 review + 2025-2026 new papers

---

## 1. Previously Missing References (Now Resolved)

### 1.1 Wilson et al. 2017 — Implicit Regularization of Adaptive Methods

**Citation**: Wilson, A.C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. (2017). *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NIPS 2017, 4148–4158. arXiv: [1705.08292](https://arxiv.org/abs/1705.08292)

**Key Contribution**: Empirical and theoretical evidence that adaptive gradient methods (Adam, AdaGrad, RMSProp) generalize *worse* than SGD with momentum on CIFAR-10, character-level language modeling, and Penn Treebank parsing — even when adaptive methods achieve lower training loss. Constructs a binary classification problem where Adam finds solutions with test error close to 50% while SGD achieves 0%. The core mechanism is that adaptive methods give undue influence to spurious features with no out-of-sample effect.

**Relevance to Iter 4**: This is the foundational reference explaining *why* dynamic WD under AdamW may behave differently than under SGD. The Phi Invariance Conjecture (null result under AdamW) should be compared against this line of work: if adaptive methods already suppress sensitivity to WD perturbations via their internal gradient scaling, this would mechanistically explain the conjecture. Direct citation needed in Discussion section (currently missing).

**Limitation**: Published 2017, predates the AdamW era. The generalization gap has since narrowed significantly, especially for Transformers where AdamW dominates. The Iteration 3 data showing CWD under SGD achieves +0.77% gain while no gain under AdamW is consistent with this framing.

---

### 1.2 van Laarhoven 2017 — L2 vs WD under Batch/Weight Normalization

**Citation**: van Laarhoven, T. (2017). *L2 Regularization versus Batch and Weight Normalization.* arXiv: [1706.05350](https://arxiv.org/abs/1706.05350)

**Key Contribution**: Shows that L2 regularization loses its regularizing effect when combined with Batch Normalization (BN), Weight Normalization, or Instance Normalization because these make the function invariant to the scale of weights. Instead, L2 (and WD) acts as a **scale controller** that indirectly modulates the effective learning rate (ELR). First paper to formally derive the ELR hypothesis. Key finding: weight decay *continuously decreases parameter scales*, thereby *increasing the scale of batch gradients*, which effectively increases the ELR. Adam partially (but not fully) eliminates normalization's influence on the learning rate.

**Relevance to Iter 4**: Provides foundational theoretical grounding for why WD acts as a training dynamics modifier rather than an explicit regularizer — the position taken by D'Angelo et al. (NeurIPS 2024). The Phi Modulator Framework's claim that dynamic WD is invariant under AdamW needs to engage with this: if WD is just an ELR modulator, and Adam already has its own ELR mechanism (adaptive scaling), then the Phi modulation may indeed be subsumed. This is exactly the quantitative mechanistic argument that the Iteration 3 reviewer (M6) requested but found missing.

**BibTeX**:
```bibtex
@article{van_laarhoven2017l2,
  title  = {L2 Regularization versus Batch and Weight Normalization},
  author = {Twan van Laarhoven},
  year   = {2017},
  eprint = {1706.05350},
  archivePrefix = {arXiv}
}
```

---

### 1.3 Kosson et al. 2023/2024 — Rotational Equilibrium

**Citation**: Kosson, A., Messmer, B., & Jaggi, M. (2024). *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks.* ICML 2024, pp. 25333–25369. arXiv: [2305.17212](https://arxiv.org/abs/2305.17212). First submitted May 2023; NeurIPS 2023 Workshop Math of Modern ML.

**Key Contribution**: Formalizes how WD and gradient updates on parameter vectors can reach a **rotational equilibrium** — a steady state where their expected effects on magnitude cancel, causing the angular update (effective LR) to remain constant. This equilibrium can be highly homogeneous across layers and neurons, explaining AdamW's advantage over Adam+L2 (where the coupling distorts this equilibrium). Demonstrates that explicitly controlling rotation can eliminate the warmup transient and reduce warmup dependency.

**Status**: Already correctly cited in Iteration 3 literature.md as reference #6. Note: The correct venue is **ICML 2024** (not just NeurIPS 2023 Workshop). The BibTeX should reference ICML Proceedings, volume 235.

---

### 1.4 Loshchilov & Hutter 2019 — AdamW

**Citation**: Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv: [1711.05101](https://arxiv.org/abs/1711.05101). OpenReview: [Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7).

**Status**: Already correctly cited in Iteration 3 literature.md as reference #1. Full BibTeX:
```bibtex
@inproceedings{loshchilov2019adamw,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Ilya Loshchilov and Frank Hutter},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  url       = {https://openreview.net/forum?id=Bkg6RiCqY7}
}
```

---

## 2. New References for Iteration 4

### 2.1 TOST Equivalence Testing Methodology

**Citation**: Lakens, D. (2017). *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses.* Social Psychological and Personality Science, 8(4), 355–362. DOI: [10.1177/1948550617697177](https://journals.sagepub.com/doi/full/10.1177/1948550617697177)

**Key Contribution**: Establishes the Two One-Sided Tests (TOST) procedure as the standard method for providing evidence of practical equivalence (not merely failing to reject null). In TOST, upper (ΔU) and lower (−ΔL) equivalence bounds are set; when both one-sided null hypotheses (Δ ≤ −ΔL and Δ ≥ ΔU) are rejected, we conclude the effect is within equivalence bounds. Key difference from p > 0.05: TOST *affirmatively demonstrates* absence of meaningful effect, whereas standard NHST only *fails to detect* one.

**Relevance to Iter 4**: The Iteration 3 reviewer (M3) explicitly demanded TOST equivalence testing to support the Phi Invariance Conjecture null result. With n=3 seeds, statistical power is insufficient even for TOST — minimum n≈10-20 for the typical 0.1-0.5% accuracy effect sizes in CIFAR experiments. Iteration 4 must either (a) increase seeds to n≥5 and implement TOST, or (b) explicitly justify the decision to rely on Cohen's d and Bonferroni correction instead of TOST.

**Implementation**: scipy.stats provides t-test infrastructure; pingouin library has direct tost() function. R TOSTER package is the reference implementation.

**BibTeX**:
```bibtex
@article{lakens2017equivalence,
  title   = {Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses},
  author  = {Daniël Lakens},
  journal = {Social Psychological and Personality Science},
  volume  = {8},
  number  = {4},
  pages   = {355--362},
  year    = {2017},
  doi     = {10.1177/1948550617697177}
}
```

---

### 2.2 Gradient-Aware AdamW for Vision Transformers (2025)

**Citation**: Anonymous Authors (2025). *Adaptive Adam-based Optimizers Using Second-Order Weight Decoupling and Gradient-Aware Weight Decay for Vision Transformer.* Machine Vision and Applications. DOI: [10.1007/s00138-025-01686-9](https://link.springer.com/article/10.1007/s00138-025-01686-9)

**Key Contribution**: Proposes adaptive Adam-based optimizers specifically targeting Vision Transformers, combining second-order weight decoupling (separate treatment of second-order moment and WD) with gradient-aware weight decay (modulating WD strength based on gradient properties). Validates on ImageNet classification with various ViT architectures.

**Relevance to Iter 4**: This 2025 paper is directly relevant to the ViT gap identified in Iteration 3 reviews. Demonstrates that gradient-aware WD *does* provide measurable gains for ViT on ImageNet — potentially contradicting the Phi Invariance Conjecture if the AdamW subsumption argument doesn't hold for Transformers with second-order decoupling.

**Risk Assessment**: Partial overlap with the Alignment-Aware WD component of our framework. However, this work is ViT-specific and lacks a unified theoretical framework — it does not address the BEM/CSI/AIS standardization contribution or the cross-optimizer comparisons.

---

### 2.3 Straight to Zero: Linear Decay-to-Zero for LLMs (2025)

**Citation**: (Cerebras, 2025). *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs.* arXiv: [2502.15938](https://arxiv.org/abs/2502.15938). Feb 2025 (updated Nov 2025).

**Key Contribution**: Large-scale empirical study showing that linear decay-to-zero (D2Z) LR schedule consistently outperforms cosine decay across model sizes 60M–1B. Critically, the paper derives this result by interpreting AdamW as an EMA of weight updates — the same EMA-timescale framing as Wang & Aitchison (2024). D2Z optimally balances early training dynamics and late gradient averaging. A 610M model with D2Z for 80 TPP achieves lower loss than 200 TPP with cosine — 60% compute savings.

**Relevance to Iter 4**: The paper demonstrates that **WD scheduling interacts strongly with LR schedule choice** — specifically, the D2Z result was derived assuming fixed WD. This is a key interaction variable that our Iteration 3 experiments did not vary. Adding D2Z as a LR schedule baseline would significantly strengthen the experimental design.

**New Finding for Framework**: Under D2Z interpretation, WD appears as the reciprocal of the EMA window length (timescale). Dynamic WD then corresponds to dynamic EMA window — which has a natural scheduling interpretation. This strengthens the WD-Scheduling sub-framework with a new mechanistic motivation.

---

### 2.4 Universal Dynamics of WSD — WD Interaction During Cooldown (2026)

**Citation**: (Jan 2026). *Universal Dynamics of Warmup Stable Decay.* arXiv: [2601.09000](https://arxiv.org/pdf/2601.09000)

**Key Contribution**: Establishes that **switching off weight decay during high-variance LR cooldown shapes** (square, mirror cosine) can significantly improve performance. The WSD (Warmup-Stable-Decay) scheduler combined with no WD during cooldown acts as an additional decaying factor for ELR, improving the bias-variance tradeoff. This is a universal phenomenon across transformers and CNNs.

**Relevance to Iter 4**: Directly contradicts the naive assumption that constant WD = baseline for comparison. If WD should be turned *off* during cooldown phases, then even "constant WD" baselines are suboptimal, and the Phi Invariance Conjecture baseline might not represent optimal constant WD behavior.

**Implication for Experimental Design**: Iteration 4 experiments should test WD=0 during the final 10-20% of training as a potentially stronger baseline.

---

### 2.5 Near-Optimal LR Schedules — WD Interaction (2026)

**Citation**: Naganuma et al. (2026). *What Do Near-Optimal Learning Rate Schedules Look Like?* arXiv: [2603.10301](https://arxiv.org/abs/2603.10301)

**Key Contribution**: Most comprehensive study of optimal LR schedule shapes across many functional families and model scales. Key finding: **once a schedule has both warmup and decay, the base learning rate is far more important than the specific schedule shape**. Critically also shows that **weight decay has a strong effect on the optimal schedule shape** — WD and LR schedule are fundamentally co-determined, not independent hyperparameters.

**Relevance to Iter 4**: Validates the approach of studying WD scheduling jointly with LR scheduling, and provides strong empirical support that WD-LR interaction is real and non-trivial. The BEM should account for this co-dependence when normalizing compute budgets.

**Already cited** in Iteration 3 literature.md as reference #25 — no change needed.

---

## 3. Weight Decay in Vision Transformers — Gap Analysis

The Iteration 3 reviewer explicitly noted missing references on ViT-specific WD behavior. Here is a synthesis:

### 3.1 Layer-wise WD in ViTs
- **DeiT III (arXiv:2204.07118)**: Shows benefits of layer-wise learning rate decay with WD=0.05 for ViT ImageNet training. Implies layer-specific WD calibration is standard practice.
- **AugReg (arXiv:2106.10270)**: WD 0.03–0.10 for ViT pre-training; regularization hurts when switching to larger datasets at fixed compute budget.
- **SAM + ViT (arXiv:2106.01548)**: Weight norms increase during ViT training, making standard WD insufficient; proposes sharpness-aware alternatives.

### 3.2 Low-Rank Effect in ViT Attention
- **Kobayashi et al. 2024 (NeurIPS 2024, arXiv:2410.23819)**: L2 regularization on attention layers (multiplicative parameters $W_K^T W_Q$ and $PW_V$) is equivalent to nuclear norm regularization, inducing low-rank attention. This can damage LM performance. Recommends **decoupling WD for attention layers** from body parameters.
  - Already cited as reference #21 in Iteration 3 literature.md.

### 3.3 Adaptive ViT-specific WD (2025, NEW)
- **Springer 2025** (DOI: 10.1007/s00138-025-01686-9): Second-order WD decoupling + gradient-aware WD for ViT. Shows measurable gains on ImageNet. First dedicated ViT-adaptive-WD paper.

**Summary for Iter 4**: The ViT gap is partially filled by the existing Kobayashi et al. citation, but the new Springer 2025 paper provides a direct 2025 competitor in the gradient-aware WD for ViT space. Recommended to cite in related work section.

---

## 4. Implicit Regularization Literature — Gap Analysis

### 4.1 SGD Implicit Regularization
- **Smith & Le 2018 (ICLR 2018, arXiv:1710.11029)**: SGD implicit regularization modeled as variational inference; iterates follow closed loops, not Brownian motion.
- **Damian et al. 2021 (arXiv:2101.12176)**: SGD with random shuffling follows gradient flow on a *modified* loss — the modification is an implicit regularizer proportional to the norm of minibatch gradients at scale `learning_rate / batch_size`.
- **Smith et al. 2024 (arXiv:2406.11110)**: "Neural networks learn the support" — weights interacting with irrelevant data components converge to zero *if and only if weight decay is added*. This shows WD is necessary for a specific implicit regularization effect, contradicting the pure "dynamics modifier" framing of D'Angelo et al.

### 4.2 AdamW Implicit Regularization (Critical for Iter 4)
- **Xie & Li 2024 (arXiv:2404.04454)**: AdamW implicitly performs ℓ∞-constrained optimization (KKT point interpretation via Frank-Wolfe). This directly explains why dynamic WD might be subsumed: AdamW already uses a specific implicit constraint geometry (ℓ∞), so WD perturbations that don't change the geometry of the constraint are indeed invisible.
  - Already cited as reference #12 in Iteration 3 literature.md.
  - **Mechanistic Argument for Iter 4**: The reviewer M6 asked for a "quantitative order-of-magnitude analysis: ratio of Phi perturbation to adaptive gradient step." The ℓ∞ implicit constraint provides the exact mechanism: Phi modulations that stay within the ℓ∞ constraint boundary are subsumed; those that would change the constraint boundary are the only ones that matter. This is the theoretical argument that should be added.

---

## 5. Field Trends 2025-2026: Dynamic WD

### 5.1 Evidence For Dynamic WD Helping
1. **SGD context**: CWD gains +0.77% on CIFAR-10 under SGD (existing Iter 3 data)
2. **LLM training**: AlphaDecay (module-wise) shows consistent gains for transformer pre-training
3. **ViT training**: Springer 2025 gradient-aware WD shows ImageNet gains
4. **LR cooldown interaction**: Turning off WD during final cooldown phase improves WSD schedules (arXiv:2601.09000)
5. **Scale effects**: Kosson et al. 2025 (arXiv:2510.19093) shows WD matters more than µP for LR transfer

### 5.2 Evidence For Dynamic WD NOT Helping (Supporting Phi Invariance)
1. **Our Iter 3 data**: All 14 dynamic WD variants indistinguishable under AdamW on CIFAR-10/100/ResNet-20
2. **D'Angelo et al. 2024 (NeurIPS)**: WD is never useful as explicit regularizer; it's all dynamics modification
3. **AdamW ℓ∞ implicit bias (Xie & Li 2024)**: AdamW subsumes WD perturbations within ℓ∞ constraint
4. **Wang & Aitchison EMA interpretation**: Optimal EMA timescale constant in epochs → static WD may be optimal if per-epoch budget is constant

### 5.3 Resolution Hypothesis for Iter 4
The evidence is consistent with: **Dynamic WD helps under SGD and in large-scale/LLM settings, but not under AdamW at small-to-medium scale**. The invariance may be scale-dependent (ResNet-20 vs. ResNet-50 vs. LLM) and optimizer-dependent (SGD vs. AdamW). Testing this is the central empirical question for Iteration 4.

---

## 6. SOTA Benchmarks and Experimental Conditions

| Setting | Architecture | Dataset | Key WD Finding |
|---------|-------------|---------|---------------|
| ResNet-20 | ResNet-20 | CIFAR-10/100 | All dynamic WD variants equivalent under AdamW (Iter 3) |
| VGG-16-BN | VGG-16-BN | CIFAR-10/100 | Unknown — needed for Iter 4 |
| ResNet-50 | ResNet-50 | ImageNet-1K | Unknown — needed for Iter 4 |
| ViT-S/B | ViT-S/B | CIFAR-100/ImageNet | Unknown — partially studied in Springer 2025 |
| LLM 60M-1B | GPT/LLaMA | C4/OWT | Dynamic WD helps (AlphaDecay, ADANA) |

---

## 7. Implications for Iteration 4 Paper

### Critical References to Add to Paper (from Review M7)
1. **Wilson et al. 2017** (arXiv:1705.08292) — Discussion section on SGD vs Adam implicit regularization
2. **van Laarhoven 2017** (arXiv:1706.05350) — Related Work section on WD as ELR modifier
3. **Lakens 2017** (SPPS 8(4)) — Statistical methods section for TOST
4. **Kobayashi et al. 2024** (arXiv:2410.23819) — Related Work for ViT-specific WD effects (already cited but venue is NeurIPS 2024)

### New Experiments to Design for Iter 4
Based on reviewer priorities:
1. **VGG-16-BN experiments** on CIFAR-10/100 (Priority 1)
2. **SGD baseline** — report CWD vs no_wd under SGD (existing data, just needs to be written up)
3. **ResNet-50 on ImageNet** — 90-epoch or shorter ablation (Priority 1, compute-intensive)
4. **More seeds** (n≥5 for key configs, n=10 for main comparisons if compute allows)
5. **TOST equivalence testing** using pingouin.tost() with bounds ±0.5% for CIFAR
6. **WD=0 during final cooldown** as a new experimental condition (informed by arXiv:2601.09000)

### New Theoretical Content
1. Connect AdamW ℓ∞ implicit bias (Xie & Li 2024) to Phi Invariance — show that Phi modulations within a norm ball are absorbed
2. Quantitative order-of-magnitude analysis: Phi perturbation magnitude vs. Adam adaptive step size
3. Prove BEM boundedness formally (fix existing error)
4. Add TOST methodology description with bounds justification

---

## 8. References Summary for New Citations JSON

All new/corrected references are documented in `new_references.json`.
