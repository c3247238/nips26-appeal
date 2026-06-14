# Novelty Check Report — Iteration 4

**Date**: 2026-03-18
**Checked by**: Sibyl Literature Researcher Agent
**Topic**: Unified Dynamic Weight Decay Framework (Phi Modulator Framework)

---

## 1. Potential Novelty Conflicts — Direct Competitors

### 1.1 Ye 2024 — "Preconditioning for Optimization and Regularization" (arXiv:2410.00232)

**Status**: Already cited in Iteration 3 literature.md (reference #28)

**Overlap Assessment**: HIGH OVERLAP in spirit, LOW overlap in execution.

- Ye (2024) derives a "unified framework" in which AdamW selects intrinsic parameters for regularization, unifying L1/L2 regularization analogues and normalization methods via a preconditioning lens.
- Our Phi Modulator Framework approaches the same space differently: we unify *dynamic scheduling strategies* (temporal, directional, spatial, target-norm axes) rather than *regularizer types*.
- Ye focuses on *what kind* of regularization AdamW performs implicitly; we focus on *how* to dynamically adapt WD strength along four axes.
- **Differentiation**: Our framework is orthogonal — Ye's is about implicit regularization geometry, ours is about explicit modulation strategies. Citation recommended in Related Work.

### 1.2 D'Angelo et al. 2024 — "Why Do We Need Weight Decay?" (arXiv:2310.04415)

**Status**: Already cited in Iteration 3 literature.md (reference #2)

**Overlap Assessment**: MEDIUM overlap — this paper provides a "unified perspective" on WD as dynamics modifier for both vision and LLM settings.

- D'Angelo et al. offer a conceptual unification (SGD: loss stabilization; LLM Adam: bias-variance tradeoff) but do not propose a formal framework or standardized metrics.
- They argue against WD as explicit regularizer but do not propose how to *dynamically adapt* WD.
- **Differentiation**: Our Phi framework formalizes the *mathematical structure* of dynamic WD strategies and proposes diagnostic metrics; D'Angelo et al. provide qualitative conceptual unification. We explicitly build on their perspective and extend it with quantitative formalism.

### 1.3 AlphaDecay 2025 — Module-wise WD Framework (arXiv:2506.14562)

**Status**: Already cited in Iteration 3 literature.md (reference #5)

**Overlap Assessment**: LOW-MEDIUM overlap.

- AlphaDecay claims to be "the first work to formally investigate and establish a framework for module-level weight decay scheduling in LLMs."
- This partially overlaps with our Spatial Modulation axis (φ_S). However, AlphaDecay is LLM-specific, uses static HT-SR spectral density (not per-iteration gradient-alignment), and does not address BEM/CSI/AIS metrics.
- **Differentiation**: Our framework is vision + LLM agnostic, provides dynamic (per-iteration) modulation, and focuses on standardized evaluation. AlphaDecay is a specific *instantiation* of the Spatial Modulation axis in our framework.

---

## 2. Potential Novelty Conflicts — Phi Invariance Conjecture

### 2.1 "Dynamic WD doesn't help under AdamW" — Prior Work

**Assessment**: No prior paper has explicitly stated or empirically demonstrated the Phi Invariance Conjecture in a systematic way.

- D'Angelo et al. 2024 argue WD is not useful as explicit regularizer, but this is a different claim (about WD itself, not dynamic vs. constant WD).
- The EMA timescale analysis (Wang & Aitchison 2024) implies optimal static WD exists, suggesting dynamic variation may not help — but this is not tested.
- The ℓ∞ implicit bias (Xie & Li 2024) provides a mechanistic reason for subsumption but does not explicitly test the conjecture.
- **Novelty Status**: The Phi Invariance Conjecture as a formal, empirically-tested, falsifiable statement is genuinely novel. However, it needs stronger evidence (more architectures, ImageNet) to be publishable.

---

## 3. Potential Novelty Conflicts — BEM / CSI / AIS Metrics

### 3.1 Budget Equivalence Metric (BEM)

**Assessment**: No prior standardized metric for normalizing compute budget across WD method comparisons.

- Wang & Aitchison 2024 discusses EMA timescale as a way to compare WD across scales, but this is a scaling rule, not a diagnostic metric.
- OUI (arXiv:2504.17160) is a diagnostic tool for overfitting detection, not compute normalization.
- **Novelty Status**: BEM fills a genuine gap. However, the Iteration 3 reviewer identified a mathematical error (BEM can exceed [0,1]; half_lambda BEM computation bug). These must be fixed.

### 3.2 Coupling Stability Index (CSI)

**Assessment**: No prior work defines a composite metric for WD-optimizer coupling stability.

- Individual components (weight norm CV, spectral condition number, effective LR CV) appear in individual papers but have never been combined into a single index.
- **Novelty Status**: Genuine novel metric, but requires CSI component normalization and weight justification (unresolved from Iter 3 review).

### 3.3 Alignment Informativeness Score (AIS)

**Assessment**: No prior work defines an alignment informativeness metric.

- CWD (Chen et al., ICLR 2026) uses binary sign-alignment; AdamO uses continuous radial/tangential decomposition.
- Neither proposes a metric quantifying *how informative* alignment is for WD decisions.
- **Novelty Status**: Genuine novel metric, but requires AIS range correction ([-1, 1] not [0, 1]).

---

## 4. Scan for New 2025-2026 Competitors (Not in Iter 3)

### 4.1 Springer 2025 — Adaptive Adam + Gradient-Aware WD for ViT

**Paper**: *Adaptive Adam-based Optimizers Using Second-Order Weight Decoupling and Gradient-Aware Weight Decay for Vision Transformer.* Machine Vision and Applications (2025). DOI: 10.1007/s00138-025-01686-9

**Overlap**: Gradient-aware WD (similar to Alignment-Aware WD axis in our framework) for ViT on ImageNet.

**Risk Level**: LOW. This paper is narrowly focused on ViT + gradient-aware, without unified framework, standardized metrics, or null-result analysis. Our framework is broader and more rigorous.

### 4.2 ADANA 2026 — Logarithmic-Time WD Scheduling

**Paper**: Ferbach et al. (2026). arXiv:2602.05298. (Already in Iter 3 literature.md #17)

**Overlap**: WD scheduling with log-time schedules.

**Risk Level**: LOW. ADANA focuses on schedule shape optimality for LLMs; we focus on dynamic adaptation based on gradient statistics. Different axis of the framework.

### 4.3 AdamO 2026 — Decoupled Orthogonal Dynamics

**Paper**: Chen, Yuan, Zhang (2026). arXiv:2602.05136. (Already in Iter 3 literature.md #9)

**Overlap**: Radial-tangential decomposition is essentially an alignment-aware WD strategy.

**Risk Level**: MEDIUM-LOW. AdamO is the closest competitor in the Alignment-Aware WD axis. However, AdamO focuses on optimizer design (decoupling the update rule) rather than modulator analysis. Our framework *analyzes* AdamO's phi_D as a special case; AdamO doesn't provide the unified framework or evaluation metrics.

### 4.4 Kosson et al. 2025 — WD > muP for LR Transfer

**Paper**: arXiv:2510.19093. (Already in Iter 3 literature.md #16)

**Overlap**: WD's role in training dynamics at scale.

**Risk Level**: LOW. Different focus (hyperparameter transfer vs. dynamic WD strategy evaluation).

---

## 5. Overall Novelty Assessment

| Contribution | Status | Risk Level | Action Needed |
|---|---|---|---|
| Phi Modulator Framework (formal unification) | Novel | LOW | Strengthen with convergence theory |
| Phi Invariance Conjecture (null result) | Novel | LOW | Expand experimental scope (ImageNet, SGD) |
| Budget Equivalence Metric (BEM) | Novel | LOW | Fix mathematical errors |
| Coupling Stability Index (CSI) | Novel | LOW | Normalize components, justify weights |
| Alignment Informativeness Score (AIS) | Novel | LOW | Fix range claim |
| Systematic visualization framework | Novel | LOW | Add diagnostic panels |

**Conclusion**: No direct competitor paper has been found that proposes a unified mathematical framework for all four WD sub-approaches with standardized evaluation metrics. The closest competitors are D'Angelo et al. 2024 (conceptual unification) and Ye 2024 (preconditioning lens), but neither covers the full scope of our framework. **The novelty case is strong**, contingent on fixing the metric errors and expanding the experimental evidence.

**Most Important Gap to Watch**: The combination of Springer 2025 (gradient-aware ViT WD) + ADANA 2026 (log-time WD scheduling) + AdamO 2026 collectively covers parts of the framework's Alignment-Aware and Scheduling axes. If any paper synthesizes these in the next 6 months, novelty could be at risk. We should accelerate publication.

---

## 6. Search Coverage Confirmation

**Searched**: arXiv, Google Scholar, Semantic Scholar, Springer, NeurIPS/ICML/ICLR proceedings
**Keywords used**:
- "unified framework weight decay"
- "dynamic weight decay scheduling adaptive 2025 2026"
- "alignment-aware weight decay"
- "Phi modulator framework" (no results — confirms name is novel)
- "weight decay invariance AdamW" (no results matching our conjecture)
- "standardized weight decay metrics evaluation"

**No direct novelty-killing papers found** as of 2026-03-18.
