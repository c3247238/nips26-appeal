# Novelty Report — Unified Dynamic Weight Decay Framework

**Date**: 2026-03-19 (updated)
**Checker**: sibyl-novelty-checker
**Search coverage**: 18+ web searches, arXiv, Google Scholar, venue proceedings checked

---

## Candidate 1: `cand_lyapunov_unified` (Front-Runner)

**Title**: Stability-Optimal Weight Decay: A Lyapunov Control Framework Unifying Adaptive Regularization in Deep Learning

### Novelty Score: 7/10

Novel with partial overlaps; differences are clear and defensible.

**Recommendation**: PROCEED

### Core Claims and Prior Art Analysis

#### Claim 1: Unified Lyapunov Convergence Certificate for General Time-Varying WD (Theorem 1)

**Prior work found**:
- **CWD (Li et al., ICLR 2026)**: Proves convergence for *binary* WD via Lyapunov + LaSalle invariance principle. Does NOT handle continuous time-varying schedules. Severity: **partial_overlap**.
- **Sun et al. (CVPR 2025)**: Proves generalization bound for *fixed* WD in nonconvex SGD. Does NOT address time-varying WD. Severity: **related_work**.
- **Kondo & Iiduka (arXiv 2508.03105, Aug 2025)**: Novel Lyapunov function for SGDM with dynamic *learning rate* and *batch size* schedules. Does NOT address weight decay schedules. Severity: **related_work** -- uses similar Lyapunov methodology but for different hyperparameters.
- **Hassan Saoud (arXiv 2510.08259, 2025)**: Composite Lyapunov criteria for nonlinear autonomous systems. Applied to inertial gradient systems, NOT to weight decay. Severity: **related_work**.

**Assessment**: **NOVEL**. No prior work derives computable certified convergence bands for arbitrary lambda(t). CWD handles binary only; Sun et al. handle fixed only; Kondo & Iiduka handle LR/batch not WD. The certified band concept is genuinely new.

#### Claim 2: Cumulative Alignment Generalization Bound (Theorem 2)

**Prior work found**:
- **Sun et al. (CVPR 2025)**: Provides worst-case alignment bound using sup_t delta_t. Our cumulative bound replaces this with per-step delta_t inside the stability product. Severity: **partial_overlap**.
- **Holzl et al. (NeurIPS 2025)**: GWA as train-time proxy for generalization. Empirical study, no formal bound. Severity: **related_work**.
- **ICLR 2025 paper on trajectory+flatness bounds**: Develops tighter generalization bounds using trajectory terms but in a different framework. Severity: **related_work**.

**Assessment**: **INCREMENTALLY NOVEL**. Direct extension of Sun et al. from worst-case to cumulative alignment. Mathematically non-trivial but conceptually straightforward. This is the weakest novelty claim.

#### Claim 3: Formal Subsumption of WD Methods (Theorem 3)

**Prior work found**:
- **Newhouse (MIT Thesis, 2025)**: Duality-based unification of *optimizers* (SGD/Adam/Muon), not WD *scheduling methods*. Severity: **related_work**.
- **CWD (ICLR 2026)**: Compares against baselines empirically but does not provide a unifying convergence theorem.
- No paper formally proves multiple WD scheduling methods satisfy a single convergence certificate.

**Assessment**: **NOVEL**. No prior work proves constant, CWD, cosine, SWD, PMP-WD satisfy a single certificate.

#### Claim 4: PMP-WD Optimality with Bang-Bang Structure (Theorem 4)

**Prior work found**:
- **E et al. (JMLR 2017)**: Foundational PMP for deep learning training (treats depth as time). Does NOT apply PMP to WD scheduling.
- **PMP for CNN training (arXiv 2504.11647, 2025)**: PMP for L0 regularization in CNNs. Different application.
- **AdamO (Chen et al., Feb 2026, arXiv:2602.05136)**: Radial/tangential decomposition of WD dynamics. Identifies "Radial Tug-of-War" but does NOT derive optimal WD schedule via PMP. Severity: **related_work** (complementary framing).
- No paper found deriving an optimal WD schedule using Pontryagin's Maximum Principle.

**Assessment**: **NOVEL**. The bang-bang prediction is sharp and falsifiable. The connection to CWD's binary mask is a non-obvious insight. Strongest novelty claim.

#### Claim 5: Diagnostic Metrics (CSI, AIS)

**Prior work found**: No prior metrics with this theoretical grounding in the WD literature. Holzl et al.'s GWA serves different purpose (early stopping, not WD method selection).

**Assessment**: **NOVEL**.

### Key Collisions Summary

| Prior Work | Overlap | Severity |
|---|---|---|
| CWD (Li et al., ICLR 2026) | Lyapunov convergence for binary WD; sliding-mode analysis | partial_overlap |
| Sun et al. (CVPR 2025) | Generalization bound with worst-case alignment for fixed WD | partial_overlap |
| AdamO (Chen et al., Feb 2026) | Radial/tangential decomposition of WD dynamics; architecture-aware rules | partial_overlap |
| Kondo & Iiduka (arXiv 2508.03105) | Lyapunov analysis for dynamic hyperparameter schedules (LR/batch, not WD) | related_work |
| Defazio (arXiv 2506.02285, 2025) | WD as gradient-to-weight ratio controller in normalized layers | related_work |
| Holzl et al. (NeurIPS 2025) | Gradient-Weight Alignment as generalization proxy | related_work |
| AlphaDecay (NeurIPS 2025) | Module-wise WD via heavy-tail spectral theory | related_work |
| CPR (Franke et al., NeurIPS 2024) | Per-parameter constrained regularization as WD alternative | related_work |
| Kobayashi et al. (NeurIPS 2024) | WD induces low-rank attention via nuclear norm equivalence | related_work |
| Kosson et al. (2024) | Rotational equilibrium from WD | related_work |
| Hassan Saoud (arXiv 2510.08259) | Composite Lyapunov criteria (mathematical tool) | related_work |
| Newhouse (MIT Thesis, 2025) | Duality-based optimizer unification | related_work |

### Time Sensitivity: MODERATE

- Kondo & Iiduka (Aug 2025): Their Lyapunov framework for LR/batch could extend to WD.
- CWD authors: Could generalize their binary Lyapunov analysis to continuous.
- AdamO (Feb 2026): Shows control-theoretic decomposition is gaining traction.
- **Recommend submission within 3-4 months.**

---

## Candidate 2: `cand_alignment_mirage` (Backup)

**Title**: The Alignment Mirage: When Gradient-Weight Geometry Misleads Dynamic Regularization

### Novelty Score: 5/10

Partial overlap; needs repositioning.

**Recommendation**: MODIFY TO DIFFERENTIATE -- better integrated as ablation within main paper

### Collision Analysis

| Prior Work | Overlap | Severity |
|---|---|---|
| Defazio (2025) | Shows normalized layers have orthogonal gradients/weights; WD = effective LR control | partial_overlap |
| D'Angelo et al. (2024, NeurIPS) | WD under BN = loss stabilization / effective LR control | partial_overlap |
| Zhu et al. (Three Mechanisms of WD) | Architecture-dependent WD mechanisms; BN confound partially characterized | partial_overlap |
| Jane Street Blog | L2 + BN = adaptive LR, not regularization | related_work |

**Assessment**: Core thesis substantially anticipated by Defazio (2025) and D'Angelo et al. (2024). What remains novel: systematic alignment SNR measurement, random mask vs CWD control on BN architectures (iter_003 data). Insufficient for standalone paper; valuable as ablation.

---

## Candidate 3: `cand_architecture_aware` (Backup)

**Title**: Rethinking Dynamic Weight Decay: Parameter Structure Matters More Than Training Dynamics

### Novelty Score: 5/10

Partial overlap; needs repositioning.

**Recommendation**: MERGE INTO MAIN CANDIDATE as discussion/experiment section

### Collision Analysis

| Prior Work | Overlap | Severity |
|---|---|---|
| Kobayashi et al. (NeurIPS 2024) | L2 on factorized attention = nuclear norm regularization; induces low-rank | partial_overlap |
| AdamO (Chen et al., Feb 2026) | Architecture-aware rules for scale-invariant layers; radial-only decay | partial_overlap |
| AlphaDecay (NeurIPS 2025) | Module-wise WD strength via spectral properties | partial_overlap |
| CPR (Franke et al., NeurIPS 2024) | Per-parameter adaptive regularization | partial_overlap |
| Standard practice | No WD on biases/norms already common | related_work |

**Assessment**: Core claim faces significant prior art from Kobayashi et al., AlphaDecay, AdamO, and CPR. The factorial ablation design (parameter-type vs scheduling vs alignment) is methodologically novel but not sufficient for standalone paper given theoretical overlap.

---

## Overall Assessment

| Candidate | Score | Recommendation |
|---|---|---|
| `cand_lyapunov_unified` | **7/10** | **PROCEED** |
| `cand_alignment_mirage` | 5/10 | Integrate as ablation |
| `cand_architecture_aware` | 5/10 | Merge as discussion |

**Overall Novelty: HIGH** (front-runner >= 7)

### Strongest Differentiators (hardest to scoop)

1. **PMP-WD bang-bang optimality** (Theorem 4): Non-obvious interdisciplinary connection. No one is close.
2. **Formal subsumption** (Theorem 3): Labor-intensive analysis of each method as a control law.
3. **Certified band explaining constant WD near-optimality**: Unique insight supported by iter_003 data.

### Must-Cite Papers

1. CWD (Li et al., ICLR 2026) -- closest predecessor
2. Sun et al. (CVPR 2025) -- direct predecessor for generalization bound
3. SWD (Xie et al., NeurIPS 2023) -- key method in unified family
4. Defazio (arXiv 2025) -- WD as gradient-to-weight controller
5. Kondo & Iiduka (arXiv 2025) -- Lyapunov for dynamic schedules
6. Holzl et al. (NeurIPS 2025) -- gradient-weight alignment
7. AdamO (Chen et al., Feb 2026) -- radial/tangential decomposition
8. AlphaDecay (NeurIPS 2025) -- spectral-aware per-module WD
9. Kobayashi et al. (NeurIPS 2024) -- WD induces low-rank attention
10. CPR (Franke et al., NeurIPS 2024) -- constrained parameter regularization
11. Kosson et al. (2024) -- rotational equilibrium
12. D'Angelo et al. (NeurIPS 2024) -- three mechanisms of WD
