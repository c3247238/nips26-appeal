# Novelty Report: Alignment-Aware Dynamic Weight Decay

## Search Methodology

Searches were conducted across arXiv (>15 targeted queries), Google Scholar (>8 queries), and general web search (>6 queries) covering:
- Dynamic/adaptive weight decay methods
- Gradient-parameter alignment in optimization
- Stability and generalization theory for SGD with weight decay
- Time-varying regularization convergence theory
- Per-layer/per-parameter weight decay scheduling

---

## Candidate 1: `cand_aadwd` — Alignment-Aware Dynamic Weight Decay (Front-Runner)

### Core Contribution Claims

1. **First convergence theory for time-varying SGDW** in nonconvex setting
2. **Cumulative contraction stability bound** replacing worst-case alignment with trajectory-weighted average
3. **Alignment-aware dynamic weight decay algorithm** using continuous gradient-parameter cosine similarity

### Prior Work Analysis

#### Directly Related Papers

**1. Sun et al., "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD" (CVPR 2025)**
- **Overlap:** This is the foundational paper that AADWD builds upon. Sun et al. establish the first generalization theory for fixed weight decay in nonconvex SGD, introducing the alignment quantity delta_T = sup_t |<nabla f_S(w_t), w_t>| / (||nabla f_S(w_t)|| ||w_t||).
- **Severity:** `related_work` — This is the direct predecessor. AADWD extends it in three non-trivial directions (time-varying decay, cumulative alignment, stochastic proxy). Sun et al. only analyze fixed lambda and use worst-case (supremum) alignment.
- **Differentiation:** AADWD's cumulative contraction framework and time-varying convergence theory are genuine extensions not present in Sun et al.

**2. Cautious Weight Decay (CWD) — Chen et al. (arXiv 2510.12402, Oct 2025)**
- **Overlap:** CWD also uses gradient-parameter alignment to modulate weight decay. It applies decay only when parameter coordinate signs align with optimizer update signs.
- **Severity:** `partial_overlap` — Both use alignment signals to modify weight decay behavior. However:
  - CWD uses binary (sign-based) alignment at coordinate level; AADWD uses continuous cosine similarity at global level
  - CWD's theory focuses on sliding-mode behavior on the stationary manifold; AADWD provides convergence rates and stability bounds for the full trajectory
  - CWD is optimizer-agnostic (AdamW, Lion, Muon); AADWD targets SGD with full theoretical analysis
  - CWD requires no new hyperparameters; AADWD introduces c, lambda_min, lambda_max
- **Risk:** CWD is a strong practical competitor. If CWD matches AADWD empirically, the differentiator must be the theory.

**3. Ghiasi, Shafahi, Ardekani, "Improving Robustness with Adaptive Weight Decay" (NeurIPS 2023)**
- **Overlap:** Proposes adaptive weight decay that adjusts lambda on the fly based on gradient norm and weight norm ratio.
- **Severity:** `partial_overlap` — Both dynamically adjust weight decay during training. However:
  - Ghiasi et al. use gradient/weight norm ratio, not gradient-parameter alignment (cosine similarity)
  - Their focus is adversarial robustness, not generalization theory
  - No convergence theory or stability bounds provided
- **Differentiation:** AADWD uses the specific alignment quantity from Sun et al.'s generalization framework, providing a direct theoretical connection. Ghiasi et al.'s adaptive rule is heuristic.

**4. AdaDecay — Nakamura & Hong, "Adaptive Weight Decay for Deep Neural Networks" (IEEE Access, 2019)**
- **Overlap:** Per-parameter adaptive weight decay based on gradient magnitude.
- **Severity:** `related_work` — Different mechanism entirely (gradient magnitude, not alignment). No convergence theory.

**5. AlphaDecay — He et al. (arXiv 2506.14562, June 2025)**
- **Overlap:** Module-wise adaptive weight decay for LLMs based on heavy-tailed spectral analysis.
- **Severity:** `related_work` — Different signal (spectral properties of weight correlation matrices, not gradient-parameter alignment). Static assignment computed periodically. No per-step dynamics, no convergence theory.

**6. Xie et al., "On the Overlooked Pitfalls of Weight Decay" — Scheduled Weight Decay (SWD) (NeurIPS 2023)**
- **Overlap:** Dynamically schedules weight decay based on gradient norm statistics. Shows weight decay should be scheduled for adaptive gradient methods.
- **Severity:** `partial_overlap` — Both dynamically adjust weight decay during training. However:
  - SWD uses gradient norm, not gradient-parameter alignment
  - SWD targets Adam/AdamW; AADWD targets SGD
  - SWD lacks convergence theory for the scheduled decay itself
- **Differentiation:** AADWD uses the theoretically motivated alignment signal from the stability bound, not gradient norms.

**7. GALA — Jiang, Kavis, Mokhtari, "Online Learning-guided Learning Rate Adaptation via Gradient Alignment" (arXiv 2506.08419, June 2025)**
- **Overlap:** Uses gradient alignment between consecutive gradients to adapt learning rate (not weight decay). Provides convergence analysis for normalized SGD with GALA.
- **Severity:** `related_work` — GALA adjusts learning rate based on gradient-to-gradient alignment (consecutive steps), not weight decay based on gradient-to-weight alignment. Different mechanism, different target hyperparameter.
- **Note:** This confirms the proposal's assessment that GALA adjusts LR rather than WD.

**8. Chou, "Correction of Decoupled Weight Decay" (arXiv 2512.08217, Dec 2025)**
- **Overlap:** Analyzes the relationship between weight decay and learning rate, argues for lambda proportional to gamma^2. Discusses orthogonality of updates to weights.
- **Severity:** `related_work` — Focuses on the correct scaling of weight decay with learning rate, not on alignment-aware dynamic modulation. The lambda proportional to gamma^2 finding is interesting because AADWD's convergence theorem also requires lambda_t = O(gamma_t^2).

**9. Wang & Aitchison, "How to set AdamW's weight decay as you scale model and dataset size" (arXiv 2405.13698, 2024)**
- **Overlap:** Studies optimal weight decay scaling with model/dataset size. Characterizes WD as EMA of recent updates.
- **Severity:** `related_work` — Different focus (scaling laws for WD, not dynamic alignment-aware scheduling).

**10. Chen et al., "Towards Better Generalization: Weight Decay Induces Low-rank Bias" (arXiv 2410.02176, 2024)**
- **Overlap:** Proves WD+SGD leads to approximately rank-two weight matrices. Derives generalization bounds.
- **Severity:** `related_work` — Different theoretical angle (low-rank bias vs. alignment-based stability).

**11. Wan et al., "Spherical Motion Dynamics" (NeurIPS 2021)**
- **Overlap:** Analyzes training dynamics of normalized neural networks with SGD and weight decay. Decouples gradient direction from magnitude.
- **Severity:** `related_work` — Studies the dynamics of weight norms under WD but does not use gradient-parameter alignment to modulate WD.

#### Papers NOT Found (Confirming Novelty)

The following searches returned NO relevant results, supporting novelty claims:
- "Time-varying weight decay convergence theory nonconvex" — No papers provide convergence rates for time-varying weight decay in nonconvex SGD
- "Cumulative contraction stability bound" — This specific concept does not appear in any prior work
- "Continuous cosine similarity dynamic weight decay" — No paper uses continuous gradient-parameter cosine similarity to dynamically modulate weight decay strength with convergence theory

### Novelty Score: **8/10**

**Justification:** The three-layer theoretical contribution is genuinely novel:
1. Time-varying SGDW convergence theory has no precedent in the nonconvex setting.
2. The cumulative contraction framework replacing worst-case alignment is a new theoretical construction.
3. The specific algorithm using continuous cosine similarity with convergence guarantees is new.

The main deduction is for partial overlap with CWD (also alignment-based WD modulation, though binary/coordinate-level) and SWD/Ghiasi (also dynamic WD, though using different signals). The theoretical contributions remain clearly differentiated.

### Recommendation: **PROCEED**

The theoretical niche is well-defended. Key risks are empirical, not novelty-related:
- CWD comparison is essential to demonstrate the value of continuous vs. binary alignment
- The theory is the primary contribution; empirical gains are secondary
- The alignment proxy reliability (H3) is a legitimate technical concern but does not affect novelty

---

## Candidate 2: `cand_empirical` — Systematic Characterization of Gradient-Parameter Alignment Dynamics (Backup)

### Core Contribution Claims

1. First comprehensive empirical characterization of delta_t evolution across architectures/datasets/optimizers
2. Per-layer alignment disaggregation revealing qualitatively different behaviors
3. Alignment variance correlating with generalization gap

### Prior Work Analysis

**1. Sun et al. (CVPR 2025)** — Introduces the alignment quantity delta_t but does not systematically characterize its behavior across settings. Only fixed WD is studied.
- **Severity:** `partial_overlap` — Sun et al. compute delta_t but do not characterize its dynamics as a primary contribution.

**2. Wan et al., "Spherical Motion Dynamics" (NeurIPS 2021)** — Studies angular dynamics of weights under SGD+WD for normalized networks but does not specifically characterize gradient-parameter alignment (cosine similarity).
- **Severity:** `related_work`

**3. Kunin et al., "Neural Mechanics" (2020)** — Studies symmetries and conservation laws in deep learning, including relationships between gradient and weight dynamics under weight decay. Relates weight decay to angular momentum.
- **Severity:** `related_work` — Different theoretical framework (Lagrangian mechanics), does not characterize delta_t.

**4. Yunis et al., "Approaching Deep Learning Through Spectral Dynamics of Weights" (2024)** — Studies spectral properties of weight matrices during training across various settings.
- **Severity:** `related_work` — Spectral analysis, not alignment (cosine similarity) characterization.

### Novelty Score: **6/10**

**Justification:** While no paper systematically characterizes delta_t dynamics as a primary contribution, the novelty is moderate because:
- Pure empirical characterization has lower novelty ceiling
- If delta_t turns out to be trivially constant or noisy, the contribution collapses
- The concept of gradient-parameter alignment is not new; the systematic study of it is the contribution

### Recommendation: **PROCEED (as backup)**

Viable if the front-runner's theory proves too hard to close, or if delta_t shows genuinely surprising phase-dependent structure. Low risk, moderate impact.

---

## Candidate 3: `cand_llm` — Alignment-Aware Weight Decay for Transformer Pre-training (Backup)

### Core Contribution Claims

1. Per-layer alignment-aware WD for AdamW in LLM pre-training
2. Distinct alignment dynamics in attention vs. FFN layers
3. Dynamic alignment-aware WD combined with AlphaDecay

### Prior Work Analysis

**1. AlphaDecay (arXiv 2506.14562)** — Module-wise weight decay for LLMs based on spectral analysis. Direct competitor in the same application domain (LLM pre-training).
- **Severity:** `partial_overlap` — Both assign different WD to different modules in LLMs. AlphaDecay uses spectral properties; this candidate uses alignment. But targeting the same problem.

**2. CWD (arXiv 2510.12402)** — Already demonstrated on language model pre-training at billion scale with AdamW, Lion, and Muon.
- **Severity:** `partial_overlap` — CWD already handles the AdamW+LLM setting this candidate targets.

**3. Chou (arXiv 2512.08217)** — Studies WD scaling for AdamW/Scion in transformer training.
- **Severity:** `related_work`

**4. Wang & Aitchison (arXiv 2405.13698)** — Studies optimal WD scaling for AdamW with transformers.
- **Severity:** `related_work`

### Novelty Score: **5/10**

**Justification:** The competitive landscape is crowded:
- AlphaDecay already does module-wise adaptive WD for LLMs
- CWD already does alignment-based WD modulation for LLM pre-training
- This candidate lacks convergence theory (purely practical)
- The main novelty would be per-layer alignment dynamics characterization in transformers, which is a subset of candidate 2

### Recommendation: **MODIFY TO DIFFERENTIATE**

To proceed, this candidate needs:
- Clear demonstration that per-layer alignment dynamics provide different/better signal than AlphaDecay's spectral analysis
- Theoretical justification for why alignment matters more than spectral properties in transformer layers
- Results at meaningful scale (>350M) where both CWD and AlphaDecay have been tested

---

## Overall Novelty Assessment

| Candidate | Score | Recommendation |
|-----------|-------|----------------|
| cand_aadwd (Front-runner) | 8/10 | **PROCEED** — Unique theoretical niche |
| cand_empirical (Backup) | 6/10 | PROCEED (backup) — Moderate novelty |
| cand_llm (Backup) | 5/10 | MODIFY — Crowded competitive landscape |

### Overall Novelty: **high**

The front-runner occupies a genuinely novel theoretical position. No prior work provides:
1. Convergence theory for time-varying weight decay in nonconvex SGD
2. A cumulative contraction stability framework replacing worst-case alignment
3. A stochastic proxy transfer theorem for alignment-based algorithms

The closest competitors (CWD, SWD, Ghiasi et al.) differ meaningfully in mechanism, scope, or theoretical depth.

### Key Prior Work to Cite

| Paper | Role | Priority |
|-------|------|----------|
| Sun et al. (CVPR 2025) | Direct foundation | Must cite |
| CWD — Chen et al. (2025) | Primary competitor | Must cite & compare |
| Ghiasi et al. (NeurIPS 2023) | Adaptive WD predecessor | Must cite |
| AdaDecay — Nakamura & Hong (2019) | Per-param adaptive WD | Should cite |
| AlphaDecay — He et al. (2025) | Module-wise WD for LLMs | Should cite |
| SWD — Xie et al. (NeurIPS 2023) | Scheduled WD by gradient norm | Must cite |
| GALA — Jiang et al. (2025) | Alignment-based LR adaptation | Should cite |
| Chou (2025) | WD proportional to gamma^2 | Should cite |
| Hardt et al. (ICML 2016) | Stability framework foundation | Must cite |
| Wan et al. (NeurIPS 2021) | Spherical motion dynamics | Should cite |
| Loshchilov & Hutter (2019) | Decoupled WD (AdamW) | Must cite |
| Chen et al. (2024) — Low-rank bias | WD induces low-rank | Could cite |
