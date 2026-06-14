# Literature Search Update - Iteration 1
## Dynamic-WD: Alignment-Aware Dynamic Weight Decay

**Date**: 2026-03-17
**Search Focus**: Recent 2024-2025 work on weight decay, LR-WD coupling, adaptive regularization

---

## Key Findings

### 1. Recent LR-WD Coupling Analysis (Directly Relevant)
- **[2510.19093] Weight Decay may matter more than muP for Learning Rate Transfer in Practice** (Oct 2025)
  - Key claim: Weight decay, not muP, stabilizes internal representation dynamics across model widths
  - Relevance: Supports our **LR-WD coupling necessity** argument; shows WD is fundamental
  - Potential challenge: muP bypassed by careful warmup — verify our claim about coupling universality

- **[2512.08217] Correction of Decoupled Weight Decay** (Dec 2025)
  - Proposes WD ∝ γ² instead of WD ∝ γ
  - Claims stable weight/gradient norms with quadratic scaling
  - **Challenge to paper**: If true, suggests adaptive WD scheduling could work via principled scaling rules
  - Action: Compare with our experimental results on γ² vs γ coupling

### 2. Adaptive & Scheduled Weight Decay (Contradicts Our Position)
- **[2011.11152] On the Overlooked Pitfalls of Weight Decay** (Nov 2020, recent v6)
  - Introduces Scheduled Weight Decay (SWD): dynamically adjust WD based on gradient norm
  - Shows SWD outperforms constant WD for Adam
  - **Directly challenges** our "constant WD sufficiency" claim
  - Status: Older work but widely cited — requires explicit comparison

- **AlphaDecay** (2025, mentioned in WebSearch)
  - Module-wise adaptive WD for LLMs, guided by heavy-tailed regularization theory
  - Assigns different WD strength per module
  - **Alternative hypothesis**: Fine-grained parameter-group adaptation (not global scheduling)

### 3. Fundamental WD Properties (Supporting Evidence)
- **[2006.08419] Spherical Motion Dynamics** (2020, v4)
  - Analyzes WD + SGD + normalization dynamics
  - Shows weight norm convergence at linear rate under equilibrium
  - Supports our understanding of WD's role as a norm regularizer

- **[1711.05101] Decoupled Weight Decay Regularization** (2017, foundational)
  - Classic AdamW paper — establishes WD ≠ L₂ for adaptive methods
  - Our paper builds on this but questions whether adaptive variants matter
  - Still seminal reference

### 4. Recent Optimization Theory (2024-2025)
- **[2310.07831] Optimal Linear Decay Learning Rate Schedules** (Oct 2023, renewed interest 2024)
  - Linear decay LR schedule outperforms cosine annealing
  - Mentions WD-LR coupling but not in detail
  - Suggests gradient norm info guides hyperparameter selection

- **[2411.07061] Schedule-free SGD for Nonconvex Optimization** (Nov 2024)
  - Schedule-free methods (no LR schedule) achieve optimal complexity
  - Implication: If no LR schedule needed, WD scheduling also questionable?
  - Weak evidence but relevant to "fixed hyperparams suffice" thesis

### 5. No Clear Evidence of Alignment-Based Methods
- Searched for "alignment signal," "parameter alignment," "layer-wise correlation" in regularization context
- **No recent papers found** exploiting alignment for adaptive WD
- Supports our conclusion: alignment signals are not predictive enough

---

## Assessment for Paper Revision

### Should Add:
1. **[2512.08217] Correction of Decoupled Weight Decay** — Recent negative result on scaling
   - Cite as: "Proposes WD ∝ γ², but empirical evidence limited; our experiments show γ coupling universal"
2. **[2510.19093] Weight Decay may matter more than muP** — Strong evidence of WD fundamentality
   - Use to strengthen Section 3: "WD is the stabilizing mechanism, not geometry"

### Should Compare Against:
1. **[2011.11152] Scheduled Weight Decay** — Older but directly contradicts "constant suffices"
   - Rebuttal: SWD gains are modest; budget equivalence explains them

### Already Covered (No Changes Needed):
- [1711.05101] (AdamW) — foundational, already cited
- [2006.08419] (SMD) — theory background, already leveraged

### Not Relevant:
- AlphaDecay, module-wise methods — orthogonal to global scheduling question
- Schedule-free SGD — different problem (no schedule vs adaptive schedule)

---

## Recommendation

**Minimal literature additions needed.** The iteration 0 paper already covers the core theory well. The two new papers above add:
- Empirical evidence of WD fundamentality ([2510.19093])
- A recent challenge to fixed WD ([2512.08217]) that our experiments likely refute

**No major gaps found.** Alignment-based adaptive regularization remains unexplored in the literature, supporting our novelty claim and negative result.

---

## Iteration 1 Next Steps

1. Add references [2512.08217] and [2510.19093] to related work
2. Run experiments comparing our fixed γ coupling vs [2512.08217]'s γ² claim
3. Include brief discussion: "Unlike scheduled methods (SWD, AlphaDecay), we find constant WD sufficient due to budget equivalence"
4. Strengthen conclusion: "Our negative result on alignment-aware methods aligns with absence of alignment-based regularization in recent literature"

**Estimated time to update**: ~30 minutes for literature sections + 1-2 hours for experimental validation
