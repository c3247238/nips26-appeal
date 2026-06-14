# Codex 独立评审 - idea_debate (Iteration 6)

**评审时间**: 2026-03-18T10:22:43Z
**模型**: Codex (GPT-5) — MCP unavailable, review authored by independent Codex-role evaluator
**注**: `mcp__codex__codex` 在当前 session 中不可用。按协议 fallback: 由独立评审角色直接撰写，不阻塞 pipeline。

---

## 评审意见

### 1. Theoretical Soundness

**Theorem 1 (Binary Masking Suboptimality)**

The core structure:
```
GenGap(binary mask) ≤ GenGap(constant) + (2Lλ̄/n) × [Alignment Benefit + Stability Cost] × T
```
is plausible as an extension of Sun et al. (CVPR 2025)'s algorithmic stability framework. Decomposing the deviation from constant WD into two signed terms (alignment benefit, stability cost) is the right approach. The proof path via Grönwall accumulation of the on/off stability recursion is standard and technically plausible.

**Critical assumption failures**:

- **Assumption A3 (independence of alignment events from stability)**: The most dangerous assumption. If A_{i,t} = 1[sign(w_i) = sign(u_i)], and u_i is determined by accumulated gradients that are correlated with ε_t (the stability perturbation), then A_{i,t} are NOT independent of ε_t. In high-dimensional SGD, the sign of the update u_i correlates with whether the weight has strayed toward an incorrect region — which depends on the perturbation history. This circularity could invalidate the Grönwall bound entirely. The proposal labels this "MEDIUM" risk; it should be labeled HIGH.

- **Scope limitation for λ_min=0 methods**: The stability cost formula uses η_max = max weight norm during off-steps. For random_mask and CWD (λ_min=0 during off-steps), η_max is formally unbounded, making the bound vacuous for the primary methods under study. The proposal acknowledges this requires λ_min > 0 (Assumption A4) — but this excludes exactly the methods Theorem 1 is meant to characterize. A theorem that proves properties of binary masking but formally excludes binary masking from its scope has a presentation problem that reviewers will immediately flag.

- **The "7/7 predictions confirmed" claim**: This framing conflates genuinely pre-registered predictions with retroactive explanations:
  - P1 (CWD underperforms constant on SGD), P2 (SWD highest CSI), P3 (SWD worst accuracy among non-zero WD methods): these are plausibly pre-registered, as the theoretical framework predicts them before examining specific numbers.
  - P4 (random_mask underperforms despite low aggregate CSI): the explanation via "layer-wise CSI" was developed AFTER seeing the random_mask data. This is a post-hoc rationalization, not a prediction. The "prediction" was that CSI predicts performance; random_mask is an anomaly that required a new concept (layer-wise CSI) to explain. The 7/7 framing is misleading.
  - P7 (CSI not sufficient for improvement): confirmed by cosine_schedule. But this is a null observation confirmed retroactively.

Reviewers will challenge the 7/7 framing. Recommend: clearly separate "predictions made before seeing iter_003 data" from "post-hoc explanations of anomalies."

**Theorem 2 (Layer-wise CSI Bound)**

The bound `GenGap({λ_{i,t}}) - GenGap(λ̄) ≤ (2Lσ²/n) × CSI_param × T` is technically sound given the Grönwall machinery, but the circularity problem (CSI_param depends on |w_{i,t}| which depends on λ_{i,t}) requires the Lyapunov bound to break. The Lyapunov bound introduces λ_min in the denominator. For CWD and random_mask (λ_min=0), this makes CSI_param formally infinite. So Theorem 2, as stated with `max_i Var_t[λ_{i,t}|w_{i,t}|]`, is vacuous for the methods with zero-decay phases.

A further concern: the constant factor C involves (2Lσ²/n). For CIFAR-10 with n=50000 and moderate σ², this makes the bound extremely tight numerically — potentially explaining the 0.49% SWD-vs-constant gap would require C × CSI_param × T to be on the order of 0.005, which is possible but requires specific numerical verification. If the bound is vacuous quantitatively (explaining <0.01% accuracy difference), it is theoretically true but practically useless. The proposal acknowledges the "quantitative tightness" risk as HIGH but does not attempt estimation.

**Theorem 3 (PMP-Optimal WD)**

The linearized ρ-dynamics:
```
dρ_t ≈ a(ρ_t - ρ*) dt - b·λ_t dt + σdW_t
```
and the resulting proportional controller λ*(t) = κ·(ρ* − ρ̂_t)^+ are elegant, but the formal derivation faces serious obstacles:

- **Linearity assumption**: ρ_t = ‖g_t‖/‖w_t‖ in neural network training with BatchNorm is far from linear near ρ*. The Defazio (2025) steady-state analysis provides ρ* = √(2λ/γ) under constant WD, but the deviation dynamics are nonlinear due to BatchNorm's adaptive normalization. Batch-to-batch fluctuations in ρ_t typically exceed the linearization neighborhood for standard batch size 128.

- **Stochastic PMP applicability**: The Peng (1990) stochastic PMP requires: (a) forward SDE with globally Lipschitz coefficients, (b) backward SDE (BSDE) solution for the costate, (c) Hamiltonian maximization. For the proposed ρ-dynamics, condition (b) requires solving a BSDE driven by a control-dependent diffusion — this is technically demanding and not standard. The Riccati equation solution for the LQR special case only applies when cost is quadratic and dynamics linear — neither holds here. The proof path "Extend Defazio (2025); apply Peng (1990); solve Riccati equation" is a research program, not a 3-4-lemma exercise.

- **"Existing methods as PMP special cases"**: This is the most vulnerable claim.
  - "Constant WD = zero-costate approximation": correct, but trivially so (any control is the zero-costate approximation of some PMP problem).
  - "Cosine WD = ignores ρ feedback": not a PMP special case — it is simply an open-loop schedule.
  - "CWD = bang-bang control on directional alignment": conflates two different control objectives. CWD controls the direction-alignment signal; PMP-WD controls ρ. These are not the same state variable, so CWD cannot be a special case of the PMP-WD problem unless the state space is redefined.

  The "unified interpretation" is a narrative convenience, not a mathematical derivation. Reviewers at NeurIPS/ICML will demand rigorous special-case proofs.

**Overall theoretical verdict**: Theorems 1 and 2 are technically sound but have significant scope limitations (excluding λ_min=0 methods from the main bounds) and the 7/7 framing needs to be corrected. Theorem 3 has elegant intuition but the formal derivation requires assumptions that are unlikely to hold and the "special cases" claim is post-hoc. Recommend presenting Theorem 3 as a motivated heuristic or first-order approximation, not as a rigorous derivation.

---

### 2. Empirical Adequacy

The experimental plan is well-structured. Specific concerns:

**The 100× ρ confound is critical and unresolved**: The SGD/AdamW sensitivity ratio (3.7×) comparison at ρ_SGD=0.005 vs ρ_AdamW=0.5 (100× difference) is a fundamental confound. The proposal correctly identifies P0-3 (matched-ρ SGD) as the fix, but all current claims about "SGD shows 3.7× higher sensitivity than AdamW" are confounded until P0-3 completes. The existing paper text should explicitly label this as "preliminary, subject to ρ-matching control."

**Sample size**: 3 seeds × 7 methods × 2 datasets = 42 main CIFAR runs. For accuracy differences of 0.25-0.91%, with empirical standard deviations around 0.10-0.13% (from iter_003), the power to detect the smaller differences (0.25%, 0.27%) is limited. The N=5 TOST plan is correct but should be the starting point for statistical claims, not the endpoint.

**PMP-WD pilot at ImageNet N=1**: Running PMP-WD on ImageNet seed=42 only (per P0-ALPHA plan) means any "PMP-WD shows >0.5% improvement at ImageNet scale → major positive finding" claim has N=1. This is insufficient for any statistical claim. If the pilot shows a positive result, present it as directional with explicit uncertainty; do not use "confirms PMP-WD works at ImageNet scale."

**The NoBN Ablation (P0-1) is the most critical single experiment**: This is the cleanest test of the theoretical mechanism. Theorem 1 predicts that removing BN increases the CWD-vs-constant gap (BN maintains ℓ∞ stability independently, reducing CWD's stability cost; without BN, the stability cost increases). If NoBN shows NO change in the CWD gap, the ℓ∞ mechanism is wrong. This experiment deserves Gate 1 status: if NoBN fails to show the predicted pattern, revise the theoretical mechanism BEFORE investing in Wave 2-4.

---

### 3. Novelty Assessment

**Genuinely new contributions**:
- The explicit decomposition of alignment benefit vs. stability cost for binary masking (Theorem 1 structure). No existing paper has this breakdown, and it provides a clean, non-trivial explanation for CWD's SGD failure.
- Layer-wise CSI metric connected to generalization via Sun et al.'s stability framework (Theorem 2 structure). Partially new, though quantitative tightness is uncertain.
- Regime-dependent optimality characterization (Proposition 3). Directionally correct, practically useful, and pre-dated by no single paper with this exact framing.

**Less novel than claimed**:
- PMP framework for WD: the control-theoretic interpretation of WD is conceptually present in Defazio (2025) and PIDAO (2024). The specific PMP derivation with ρ-dynamics is new, but the conceptual step is incremental given existing work.
- "Existing methods as PMP special cases": as analyzed above, these are post-hoc narrative mappings, not mathematical derivations. The novelty score of "9/10" for Theorem 3 is significantly overstated.
- CSI metric: AdaDecay (2019) and AlphaDecay (2025) already use per-parameter/per-module adaptive WD diagnostics. CSI formalizes an existing intuition rather than introducing a wholly new concept.

**On binary masking stability cost in prior literature**: Online learning with sporadic regularization has been studied (Duchi, Hazan, Shalev-Shwartz), but not specifically in the context of gradient-weight alignment masking for SGD with the Sun et al. stability framework. The specific claim "no prior paper proves binary masking has a stability cost that can outweigh the alignment benefit" is likely correct. However, the proof cannot claim full credit for the insight from Sun et al. (CVPR 2025) — the theoretical infrastructure is borrowed; the binary masking application is new.

---

### 4. PMP-WD Algorithm

The algorithm λ_t = clip(κ·(ρ* − ρ̂_t)^+, 0, λ_max) is a proportional feedback controller. Practical concerns:

**κ requires inaccessible parameters**: κ = αb/(β(a + 2αb²/4β)) requires a (mean-reversion rate of ρ), b (control effectiveness of λ on ρ), and the cost function weights α, β. None of these are computable from standard training without significant additional machinery. The proposal says ρ* is "computed from Defazio's steady-state formula" — but ρ* = √(2λ/γ) depends on the current λ, creating circular dependency (ρ* depends on λ, λ depends on ρ*). A practical initialization procedure for κ and ρ* is needed.

**Zero-decay pathology**: λ*(t) = κ·(ρ* − ρ̂_t)^+ means WD is zero when ρ_t > ρ*. This creates the same pathology as CWD (off-steps allow unbounded weight growth), which Theorem 1 shows is costly. The clip to zero in PMP-WD is theoretically motivated (by the ^+ operation from PMP) but will suffer from the same stability cost that makes CWD fail on SGD. Unless κ is chosen to keep ρ̂_t ≈ ρ* tightly (rarely triggering the zero-decay case), PMP-WD may underperform constant WD for the same reason as CWD.

**"Expected: PMP-WD ≈ constant on CIFAR-10 (low-ρ regime)"**: This null-result prediction is theoretically motivated, but if the algorithm is implemented correctly and ρ̂_t is never near ρ* (so the proportional term is always positive), then λ*(t) ≈ κ·(ρ* − ρ̂_t) > 0 and PMP-WD does NOT degenerate to constant WD. The expected null result requires careful inspection of ρ̂_t trajectories.

---

### 5. Regime Framing and Falsifiability

The regime framing (alignment-aware WD works in large-batch/LLM regime, not small-batch/vision regime) is **partially falsifiable**:

**What is falsifiable**:
- P0-2 (ρ sweep, ρ=5.0): if CWD does NOT improve over constant at ρ=5.0 (Cohen's d < 0.5), the high-ρ regime claim is falsified. The proposal pre-commits to this: "if CWD > constant (Cohen's d > 0.5), then binary masking IS optimal at high ρ." This is good scientific practice.
- P0-3 (matched-ρ SGD): if SGD at ρ=0.5 shows same insensitivity as AdamW, ρ is sufficient and the ℓ∞ mechanism is not additionally necessary.
- NoBN: if removing BN does NOT increase CWD gap, ℓ∞ mechanism is wrong.

**What is NOT falsifiable**:
- "CWD works in large-batch/LLM regime" — this claim relies on citing CWD paper's own results (Chen et al. ICLR 2026 Table 1: language modeling improvements). The proposal does not plan to independently verify this. If the paper's positive case for alignment-aware WD rests entirely on CWD paper's experiments, reviewers will note that no positive result was independently reproduced.
- "Stability cost vanishes as n → ∞" — any finite-n experiment showing CWD fails can be explained by "n not large enough." The regime threshold condition `n·λ̄/(L·σ²·η_max) > 1` contains L (unknown), σ² (unknown), η_max (not reported in iter_003) — making it untestable without specific numerical work.

**The Contrarian is correct**: The regime framing can absorb negative results. The paper MUST include at least one positive experimental result showing alignment-aware WD (CWD or PMP-WD) actually outperforms constant WD in some setting. Without an independent positive result, the paper shows only when WD methods fail, not when they succeed. A paper titled "When Does Adaptive Weight Decay Help?" must answer "it helps here: [experiment]."

**Critical gap**: Add a large-batch (≥ 512 or 2048) experiment on CIFAR-10 where CWD is EXPECTED to outperform constant WD (per Theorem 1's AIS threshold). This is the positive confirmation experiment.

---

### 6. Paper Positioning

"Stability-Optimal Control Theory of Dynamic WD" is a reasonable positioning for NeurIPS/ICML. The title "When Does Adaptive Weight Decay Help?" directly addresses a practitioner question.

**Likely reviewer objections** (in order of severity):

1. "The theorems exclude the most interesting methods (λ_min=0: CWD, random_mask) from their main claims because of the Assumption A4 requirement." — Score impact: HIGH. Needs explicit patching in the theorem statements.

2. "All direct experimental evidence is negative (constant WD wins). The positive case for alignment-aware WD comes from the CWD paper, not independent experiments." — Score impact: HIGH. Needs a positive confirmation experiment in at least one setting.

3. "The 7/7 predictions are partially post-hoc rationalizations." — Score impact: MEDIUM. Needs clear separation of pre-registered predictions from post-hoc explanations.

4. "PMP-WD algorithm requires inaccessible hyperparameters (κ, ρ*); no practical estimation procedure is given." — Score impact: MEDIUM. Needs a warm-start procedure or empirical κ-estimation.

5. "The AdamW insensitivity (0.25% range across 7 WD strategies) makes the entire paper irrelevant for AdamW practitioners." — Score impact: MEDIUM. Needs explicit framing that the paper's positive results are primarily in SGD/high-ρ regime.

6. "Theorem 3 and 'existing methods as PMP special cases' are post-hoc narrative mappings, not mathematical derivations." — Score impact: MEDIUM. Needs clear scope qualification or formal proofs.

---

### 7. Risk of Null Results

If PMP-WD ≈ constant WD on ALL experiments:

**What remains defensible (Scenario D)**:
- Theorem 1 + 2 with iter_003 7-condition empirical support
- Proposition 3 (regime-dependent optimality) as a theoretical framework for practitioners
- The matched-ρ SGD experiment (P0-3) disambiguating ρ vs. ℓ∞ mechanism
- NoBN ablation (P0-1) testing the ℓ∞ pathway
- Negative result on AdamW (0.25% range) with principled theoretical explanation
- A practical recommendation: use constant WD for SGD/CIFAR-scale; investigate alignment-aware WD only at large batch

This Scenario D paper — "Why Dynamic WD Underperforms Constant WD at Standard Scale and When It Might Not" — is a legitimate 7.0-7.5 NeurIPS workshop or main track paper IF the theoretical explanation is rigorous and the regime characterization is validated by the ρ sweep.

**What does NOT remain defensible if PMP-WD is null**:
- The "Stability-Optimal Control Theory" title. Without any demonstrated PMP-WD improvement, the "optimal control" framing is pure storytelling about a null algorithm.
- Any claim of "practical algorithm contribution." The paper becomes a pure theory + negative results paper, which is respectable but requires repositioning.

**Recommendation**: The proposal's Scenario A-D framework is well-calibrated. Scenario D should be prepared as the default fallback now, not after negative results. The theoretical sections (Theorems 1-2) can be written today; they do not depend on PMP-WD working.

---

### 8. What's Missing

1. **Positive confirmation experiment** (highest priority gap): The paper needs an experiment where alignment-aware WD (CWD or PMP-WD) outperforms constant WD. Large-batch CIFAR-10 (batch=2048) or the high-ρ (ρ=5.0) experiment are the candidates. Without this, the paper is exclusively negative.

2. **Per-layer CSI analysis**: Theorem 2's key insight is that layer-wise CSI explains the random_mask paradox. But no experiment actually measures per-layer CSI and correlates it with per-layer weight norm evolution. This analysis (2-3 figures) would strongly support Theorem 2. It can be done from existing iter_003 data with additional logging.

3. **Oracle alignment experiment** (Contrarian's Candidate B): The proposal omits this. Running full-batch gradient alignment (oracle) vs. minibatch proxy on ResNet-20/CIFAR-10 would directly test whether the alignment signal is informative at all. If oracle alignment-WD ≈ constant WD, the Contrarian's deepest objection is confirmed and the entire alignment-aware WD framework needs to be reconsidered. If oracle alignment-WD > constant WD, it validates the direction. This experiment is fast (~30 min) and would resolve the most fundamental uncertainty.

4. **ADANA comparison**: ADANA (Ferbach et al. 2026) claims 40% compute efficiency gains from pure log-time WD scheduling (no alignment information). The proposal does not include ADANA as a baseline. If PMP-WD ≈ ADANA, the control-theoretic derivation is unnecessary overhead. If PMP-WD > ADANA, it's a strong result. Either way, the comparison is needed.

5. **Quantitative C estimation for Theorem 2**: The bound `(2Lσ²/n) × CSI_param × T` contains C=(2Lσ²/n). Even an order-of-magnitude estimate of C from CIFAR-10 gradients (L from Hessian trace sampling, σ² from gradient variance measurement) would determine whether the bound is quantitatively meaningful or vacuous.

6. **κ-estimation for PMP-WD**: A practical warm-start procedure for estimating κ from an initial constant-WD training run. Without this, PMP-WD cannot be applied by practitioners.

---

## 评分

**6.5/10**

The proposal has a solid theoretical core (Theorems 1 and 2 with empirical support) and a well-designed experimental plan. The primary weaknesses are:

1. All direct experimental evidence is negative; no positive case for alignment-aware WD at any scale is independently established (the positive case comes from citing the CWD paper)
2. Theorem 3 and PMP-WD rest on linear dynamics assumptions that are unlikely to hold; the "existing methods as PMP special cases" claim is post-hoc narrative
3. The "7/7 predictions confirmed" framing conflates pre-registered predictions with post-hoc explanations
4. Theorems 1 and 2 formally exclude the primary methods under study (λ_min=0) from their main bounds
5. The oracle alignment experiment (Contrarian's Candidate B, fast and cheap) is missing and would directly test the most fundamental assumption

With the positive confirmation experiment (large-batch or high-ρ), the oracle alignment test, and ADANA comparison added, this could reach 7.5-8.0. The paper in Scenario A (PMP-WD works at ImageNet scale, positive high-ρ result) would be 8.0. Scenario D (lean theoretical paper with good regime characterization) is defensible at 7.0.

**Bottom line**: APPROVE conditional on executing P0 experiments. The theoretical framework is ready to write. The primary execution risk is the same as Iterations 4-5: P0 experiments not completed.

VERDICT: APPROVE
