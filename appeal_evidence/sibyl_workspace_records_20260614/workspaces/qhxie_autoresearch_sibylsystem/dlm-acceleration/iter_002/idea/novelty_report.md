# Novelty Report: DLM Acceleration Candidates

**Date**: 2026-04-14  
**Scope**: All 4 candidates in `candidates.json`  
**Sources searched**: arXiv, Google Scholar, Web (blogs, repos, workshops)

---

## Candidate 1: `cand_composeaccel` — ComposeAccel (Front-Runner)

**Novelty Score: 6/10**

### Core Contribution Claims
1. First controlled factorial composition study of 3+ training-free DLM acceleration methods
2. IGSD (Information-Geometric Step Distillation) as a novel step scheduler using inter-step KL divergence
3. Speedup decomposition framework with quantified orthogonality metrics
4. Task-specific acceleration recipes

### Collision Analysis

#### Collision 1: SlowFast Sampling + dLLM-Cache composition (34.22x speedup)
- **Paper**: Wei et al., "Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles" (arXiv:2506.10848, Jun 2025)
- **Overlap**: SlowFast explicitly combines adaptive step scheduling (slow/fast phases) with dLLM-Cache (KV caching), achieving 34.22x combined speedup on LLaDA. This IS a composition of methods from two axes (step scheduling + KV caching), with empirical evaluation of the combined speedup.
- **Severity**: **partial_overlap** — SlowFast combines exactly 2 methods (their own scheduler + dLLM-Cache), not a systematic factorial study across 3+ axes. No orthogonality quantification or decomposition framework. However, it significantly weakens the "no one has combined methods" claim.

#### Collision 2: Fast-dLLM (ICLR 2026) — Combined KV cache + parallel decoding
- **Paper**: Wu et al., "Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding" (arXiv:2505.22618, ICLR 2026)
- **Overlap**: Combines block-wise KV caching with confidence-aware parallel decoding, reporting 27.6x combined speedup. Explicitly shows the two methods are "highly complementary." This is a 2-axis composition with combined evaluation.
- **Severity**: **partial_overlap** — Again a 2-method combination, not a systematic 3+ axis factorial study. No orthogonality metric or decomposition framework. But it already claims complementarity and measures combined speedup.

#### Collision 3: dInfer — Modular decomposition of dLLM inference
- **Paper**: dInfer (arXiv:2510.08666, Oct 2025)
- **Overlap**: Decomposes dLLM inference into four modular components (model, diffusion iteration manager, decoding strategy, KV-cache manager) and tests different combinations. This is a partial composition framework.
- **Severity**: **partial_overlap** — dInfer provides a modular decomposition and tests combinations, but focuses on system-level engineering rather than a controlled scientific composition study with orthogonality metrics.

#### Collision 4: KLASS (NeurIPS 2025 Spotlight) — KL-guided adaptive unmasking
- **Paper**: Kim et al., "KLASS: KL-Guided Fast Inference in Masked Diffusion Models" (arXiv:2511.05664, NeurIPS 2025)
- **Overlap**: KLASS uses token-level KL divergence between consecutive timesteps to guide unmasking decisions. ComposeAccel's IGSD uses inter-step logit KL divergence for step scheduling (whether to skip entire steps). Both use KL divergence as a temporal consistency signal for acceleration.
- **Severity**: **partial_overlap** — KLASS operates at the token level (which tokens to unmask) while IGSD operates at the step level (whether to skip entire steps). Different granularity, different mechanism. But the core insight (KL divergence between consecutive steps indicates temporal stability) is shared.

#### Collision 5: "Not All Denoising Steps Are Equal" — Model scheduling via step importance
- **Paper**: Sedykh et al. (arXiv:2604.02340, Feb 2026)
- **Overlap**: Measures step importance via loss and KL divergence between small and large models across timesteps; identifies middle steps as most important, early/late as skippable. Shares the "inverted-U" KL divergence profile insight with ComposeAccel's H6.
- **Severity**: **partial_overlap** — Uses KL divergence to identify step importance (same core signal as IGSD), but for model scheduling (swapping to a smaller model) rather than step skipping. The step-importance characterization directly overlaps with H6.

#### Collision 6: TORS (arXiv:2603.00763) — Composition compatibility study for text-to-image
- **Paper**: "Analyzing and Improving Training-Free Fast Sampling of Text-to-Image Diffusion Models"
- **Overlap**: TORS does exactly what ComposeAccel proposes but for text-to-image diffusion: systematic design space analysis of training-free acceleration methods, composition compatibility evaluation. The framing is almost identical: "training-free acceleration methods are developed independently, leaving overall performance and compatibility unexplored."
- **Severity**: **partial_overlap** — Same methodology applied to a different domain (T2I vs. text DLMs). ComposeAccel can differentiate by domain specificity, but the conceptual framework is not novel.

#### Collision 7: "How Efficient Are Diffusion Language Models?" — Critical evaluation study
- **Paper**: arXiv:2510.18480
- **Overlap**: Systematic benchmarking of DLM efficiency claims with standardized evaluation. Notes that acceleration techniques show diminishing gains at larger batch sizes.
- **Severity**: **related_work** — Evaluation critique paper, does not compose methods or measure orthogonality. Should be cited as motivation.

#### Collision 8: EntropyCache (Mar 2026) — Entropy-guided KV caching with 15-26x speedup
- **Paper**: arXiv:2603.18489
- **Overlap**: EntropyCache uses entropy of decoded token distributions as a signal for KV cache staleness. Achieves 15.2-26.4x speedup. This is a stronger KV caching method than what ComposeAccel's M1 axis proposes.
- **Severity**: **related_work** — Should be included as an M1 candidate in the composition study, potentially replacing d2Cache.

### Assessment

The composition study concept has moderate novelty. The core gap (no systematic factorial composition study across 3+ axes) still holds, but multiple papers now compose 2 methods and report combined speedups (SlowFast+dLLM-Cache at 34.22x, Fast-dLLM at 27.6x, Learn2PD+EoTP at 57.51x). The field is converging toward composition naturally.

**IGSD's novelty is weakened** by KLASS (same KL-divergence signal, different granularity) and the model scheduling paper (same step-importance characterization). The name "Information-Geometric Step Distillation" overstates the contribution — it is a simple threshold-based step skipper using KL divergence, and the information-geometric framing is cosmetic.

**Key differentiation that remains**: (a) controlled 3-way factorial design, (b) quantified orthogonality metric, (c) task-specific recipes, (d) speedup decomposition framework. No existing paper provides all four.

**Recommendation**: PROCEED with modifications:
- Acknowledge SlowFast, Fast-dLLM, and dInfer as partial composition studies
- Reframe IGSD more modestly; acknowledge KLASS as the originator of KL-guided acceleration for DLMs and clearly differentiate (step-level vs. token-level)
- Update M1 to include EntropyCache and d2Cache as axis candidates (not just the simplified EntropyCache from iter_001)
- Emphasize the novelty of the orthogonality metric and speedup decomposition framework, not the composition study framing alone

---

## Candidate 2: `cand_convergence_theory` — Token-Level Information-Theoretic Convergence Bounds

**Novelty Score: 4/10**

### Core Contribution Claims
1. Per-token convergence rate bounds via conditional mutual information
2. Entropy as near-optimal proxy for per-token denoising budget
3. Unifying framework for EntropyCache, KLASS, DyLLM, ES-dLLM
4. Water-filling optimal per-token budget allocation

### Collision Analysis

#### Collision 1 (EXACT): "A Convergence Theory for Diffusion Language Models" (arXiv:2505.21400)
- **Paper**: Chen & Cai, "Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models" (May 2025, CUHK)
- **Overlap**: Establishes convergence guarantees for DLMs from an information-theoretic perspective. Proves KL divergence decays as O(1/T), scales linearly with mutual information between tokens. Provides matching upper and lower bounds. Shows balanced mask schedules achieve optimal convergence.
- **Severity**: **exact_match** — This paper already provides the information-theoretic convergence theory for masked diffusion LMs. The per-token mutual information framing and the O(1/T) convergence rate with tight bounds are directly established.

#### Collision 2: "Optimal Inference Schedules for Masked Diffusion Models" (arXiv:2511.04647)
- **Paper**: Chen et al. (Nov 2025)
- **Overlap**: Provides optimal scheduling bounds in terms of total correlation and dual total correlation, showing O(log n) step sampling is possible in certain natural settings.
- **Severity**: **exact_match** — Directly addresses optimal scheduling from an information-theoretic perspective, with tighter bounds than the proposal suggests.

#### Collision 3: KLASS (NeurIPS 2025) — KL-guided token selection
- **Paper**: Kim et al. (arXiv:2511.05664)
- **Overlap**: KLASS already uses KL divergence as a per-token convergence signal and demonstrates empirically that it works as a proxy for "when a token is converged." The proposal's H5 (entropy as proxy for convergence budget) is largely validated by KLASS.
- **Severity**: **partial_overlap** — KLASS is empirical, not theoretical. The proposal could still contribute formal proofs. But the key practical insight (KL/entropy as convergence signal) is no longer novel.

#### Collision 4: DyLLM, ES-dLLM, Dynamic-dLLM — Per-token adaptive compute
- **Papers**: DyLLM (arXiv:2603.08026), ES-dLLM (arXiv:2603.10088), Dynamic-dLLM
- **Overlap**: All three frameworks already implement per-token adaptive compute allocation based on various convergence signals (cosine similarity, confidence scores, intermediate tensor variation). The proposal's claim of "unifying" these under a theoretical framework is still possible but the practical impact is diminished — the methods already work without the theory.
- **Severity**: **partial_overlap** — Theory could add value, but the practical methods already exist and perform well.

### Assessment

This candidate has been **substantially preempted** by the convergence theory paper (arXiv:2505.21400), which provides exactly the information-theoretic convergence bounds the proposal claims. The optimal scheduling paper (arXiv:2511.04647) further closes the gap. The remaining novelty would be in the specific "water-filling" per-token allocation and the formal proof that entropy is near-optimal among proxy signals, but these are incremental contributions over the existing theory.

**Recommendation**: DROP or radically revise. The core contribution — information-theoretic convergence bounds for masked diffusion LMs — has been published. If pursued, must build on (not re-derive) the existing bounds and focus on a genuinely new angle.

---

## Candidate 3: `cand_order_first` — Order-First Acceleration

**Novelty Score: 5/10**

### Core Contribution Claims
1. Challenges the DLM acceleration paradigm by showing unmasking order is fundamentally flawed
2. Training-free entropy-based priority inversion as a preprocessing step
3. Tests whether acceleration methods compose better under corrected order

### Collision Analysis

#### Collision 1 (EXACT): "The Flexibility Trap" (arXiv:2601.15165)
- **Paper**: Ni et al., "The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models" (Jan 2026)
- **Overlap**: Directly identifies the same problem: confidence-based unmasking order is fundamentally flawed for reasoning tasks. Shows that order flexibility narrows rather than expands the reasoning boundary. This IS the core thesis of `cand_order_first`.
- **Severity**: **exact_match** — The "broken baseline" framing is exactly what The Flexibility Trap demonstrates. The core observation has been published.

#### Collision 2 (NEAR-EXACT): LogicDiff (arXiv:2603.26771)
- **Paper**: Shaik et al., "LogicDiff: Logic-Guided Denoising Improves Reasoning in Masked Diffusion Language Models" (Mar 2026)
- **Overlap**: Proposes logic-role-guided unmasking with a trained classification head. Achieves +38.7pp improvement on GSM8K (from 22.0% to 60.7%) with <6% speed overhead. This SOLVES the problem `cand_order_first` proposes to address, with a concrete method and massive empirical results.
- **Severity**: **exact_match** — LogicDiff's results (+38.7pp on GSM8K) far exceed the proposal's H8 target (>10pp). The problem has been identified AND solved.

#### Collision 3: Learning Unmasking Policies via RL (arXiv:2512.09106, ICLR 2026)
- **Paper**: Olausson et al.
- **Overlap**: Trains unmasking policies via reinforcement learning, formalizing masked diffusion sampling as an MDP.
- **Severity**: **partial_overlap** — RL-based approach rather than training-free entropy inversion, but addresses the same problem (learning better unmasking order).

#### Collision 4: Ground-Truth-Guided Unmasking Order Learning (arXiv:2602.09501)
- **Paper**: Feb 2026
- **Overlap**: Addresses the train-inference mismatch where standard MDLM training optimizes what-to-unmask but leaves where-to-unmask implicit.
- **Severity**: **partial_overlap** — Training-based rather than training-free, but the same problem framing.

#### Collision 5: TRIMS — Trajectory-Ranked Instruction Masked Supervision (arXiv:2604.00666)
- **Paper**: Apr 2026
- **Overlap**: Explicitly addresses the train-inference mismatch in unmasking order by providing explicit supervision over token reveal order during training.
- **Severity**: **partial_overlap** — Training-based, but directly targets the same gap.

### Assessment

This candidate is **heavily preempted**. The Flexibility Trap (Jan 2026) identifies the exact problem. LogicDiff (Mar 2026) solves it with a concrete method achieving +38.7pp on GSM8K — far exceeding the proposal's aspirations. Multiple concurrent works (RL-based policies, ground-truth-guided learning, TRIMS) attack the same problem from different angles.

The remaining potential novelty — testing whether acceleration methods compose better under corrected order (the "composition under corrected order" angle) — is thin. It would need LogicDiff or similar as a prerequisite, reducing this to a follow-up study.

**Recommendation**: DROP. The problem has been identified and solved by multiple groups. The +38.7pp result from LogicDiff makes the proposal's >10pp hypothesis look unambitious.

---

## Candidate 4: `cand_multigrid` — Multigrid Denoising

**Novelty Score: 8/10**

### Core Contribution Claims
1. Cross-disciplinary transplant from numerical analysis (multigrid V-cycle)
2. Decomposes each denoising step into smooth/restrict/correct/prolongate/post-smooth
3. Unifies KV caching (smoothing) and token selection (correction) in a principled hierarchical framework

### Collision Analysis

#### Collision 1: Classical multigrid for image denoising (well-known)
- **Papers**: Various (Rudin-Osher-Fatemi 1992, multigrid PDE solvers for anisotropic diffusion)
- **Overlap**: Multigrid methods are well-established for PDE-based image denoising. The V-cycle idea comes from this literature.
- **Severity**: **related_work** — Classical inspiration, not competing. These are for continuous PDEs, not discrete masked diffusion.

#### Collision 2: DyLLM — Saliency-based token selection with cached activations
- **Paper**: DyLLM (arXiv:2603.08026)
- **Overlap**: DyLLM identifies "salient tokens" via cosine similarity and recomputes only those while caching the rest. This is functionally similar to the restrict-correct cycle (identify uncertain tokens, compute only those, reuse cached for the rest).
- **Severity**: **partial_overlap** — DyLLM achieves a similar practical outcome (selective compute on a subset of tokens + cached activations for the rest) but without the multigrid framing. It lacks the explicit smooth/restrict/correct/prolongate/post-smooth decomposition and the principled hierarchical theory. However, DyLLM achieves 9.6x throughput improvement, so the practical gap may be small.

#### Collision 3: ES-dLLM — Early-skipping based on token importance
- **Paper**: ES-dLLM (arXiv:2603.10088)
- **Overlap**: Skips tokens in early layers based on estimated importance. Functionally related to "restriction" (identifying which tokens need full compute).
- **Severity**: **related_work** — Layer-level skipping, not the full V-cycle structure.

#### Collision 4: Token merging / hierarchical attention (DiT literature)
- **Papers**: "Attend to Not Attended" (CVPR 2025), "LazyDiT"
- **Overlap**: Token merging and hierarchical processing in Diffusion Transformers, but for text-to-image, not text DLMs.
- **Severity**: **related_work** — Different domain and different mechanism (token merging vs. V-cycle decomposition).

### Assessment

This candidate has the **highest genuine novelty** among all four. No existing work frames DLM denoising as a multigrid V-cycle. The closest competitor, DyLLM, achieves a similar practical outcome (selective token computation + caching) but lacks the theoretical multigrid framework and the principled smooth/correct decomposition.

The risk is primarily around **impact**: if the V-cycle reduces to a marginal improvement over DyLLM (which already gets 9.6x), the multigrid framing may be seen as theoretical overhead without practical benefit. The feasibility score of 5 reflects real implementation complexity.

**Recommendation**: PROCEED as high-novelty backup. The multigrid V-cycle framing is genuinely novel for DLMs. If the composition study (ComposeAccel) encounters problems, this is the strongest pivot target.

---

## Overall Summary

| Candidate | Novelty Score | Key Collisions | Recommendation |
|-----------|--------------|----------------|----------------|
| `cand_composeaccel` | 6 | SlowFast+cache (34x), Fast-dLLM (27x), KLASS (KL-guided), TORS (T2I composition study) | **PROCEED with modifications** |
| `cand_convergence_theory` | 4 | Convergence theory paper (exact), Optimal schedules (exact) | **DROP** |
| `cand_order_first` | 5 | Flexibility Trap (exact), LogicDiff (+38.7pp, exact solution) | **DROP** |
| `cand_multigrid` | 8 | DyLLM (partial practical overlap, no theory) | **PROCEED as backup** |

**Overall Novelty**: **medium** — The front-runner has defensible but weakened novelty; two backups are preempted; one backup has high novelty but low feasibility.

### Key Recommendations for ComposeAccel

1. **Acknowledge existing compositions**: SlowFast+dLLM-Cache (34.22x), Fast-dLLM (27.6x), and Learn2PD+EoTP (57.51x) already compose 2 methods. The "no one has combined methods" framing is outdated. Reframe as "no controlled factorial study with orthogonality quantification."

2. **Rename/reposition IGSD**: The "Information-Geometric Step Distillation" name overstates the contribution. It is a KL-threshold step skipper. Acknowledge KLASS as prior art on KL-guided DLM acceleration. Differentiate clearly: KLASS = token-level unmasking decisions, IGSD = step-level scheduling decisions.

3. **Update the method pool**: Include EntropyCache (15-26x, Mar 2026), d2Cache, Sparse-dLLM as M1 candidates. Include SchED, Saber, SlowFast as step scheduling candidates. Include LogicDiff as M3 candidate (far stronger than AR guidance).

4. **Add LogicDiff to the composition study**: LogicDiff's +38.7pp on GSM8K makes it the most impactful single-axis method. Composing LogicDiff + EntropyCache + step scheduling would be far more interesting than the current M1+M3+IGSD proposal.

5. **Strengthen the decomposition framework**: The speedup decomposition and orthogonality metric are the most novel contributions. Make these the paper's centerpiece, not the individual methods.
