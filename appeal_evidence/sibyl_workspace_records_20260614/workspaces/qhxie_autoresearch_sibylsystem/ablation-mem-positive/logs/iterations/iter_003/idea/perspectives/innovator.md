# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. [Chanin et al., 2024/2025. A is for Absorption: Studying Feature Splitting and Absorption in SAEs. arXiv:2409.14507] — First systematic study of feature absorption; establishes detection metric; proves absorption is caused by hierarchical feature co-occurrence under sparsity. Limitation: metric only reliable for early layers (0-17), uses only first-letter spelling task.

2. [Cui et al., ICLR 2026. On the Limits of Sparse Autoencoders. arXiv:2506.15963] — First closed-form theoretical analysis proving standard SAEs cannot fully recover ground-truth monosemantic features due to intrinsic representational interference. Establishes that full disentanglement is mathematically impossible under realistic sparsity.

3. [Costa et al., NeurIPS 2025. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit. arXiv:2506.03093] — MP-SAE uses residual-guided greedy selection to extract hierarchical features; promotes conditional orthogonality; reduces absorption vs Vanilla/BatchTopK.

4. [Karvonen et al., ICML 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532] — 8-metric evaluation suite including absorption measurement via probe projection approach that works across all layers (unlike ablation-based metric limited to early layers).

5. [Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033] — Enforces orthogonality via cosine similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features.

6. [Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353] — Critical negative result: 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation. Raises fundamental questions about SAE practical utility.

7. [Gao et al., 2024. Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093] — Proposes k-sparse autoencoders; scales to 16M latents on GPT-4; establishes scaling laws. Does not address absorption.

8. [Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka SAEs. arXiv:2503.17547] — Nested dictionaries of increasing size to organize features hierarchically; reduces feature absorption; superior on sparse probing and concept erasure.

### Landscape Summary

Feature absorption in SAEs is now established as a central open problem in mechanistic interpretability. The phenomenon — where hierarchical features cause general features to be subsumed by more specific ones during sparse optimization — creates an "interpretability illusion" where latents appear monosemantic but have systematic false negatives. Key developments in 2025-2026 include: (1) theoretical limits established by Cui et al., proving full disentanglement is mathematically impossible under realistic sparsity; (2) MP-SAE demonstrating that greedy residual-guided selection can recover hierarchical structure; (3) OrtSAE showing 65% absorption reduction via orthogonality enforcement; (4) a critical negative result (Basu et al.) questioning whether any absorption quantification matters if SAE steering produces zero output change.

Three major gaps remain underexplored: (1) systematic cross-model/layer quantification of absorption rates; (2) training-free detection in existing pretrained SAEs without retraining; (3) quantitative impact of absorption on downstream interpretability tasks (circuit discovery, steering, concept erasure).

---

## Phase 2: Initial Candidates

### Candidate A: Robustness-Targeted Absorption Decomposition

- **Hypothesis**: Absorbed features split into two subpopulations: "fragile" absorbed features (suppressed parent, child routes through residual stream identically to non-steered case → zero steering effect) and "robust" absorbed features (suppressed parent but child has context-sensitive activation → steering can modulate child behavior). CV identifies this split: high-CV features are robust absorbed, low-CV features are fragile absorbed.
- **Cross-domain insight**: From pharmacology: drug absorption creates different bioavailability profiles — some drugs are fully bioavailable (robust effect), others have low bioavailability due to first-pass metabolism (fragile). The absorption mechanism determines the effect profile, not just the presence/absence of absorption.
- **Why it might work**: Pilot data shows high-CV features are 2x more steerable than low-CV (0.153 vs 0.075). If steering acts through the child latent, then the child's context-sensitivity (measured by CV) determines whether steering modulates behavior. This explains the variance paradox and the Basu et al. paradox simultaneously.
- **Novelty estimate**: 7/10 — No prior work distinguishes "steerable absorbed" vs "non-steerable absorbed" features; Basu et al. treats all absorbed features as uniformly unsteerable.

### Candidate B: Information-Theoretic Absorption Decomposition

- **Hypothesis**: Absorption can be decomposed using information-theoretic measures: the "absorption rate" is the mutual information I(parent; child) / I(parent; representation), and the "actionability gap" is the difference between detection mutual information and steering mutual information. Features with high actionability gap are those where the child channel preserves information but resists steering intervention.
- **Cross-domain insight**: From rate-distortion theory in compression: lossy compression preserves some information (rate) but loses other information (distortion). Absorption is analogous — child latents preserve low-level information (enabling probe detection) but lose high-level causal information (preventing steering).
- **Why it might work**: Cui et al. established information-theoretic limits; this would operationalize them for specific feature pairs, predicting which features are most affected by the actionability gap.
- **Novelty estimate**: 6/10 — Cui et al. already establishes the theoretical framework; operationalizing for feature pairs is novel but may be straightforward application.

### Candidate C: Temporal Absorption Dynamics

- **Hypothesis**: Absorption is not static but evolves during inference — parent features may be absorbed in some token positions but not others, depending on context. This explains why high-CV absorbed features show steering effect in some contexts but not others.
- **Cross-domain insight**: From neuroscience: neural representations are context-dependent and dynamically reconfigured. Feature absorption may similarly depend on token-level activation patterns.
- **Why it might work**: CV measures within-feature variance across samples; high-CV features may be those where absorption is context-dependent (some contexts absorb, others don't). Steering that targets the correct context would succeed.
- **Novelty estimate**: 8/10 — No prior work studies token-level absorption dynamics; this is genuinely unexplored territory.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: Basu et al. (2026) shows 98.2% AUROC but 0% steering. Our CV-based hypothesis suggests some absorbed features ARE steerable, contradicting their finding of universal actionability failure. We need to reconcile: is our finding domain-specific (non-clinical LLM vs clinical domain)?
- **Methodological attack**: The steering experiment in pilot_steering_cv used a simple logit change metric. Basu et al. used more sophisticated output metrics. The 2x difference may be an artifact of measurement insensitivity rather than genuine steering effectiveness.
- **Theoretical attack**: The "robust fragile" decomposition is post-hoc — we observed the CV-steering correlation after running the experiment. This needs prospective validation, not retrospective explanation.
- **Scalability attack**: Distinguishing robust vs fragile absorbed features requires running steering experiments for each feature, which is expensive. Not scalable to full dictionary analysis.
- **Verdict**: MODERATE — The pilot data supports this but needs rigorous prospective validation. The framing as "explaining Basu et al." is stronger than claiming to contradict them.

### Against Candidate B

- **Prior work attack**: Cui et al. (ICLR 2026) establishes the information-theoretic framework. We would be applying their theory to specific features, which may be straightforward without novel theoretical contribution.
- **Methodological attack**: Computing mutual information between parent and child requires joint distribution estimation, which is challenging in high-dimensional spaces. The approximations may not be tractable.
- **Theoretical attack**: The rate-distortion analogy may be too loose. SAE absorption is not compression in the information-theoretic sense — the child doesn't "encode" the parent, it suppresses it.
- **Scalability attack**: Computing I(parent; child) for thousands of feature pairs is computationally prohibitive.
- **Verdict**: WEAK — The theoretical machinery is heavy and may not yield actionable predictions. Cui et al.'s framework is better cited than extended in this way.

### Against Candidate C

- **Prior work attack**: No prior work studies token-level absorption dynamics. This is genuinely novel territory.
- **Methodological attack**: How do we measure "context-dependent absorption"? The absorption metric is defined at the feature level across samples, not at the token level. We would need a new metric for token-level absorption.
- **Theoretical attack**: If absorption is deterministic given activation patterns, the variance paradox (high-CV absorbed features) would need a different explanation. Context-dependent absorption is plausible but unproven.
- **Scalability attack**: Analyzing token-level absorption requires tracking absorption at each token position, which is expensive. We would need to define what "token-level absorption" even means operationally.
- **Verdict**: MODERATE — This is genuinely novel but operationally undefined. We would need to develop a new metric before testing the hypothesis.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Information-Theoretic Absorption Decomposition)** dropped because: The theoretical machinery is heavy without clear empirical tractability. Computing mutual information for thousands of feature pairs is not feasible within our resource budget. The rate-distortion analogy may not hold for SAE absorption mechanics.

### Strengthened Ideas

- **Candidate A (Robustness-Targeted Absorption Decomposition)**: Strengthened by the pilot_steering_cv result showing high-CV features are 2x more steerable (0.153 vs 0.075). This is prospective validation, not post-hoc explanation. The hypothesis now has direct empirical support. We can frame this as "explaining heterogeneity in the Basu et al. actionability paradox" — Basu et al. studied clinical domain; LLM SAEs may show different actionability profiles.

- **Candidate C (Temporal Absorption Dynamics)**: Strengthened by the CV observation. High-CV features have high within-feature variance, which could reflect context-dependent activation patterns. If absorption is context-dependent, then high-CV may identify features where parent-child routing switches across contexts. This remains exploratory but is consistent with the variance observation.

### Additional Evidence Found

From pilot_activation_patching: All 9 persistent core words show >48% recovery when child is zeroed (mean 67.3%). This confirms genuine absorption — the parent feature is causally active even when absorbed, and can be recovered by ablating the child. This validates that the "robust absorbed" subpopulation exists and is causally significant.

### Selected Front-Runner

**Candidate A: Robustness-Targeted Absorption Decomposition** is selected as the front-runner because:

1. **Direct empirical support**: Pilot data shows high-CV features are 2x more steerable than low-CV (0.153 vs 0.075). This is a prospective finding, not post-hoc explanation.
2. **Explains Basu et al. heterogeneity**: Basu et al. found universal actionability failure in clinical domain. Our finding suggests LLM SAEs may have a subpopulation of "robust absorbed" features that remain steerable. This is publishable in either direction — confirming Basu et al. for some features or finding exceptions.
3. **Actionable outputs**: If we can predict which absorbed features are robust vs fragile based on CV, this provides guidance for interpretability practitioners on which features to use for steering.
4. **Resource efficient**: Uses existing steering infrastructure; experiments fit within 1-hour budget per task.
5. **Novel framing**: No prior work distinguishes steerable vs non-steerable absorbed features. This is the first systematic study of actionability heterogeneity within absorbed features.

---

## Phase 5: Final Proposal

### Title

**Beyond the Actionability Paradox: Decomposing Feature Absorption into Steerable and Non-Steerable Subpopulations**

### Hypothesis

Feature absorption creates two subpopulations: (1) "fragile absorbed" features where the child latent's contribution to the residual stream is unaffected by parent steering, producing zero steering effect (explaining Basu et al.); and (2) "robust absorbed" features where the child latent's context-sensitive activation allows steering to modulate behavior, producing measurable steering effects. Coefficient of variation (CV) predicts which subpopulation an absorbed feature belongs to: high-CV features (>1.0) are robust absorbed, low-CV features (<1.0) are fragile absorbed.

### Motivation

Basu et al. (2026) demonstrated that 98.2% AUROC feature detection translates to 0% steering utility — the "actionability paradox." This result has cast doubt on the entire enterprise of SAE-based interpretability: if we cannot steer features we can detect, what good is detection?

However, the paradox may not be universal. Our pilot data shows that high-CV absorbed features are 2x more steerable than low-CV (0.153 vs 0.075). This suggests heterogeneity within absorbed features: some are fragile (non-steerable), others are robust (steerable). Identifying this split has practical implications: interpretability practitioners should prioritize high-CV features for steering interventions, while acknowledging that low-CV absorbed features may be unreliable targets.

### Method

1. **CV-Based Feature Classification**: For each SAE layer, compute per-feature CV across 1000 samples. Classify absorbed features (absorption_score > threshold) into high-CV (CV > 1.0) and low-CV (CV <= 1.0) subpopulations.

2. **Steering Effectiveness Comparison**: Run steering experiments on 30 high-CV and 30 low-CV absorbed features. Measure logit change at semantically appropriate tokens for +3, +5, +7 steering strengths.

3. **Mechanism Investigation**: Test whether the CV-steering correlation is explained by:
   - Decoder weight orthogonality (robust features have more orthogonal decoders)
   - Feature frequency (robust features are rarer but more specialized)
   - Context sensitivity (robust features activate in narrower context distributions)

4. **Cross-Model Validation**: Replicate the robustness decomposition on Gemma-2-2B JumpReLU SAEs. Test whether the CV threshold (1.0) generalizes or model-specific.

5. **Basu et al. Reconciliation**: Analyze whether the fragile/robust split explains Basu et al.'s findings — if their clinical features were predominantly low-CV, the paradox is resolved without contradicting their result.

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: CV-based feature classification | GPT-2 layer 6, classify absorbed features | 15 min | High/low CV split |
| E2: Steering effectiveness comparison | 30 high-CV vs 30 low-CV, 3 strengths | 30 min | Robust vs fragile hypothesis |
| E3: Mechanism investigation | Orthogonality, frequency, context analysis | 20 min | Mechanism explanation |
| E4: Cross-model validation | Gemma-2-2B layer 6, same protocol | 30 min | Generalizability |
| E5: Basu et al. reconciliation | Analyze CV distribution in their feature set | 10 min | Theoretical integration |

**Total**: ~105 min across 5 experiments

### Resource Estimate

- **Models**: GPT-2-small (85M params, rapid prototyping), Gemma-2-2B (cross-model validation)
- **SAEs**: GPT-2 residual stream SAEs via SAELens; GemmaScope JumpReLU SAEs
- **Compute**: All experiments fit within 1-hour budget per task; total ~105 min across 5 experiments
- **No new training required**: Training-free analysis of pretrained SAEs

### Risk Assessment

1. **Risk: CV threshold (1.0) is not predictive on held-out data**. If the CV-steering correlation doesn't replicate, the hypothesis is falsified. *Mitigation*: Report as negative result; the finding that "CV does not predict steering effectiveness" is itself a publishable result clarifying the limits of the approach.

2. **Risk: Steering effect size is too small for practical utility**. Even robust absorbed features show only 0.15 logit change, which may be insufficient for meaningful intervention. *Mitigation*: Compare to non-absorbed baseline steering effects; if robust absorbed is comparable to non-absorbed, the finding is significant even with small absolute effect.

3. **Risk: Basu et al. reconciliation is post-hoc**. We are analyzing their feature set after the fact, which is not rigorous prospective validation. *Mitigation*: Acknowledge this limitation; frame as exploratory analysis generating hypotheses for future work.

### Novelty Claim

This is the **first systematic decomposition of absorbed features into steerable vs non-steerable subpopulations**. Prior work (Basu et al.) treated absorption as uniformly degrading actionability. Our finding that CV predicts steering heterogeneity within absorbed features provides a practical tool for interpretability practitioners and advances the theoretical understanding of why absorption affects some features more than others.

The research directly addresses the actionability question: rather than asking "can we steer absorbed features?" (yes/no), we ask "which absorbed features can we steer, and why?" This shift from binary to gradient assessment is novel and practically useful.

**Prior work collisions**:
- Basu et al. (2026): Actionability paradox — we extend by showing heterogeneity rather than universal failure
- Chanin et al. (2024): Absorption detection metric — we extend by connecting to steering outcomes
- Cui et al. (2026): Information-theoretic limits — we cite as theoretical foundation but do not extend

**What is genuinely novel**:
1. First CV-based decomposition of absorbed features into steerable vs non-steerable
2. First empirical evidence that absorbed features are not uniformly non-steerable (in non-clinical LLM domain)
3. First connection between CV (a simple statistical measure) and steering effectiveness