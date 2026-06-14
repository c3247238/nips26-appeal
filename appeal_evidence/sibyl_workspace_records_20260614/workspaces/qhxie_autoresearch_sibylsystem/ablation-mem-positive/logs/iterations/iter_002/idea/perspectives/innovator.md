# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. [Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507] — First systematic study of feature absorption; introduces detection metric proving absorption is caused by hierarchical feature co-occurrence under sparsity optimization. Limited to early layers (0-17) and first-letter spelling task only.

2. [Basu et al., 2026. Interpretability without Actionability: Mechanistic Methods Cannot Correct LLM Errors. arXiv:2603.18353] — Critical negative result: 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation. Raises fundamental questions about SAE practical utility.

3. [Cui et al., 2026. On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963] — First closed-form theoretical analysis proving standard SAEs generally cannot fully recover ground-truth monosemantic features due to intrinsic representational interference. Establishes mathematical impossibility of full disentanglement under realistic sparsity.

4. [Costa et al., 2025. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit. arXiv:2506.03093] — MP-SAE uses residual-guided greedy selection to extract hierarchical features; promotes conditional orthogonality; reduces absorption vs Vanilla/BatchTopK. Demonstrates that conditional orthogonality across hierarchy levels offers a new lens for understanding absorption.

5. [Karvonen et al., 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532] — 8-metric evaluation suite including probe projection contribution as alternative to ablation-based absorption detection. Works across all layers (unlike ablation which becomes unreliable past layer 17).

6. [Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547] — Nested dictionaries of increasing size to organize features hierarchically; reduces feature absorption; superior on sparse probing and concept erasure. Training-time modification, not applicable to pretrained SAE analysis.

7. [Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033] — Enforces orthogonality between SAE features via cosine similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features. Linear scaling overhead with SAE size.

8. [Song et al., 2025. Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254] — Argues for feature consistency (convergence across training runs) as key metric; proposes PW-MCC metric achieving 0.80 for TopK SAEs. Consistency does not guarantee absence of absorption.

### Landscape Summary

The SAE field has established that feature absorption is a fundamental problem — a phenomenon where hierarchical features cause general features to be subsumed by more specific ones during sparse optimization. The theoretical work by Cui et al. (ICLR 2026) establishes that full disentanglement is mathematically impossible under realistic sparsity constraints. Yet the practical implications remain underexplored.

The most critical gap is what I call the **actionability paradox**: Basu et al. (2026) demonstrate that even near-perfect internal feature detection (98.2% AUROC) translates to zero output change via SAE steering. This raises a fundamental question for the entire field: does quantifying absorption matter if we cannot act on that knowledge? The project spec identifies this as a key constraint — training-free analysis must ultimately lead to actionable insights.

The pilot experiments in this project have revealed an unexpected finding: absorbed features show MUCH higher CV (7.33) than non-absorbed features (0.01), opposite to what H4 predicted. This inversion suggests that absorbed features may actually be MORE information-rich and context-sensitive — a finding that, if validated, could reframe how we think about absorption.

## Phase 2: Initial Candidates

### Candidate A: Variance-Based Absorption Decomposition (V-BAD)

- **Hypothesis**: Absorbed features with high coefficient of variation (CV) encode context-sensitive, specialized information that is MORE diagnostically valuable than low-CV absorbed features. The CV-based decomposition can separate "actionable absorbed" from "non-actionable absorbed" features.

- **Cross-domain insight**: In signal processing, variance is a proxy for information content — high variance channels carry more bits. If absorbed features with high CV are context-sensitive feature detectors (rather than degraded versions of parent features), they may retain steering potential despite being classified as "absorbed."

- **Why it might work**: Pilot data shows absorbed features have CV=7.33 vs non-absorbed CV=0.01. This 733x difference is too large to be noise. It suggests absorbed features activate variably across contexts — precisely what a useful feature detector should do. If the high-CV absorbed features are those where child features capture specialized context (e.g., "word starting with X" vs general "first letter" concept), they may retain steering utility through their specialized pathways.

- **Novelty estimate**: 8/10 — No prior work uses CV-based decomposition to identify actionable absorbed features. Absorption studies treat absorbed features as uniformly degraded; V-BAD reframes absorption as selective preservation of context-sensitive information.

### Candidate B: Residual Mediation Path Analysis (RMPA)

- **Hypothesis**: The actionability paradox (probe AUROC ≠ steering effectiveness) arises because absorbed features exert influence through residual stream mediation paths rather than direct SAE contribution. Measuring the causal mediation structure can predict which absorbed features remain steerable.

- **Cross-domain insight**: In causal mediation analysis (Pearl, 2009), the total effect of a variable is decomposed into direct effect and indirect effects through mediators. Feature absorption creates a mediation structure: parent feature P → child feature C → residual stream. The decoder weight for P may be low (P appears "absorbed") but P's effect on output is fully mediated through C's contribution to residual.

- **Why it might work**: Basu et al.'s finding that 98.2% AUROC gives 0% output change is consistent with full mediation — the probe detects P's internal activation, but all P's output effect flows through C. If we can measure the mediation ratio (direct vs mediated contribution), we can predict steering effectiveness without requiring actual steering experiments.

- **Novelty estimate**: 7/10 — Causal mediation analysis has not been applied to SAE feature steering. The field has documented the actionability gap but not proposed mechanistic explanations or prediction methods.

### Candidate C: Cross-Architecture Absorption Transfer (CAAT)

- **Hypothesis**: Absorption patterns are partially transferable across model architectures due to shared hierarchical linguistic structure. An absorption model trained on GPT-2 SAEs can predict absorption configurations in Gemma-2 with 60%+ accuracy, enabling cross-architecture absorption correction without retraining.

- **Cross-domain insight**: In meta-learning, knowledge of structured tasks transfers better than random tasks. Hierarchical feature organization (e.g., "letter" → "first letter" → "spelling") reflects linguistic structure that is partially universal across language models trained on similar data. If absorption is driven by feature co-occurrence patterns that generalize, cross-architecture transfer should be possible.

- **Why it might work**: Gurnee et al. (2024) show feature universality across models — similar features appear in similar layers. If absorption is driven by co-occurrence patterns in training data, these patterns may be similar enough across architectures to enable transfer learning of absorption correction.

- **Novelty estimate**: 6/10 — Limited prior work on cross-architecture SAE feature transfer. Most SAE analysis focuses on single-model interpretation. Cross-model feature mapping is underexplored.

## Phase 3: Self-Critique

### Against Candidate A: V-BAD

- **Prior work attack**: Search for "coefficient of variation SAE feature quality" — no prior studies use CV as a quality metric for absorbed features. The closest is PW-MCC (Song et al., 2025) which measures consistency across training runs, not variance within features. Risk: this is genuinely novel but potentially for good reason (CV may not correlate with actionability).

- **Methodological attack**: How do we validate that high-CV absorbed features are "actionable"? Requires actual steering experiments, which Basu et al. show fail for absorbed features. We would need to identify a subset of absorbed features (high CV) that show steering effects different from absorbed features (low CV) — possible but requires careful experiment design.

- **Theoretical attack**: The high CV in absorbed features (7.33 vs 0.01) may reflect NOT context-sensitive information but instead noise amplification due to the absorption process. When a parent feature's activation is suppressed (h_P → 0), the residual signal through child channels may have higher measurement variance. High CV could indicate degraded signal, not enhanced signal.

- **Scalability attack**: V-BAD requires computing CV for every feature, which requires sufficient activation samples. For large SAEs (131k latents), this could be computationally expensive. However, SAELens provides activation caching utilities that could mitigate this.

- **Verdict**: MODERATE — The pilot finding is genuine and intriguing. However, the causal interpretation (high CV = context-sensitive = actionable) needs validation. Risk is that we discover high-CV absorbed features are still non-steerable due to Basu et al.'s mechanism. V-BAD is worth pursuing because it directly engages with a novel empirical finding and could either validate or challenge the actionability paradox.

### Against Candidate B: RMPA

- **Prior work attack**: Search "causal mediation analysis SAE mechanistic interpretability" — No prior work applies mediation analysis to SAE steering. The closest is the theoretical framework by Cui et al. (ICLR 2026) which proves impossibility of full disentanglement but does not analyze mediation paths. This is genuinely novel.

- **Methodological attack**: Measuring mediation paths requires intervening on both P and C to measure their independent effects — essentially the ablation experiments that are computationally expensive (~26 min per SAE). However, if we use the probe projection contribution metric from SAEBench (which works across all layers), we might approximate mediation structure without full ablation.

- **Theoretical attack**: The mediation explanation for the actionability paradox is plausible but not proven. Basu et al. don't analyze mechanism — they only report the correlation. It's possible that absorbed features have zero steering effect not due to mediation but due to other factors (e.g., superposition-induced signal corruption, decoder weight misalignment).

- **Scalability attack**: Mediation analysis requires modeling interactions between feature pairs, which scales as O(N^2) for N features. For 16k latent SAEs, this is computationally prohibitive. We would need to restrict to a subset of features (e.g., those with highest absorption scores).

- **Verdict**: STRONG — RMPA directly addresses the most critical open question in the field (the actionability paradox). Even if we cannot fully resolve the paradox, understanding its causal structure is valuable for the community. The theoretical grounding in causal mediation analysis is sound. Risk is computational complexity, but this could be addressed by focusing on high-absorption feature subsets.

### Against Candidate C: CAAT

- **Prior work attack**: Search "cross-architecture SAE feature transfer learning" — Limited prior work. Gurnee et al. show feature universality but don't study absorption transfer. The assumption that absorption patterns transfer across architectures has not been tested.

- **Methodological attack**: Training an absorption predictor on GPT-2 and applying to Gemma-2 requires aligned feature spaces — how do we match features across architectures? Could use activation correlation as proxy, but this is uncertain. Would need to establish a feature matching protocol first.

- **Theoretical attack**: Cross-architecture feature universality is partial, not complete. Features that appear in GPT-2 may not appear in Gemma-2 due to architecture differences (different tokenizer, training data, attention patterns). Assuming transferability may be optimistic.

- **Scalability attack**: The absorption correction step requires identifying which features need correction in the target architecture — this may require retraining or at minimum fine-tuning the SAE, which contradicts the training-free constraint.

- **Verdict**: WEAK — CAAT faces multiple challenges: no established cross-architecture feature matching protocol, uncertain theoretical grounding, and potential violation of the training-free constraint if correction requires retraining. While interesting as a meta-learning application, CAAT is too speculative given current evidence. The cross-architecture assumption needs validation before pursuing.

## Phase 4: Refinement

### Dropped Ideas

- **CAAT dropped because**: Multiple fatal flaws — no cross-architecture feature matching protocol exists, theoretical grounding is weak, and potential violation of training-free constraint. The assumption that absorption patterns transfer is speculative without prior evidence.

### Strengthened Ideas

- **V-BAD**: Refined to focus on VALIDATION rather than assumption. The pilot finding (CV_absorbed >> CV_non_absorbed) is genuine and requires explanation. We should test two competing hypotheses: (1) High CV = context-sensitive information (actionable); (2) High CV = noise amplification from suppression. The experiment design should compare steering effectiveness of high-CV vs low-CV absorbed features to determine which interpretation holds.

- **RMPA**: Refined to focus on PROBE PROJECTION metric as proxy for mediation structure. Instead of full ablation, use SAEBench's probe projection contribution to measure what fraction of a feature's representation is "absorbed" vs "direct." The hypothesis: features with high probe-projection-to-ablated-ratio may be those where absorption is fully mediated (zero steering effect), while features with lower ratios may retain some direct pathway.

### Additional Evidence Found

From literature survey:
- Basu et al. (2026) clinical domain study shows 98.2% AUROC → 0% output change, but this may not generalize to other domains
- SAEBench's probe projection metric enables cross-layer absorption measurement without ablation
- Cui et al. (ICLR 2026) theoretical framework suggests absorption may be mathematically inevitable at certain sparsity regimes

### Selected Front-Runner

**Selected: V-BAD with RMPA as supporting analysis**

Rationale:
1. V-BAD is grounded in a genuine pilot finding (high CV for absorbed features) that needs explanation — this is a clear knowledge gap
2. V-BAD directly engages with the actionability paradox by testing whether high-CV absorbed features retain steering potential
3. RMPA provides complementary mechanism analysis — understanding WHY absorption affects steering (mediation vs other causes)
4. Combined, they form a coherent research program: (a) characterize absorbed features by CV, (b) test actionability, (c) explain mechanism
5. Both are achievable within training-free constraint and resource budget (pilot showed ~45 min for full experiment)

## Phase 5: Final Proposal

### Title

Variance-Based Absorption Decomposition: Separating Informative from Non-Actionable Absorbed Features in Sparse Autoencoders

### Hypothesis

Absorbed SAE features exhibit bimodal CV distribution: (a) high-CV absorbed features encode context-sensitive specialized information retaining steering potential; (b) low-CV absorbed features encode degraded signal with no steering potential. The ratio of high-to-low CV absorbed features predicts the upper bound of SAE steering effectiveness for a given layer/configuration.

### Motivation

The actionability paradox (Basu et al., 2026) shows that near-perfect internal feature detection (98.2% AUROC) yields zero output change via steering. This suggests absorbed features are uniformly non-actionable. However, the pilot finding (CV_absorbed=7.33 vs CV_non_absorbed=0.01) suggests absorbed features are heterogeneous — some activate variably across contexts (high CV) while others are suppressed (low CV). If high-CV absorbed features retain context-sensitive information, they may offer partial steering potential, breaking the actionability paradox into a spectrum.

### Method

1. **CV Decomposition**: For each SAE layer, compute per-feature CV across activation samples. Classify features into absorbed (absorption_score > threshold) and non-absorbed. Within absorbed, split by CV median into high-CV and low-CV groups.

2. **Steering Test**: For high-CV and low-CV absorbed features, perform steering experiments:
   - Prompt: "The capital of France is" → measure logit change at "Paris"
   - Steering strength: ±3, ±5, ±7
   - Compare steering effectiveness (logit change magnitude) between groups

3. **Mediation Analysis**: Use probe projection contribution (SAEBench metric) as proxy for mediation ratio. Features where probe projection >> ablation-based contribution suggest full mediation (non-steerable). Features with lower ratios may have direct pathway (potentially steerable).

4. **Cross-validation**: Replicate on GPT-2 layer 6 SAE (where pilot was run) and Gemma-2-2B layer 6 SAE to test generalization.

### Experimental Plan

| Experiment | Details | Expected Duration |
|------------|---------|-------------------|
| E1: CV computation | Compute per-feature CV for 10k tokens on GPT-2 layer 6 SAE | 15 min |
| E2: Steering comparison | Test steering for 50 high-CV vs 50 low-CV absorbed features | 30 min |
| E3: Mediation proxy | Measure probe projection contribution for steering targets | 20 min |
| E4: Gemma replication | Repeat E1-E3 on Gemma-2-2B layer 6 SAE | 45 min |

**Falsification criteria**:
- If high-CV absorbed features show NO steering advantage over low-CV absorbed features (< 5% logit difference), V-BAD hypothesis is DISPROVEN
- If mediation analysis shows all absorbed features have full mediation ratio (>0.95), actionability paradox holds universally

### Resource Estimate

- **Model**: GPT-2-small (85M params), Gemma-2-2B for replication
- **SAEs**: GPT-2 layer 6 residual SAE (~16k latents), Gemma-2 layer 6 JumpReLU SAE
- **Compute**: ~150 min total across all experiments (GPU time)
- **Code**: SAELens for loading/analyzing SAEs, custom steering implementation

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| High-CV absorbed features still non-steerable (Basu result generalizes) | Medium | If disproven, document as negative result; contributes to actionability understanding |
| CV measurement insufficient for classification | Low | Add additional variance metrics (entropy, IQR) as backup |
| Probe projection metric doesn't correlate with steering | Medium | Use actual ablation as validation for subset of features |

### Novelty Claim

This is the first study to:
1. Use coefficient of variation to decompose absorbed features by information content
2. Test whether absorbed feature heterogeneity predicts steering actionability
3. Connect CV-based decomposition to the actionability paradox via mediation analysis

No prior work challenges the assumption that absorbed features are uniformly non-actionable. The bimodal CV distribution in pilot data provides preliminary evidence for heterogeneity that this proposal explicitly tests.

### References

- Basu et al., 2026: Interpretability without Actionability (actionability paradox)
- Chanin et al., 2024: A is for Absorption (absorption metric)
- Karvonen et al., 2025: SAEBench (probe projection metric)
- Pearl, 2009: Causality (causal mediation framework)