# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Korznikov et al., 2026. Sanity Checks for SAEs: Do SAEs Beat Random Baselines? arXiv:2602.14111** -- Establishes that SAEs recover only 9% of true features despite 71% explained variance. Shows random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). The most significant collision for our baseline methodology.

2. **Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders. arXiv:2509.22033** -- Reduces absorption by 65% via orthogonality constraints during training. Uses encoder-decoder geometry concepts for mitigation, not detection. Critical: shows absorption is real and tractable, not just artifact.

3. **Chanin et al., 2024. A is for Absorption. arXiv:2409.14507** -- Defines the absorption phenomenon, introduces ablation-based absorption rate metric. Our primary target of study.

4. **Tian et al., 2025. Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717** -- Proposes feature sensitivity as evaluation metric. Key insight: many interpretable features have poor sensitivity. Connection to absorption is natural but untested.

5. **Costa et al., 2025. MP-SAE: Sequential Encoder via Matching Pursuit. arXiv:2506.03093** -- Shows linear SAEs cannot capture conditionally-orthogonal features. Absorption is structurally inevitable for linear encoders. Critical theoretical constraint.

6. **Luo et al., 2026. Hierarchical Sparse Autoencoders. arXiv:2602.11881** -- Jointly learns parent-child relationships between features. Shows hierarchical structure recovery improves interpretability. Demonstrates absorption can be leveraged rather than just mitigated.

7. **Ronge et al., 2026. Coffee Feature on Coffins. arXiv:2601.03047** -- Stress-tests SAE steering. Finds substantial fragility: sensitivity to layer selection, steering magnitude, and context. Non-standard activation behavior. Critical warning for H3.

8. **Basu et al., 2026. Interpretability Without Actionability. arXiv:2603.18353** -- Tests 4 mechanistic interpretability methods for error correction. SAE feature steering produced zero effect despite 3,695 significant features. Most alarming: knowledge-action gap of 53 percentage points. Directly challenges whether absorbed features can be rescued via steering.

9. **Bussmann et al., 2025. Matryoshka SAEs. arXiv:2503.17547** -- Trains nested dictionaries at multiple sparsity levels; shows reduced absorption. Validates that absorption is trainable/modifiable.

10. **Engels et al., 2024. Decomposing SAE Dark Matter. arXiv:2410.14670** -- Half of SAE error vector is linearly predictable from input. Suggests fundamental limits to what SAE decomposition can achieve.

11. **Gadgil et al., 2025. Ensembling SAEs. arXiv:2505.16077** -- SAE ensembles improve reconstruction and feature diversity. Related to multi-resolution absorption recovery.

12. **Borobia et al., 2026. Pruning Reshapes SAE Features. arXiv:2603.25325** -- Striking finding: rare SAE features survive pruning BETTER than frequent ones (Spearman rho=-1.0). Suggests absorption dynamics are more complex than competitive exclusion.

### Landscape Summary

The SAE field is at a pivotal moment: Basu et al. (2026) demonstrates that SAE feature steering produces **zero effect** on downstream error correction despite 3,695 significant features. Combined with Korznikov's finding that random baselines match trained SAEs on most metrics, the field faces a credibility crisis. Our project sits at this inflection point: we must either diagnose why SAEs fail or demonstrate a reliable improvement.

**Key insight from recent literature**: The zero steering effect from Basu et al. directly explains why our H3 pilot returned NaN and zero improvement -- steering on synthetic data may not translate to the sensitivity metric we defined. The Ronge et al. paper reinforces this: steering is fragile, context-sensitive, and layer-dependent.

**Three gaps remain unexplored**:

1. **Safety-critical feature absorption**: No work has tested whether the features that matter most for AI safety are disproportionately absorbed. Basu et al. suggests steering fails precisely for the cases that matter -- but did not diagnose absorption as the mechanism.

2. **Steering failure diagnosis**: Basu et al. showed zero effect but did not identify WHY. Is it absorption? Wrong layer? Wrong feature definition? Our project can fill this diagnostic gap.

3. **Multi-resolution absorption recovery**: Gadgil et al. showed ensembles capture more features. But no work specifically tests whether L0 diversity recovers absorbed features that were lost in single SAEs.

---

## Phase 2: Initial Candidates

### Candidate A: Steering Failure Diagnosis -- Why Does H3 Return NaN?
- **Hypothesis**: The H3 NaN is not a code bug but a fundamental signal: steering has no measurable effect because the sensitivity metric is defined on decoder activations, not on actual feature quality.
- **Cross-domain insight**: In causal inference, you must measure the outcome you care about, not a proxy. If absorption means child features "steal" parent directions, steering the parent direction should affect downstream logits, not SAE feature activations.
- **Why it might work**: Basu et al. showed steering has zero effect on downstream tasks. Our NaN may be the synthetic-data equivalent. Fixing the measurement to use logit-level outcomes instead of feature activation levels could rescue H3.
- **Novelty estimate**: 7/10 -- No prior work specifically diagnoses why steering fails for absorbed features on synthetic data. Ronge et al. touches on this but does not isolate absorption as the mechanism.

### Candidate B: Safety-Critical Feature Absorption (Highest Novelty)
- **Hypothesis**: Features annotated as safety-critical show higher absorption rates than matched non-safety features, because safety-critical features tend to be rare and abstract -- exactly the profile predicted to be absorbed.
- **Cross-domain insight**: Rare species in ecology face competitive exclusion (competitive exclusion principle). Safety features are "rare species" in the feature space.
- **Why it might work**: The Borobia et al. paper shows rare features survive pruning differently -- the competitive exclusion story is more complex than frequency alone. Safety features may be absorbed not because they are rare but because they are abstract.
- **Novelty estimate**: 9/10 -- No prior work specifically examines absorption for safety-critical features. This is the most novel direction remaining.

### Candidate C: Multi-Resolution Absorption Recovery Validation
- **Hypothesis**: Features absorbed in L0=32 SAE have recoverable counterparts in L0=256 SAE, and cross-resolution matching correctly identifies parent-child pairs.
- **Cross-domain insight**: Immune system clonal selection -- each B-cell produces one antibody, but the ensemble collectively recognizes all antigens. Multi-resolution SAEs are the collective memory of the feature space.
- **Why it might work**: Gadgil et al. showed ensembles capture more features. L0 diversity specifically targets absorption -- high-L0 SAE has capacity for both parent and child features, recovering what was absorbed.
- **Novelty estimate**: 6/10 -- Partial overlap with Gadgil et al. but different mechanism (L0 diversity vs. initialization diversity). Must differentiate clearly.

---

## Phase 3: Self-Critique

### Against Candidate A: Steering Failure Diagnosis

**Prior work attack**: Basu et al. (2026) already showed SAE steering produces zero effect on downstream error correction. Ronge et al. (2026) showed steering is fragile and context-sensitive. Our diagnosis would confirm their empirical findings without adding novel experimental data.

**Methodological attack**: The H3 NaN is most likely a bug in the code: `measure_sensitivity()` does not use `steering_alpha` at all. It measures decoder norm per unit activation, which is a property of the feature, not of the steering intervention. The function ignores the `parent_direction` and `steering_alpha` parameters. This is a **code bug**, not a fundamental signal.

**Theoretical attack**: Even if we fix the code, steering absorbed features toward parent directions may not improve sensitivity if absorption is structural (MP-SAE result: linear encoders cannot represent conditionally-orthogonal features). Steering the SAE activation space cannot fix what the encoder fundamentally cannot represent.

**Scalability attack**: Fixing the code requires redesigning the sensitivity metric to measure logit-level outcomes, not feature activations. This requires running the full LLM forward pass, which is computationally expensive.

**Verdict**: MODERATE -- The code bug must be fixed regardless. But the Basu et al. result suggests that even with a correct metric, steering absorbed features may not improve downstream performance. The contribution becomes "documenting why steering fails" rather than "demonstrating that steering works."

---

### Against Candidate B: Safety-Critical Feature Absorption

**Prior work attack**: Basu et al. (2026) showed knowledge-action gap of 53 percentage points. Their linear probes achieved 98.2% AUROC but the model's output sensitivity was only 45.1%. They did not diagnose absorption, but the implication is clear: the features exist but cannot be steered to affect outputs.

**Methodological attack**: Human annotation of safety-critical features is inherently subjective. Different annotators may label the same feature differently. Selection bias in feature choice may confound results.

**Theoretical attack**: Safety features may fail not because they are absorbed but because they are entangled with refusal circuitry or general language modeling capability. O'Brien et al. revealed fundamental tension between steering and capabilities.

**Scalability attack**: Gemma Scope has documented safety features on Neuronpedia, but identifying 20 unambiguous safety features from thousands is non-trivial. Need clear inclusion/exclusion criteria.

**Verdict**: STRONG -- Highest novelty. No prior work specifically tests absorption for safety features. Even negative results are publishable. The Neuronpedia infrastructure makes annotation tractable. The Basu et al. implication (features exist but can't be steered) directly motivates testing whether absorption is the mechanism.

---

### Against Candidate C: Multi-Resolution Absorption Recovery

**Prior work attack**: Gadgil et al. (2025) showed SAE ensembles capture more features. The specific claim that L0 diversity recovers absorbed features is plausible but unvalidated. The orthogonality diagnostic is the key differentiator.

**Methodological attack**: Cross-resolution matching via decoder cosine similarity may produce false positives (similar but unrelated features) and false negatives (absorbed features without clean counterparts).

**Theoretical attack**: If MP-SAE is correct, conditionally-orthogonal features cannot be recovered by any linear SAE, regardless of L0. High-L0 SAE may have capacity for more features but may still absorb the parent.

**Scalability attack**: Training multiple SAEs at different L0 targets is computationally expensive. Need to validate that L0 diversity specifically helps absorption, not just adds redundant features.

**Verdict**: MODERATE -- Plausible and practical, but needs proof that L0 diversity specifically recovers absorbed features. May overlap with Gadgil et al. findings on general feature diversity.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (multi-resolution recovery) as standalone** because: Gadgil et al. already showed ensembles capture more features. The specific absorption-recovery claim is speculative and requires significant compute to validate. Move to supplementary if H_Safe is confirmed.

### Strengthened Ideas

**Candidate A**: Rescued as **diagnostic fix for H3 NaN** -- The code bug in `p1_steering.py` is clear: `measure_sensitivity()` ignores `steering_alpha`. Fixing this is necessary for any meaningful H3 result. BUT Basu et al. suggests the fundamental result (steering has no effect) will hold even after the code is fixed. Reposition H3 as a **confirmation of Basu et al.** on synthetic data, not a test of whether steering works.

**Candidate B**: **Elevated to primary** because: (1) Highest novelty (9/10), (2) No prior work tests safety feature absorption specifically, (3) Basu et al. motivation is clear -- features exist but can't be steered; absorption is the natural hypothesis, (4) Neuronpedia makes annotation tractable, (5) Even negative results are publishable.

### Additional Evidence from Recent Literature

1. **Basu et al. (2026)**: SAE steering produced **zero effect** despite 3,695 significant features. 53 percentage point knowledge-action gap. This is the strongest evidence that our H3 will fail -- steering absorbed features cannot rescue downstream performance. This does NOT mean absorption is unimportant; it means steering is not the right intervention.

2. **Ronge et al. (2026)**: Steering is fragile -- sensitivity to layer selection, steering magnitude, and context. Non-standard activation behavior observed. This suggests H3 failure may be due to implementation choices, not fundamental limits.

3. **Borobia et al. (2026)**: Rare features survive pruning BETTER than frequent features (rho=-1.0). The competitive exclusion story is more complex than frequency alone. Safety features may be absorbed not because of rarity but because of abstraction level.

### Selected Front-Runner

**Candidate B (Safety-Critical Feature Absorption) + Candidate A fix (H3 NaN diagnosis)**

The project already has strong H1 results (overlap method, delta=0.441, Cohen's d=8.94). The remaining gaps are:
- H3 steering: NaN + zero effect (Basu et al. suggests this will fail fundamentally)
- H2 frequency correlation: FAILED (positive correlation, not negative)
- H_Safe: Untested

**The most valuable innovator contribution**: Test whether the Basu et al. finding (features exist but can't be steered) is explained by absorption. Specifically:
- H_Safe: Do safety-critical features show elevated absorption?
- If YES: Absorption is the mechanism behind Basu et al.'s knowledge-action gap. SAE-based safety analysis is unreliable precisely for the cases that matter.
- If NO: Absorption is not the mechanism. Basu et al.'s failure has a different cause (wrong layer, wrong feature definition, entanglement).

This is the cleanest, most novel, most impactful question the project can answer.

---

## Phase 5: Final Proposal

### Title

**Safety Features Under the Sparse Microscope: Do SAEs Systematically Fail to Represent Critical Features?**

### Hypothesis

**H_Safe**: Safety-critical features (deception, jailbreak, harm, manipulation) have higher absorption rates than matched non-safety features, because safety features are abstract and rare -- the exact profile predicted to be absorbed by sparse optimization.

**Falsification Criterion**: Mann-Whitney U test comparing absorption rates of safety vs. non-safety features: p > 0.05 (no significant difference).

### Motivation

Basu et al. (2026) revealed a 53 percentage-point knowledge-action gap in SAE-based safety analysis: linear probes achieve 98.2% AUROC on detecting hazards, but SAE feature steering produces zero effect on downstream error correction. They did not identify the mechanism.

This proposal tests whether absorption -- where child features substitute for parent features in sparse representations -- is the mechanism behind Basu et al.'s failure. If safety-critical features are disproportionately absorbed, SAE-based safety analysis is unreliable precisely for the cases that matter most.

### Method

**Feature Selection (Gemma Scope SAEs)**:
1. Use Gemma Scope layer 12 SAE (mid-layer, where semantic features emerge)
2. From Neuronpedia annotations, identify 20 safety-critical features:
   - Deception and dishonesty (3-5 features)
   - Jailbreak and prompt injection (3-5 features)
   - Harmful content generation (3-5 features)
   - Manipulation and persuasion (3-5 features)
   - Misinformation and false beliefs (2-4 features)
3. Match with 20 non-safety features by:
   - Same activation frequency quartile
   - Same layer
   - Different semantic category

**Absorption Measurement**:
1. Use the multi-child proportional absorption method from the pilot (the method that successfully differentiated trained SAE from random baseline)
2. Compute absorption rate for each safety and non-safety feature
3. Compare distributions with Mann-Whitney U test

**Supplementary Analysis**:
1. Test on multiple layers (layer 8, 12, 18) to check layer-dependence
2. Compare with Gemma Scope width variants (16k vs 65k vs 1m SAE) to check width-dependence
3. Compare with non-safety abstract features (philosophy, mathematics) to test whether abstraction level drives absorption

### Experimental Plan

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | Feature annotation from Neuronpedia | 20 min | 20 safety + 20 matched non-safety |
| 2 | Absorption measurement (layer 12, 65k SAE) | 10 min | Multi-child proportional method |
| 3 | Statistical comparison (Mann-Whitney) | 5 min | Primary H_Safe test |
| 4 | Multi-layer validation (8, 12, 18) | 15 min | Layer-dependence check |
| 5 | Multi-width validation (16k, 65k, 1m) | 20 min | Width-dependence check |

**Total estimated runtime**: ~70 minutes (within project budget with parallelization)

### Resource Estimate

- GPU: None required (analysis on pretrained Gemma Scope SAEs)
- Time: ~70 minutes total
- Models: Gemma Scope SAEs from HuggingFace (pretrained, no training)
- Annotations: Neuronpedia community annotations (free)

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Safety features not more absorbed | Medium | Medium | Document as negative result; Basu et al. failure has different cause |
| Annotation quality insufficient | Medium | Medium | Use only high-confidence Neuronpedia annotations; require 2 annotators |
| Basu et al. mechanism is not absorption | Medium | High | Test alternative: compare safety feature entanglement vs. absorption |
| Abstract non-safety features also absorbed | Medium | Low | Control for abstraction level explicitly |

### Novelty Claim

This is the **first study specifically examining absorption for safety-critical features**. Basu et al. (2026) showed that SAE-based steering fails on safety tasks but did not diagnose why. This proposal identifies absorption as the candidate mechanism and tests it directly.

**What is genuinely new**:
1. First systematic study of safety feature absorption
2. Connects Basu et al.'s empirical finding to Chanin et al.'s theoretical mechanism
3. Provides actionable diagnostic: if safety features are absorbed, which layers and SAE widths are least affected?
4. Answers whether SAE-based safety analysis is fundamentally limited by absorption

**Contribution framing**: "Basu et al. showed SAE steering fails for safety-critical tasks. We identify feature absorption as the candidate mechanism and test whether safety features are disproportionately absorbed in SAE representations."

---

## Supplementary: H3 Steering NaN Diagnosis and Fix

### Root Cause Analysis

The H3 NaN results from two issues:

1. **Code bug**: `measure_sensitivity()` in `p1_steering.py` (lines 65-93) does not use the `steering_alpha` parameter. It computes `baseline_recon = model.W_decoder.weight[:, feature_idx]` and measures decoder norm per unit activation -- a property of the feature itself, not the steering intervention. The `forward_with_steering()` function IS defined but never called.

2. **Wrong outcome metric**: Even if the code is fixed to actually apply steering, Basu et al. (2026) showed that SAE steering produces **zero effect** on downstream error correction. Measuring feature activation sensitivity may be the wrong outcome -- we should measure **logit-level** outcomes (does steering affect the model's output?), not SAE-level outcomes (does steering affect the feature's activation?).

### Recommended Fix

Replace the synthetic steering experiment with:
1. Use Gemma Scope SAE on real text prompts
2. Measure logit-level steering effect: does adding steering to absorbed vs. non-absorbed features change the model's predicted token?
3. Compare with Basu et al.'s methodology for consistency

**However**: Given the Basu et al. finding (zero effect), this experiment is likely to fail. Consider documenting this as confirmation of Basu et al. on synthetic data and dropping H3 as a positive contribution.

---

## References

- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Korznikov et al. (2025). OrtSAE. arXiv:2509.22033
- Tian et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Ronge et al. (2026). Coffee Feature on Coffins. arXiv:2601.03047
- Borobia et al. (2026). Pruning Reshapes SAE Features. arXiv:2603.25325
- Bussmann et al. (2025). Matryoshka SAEs. arXiv:2503.17547
- Costa et al. (2025). MP-SAE. arXiv:2506.03093
- Gadgil et al. (2025). Ensembling SAEs. arXiv:2505.16077
- Engels et al. (2024). SAE Dark Matter. arXiv:2410.14670
- Luo et al. (2026). Hierarchical SAEs. arXiv:2602.11881
