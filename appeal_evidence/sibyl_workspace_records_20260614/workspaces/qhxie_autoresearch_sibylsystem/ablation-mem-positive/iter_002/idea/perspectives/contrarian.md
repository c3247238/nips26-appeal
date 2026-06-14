# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a significant, actionable problem in SAEs**
   - Evidence challenging it:
     - Basu et al. (2026) "Interpretability without Actionability": 98.2% internal feature detection AUROC translates to **zero output change** via SAE steering
     - Chanin et al. (2024) only tested on early layers (0-17) with ablation metric that becomes unreliable past layer 17
     - suppression_ratio is uniformly 1.0 in pilot experiments, indicating the detection metric itself is degenerate
   - **Verdict**: The field measures absorption but cannot act on that knowledge

2. **Assumption: Detecting absorption enables downstream improvement**
   - Evidence challenging it:
     - All absorption mitigation methods (Matryoshka SAE, OrtSAE, ATM, MP-SAE, Masked Regularization) require **retraining from scratch**
     - The pretrained SAE ecosystem (GemmaScope, LlamaScope) cannot benefit from these advances without retraining
     - Cui et al. (ICLR 2026) prove standard SAEs generally cannot fully recover ground-truth features — absorption may be **mathematically inevitable** under realistic sparsity constraints
   - **Verdict**: Detection without remediation is academically interesting but practically useless

3. **Assumption: Feature absorption degrades interpretability reliability**
   - Evidence challenging it:
     - E5 downstream impact: absorbed features have **lower** coefficient of variation (1.07 vs 1.46, p=0.005) — they are MORE stable, not less
     - Chanin et al. only tested first-letter spelling task; generalization to semantic hierarchies unproven
     - SAEBench's absorption metric is computationally expensive (~26 min/SAE) and noisy
   - **Verdict**: The "degradation" narrative is unproven; absorbed features may be simply different, not worse

4. **Assumption: Cross-layer absorption patterns are real and generalizable**
   - Evidence challenging it:
     - Layer 6 shows 0.549% absorption rate vs layer 0 at 0.024% — but this is a 23x range on very small absolute numbers
     - No high-confidence pairs (>0.7) found anywhere; max score is 0.63
     - Cross-model validation was skipped (Gemma SAE unavailable)
   - **Verdict**: Patterns could be noise; cannot generalize without cross-model confirmation

5. **Assumption: The ablation-based absorption metric accurately measures the phenomenon**
   - Evidence challenging it:
     - suppression_ratio = 1.0 uniformly across all pairs in E1 — the metric has zero discrimination
     - E4 shows co-occurrence correlates NEGATIVELY with absorption score (r=-0.52), contradicting the theoretical model
     - The scoring formula weights are likely wrong; the phenomenon definition may be inverted
   - **Verdict**: The measurement instrument is broken; we may be measuring something else entirely

### Landscape of Doubt

The feature absorption research program rests on four shaky pillars:

1. **Measurement validity**: The primary detection metric (ablation-based suppression_ratio) is degenerate. If we cannot measure the phenomenon accurately, we cannot study it.

2. **Actionability gap**: Basu et al. (2026) demonstrate that even PERFECT internal detection (98.2% AUROC) yields zero steering effect. The field is quantifying a phenomenon it cannot interventionally address.

3. **Theoretical limits**: Cui et al. prove full disentanglement is mathematically impossible under realistic sparsity. High absorption may be an *inevitable* consequence of representational interference, not a fixable training artifact.

4. **Ecological validity**: All absorption studies use spelling tasks or synthetic proxies. Real-world interpretability applications (circuit discovery, model steering, concept erasure) show the opposite of what the absorption metric predicts.

**The emperor has no clothes**: The field measures absorption, publishes benchmarks, designs mitigations — but nobody has shown that reducing absorption actually improves downstream interpretability tasks in pretrained SAEs. The entire research program may be a sophisticated circularity.

---

## Phase 2: Initial Candidates

### Candidate A: "Feature Absorption May Be a Phantom — The Measurement Is Broken"
- **Challenged assumption**: The ablation-based absorption metric accurately captures a real phenomenon
- **Evidence against**: suppression_ratio = 1.0 uniformly; co-occurrence negatively correlates with score (r=-0.52); no high-confidence pairs found
- **Contrarian hypothesis**: The absorption "signal" is a statistical artifact of sparse coding geometry, not a distinct failure mode
- **Exploitation plan**: Replicate the measurement with synthetic ground-truth data (SynthSAEBench) and prove the metric can/cannot recover known absorption
- **Novelty estimate**: 8/10 (nobody has questioned the measurement validity head-on)

### Candidate B: "Actionability Crisis — Absorption Quantification Is Pointless"
- **Challenged assumption**: Quantifying absorption enables better interpretability outcomes
- **Evidence against**: Basu et al. (2026): 98.2% internal AUROC → 0% output change; all mitigations require retraining; pretrained SAEs are stuck
- **Contrarian hypothesis**: The field is solving an academically interesting but practically irrelevant problem
- **Exploitation plan**: Demonstrate that absorbed features, when used for steering, produce indistinguishable outputs from non-absorbed features — the "absorption" label is taxonomically interesting but causally inert
- **Novelty estimate**: 7/10 (actionability critique exists in Basu but has not been connected to absorption specifically)

### Candidate C: "Theoretical Inevitability — Absorption Is Not a Bug But a Feature"
- **Challenged assumption**: SAEs should fully disentangle hierarchical features for interpretability to work
- **Evidence against**: Cui et al. prove full disentanglement is mathematically impossible; absorbed features show lower CV (more stable); sparsity requires compression
- **Contrarian hypothesis**: Absorption represents efficient hierarchical encoding; "splitting" features to avoid absorption would reduce representation efficiency and hurt model performance
- **Exploitation plan**: Show that SAE configurations with higher absorption also have better reconstruction quality and lower loss recovered penalty
- **Novelty estimate**: 6/10 (Cui touches on this but does not frame it as "absorption is good")

---

## Phase 3: Self-Critique

### Against Candidate A (Measurement Is Broken)
- **Steelman**: The metric may be degenerate in early layers with small models (GPT-2-small), but ablation-based absorption was validated on Gemma-2-2B by Chanin et al. with real signal. The issue may be model-specific, not fundamental.
- **Cherry-picking attack**: E1-E5 all used GPT-2-small SAEs. If the metric works on Gemma but not GPT-2, this is a model architecture confound, not evidence against the metric.
- **Confounding attack**: The negative co-occurrence correlation (r=-0.52) could reflect a real structural pattern — absorbing pairs have HIGH co-occurrence (they appear together), which is WHY absorption happens. The correlation may be correctly signed but interpreted incorrectly.
- **Actionability attack**: Even if the metric is broken, this does not suggest a constructive alternative. A paper saying "we cannot measure absorption" is a dead end.
- **Verdict**: MODERATE — The measurement critique is valid for GPT-2-small but does not generalize. Needs cross-model validation before declaring the metric broken.

### Against Candidate B (Actionability Crisis)
- **Steelman**: Basu et al. is a single paper on a clinical domain (medical reasoning). The zero-output-change result may be domain-specific. Generalizing one negative result is exactly the kind of reasoning the contrarian should avoid.
- **Cherry-picking attack**: The field HAS produced actionable results — e.g., Steering GPT-2 with SAE features changes behavior (Cunningham et al., 2023). The actionability crisis may be the exception, not the rule.
- **Confounding attack**: Basu et al. uses probes that achieve 98.2% AUROC internally but the probe may be detecting a correlate, not the causal feature. This does not prove absorption is irrelevant — it proves that probe quality does not equal actionability.
- **Actionability attack**: This critique, if valid, KILLS the entire research direction. If we cannot act on absorption knowledge, why study it? This is a "gotcha" paper at best.
- **Verdict**: STRONG — The Basu result is a genuine crisis for the field. But the candidate framing is too nihilistic. The correct framing is "absorption does not imply actionability" — this is a nuance worth exploring, not a kill switch.

### Against Candidate C (Absorption Is Inevitable/Good)
- **Steelman**: If absorption is mathematically inevitable (Cui et al.), then studying it is pointless and we should study something else. But Cui's result is about general disentanglement, not absorption specifically. Absorption may be inevitable BUT still a valid object of study.
- **Cherry-picking attack**: The CV difference (absorbed features more stable) is a single result on a single model. This could be noise or a specific property of GPT-2 architecture.
- **Confounding attack**: Lower CV could reflect that absorbed features are simply lower-magnitude or more frequently active, not that they are "better" encoded. This is a confound, not evidence.
- **Actionability attack**: If absorption is "good" (efficient encoding), this is interesting but does not lead to a publishable improvement. The paper would be "we found something interesting but it does not help."
- **Verdict**: WEAK — The theoretical inevitability claim is overstated. The CV evidence is suggestive but confounded. This is too nihilistic even for a contrarian.

---

## Phase 4: Refinement

### Dropped Candidates
- **Candidate C**: Too nihilistic; "absorption is good" does not lead to actionable research. The CV evidence is confounded by feature frequency.

### Strengthened Candidates
- **Candidate B** (Actionability Crisis): Connect specifically to Basu et al. framing. The key insight: "absorption ≠ steerability" is a publishable finding that differentiates from previous work.
- **Candidate A** (Measurement Broken): Reframe as "measurement domain mismatch" — ablation works on Gemma (large model, JumpReLU SAE) but not GPT-2 (small model, TopK SAE). This is a discovery about generalization limits, not a kill switch.

### Additional Corroboration
- Gurnee et al. (2023) k-sparse probing shows feature splitting is detectable
- Costa et al. (NeurIPS 2025) MP-SAE demonstrates conditional orthogonality — features CAN be extracted hierarchically
- The negative co-occurrence correlation (r=-0.52) is REAL and unexplained — this alone is a publishable empirical finding

### Selected Front-Runner
**Candidate B + A hybrid**: "Rethinking Feature Absorption: Measurement Validity and Actionability Limits in Pretrained SAEs"

The key contrarian insight: The field assumes (1) absorption is measurable, (2) absorption implies actionability, and (3) reducing absorption improves interpretability. Evidence suggests all three assumptions fail for pretrained SAEs:
- (1) fails: ablation metric degenerate on GPT-2-small, no cross-model validation
- (2) fails: Basu et al. (2026) — 98.2% AUROC but zero steering effect
- (3) untested: No paper has shown that reducing absorption improves any downstream task in pretrained SAEs

---

## Phase 5: Final Proposal

### Title
**Rethinking Feature Absorption: On the Measurement Validity and Practical Limits of SAE Feature Disentanglement**

Alternative titles:
- "When Absorption Meets Actionability: A Critical Re-examination of SAE Feature Decomposition"
- "The Phantom Menace: Does Feature Absorption Actually Degrade Interpretability Outcomes?"

### Challenged Assumption
The field assumes that: (a) feature absorption is a measurable, significant problem in pretrained SAEs, (b) quantifying absorption enables practical improvements in interpretability applications, and (c) reducing absorption through architectural changes improves downstream task performance. This proposal challenges all three assumptions.

### Evidence

**For the assumption**:
- Chanin et al. (NeurIPS 2025) provides the first systematic study, demonstrating absorption exists and is detectable via ablation on Gemma-2-2B
- Cross-layer variation (0.024% to 0.549%) shows structured patterns not random noise
- OrtSAE reduces absorption by 65% in controlled settings
- MP-SAE recovers hierarchical structure that standard SAEs miss

**Against the assumption**:
- **Measurement**: suppression_ratio = 1.0 uniformly in GPT-2-small pilots; co-occurrence negatively correlates with score (r=-0.52); no high-confidence pairs found
- **Actionability**: Basu et al. (2026) — 98.2% internal probe AUROC but 0% output change via steering
- **Generalization**: Chanin only tested early layers (0-17) on Gemma; GPT-2 validation absent; ablation unreliable past layer 17
- **Theoretical limit**: Cui et al. (ICLR 2026) prove full disentanglement is mathematically impossible under realistic sparsity
- **Downstream impact**: Absorbed features show LOWER CV (more stable), contradicting the "degradation" narrative

### Hypothesis
Feature absorption is a real phenomenon under specific conditions (large models, JumpReLU SAEs, early layers) but: (1) current detection metrics fail to generalize across architectures, (2) absorption does not imply loss of steerability in pretrained SAEs, and (3) the "degradation" framing is unsupported — absorbed features may be differently encoded rather than degraded.

### Method
1. **Systematic measurement comparison**: Run ablation-based absorption metric AND SAEBench's probe projection metric on SAME pretrained SAEs (GPT-2-small, Gemma-2-2B, Llama-3.2-1B) to establish metric agreement/disagreement
2. **Actionability test**: For top-scoring "absorbed" pairs, perform SAE steering and measure output change — does absorption predict steering failure?
3. **Downstream impact quantification**: Compare circuit discovery reliability, steering effect size, and concept erasure success for absorbed vs matched non-absorbed features

### Experimental Plan
| Experiment | Model | SAE | Duration |
|---|---|---|---|
| E1: Metric Comparison | GPT-2-small, Gemma-2-2B | Layer 6 SAEs | 30 min |
| E2: Steering Actionability | GPT-2-small | Top 20 absorbed pairs, top 20 matched controls | 40 min |
| E3: Cross-Model Validation | Llama-3.2-1B | Per-layer SAEs | 30 min |

Total: ~100 minutes, within 1-hour-per-task budget

### Baselines
- Mainstream method: Chanin et al. ablation-based absorption metric
- State-of-the-art: SAEBench probe projection metric
- Comparison: Steering effect size correlation with absorption score

### Risk Assessment
**If mainstream view is correct**:
- Absorption IS measurable and actionable: We publish a replication/cross-validation study confirming existing results
- Risk: Low (confirmatory publication is still valuable)

**If contrarian view is correct**:
- Measurement fails to generalize: We publish a critical re-examination prompting methodological reform
- Risk: Field may reject as "negative" or "kill the research program" — but this is a genuine contribution if the criticism is warranted

**Key risk**: We may find that absorption IS real and actionable after all, and this paper becomes a "defense of the mainstream" rather than a contrarian contribution. This is acceptable if the evidence leads there.

### Novelty Claim
This is the **first study to directly connect absorption quantification to downstream actionability in pretrained SAEs**, and the **first to systematically compare ablation-based and projection-based absorption metrics across model architectures**. The negative co-occurrence/score correlation (r=-0.52) is a **new empirical finding** that challenges the current theoretical model of absorption.
