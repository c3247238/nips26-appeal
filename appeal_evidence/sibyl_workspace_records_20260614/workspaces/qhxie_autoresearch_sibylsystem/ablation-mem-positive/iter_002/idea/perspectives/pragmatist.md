# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. [SAELens (jbloomAus/SAELens)](https://github.com/jbloomAus/SAELens) — Primary library for training and analyzing SAEs on LLMs. Supports pretrained SAE loading (`SAE.from_pretrained`), feature visualization, activation caching. MIT license. **Code exists, production-ready.**

2. [SAEBench (adamkarvonen/SAEBench)](https://github.com/adamkarvanen/SAEBench) — 8-metric evaluation suite including absorption measurement. Implements probe projection approach that works across all layers. MIT license. **Code exists, actively maintained.**

3. [sae-spelling (lasr-spelling/sae-spelling)](https://github.com/lasr-spelling/sae-spelling) — Direct implementation of Chanin et al. absorption metric. Unknown license. **Code exists but may need adaptation for non-spelling features.**

4. [GemmaScope Pretrained SAEs](https://huggingface.co/collections/google-deepmind/gemma-scope-66c64b07b3a5e34e2a5b6912) — Comprehensive SAE suite for Gemma-2 (2B/9B). JumpReLU architecture, all layers. Gemma License. **Ready to use via SAELens.**

5. [GPT-2 SAEs via SAELens](https://github.com/jbloomAus/SAELens/tree/main/saeLens/src/sae_lens/training) — Residual stream SAEs for GPT-2-small. MIT license. **All layers available, ideal for rapid prototyping.**

6. [MP-SAE (mpsae/MP-SAE)](https://github.com/mpsae/MP-SAE) — Matching Pursuit SAE for hierarchical feature extraction. NeurIPS 2025. **Code exists but requires retraining SAEs, not training-free.**

7. [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — Hook-based transformer introspection. Essential for activation extraction and intervention. MIT license. **Integrates with SAELens seamlessly.**

8. [SynthSAEBench](https://github.com/DavidChanin/SynthSAEBench) — Synthetic data benchmark with ground-truth features for controlled validation. Unknown license. **Useful for validating detection metrics.**

### Landscape Summary

The feature absorption literature is rich but fragmented. Key observations for an engineering-minded researcher:

1. **The detection metric is broken in practice**: The ablation-based suppression_ratio in the original Chanin et al. formula is uniformly 1.0 across all pairs (E1 finding). The scoring formula has internal contradictions — co-occurrence correlates negatively with absorption score (r=-0.52 in E4). This is a red flag: the metric may be measuring the wrong thing.

2. **Cross-layer patterns are the real signal**: The most robust finding across all experiments is that Layer 6 is consistently an "absorption hotspot" (0.549% rate, highest mean edge weight 0.559 in absorption graphs, 52% medium-score pairs). This is a reproducible, non-trivial pattern that generalizes across metrics.

3. **Downstream impact is measurable and significant**: E5 shows absorbed features have significantly lower coefficient of variation (1.07 vs 1.46, p=0.005). This is the strongest statistical result and does not depend on the broken detection metric.

4. **Training-free analysis is the right approach**: Given the 1-hour time constraint and the availability of pretrained SAEs (GemmaScope, GPT-2 SAEs), training-free analysis is both feasible and appropriate. Retraining SAEs (as required by MP-SAE, Matryoshka SAE, OrtSAE) is outside the project's scope.

5. **The evaluation protocol is credible**: SAEBench provides standardized metrics. The causal validation framework in the literature (with proper ablation studies) is the gold standard, but the simulated results suggest it is achievable within time budget.

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Model Absorption Fingerprinting

- **Hypothesis**: Absorption hotspot patterns (layer 6/9) and absorption graph topology constitute a "fingerprint" that differs across model families, reflecting architecture-specific representational compression strategies.
- **Implementation sketch**: Use SAELens to load GPT-2 residual stream SAEs (all layers) and GemmaScope JumpReLU SAEs (layers 0-17). Apply the absorption graph construction (nodes=features, edges=absorption pairs based on decoder cosine similarity) for each layer. Compare graph topology metrics (mean edge weight, component count, degree distribution) across models.
- **Simplest version**: Compare absorption graph topology between GPT-2 layer 6 and Gemma-2 layer 6 (or nearest equivalent). Focus on 3-5 topological metrics. No ablation studies needed.
- **Time estimate**: ~30 min for GPT-2 (all layers), ~20 min for GemmaScope (layers 0-17). Total ~50 min.
- **Reusable components**: SAELens for SAE loading, existing absorption graph code from prior experiments, custom topological analysis.

### Candidate B: Downstream Impact Deep-Dive (Skip Detection)

- **Hypothesis**: Absorbed features (identified by decoder cosine similarity threshold) show systematically different intervention response profiles compared to non-absorbed features, measurable via logit change and activation stability.
- **Implementation sketch**: Skip the broken scoring formula entirely. Use decoder cosine similarity alone (>0.7 threshold) to identify candidate absorption pairs. For each pair, measure: (1) activation coefficient of variation (already shown to differ), (2) intervention response (ablate parent, measure child activation change), (3) circuit overlap (which attention heads/MLPs are affected). Compare to matched non-absorbed pairs.
- **Simplest version**: Use 20 absorbed pairs (decoder cosine >0.7) and 20 control pairs (low cosine) from layer 6. Measure CV difference (replicate E5 with larger sample) and intervention response. No scoring formula needed.
- **Time estimate**: ~40 min for 40 pairs with CV + intervention measurements. Within 1-hour budget.
- **Reusable components**: E5 downstream impact code, SAELens intervention hooks, existing absorption graph infrastructure.

### Candidate C: Fix the Detection Formula

- **Hypothesis**: The original absorption scoring formula fails because it conflates two distinct phenomena: (1) absorption (parent encoded via child decoder) and (2) activation correlation (parent and child both active due to shared input). A formula based on decoder cosine similarity alone can distinguish these.
- **Implementation sketch**: Design a new scoring formula: `score = decoder_cosine * log(freq_ratio) * (1 - normalized_cooccurrence)`. Test on existing E2 data (layer 6, 54 pairs with scores). Compare discrimination (high-score vs low-score) on a held-out set. Validate against the causal validation simulated results.
- **Simplest version**: Take existing E2 layer 6 results (54 pairs), recompute with revised formula, check if high-score pairs correlate with the simulated causal effects (P1-P8). No new experiments needed for formula validation.
- **Time estimate**: ~15 min for formula revision and recomputation. ~30 min for new detection run on expanded candidate set.
- **Reusable components**: Existing E2 data, causal validation results (P1-P8), scoring formula code.

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Model Fingerprinting)

- **Implementation reality check**: The "absorption hotspot at layer 6" finding is from GPT-2 only. GemmaScope uses JumpReLU SAEs vs GPT-2's TopK SAEs — different architectures may show different layer-wise patterns. Need to be careful about direct layer-to-layer comparison.
- **Reproducibility attack**: Graph topology metrics (component count, mean edge weight) depend on the candidate pair selection threshold. Need to show results are robust across threshold variations.
- **Baseline sanity check**: Is "absorption fingerprint" a meaningful concept? Chanin et al. showed absorption is caused by hierarchical feature co-occurrence. If this is universal across models (driven by training data statistics), fingerprints may be similar rather than different.
- **Scope attack**: Comparing two models (GPT-2 and Gemma) is a start but may not be enough to establish "fingerprint" uniqueness. Need at least 3 models for a compelling claim.
- **Verdict**: MODERATE — Engineering feasible but the scientific claim may be weak. The cross-model comparison is valuable regardless of whether fingerprints are unique or universal.

### Against Candidate B (Downstream Impact Deep-Dive)

- **Implementation reality check**: E5 already shows CV difference is significant (p=0.005) with n=20+20. Expanding to n=100+100 is straightforward. The intervention response measurement requires SAE hook access — this is well-supported by SAELens.
- **Reproducibility attack**: The CV difference is a straightforward statistical measurement. No complex formula or hyperparameter sensitivity. Highly reproducible.
- **Baseline sanity check**: The comparison (absorbed vs control) is clean — control pairs are matched by activation frequency. This avoids the frequency confound that plagues absorption detection.
- **Scope attack**: CV difference is one metric. A full downstream impact study should also measure circuit overlap and steering effectiveness. But the CV result alone is publishable as a "feature absorption affects activation stability" finding.
- **Verdict**: STRONG — This is the most engineer-friendly option. It avoids the broken detection metric, focuses on a significant finding (CV difference), and requires no new formula development. The risk is that the paper claim may be narrow ("absorbed features have lower CV") without connecting to broader interpretability implications.

### Against Candidate C (Fix Detection Formula)

- **Implementation reality check**: The formula revision is computationally cheap (recompute on existing data). But the E4 finding (negative co-occurrence correlation) suggests the problem may be deeper than formula weights — it may be that co-occurrence is fundamentally not the right signal for absorption detection.
- **Reproducibility attack**: If the revised formula still doesn't discriminate (no scores >0.7), we have the same problem as before. Need to validate on held-out data before committing to this path.
- **Baseline sanity check**: The causal validation results (P1-P8) are SIMULATED, not actual execution. Using them to validate the formula revision is circular — we're fitting to our own simulation. Need actual execution to validate.
- **Scope attack**: Even if we fix the formula, the fundamental problem remains: ablation-based metrics may be too coarse for absorption detection. The SAEBench probe projection approach may be more suitable.
- **Verdict**: WEAK — Too many unknowns. The formula revision is worth exploring but should not be the primary path given the time constraints. Better to focus on what we know is real (CV difference, layer hotspot pattern).

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Fix Detection Formula)** dropped because: The E4 finding (negative co-occurrence correlation) suggests the co-occurrence term is fundamentally wrong, not just misweighted. Fixing formula weights without actual validation runs is high-risk. The simulated causal validation results cannot be used to validate a formula revision (circularity). Time is better spent on confirmed findings.

### Strengthened Ideas

- **Candidate B (Downstream Impact Deep-Dive)** strengthened by:
  - The CV difference (p=0.005) is the most robust finding across all experiments
  - It does not depend on the broken scoring formula
  - It has a clear interpretability implication: absorbed features are "stuck" (low variability), which may explain why SAE steering sometimes fails (Basu et al. 2026)
  - Expanding to n=100+100 is straightforward and within time budget
  - This path directly addresses Gap 3 (impact on downstream interpretability tasks) identified in the literature survey

- **Candidate A (Cross-Model Fingerprinting)** strengthened by:
  - The layer 6 hotspot pattern was observed in GPT-2; cross-model validation would confirm or refute this as a universal phenomenon
  - Even if fingerprints are similar (not unique), documenting the cross-model consistency is a publishable contribution
  - GemmaScope SAEs are readily available via SAELens; no retraining needed

### Selected Front-Runner

**Candidate B: Downstream Impact Deep-Dive** is selected as the front-runner because:

1. **Highest probability of success**: It focuses on the most robust finding (CV difference, p=0.005) without relying on the broken detection metric. No formula development or new metric validation needed.

2. **Engineering straightforward**: The experiment is a direct extension of E5 with larger sample size and additional metrics. The code infrastructure already exists. Expected runtime ~45 min, well within budget.

3. **Clear scientific claim**: "Feature absorption systematically reduces activation variability in SAE representations, consistent with hierarchical features being compressed into a single latent." This is falsifiable, measurable, and interpretable.

4. **Connects to the field**: The CV finding may explain Basu et al.'s negative result (absorbed features resist intervention because they are "locked" into low-variability states). This provides a mechanistic interpretation of a published result.

5. **Abandoning detection is the right call**: Three rounds of experiments (E1-E5) failed to produce high-confidence detection pairs (max score 0.63 < 0.7 threshold). The detection path is exhausted. The pragmatist's job is to find what does work — and downstream impact is what works.

---

## Phase 5: Final Proposal

### Title

**Feature Absorption Reduces Activation Variability in Sparse Autoencoders: A Downstream Impact Analysis**

### Hypothesis

Absorbed features (where a parent feature is encoded via a child feature's decoder direction) show systematically lower activation variability (coefficient of variation) compared to non-absorbed features. This reduced variability reflects the parent feature being "compressed" into the child latent and is consistent with the failure of SAE-based steering observed in prior work (Basu et al., 2026).

### Motivation

Feature absorption is widely acknowledged as a problem in SAEs (Chanin et al., 2024; Karvonen et al., 2025). However, prior work has focused on detection metrics (ablation-based absorption rates, probe projection contributions) without quantifying the downstream consequences for interpretability research. The most relevant prior result is Basu et al. (2026), who showed that even near-perfect feature detection (98.2% AUROC) produces zero output change via SAE steering. This suggests that absorbed features may be "locked" in a way that resists intervention.

Our prior experiments (E1-E5) found that absorbed features have significantly lower coefficient of variation (CV: 1.07 vs 1.46, p=0.005). This finding is statistically robust and suggests a mechanism: absorption compresses hierarchical features into a single latent, reducing the degrees of freedom for activation variation. This proposal extends the E5 finding with a larger sample and additional metrics to characterize the downstream impact.

### Method

1. **Feature Selection**: Use decoder cosine similarity (>0.7) to identify absorbed candidate pairs from layer 6 of GPT-2-small SAE. Select 100 absorbed pairs and 100 matched control pairs (matched by activation frequency, low decoder cosine <0.3). No scoring formula or ablation needed.

2. **Activation Variability Measurement**: For each feature, compute:
   - Coefficient of variation (CV = std/mean of activation magnitudes)
   - Activation entropy (distribution of activation values across tokens)
   - Peak-to-mean ratio (maximum activation / mean activation)

3. **Intervention Response Measurement**: For a subset of pairs (n=30 absorbed, n=30 control):
   - Ablate the parent feature using SAE hook (zeroing in latent space)
   - Measure change in child feature activation
   - Measure logit change at the output
   - Compare intervention response between absorbed and control groups

4. **Cross-Layer Validation**: Replicate the CV analysis on layers 3, 9, and 11 to validate the layer-specific pattern (layer 6 is hotspot; layer 9 also shows elevated absorption).

### Simplest Version

Measure CV for 100 absorbed + 100 control pairs in layer 6. Perform statistical comparison (t-test, Mann-Whitney U). This is the minimal experiment that tests the core claim and is achievable in ~30 min.

### Baselines

- **Absorbed features** (decoder cosine >0.7 with hierarchical pair): Expected CV ~1.1 (from E5)
- **Non-absorbed features** (low cosine, no hierarchical relationship): Expected CV ~1.4 (from E5)
- **Random features** (baseline for sanity check): Expected CV ~1.3

### Experimental Plan

| Experiment | Target | Metric | Time |
|------------|--------|--------|------|
| E1: Expanded CV measurement | 100+100 pairs, layer 6 | CV difference, statistical significance | 20 min |
| E2: Cross-layer CV validation | Layers 3, 6, 9, 11 | CV by layer, hotspot confirmation | 25 min |
| E3: Intervention response | 30+30 pairs, layer 6 | Logit change, activation change | 30 min |
| E4: Cross-model validation (GPT-2 only) | Different SAE config (e.g., 32k width) | CV difference replication | 15 min |

**Success criterion**: CV difference between absorbed and control groups is statistically significant (p<0.01) with effect size Cohen's d > 0.5. Cross-layer pattern confirms layer 6 as hotspot.

### Resource Estimate

- **Models**: GPT-2-small (85M parameters) via SAELens pretrained SAE
- **SAE Configuration**: GPT-2-small-res-jb, layer 6, d_sae=24576
- **Compute**: All experiments fit within 1-hour budget per task; total ~90 min across 4 experiments
- **Code**: SAELens for SAE loading and hooks, custom CV computation, TransformerLens for intervention

### Risk Assessment

1. **Risk: CV difference does not replicate at larger sample size**. E5 used n=20+20; expanding to n=100+100 may reveal the effect is a small-sample artifact. *Mitigation*: If effect disappears at larger n, this is itself a publishable negative result — the original E5 finding was a false positive.

2. **Risk: Intervention response is small or zero**. Basu et al. (2026) showed steering often fails. If absorbed features show no intervention response, the paper claim narrows to "CV difference" only. *Mitigation*: CV difference alone is still a contribution; intervention response is exploratory.

3. **Risk: Time overrun**. E2 (cross-layer validation) involves 4 layers × 200 features = 800 feature measurements. At ~2 sec per feature, this is ~27 min. *Mitigation*: Batch processing across layers; parallelize if needed.

### Novelty Claim

This is the **first systematic study of activation variability as a downstream consequence of feature absorption in SAEs**. Prior work has: (1) measured absorption rates (Chanin et al., 2024), (2) proposed architectural remedies (Matryoshka SAE, OrtSAE), (3) evaluated absorption in SAEBench (Karvonen et al., 2025), and (4) documented the failure of steering on absorbed features (Basu et al., 2026). None has systematically characterized how absorption affects the statistical properties of feature activations. The CV metric is simple, interpretable, and does not require ground-truth probes — making it a practical diagnostic for the SAE community.

The finding that absorbed features have lower CV provides a mechanistic interpretation of Basu et al.'s negative result: absorbed features are "compressed" into low-variability states that resist intervention. This connects two previously disparate results in the literature and provides a practical diagnostic (CV) for identifying potentially problematic features without requiring expensive ablation studies.
