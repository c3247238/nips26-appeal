# Research Proposal

## Title
**Construct Validity of the SAEBench Feature Absorption Metric: Does First-Letter Absorption Predict Semantic-Hierarchy Absorption?**

## Abstract

Feature absorption is one of the most consequential failure modes identified in sparse autoencoders (SAEs). The dominant metric for measuring absorption, introduced by Chanin et al. (2024) and adopted by SAEBench (Karvonen et al., 2025), relies on first-letter classification tasks with ground-truth logistic probes. While this metric has shaped architectural development (Matryoshka SAEs, OrtSAE, HSAE), it has never been validated on real semantic hierarchies. We propose the first systematic construct-validity study of the SAEBench absorption metric, testing whether first-letter absorption scores generalize to matched-frequency semantic hierarchies drawn from WordNet. If the metric fails to generalize, it undermines a large body of follow-up work that optimizes for it. If it generalizes, the community gains confidence in a widely used benchmark. The study is entirely training-free, uses existing pretrained SAEs, and can be completed within 1 GPU-hour.

## Motivation

The SAE community has rapidly coalesced around feature absorption as a central pathology. Chanin et al. (2024) proved analytically that absorption is incentivized by sparsity loss for hierarchical features, and introduced a rigorous detection protocol based on first-letter probes. SAEBench (Karvonen et al., 2025) standardized this protocol, making it one of eight canonical SAE evaluations. Follow-up architectures—Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), Hierarchical SAEs (Zhan et al., 2026)—all report absorption reductions as primary contributions.

Yet a critical methodological question remains unasked: **Does the first-letter absorption metric actually measure a general phenomenon, or is it an artifact of a narrow, artificial task?** First-letter hierarchies (e.g., "starts with S" ⊃ "short") are convenient because they have ground-truth labels and causal ablations are tractable. But real-world semantic hierarchies (e.g., "animal" ⊃ "mammal" ⊃ "dog") differ in frequency structure, semantic coherence, and representational geometry. If first-letter absorption is uncorrelated with semantic-hierarchy absorption, then architectures optimized for the former may not improve behavior on the latter.

This question is urgent because:
1. **Benchmark validity shapes research direction.** If the metric lacks construct validity, the community may be optimizing the wrong target.
2. **Training-free constraints align perfectly.** We can answer this using existing pretrained SAEs and the SAEBench codebase.
3. **The stakes are high either way.** A positive result validates a cornerstone of the field; a negative result reveals a methodological blind spot with immediate implications for benchmark design.

## Research Questions

1. Do SAEs with low first-letter absorption rates (as measured by SAEBench) also exhibit low absorption rates on matched-frequency semantic hierarchies from WordNet?
2. Is the SAEBench absorption metric specific to hierarchical features, or does it also detect absorption-like behavior in non-hierarchical correlated features?
3. How robust is the first-letter vs. semantic-hierarchy correlation across different SAE architectures, layers, and feature-splitting thresholds?

## Hypotheses

**H1 (Construct Validity):** The Pearson correlation between first-letter absorption scores and semantic-hierarchy absorption scores across a diverse set of 6–8 SAEs will be greater than 0.6. A correlation below 0.6 would falsify the hypothesis and suggest the metric lacks construct validity as a general measure of feature absorption.

**H2 (Hierarchy Specificity):** Semantic-hierarchy absorption scores will be significantly higher than non-hierarchy correlated-feature absorption scores, indicating the metric is specific to hierarchical rather than merely correlated features.

**H3 (Robustness):** The correlation between first-letter and semantic-hierarchy absorption will be stable across τ_fs values (0.01, 0.03, 0.05) and across architectures.

## Method

### SAE Selection
Select 6–8 publicly available pretrained SAEs that span the absorption-rate spectrum:
- Matryoshka SAE (very low absorption)
- OrtSAE (low absorption)
- BatchTopK SAE (moderate absorption)
- Standard ReLU SAE (moderate-high absorption)
- TopK SAE (moderate)
- Gated SAE (moderate)
- JumpReLU SAE (moderate-high)
- Optionally a narrow SAE (high absorption / hedging)

Source: SAELens releases for GPT-2 small and Gemma-2-2B.

### First-Letter Absorption
Compute absorption scores using the standard SAEBench protocol (ground-truth logistic probes, k-sparse probing with τ_fs = 0.03, absorption formula with τ_pa = 0, τ_ps = -1).

### Semantic-Hierarchy Construction
1. Extract 8–10 parent-child pairs from WordNet where the parent is a direct hypernym of the child (e.g., animal → mammal, mammal → dog, vehicle → car).
2. Ensure tokens are single tokens in the model vocabulary.
3. **Frequency matching:** For each hierarchy, create a synthetic balanced dataset where parent and child tokens appear with equal frequency, drawn from a background corpus. This controls the frequency-confound identified by the empiricist and contrarian perspectives.

### Semantic-Hierarchy Absorption Measurement
1. For each parent concept, train a logistic regression ground-truth probe on the base model's residual-stream activations.
2. Apply the exact SAEBench absorption formula to the SAE latents, using the same k-sparse probing protocol but on the semantic parent-child pairs.
3. Compute absorption scores per parent, then average across parents.

### Control Condition (Non-Hierarchical Correlated Features)
1. Select 4–5 pairs of semantically related but non-hierarchical concepts (e.g., synonyms like "happy/joyful", or co-occurring attributes like "doctor/hospital").
2. Match frequencies and compute "absorption" scores using the same formula.
3. If the metric is hierarchy-specific, these scores should be near-zero.

### Random-SAE Control
Compute absorption on an SAE with randomly permuted decoder directions. Should yield near-zero absorption on all tasks; if not, the metric is not specific to learned structure.

## Evaluation Protocol

**Primary benchmarks:**
- SAEBench Feature Absorption (first-letter) — established public benchmark.
- Custom Semantic-Hierarchy Absorption — novel construct-validity test.
- Custom Non-Hierarchy Correlated-Feature Absorption — control benchmark.

**Metrics:**
- Mean absorption score (first-letter)
- Mean absorption score (semantic hierarchy)
- Mean absorption score (non-hierarchy control)
- Pearson correlation r (first-letter vs. semantic hierarchy) with bootstrap 95% CI (B = 10,000).
- Pearson correlation r (first-letter vs. non-hierarchy control) with bootstrap 95% CI.
- Paired t-test comparing semantic-hierarchy absorption to non-hierarchy control absorption.

**Random seeds:** 3 random seeds for probe training and data sampling; report mean ± std.

**Statistical test plan:**
- Bootstrap CI for correlation to handle small-n SAE sampling.
- If correlation CI excludes 0.6, hypothesis is supported; if it includes values < 0.3, hypothesis is rejected.
- Report all raw scores in an appendix table for full transparency.

## Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| **Frequency-matched vs. natural-frequency hierarchies** | Whether frequency imbalance drives absorption-like effects | Frequency-matched should show lower variance and clearer hierarchy-specificity |
| **Single-token vs. multi-token concepts** | Whether tokenization artifacts confound the metric | Single-token should be cleaner; multi-token may show inflated absorption |
| **Different base models (Gemma-2-2B vs. GPT-2 small)** | Whether construct validity is model-specific | Correlation pattern should replicate across models if the metric is general |
| **Varying τ_fs (0.01, 0.03, 0.05)** | Whether feature-splitting threshold changes the correlation | Correlation should be robust to τ_fs if the metric is stable |

## Pilot Design
- **Scope:** 2 SAEs (Matryoshka + BatchTopK) × 3 semantic parent-child pairs × 1 control pair.
- **Target runtime:** 10–15 minutes on a single GPU.
- **Success criterion:** Pilot successfully computes semantic-hierarchy absorption scores that are numerically stable and show the expected ordering (Matryoshka < BatchTopK). If scores are noisy or inverted, refine the probe training or concept selection before scaling.

## Resource Estimate
- **GPU-hours:** ~0.5–1.0 GPU-hour for the full experiment (6–8 SAEs, probe training is lightweight).
- **Model sizes:** Gemma-2-2B (primary), GPT-2 small (replication control).
- **All tasks well under the 1-hour limit.**

## Risk Assessment

| Threat | Mitigation |
|--------|------------|
| **Probe quality is poor for semantic concepts** | Filter concepts to those with probe AUROC > 0.7; report probe-quality table |
| **WordNet concepts are not single tokens** | Pre-filter using model tokenizer; exclude multi-token concepts |
| **Frequency matching is imperfect** | Use synthetic balanced datasets; report token frequencies |
| **Small-n correlation is noisy** | Report bootstrap CIs; interpret cautiously; emphasize effect-size bounds |
| **First-letter and semantic tasks use different probe difficulties** | Standardize absorption scores by probe AUROC before correlating |

## Novelty Assessment

This would be the **first systematic construct-validity study of the dominant feature-absorption metric in the SAE literature**. We searched arXiv and Google Scholar for papers extending the Chanin/SAEBench absorption metric beyond first-letter tasks to semantic hierarchies. Key findings:

- **Chanin et al. (2024)** explicitly note the limitation: *"Our metric cannot capture absorption past layer 17... [and] requires ground-truth knowledge of true labels."* They call for *"finding examples of feature absorption unrelated to character identification"* as future work.
- **SAEBench (2025)** adapted the metric technically by replacing ablation with probe-projection criteria, enabling all-layer evaluation. However, the underlying evaluation task remains **first-letter classification**.
- **Hierarchical SAEs (2025–2026)** and **Matryoshka SAEs (2025)** report absorption improvements, but all use the first-letter benchmark.
- **RAVEL (2024)** evaluates factual disentanglement (country/continent) but is not used as an absorption metric.

**No prior work has systematically tested whether first-letter absorption predicts semantic-hierarchy absorption.** This makes our proposal both novel and timely.

## Revisions from Prior Feedback

This is the first synthesis round for this project; no prior proposal, novelty report, or Codex feedback exists to revise from. The proposal synthesizes six fresh perspectives generated in the current `idea_debate` stage.

## Synthesis Rationale

### How the Perspectives Were Weighted

**Highest weight: Empiricist + Contrarian.** The empiricist identified the core methodological gap (nobody has validated the absorption metric on real hierarchies) and designed a rigorous, falsifiable experiment. The contrarian reinforced this by challenging the assumption that lower absorption scores mean better features, and by highlighting the metric's blind spots (conservative underestimate, inapplicability past layer 17, ground-truth scarcity).

**Strong weight: Pragmatist.** The pragmatist confirmed that SAEBench and SAELens provide ready-to-use infrastructure, making the experiment feasible within tight constraints. The screening-tool idea (FastProbe-Absorb) is folded in as a potential follow-up rather than the main contribution.

**Moderate weight: Theoretical + Interdisciplinary.** Both perspectives provide conceptual framing: absorption is structurally inevitable under sparsity (theoretical), and it can be understood as a phase-transition phenomenon (interdisciplinary). These insights strengthen the introduction and motivation but are not the empirical core of the proposal.

**Lower weight: Innovator.** The innovator's cross-layer dose-response framework is ambitious and valuable, but its scope (25–30 SAE cohort + causal validation + cross-layer modeling) exceeds the project's 1-hour-per-experiment constraint. Its core insight—distinguishing "truly missing" from "merely hidden" absorbed features—is retained as a conceptual thread but not as the primary methodology.

### Why This Idea Was Selected

The empiricist's front-runner was selected because it:
1. **Addresses a genuine field-wide blind spot** with high stakes for both positive and negative results.
2. **Is perfectly aligned with training-free constraints** and existing infrastructure.
3. **Has clear falsification criteria** and a tight statistical protocol.
4. **Can be completed within 1 GPU-hour**, making it ideal for rapid iteration.
5. **Generates actionable implications**: either validate a cornerstone benchmark or reveal that architectures have been optimizing the wrong target.

### What Was Dropped or Deferred

- **Innovator's dose-response framework:** Deferred to a future, larger-scale follow-up study.
- **Pragmatist's FastProbe-Absorb tool:** Retained as a backup idea and potential follow-up contribution.
- **Contrarian's label-free geometric proxy:** Interesting but requires substantial validation; deferred.
- **Theoretical's combinatorial bound:** A strong standalone theory contribution; retained as Backup Idea B.
- **Interdisciplinary's phase-transition experiment:** Requires training SAEs with varying λ; deferred due to training-free constraint.

## Expected Contributions

1. **First construct-validity test** of the dominant SAE absorption metric on real semantic hierarchies.
2. **Empirical evidence** on whether first-letter absorption generalizes, with direct implications for benchmark design.
3. **Open-source replication materials** (WordNet hierarchy dataset, frequency-matching protocol, evaluation code).
4. **Guidance for the community** on whether absorption-reducing architectures should be adopted as default.
