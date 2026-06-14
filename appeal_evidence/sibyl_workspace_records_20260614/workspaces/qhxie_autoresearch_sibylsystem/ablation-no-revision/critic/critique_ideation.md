# Critique of Ideation: Quantifying Feature Absorption in Sparse Autoencoders

## Summary

The research proposal (`proposal.md`) is well-structured with a clear problem statement, systematic methodology, and appropriate baselines. The training-free analytical approach is a sound choice for rapid iteration. However, three issues reduce the proposal's strength: (1) the novelty claim is overstated given prior work on feature interaction phenomena; (2) the hypotheses are not fully independent (H1 failure cascades to H2); (3) the H4 experiment design was flawed from the start — task-agnostic latent selection cannot test whether absorption predicts circuit importance.

---

## Strengths

### Clear problem statement
The proposal correctly identifies the gap: "no systematic empirical quantification of its prevalence exists." This is a legitimate contribution framing.

### Training-free approach
Using pretrained SAEs from SAELens is efficient and appropriate. This enables rapid iteration without costly retraining.

### Appropriate baselines
The random dictionary control is excellent — by construction it yields zero absorption, validating the metric.

---

## Critical Issues

### 1. Novelty Claim Overstatement

The proposal claims "first systematic quantification of feature absorption." However, the Background section (Section 2.2) correctly notes prior work on superposition (Elhage et al., 2022), correlation-based feature analysis (Sharkey et al., 2023), and internal geometry (Templeton, 2025). The specific contribution is the **quantification method** (RVE-based absorption score), not the observation that feature interactions exist. This should be clarified: the contribution is "first systematic quantification using the RVE-based absorption score metric," not "first observation of feature interaction phenomena."

### 2. Hypothesis Dependency (Cascade Failure)

H1's failure blocks H2 by design: if absorption is rare at the chosen layer, there is no variance to correlate with token frequency. The proposal does not pre-register a fallback plan for H2 if H1 fails. A well-designed proposal should either:
- Pre-register "if H1 shows <1% prevalence at layer 8, H2 will use layer 4 as the primary test layer" (with explicit justification for why layer 4 would work when layer 8 fails), or
- Acknowledge that H2 is contingent on H1 and cannot be evaluated independently.

The current proposal leaves H2 in an ambiguous state when H1 fails, which is exactly what happened.

### 3. H4 Experiment Design Flaw

The H4 hypothesis asks whether "high-absorption SAE patching reduces faithfulness by >=5pp vs. low-absorption." The operationalization (Section 4.3) selects low/high absorption latents corpus-wide, then patches them on a specific circuit (France/Paris). This two-step selection means:
- Corpus-wide low/high absorption latents are selected based on general co-firing patterns, not circuit relevance
- The circuit (France/Paris) may be driven by latents that are neither high-absorption nor low-absorption by corpus standards
- Therefore, the experiment tests reconstruction capacity (can a 10% subset sustain patching?) not absorption quality (does high absorption impair circuit tracing?)

This design flaw was predictable and is indeed predictable in retrospect given the SAE patching literature. The proposal should have pre-registered a task-aware design: "if corpus-wide absorption selection proves irrelevant to circuit-level importance, H4 will be redesigned to compare full SAEs at layers with different absorption levels (layer 4 vs. layer 8)."

---

## Minor Issues

### H3 L0 proxy concern

The proposal acknowledges that L0 is a proxy for lambda, but does not pre-register what happens if L0 and lambda are not monotonic across layers (which is exactly what was observed — layer 8 has the highest L0 but the lowest absorption among mid layers). This is a genuine risk that should be addressed in the proposal.

### H5 subsampling vs retraining

The proposal does not address the limitation that dictionary sizes <24K were simulated by subsampling, not retraining. Results may differ for independently trained SAEs with different dictionary sizes. This is acknowledged in the paper but not in the proposal, which should pre-register this limitation.

---

## Assessment

The proposal is solid but has two structural weaknesses: hypothesis dependency (H1 failure cascades to H2 without a fallback plan) and a predictable H4 design flaw (task-agnostic selection cannot test circuit-level importance). The novelty claim is also slightly overstated — the contribution is the quantification method, not the underlying phenomenon.