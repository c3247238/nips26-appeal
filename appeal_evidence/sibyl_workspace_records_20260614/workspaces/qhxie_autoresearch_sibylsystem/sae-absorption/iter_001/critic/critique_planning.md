# Critique: Planning and Methodology (Post-R4 Update)

**Updated:** 2026-04-13 | **Round:** Post-R4 (final experimental state)

## Overall Planning Assessment

The R4 planning addressed the most critical R3 gaps: a proper shuffled hierarchy control was designed and executed (correctly falsifying H3), direct-label EDA validation was attempted (limited by model access), and cross-model Llama weight statistics were collected. The go/no-go gate framework functioned correctly. The primary remaining planning failures are inherited from earlier rounds and cannot be fully resolved without model access.

---

## R4 Planning: What Worked

### R4B Shuffled Control Design is Methodologically Correct

The R4B plan specified a proper shuffled hierarchy control: "Randomize parent-child label assignments while holding all other aspects of the pipeline constant." This is the right null test for cross-domain absorption. The R3 plan compared against fixed random probe directions (not shuffled hierarchy labels), which was an invalid null test. The R4 correction is a genuine planning improvement.

### Decision Gate Framework Executed Correctly

The go/no-go gate (r4_writing_gate.json) correctly:
- Gates A and C pass (EDA: 3/8 configs pass; cross-model: 2 families)
- Gate B fails (H3: 0/9 domain-SAE combos pass shuffled control)
- Activates the two-contribution framing pivot per fallback_decisions
- Sets go_write=true

This is correct pivot management. The gate thresholds were calibrated pre-experiment and honored.

---

## Inherited Planning Failures (Unresolved)

### Failure 1: Taxonomy Was Never Expanded Beyond Two Layer-12 Configurations

The R3 plan ran taxonomy on L12-16k and L12-65k only. The R4 plan did not include taxonomy expansion despite identifying it as a gap. GPT-2-L6 (n_pos=67, exact labels) was available and would have been a low-cost third data point. The plan should have scheduled: "If time permits, run Phase 2 taxonomy on GPT-2-L6 using exact Chanin et al. labels." This was not included, leaving the taxonomy's central claim with n=16 and n=65 from a single layer (L12) of a single model family (Gemma 2B SAEs).

**Residual risk:** A reviewer will note the taxonomy is "Layer 12-only, Gemma-only" and question whether early dominance generalizes.

---

### Failure 2: Tau Threshold for Taxonomy is Not Scientifically Justified

The pre-registered tau=0.3 for the decoder-coverage threshold lacks a scientific basis in the plan documents. The plan acknowledges the threshold was "tested at {0.20, 0.25, 0.30, 0.35, 0.40}" — but does not specify how tau=0.3 was selected as canonical. The planning process should have:
1. Computed the 95th percentile of random pairwise cosine similarities in the decoder dictionary (to establish a random-baseline for "near the parent probe")
2. Selected tau as a multiple of this percentile
3. Pre-registered tau before seeing the subtype proportion results

Without this, tau=0.3 could be criticized as chosen post-hoc to maximize the early-dominance finding. At tau=0.2, early is 32%; at tau=0.3, it is 72%. The tau choice multiplies the reported early fraction by 2.25×.

---

### Failure 3: ITAC Evaluation Mode Was Not Pre-Registered Carefully Enough

The R3 plan specified ITAC would be evaluated on "absorption testing applied to up to 20 words per letter" — but the actual implementation used synthetic activations (decoder columns as parent-positive inputs), not real text activations. This operationalization gap was never pre-registered and represents a deviation from what a practitioner would consider "testing ITAC."

The R4D plan specified testing on real activations (10,000 tokens of Pile/Wikitext-103) but was not completed ("optional task, no scheduler time allocated"). This leaves the ITAC evaluation on synthetic activations only — a significant limitation that should have been closed in R4.

---

### Failure 4: ITAC Null Test Operationalization Was Incorrect

The plan specified "null test on early-type latents" but this was operationalized as "measuring baseline FN rate of early-type latents without applying ITAC." The correct null test is "apply ITAC to early-type latents and verify FN reduction = 0%." This planning ambiguity produced a misleading result in the paper.

---

## Methodological Decisions That Held Up Under Scrutiny

1. **Bootstrap CI method** (10,000 resamples, seed=42): Correct and reproducible.

2. **Multiple baseline comparisons** (decoder cosine, shuffled EDA, random probe directions, shuffled hierarchy labels in R4): Well-designed baseline suite.

3. **Phase 0 metric validation before Phase 1** (threshold sensitivity, random baseline, SynthSAEBench): Correct sequencing.

4. **Using AUPRC as a secondary metric** (reported in JSON files, though not in main text): Shows methodological awareness of the class-imbalance problem.

5. **Proxy label transparency**: The plan correctly distinguished Neuronpedia proxy labels from exact Chanin et al. labels and specified the L0_source for all configurations.

---

## Methodology Red Flags

### 1. L0 Covariate in H6 Is an Estimate, Not Measured

Phase 4 uses "gemma_scope_paper_canonical" L0 values (65-72 across all configs). These are design targets, not measured values from evaluation runs. The partial correlation test rho(width, absorption | L0) treats this estimated L0 as if it were an observed variable — an assumption that undermines the H6 analysis.

### 2. Positive-Class Prevalence Never Had a Minimum Sample Size Requirement

The plan should have required n_pos >= 50 as a minimum for reliable AUROC measurement. L12-16k (n_pos=16) and L12-65k (n_pos=16 despite 65k latents) would have been flagged as insufficient under this criterion, correctly directing experimental effort toward configurations with adequate positive prevalence.

### 3. R4A Defines "Direct Labels" Using GPT-2 Only

R4A's "direct label generation" used GPT-2 Small as the host model because Gemma 2B and Llama-3.1-8B were gated. This means R4A produced direct labels for GPT-2 configurations only — the Gemma Scope configurations (the paper's primary evaluation set) still use proxy labels. The planning should have specified: "R4A ONLY counts as resolving the direct-label gap for Gemma configurations. GPT-2 direct labels were already available in R3."

---

## Planning Quality vs. Execution Quality (Updated)

| Phase | Planning Quality | Execution Quality | Post-R4 Status |
|-------|-----------------|-------------------|----------------|
| R3 Phase 0 metric validation | Good | Good | Complete |
| R3 Phase 1 EDA validation | Good, threshold too permissive | Mixed (3/8 pass) | Complete; Gemma proxy label gap persists |
| R3 Phase 2 taxonomy | Adequate scope, too few configs | Poor (n=16 L12-16k, ordering fails) | Unextended; GPT-2 taxonomy not run |
| R3 Phase 2 ITAC | Pre-registration too optimistic; synthetic eval not flagged | Poor (3.14%, near-null) | Null test still incorrect |
| R3 Phase 3 cross-domain | No hard pivot on proxy model failure | Compromised | R4B correctly resolved with shuffled control |
| R3 Phase 4 scaling | H6 design underpowered by construction | Poor (H6 untestable) | Reported as limitation; not fixed |
| R3 Phase 5 GPT-2 | Well-designed | Best result in paper | Confirmed by R4A |
| R4A direct label | Good intent; limited by model access | Partial (GPT-2 only) | Gemma direct labels remain unavailable |
| R4B RAVEL proper probes + shuffled control | Good design; correct null test | Good (H3 properly falsified) | Complete |
| R4C Llama weight-only | Correctly scoped given model access constraints | Adequate | Weight-only statistics collected |
| R4D ITAC real activations | Optional, correctly low-priority | Not completed | Limitation acknowledged |
