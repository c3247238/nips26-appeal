# Methodology: Iteration 7 — Breaking the Stagnation

## Executive Summary

This iteration has a single overriding objective: **execute the three critical experiments that have blocked score improvement for 3 consecutive reviews, plus the zero-GPU analyses that cost nothing but remain undone**. Writing revision is secondary and gated behind experiment completion.

The project has accumulated extensive experimental results across 6 iterations (23/23 tasks succeeded in iter_006). The paper currently scores 6.5 with two strong empirical pillars (universal control failure, L0 phase transition). The path to 7.5–8.0 is experimentally precise and computationally cheap (3 GPU-hours + 3 hours CPU analysis).

## Context: Why Experiments Are the Binding Constraint

**Score trajectory**: 5.5 (x4) → 6.5 → 6.0 → 6.5 → 6.5 → 6.5 (3 consecutive stagnation)

**Key pattern**: Every score improvement was driven by experimental execution. Writing-only iterations produce +0.0.

**Blocking issues (recurring 3+ reviews)**:
1. Activation patching on 9 core words — NEVER EXECUTED
2. Tightened hedging classification — NEVER EXECUTED
3. CMI at L0=22 replication — NEVER EXECUTED
4. Partial correlations, leave-one-out — NEVER COMPUTED (zero GPU)
5. Threshold sensitivity — COMPUTED BUT NEVER REPORTED (zero GPU)
6. validate_integration.py — NEVER IMPLEMENTED (8 iterations!)

## Experimental Design

### Gate 0: Zero-GPU Data Integrity & Analysis (BLOCKING — must complete before any other work)

**0A. validate_integration.py**: Cross-check all numbers in paper.md against source JSON files. Flag any mismatches. This has been recommended for 8 iterations and never implemented. Cost: 1.5 hours, 0 GPU.

**0B. Partial correlation (CMI × absorption | probe_F1)**: The rho=-0.67 correlation between absorption rate and probe F1 is an uncontrolled confound. Compute partial Spearman rho(CMI, absorption | probe_F1) across all 25 letters. Cost: 30 min, 0 GPU.

**0C. Leave-one-out sensitivity**: For CMI-absorption correlation, remove each letter in turn and recompute rho. Identify whether letters S (high CMI, high absorption) and K (low CMI, zero absorption) are driving the overall correlation. Cost: 30 min, 0 GPU.

**0D. Threshold sensitivity reporting**: The ablation_threshold_sensitivity.json (141KB, 5×4 grid) already exists from iter_006 but has never been incorporated into the paper. Extract key results: CV across thresholds, whether control failure is threshold-dependent or structural, optimal threshold region. Cost: 30 min, 0 GPU.

**0E. Control failure diagnosis**: Analytical computation — sample 1,000 random unit vectors in R^2304, count decoder columns with cosine ≥ 0.025. Compare candidate counts from true probes vs. shuffled probes. Explains WHY controls fail. Cost: 30 min, 0 GPU.

### Gate 1: Three Critical Experiments (BLOCKING for writing)

**1A. Activation Patching on 9 Core Words** (highest priority, resolves central ambiguity)

- **What**: For each of the 9 persistent core words (tokens that are false negatives at ALL four L0 values {22, 41, 82, 176}), perform causal intervention:
  1. Identify the child feature (specific-token feature) from hierarchy_details
  2. Zero the child feature's activation in the SAE encoding
  3. Re-decode through the SAE
  4. Check whether the parent feature (first-letter feature) recovers (activation goes from 0 to >0)
- **Why**: This is the ONLY metric-independent causal evidence for competitive exclusion. Without it, the paper cannot distinguish interpretation (a) "JumpReLU genuinely has minimal absorption" from interpretation (b) "the metric is miscalibrated."
- **Model/SAE**: Gemma 2 2B + Gemma Scope L12 16k (L0=82, the configuration used in the main analysis)
- **Expected output**: Per-word recovery table (word, letter, child_feature_idx, parent_recovered_yes_no, recovery_magnitude)
- **Positive result**: Parent recovers for ≥5/9 words → competitive exclusion confirmed at small scale
- **Negative result**: Parent does not recover → all-hedging narrative strengthened (also publishable)
- **GPU cost**: ~0.5–1 hour on single GPU
- **Data source for 9 words**: confound_decomposition_multi_l0.json — tokens with FN at all 4 L0 values. 5 named (eight/E, lower/L, liked/L, offer/O, often/O) + 4 unnamed (must identify from raw data)

**1B. Tightened Hedging Classification** (validates headline 98.6% claim)

- **What**: For each of the 657 false negatives at L0=22, check whether the SPECIFIC parent-associated latent (the k=5 probe latents with highest cosine to probe direction) fires at L0=176. Classify as "strict hedging" ONLY if the parent latent fires at higher L0.
- **Why**: The current 98.6% hedging figure uses a permissive definition — any token that ceases to be a false negative at any higher L0 is classified as "hedging." At L0=176, only 10/1195 tokens remain FN, so 99.2% trivially resolve. The classification does not check whether the SPECIFIC parent latent fires.
- **Model/SAE**: Gemma 2 2B + Gemma Scope L12 16k at L0=22 and L0=176
- **Expected output**: strict_hedging_rate, permissive_hedging_rate, comparison table
- **Key insight**: If strict rate << permissive rate, the 98.6% claim needs qualification. If strict rate ≈ permissive rate, the claim is validated.
- **GPU cost**: ~0.5–1 hour (activation data from L0=22 and L0=176 already partially cached)

**1C. CMI Replication at L0=22** (tests theoretical pillar with perfect probes)

- **What**: Re-run CMI estimation at L0=22 (where all 25 probes have F1=1.0) instead of L0=82 (where only 10/25 pass F1>0.85). Pre-register d'=10 as primary subspace dimension.
- **Why**: The current CMI result (rho=-0.383, Bonferroni p=0.236, sign reversal at d'≥20) is computed on L0=82 where probe quality confounds the result. At L0=22, all probes are perfect (F1=1.0), eliminating this confound.
- **Model/SAE**: Gemma 2 2B + Gemma Scope L12 16k at L0=22
- **Expected output**: CMI per letter, Spearman rho with absorption, bootstrap CIs, convergence curves
- **If significant**: Theoretical pillar secured; overclaiming can be resolved
- **If non-significant**: Cleanly downgrade to exploratory; paper stands on two empirical pillars
- **GPU cost**: ~1 hour

### Gate 2: Writing Revision (ONLY after Gates 0 and 1 complete)

- Incorporate all new experimental results
- Retitle Section 6 to "Exploratory CMI-Absorption Association" (unless CMI at L0=22 is significant)
- Report Bonferroni-corrected p-values everywhere
- Include threshold sensitivity results
- Add two-interpretation paragraph in Discussion
- Name all 9 persistent core words
- Compress Section 5.3 (confounded cross-architecture comparison)
- Add reconciliation footnote for vocabulary size differences
- Fix abstract overlength

### Gate 3: Review

- Only after writing revision incorporates all new results

## Baselines and Controls

### Existing (from iter_006, to be preserved)
- **First-letter spelling baseline**: Absorption rate 15.96% (L0=82), 42.85% (L0=22) on Gemma 2 2B
- **Controls C1–C4**: Random probe (9.2%), shuffled labels (59.5%), dense probe (0.617 F1), untrained SAE
- **Cross-domain**: City-country, city-continent, city-language, animal-class absorption rates

### New Controls for Gate 1 experiments
- **Activation patching control**: Zero a random (non-child) feature and check parent recovery — expect no change
- **Tightened hedging null**: Apply strict classification to shuffled-label false negatives — expect similar strict/permissive ratio as true labels (validates methodology)
- **CMI estimation controls**: Bootstrap CIs (10k resamples), convergence curves for 3 representative letters, k-sensitivity analysis (k=3,5,10,20)

## Metrics

| Metric | Source | Status |
|--------|--------|--------|
| Absorption rate (per letter, per domain) | sae-spelling adapted metric | DONE (iter_006) |
| Bootstrap 95% CIs | 10k resamples | DONE |
| L0 phase transition | Multi-L0 decomposition | DONE |
| Cross-domain rates (5 domains) | Cross-domain experiments | DONE |
| CMI-absorption Spearman rho | k-NN MI estimator | DONE (L0=82), TODO (L0=22) |
| Partial correlation (CMI | probe_F1) | Partial Spearman | TODO (zero GPU) |
| Leave-one-out sensitivity | Jackknife | TODO (zero GPU) |
| Threshold sensitivity CV | 5×4 grid | DONE (unreported) |
| Activation patching recovery | Causal intervention | TODO (GPU) |
| Strict hedging rate | Parent-specific check | TODO (GPU) |

## Expected Visualizations

- **Table 1**: Activation patching results — 9 core words × recovery outcome
- **Table 2**: Hedging classification comparison — permissive (98.6%) vs strict rate
- **Table 3**: Threshold sensitivity heatmap (5×4 grid) — already computed
- **Figure (updated)**: CMI-absorption scatter with partial correlation annotation
- **Table (updated)**: Leave-one-out sensitivity showing rho stability

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Child features not identifiable for 4 unnamed core words | Medium | Medium | Inspect confound_decomposition code to extract all 9 word identities from raw token data |
| Activation patching shows no parent recovery | Medium | Low | Negative result is equally publishable — strengthens "all-hedging" narrative |
| Strict hedging rate << permissive rate | Medium | High | Report BOTH rates honestly; reframe narrative if needed |
| CMI at L0=22 still non-significant | Medium | Low | Cleanly downgrade to exploratory; paper stands on empirical pillars |
| L0=22 SAE has different activation patterns | Low | Medium | Use same SAE config as confound_decomposition_multi_l0 (already tested) |

## Resource Estimate

| Task | GPU Hours | CPU Hours | Priority |
|------|-----------|-----------|----------|
| Gate 0 (all zero-GPU) | 0 | 3.5 | P0 BLOCKING |
| 1A Activation patching | 0.5–1 | — | P1 BLOCKING |
| 1B Tightened hedging | 0.5–1 | — | P1 BLOCKING |
| 1C CMI at L0=22 | 1 | — | P1 BLOCKING |
| Gate 2 Writing revision | 0 | 3 | P2 (after Gate 1) |
| **Total** | **2–3** | **6.5** | |

Total wall-clock: ~10 hours including 3 GPU-hours. Less than the time consumed by the last 2 empty review cycles.
