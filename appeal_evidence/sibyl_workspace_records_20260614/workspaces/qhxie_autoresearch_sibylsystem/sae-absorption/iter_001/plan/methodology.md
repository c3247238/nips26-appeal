# Experimental Methodology: Characterizing SAE Absorption — Regime-Specific Detector and Three-Subtype Taxonomy

## Overview

This document describes the full experimental methodology for the paper
"Characterizing SAE Absorption: A Regime-Specific Detector and Three-Subtype Taxonomy".
All experiments are **training-free** and operate on pre-trained SAEs via SAELens/Gemma Scope.

**Experimental rounds:**
- Round 1–3 (complete): All 15 tasks finished. Results in `exp/results/full/`.
- Round 4 (complete): All blocking tasks finished. Writing gate: `go_write=true` (r4_writing_gate.json).
  - EDA direct label validation on GPT-2 (AUROC=0.650, direct labels)
  - H3 cross-domain: FALSIFIED — all RAVEL probe quality gates failed (best 59.5%, gate 85%)
  - H3 shuffled control: real absorption ≈ shuffled null for all domains
  - Paper pivoted to TWO-CONTRIBUTION structure: (1) EDA regime-specific detector + (2) three-subtype taxonomy
- Round 5 (this plan): Pre-writing consolidation experiments to address R4-identified limitations.

**Remaining limitations from R4 (to address in R5):**
1. ITAC on real activations (R4D) was skipped — run once to conclusively dismiss or rescue H5
2. Taxonomy rests on 2 SAE configs (n=16 and n=65); early-dominance finding is threshold-sensitive — extend to 2 additional configs
3. Controlled dictionary experiment (amortization gap) was proposed but never run — adds mechanistic insight to the 75% early-dominance finding
4. EDA AUROC on Gemma configs still uses proxy labels — no Gemma 2B direct validation possible (model gated); acknowledge and characterize the label-quality limitation with a systematic sensitivity analysis

---

## Candidate

**Primary candidate:** `cand_eda_crossdomain` — Encoder-Decoder Alignment (EDA) metric,
three-subtype taxonomy (early/late/diffuse), and ITAC training-free correction.
Two-contribution framing per R4 writing gate.

**Secondary (supplementary):** `cand_amortization_gap_experiment` — controlled dictionary experiment
separating encoder-architecture from dictionary-coverage causes of early absorption.

---

## Environment and Dependencies

```bash
pip install sae-lens transformer-lens datasets scikit-learn scipy statsmodels
pip install sae-spelling   # Chanin et al. first-letter absorption benchmark
```

**Models:**
- GPT-2 Small (fully accessible, direct label validation confirmed)
- Gemma 2 2B SAE weights only (model gated, SAE weights loadable)
- Llama Scope SAE weights only (model gated, weights loadable)

**SAEs:**
- Gemma Scope: `gemma-scope-2b-pt-res` at layers 5, 12, 19 × widths 16k, 65k (6 configs)
- GPT-2 Small: `gpt2-small-res-jb` layers 2, 6, 8, 10 (extend from current 2 configs)

---

## Round 5 Experimental Phases

### Phase R5-A: ITAC on Real Text Activations (H5 Final Verdict)

**Goal:** Run ITAC on real text activations to conclusively dismiss or rescue ITAC as a minor contribution.
R4D was skipped (optional in R4 task plan). R3 ITAC used synthetic activations only (decoder column inputs).

**Setup:**
- Load 10,000-token subset of Wikitext-2 through GPT-2 Small (publicly accessible)
- Cache residual stream activations at layer 6 (where EDA AUROC is confirmed 0.650)
- For each late-absorbed latent identified in Phase 2 taxonomy (target: L6 equivalent):
  - Identify parent-positive activation windows using first-letter probe
  - Measure parent latent FN rate on real text
  - Apply ITAC correction to each late-absorbed latent
  - Measure FN rate after ITAC on same parent-positive windows
- Run null test: ITAC effect on early-absorbed latents (expected: negligible)
- FVU constraint: reconstruction error change < +5%

**Decision gate:**
- FN reduction > 10% (relative) on late-absorbed latents → restore ITAC as minor contribution
- FN reduction < 5% → confirm H5 falsification generalizes to real activations
- Either outcome adds value: the negative result on real activations is stronger evidence than synthetic-only

**Estimated cost:** 1-2 GPU-hours

---

### Phase R5-B: Taxonomy Robustness — Extended SAE Configs

**Goal:** Extend three-subtype taxonomy beyond 2 SAE configs (current: L12-16k n=16, L12-65k n=65)
to address the reviewability concern that 75% early-dominance rests on only 2 configs.

**Setup:**
- Run taxonomy classification on 2 additional Gemma Scope configs:
  - L5-16k (already used for EDA AUROC; add taxonomy measurement)
  - L19-16k (failure case for EDA; check if early-dominance pattern still holds)
- Taxonomy classification uses weight-only criterion: early if max cos(d_k, parent_probe) < tau
  - Run at tau = {0.2, 0.3, 0.4} for threshold sensitivity
- Use Neuronpedia proxy labels as starting set; apply geometry-based taxonomy criteria
- Compute: early/late/diffuse fractions per config; KW test for EDA ordering across subtypes

**Pilot:** 100 latents per config, seed=42, timeout=900s
**Pass criteria:** Early fraction > 50% at tau=0.3 in >= 3/4 total configs (extends from 2/2)

**Estimated cost:** 1-2 GPU-hours (weight-only, no activation caching needed)

---

### Phase R5-C: Controlled Dictionary Experiment (Supplementary: Amortization Gap vs. Sparsity Landscape)

**Goal:** Run the `cand_amortization_gap_experiment` — hold the decoder dictionary D constant and
compare absorption rates under feedforward vs. OMP vs. 2-pass encoding.
This directly adjudicates between Tang et al. (sparsity landscape causes early absorption) and
O'Neill et al. (amortization gap causes early absorption).

**Why now:** The 75% early-dominance finding is the paper's highest-impact result. Reviewers will ask
"why is early absorption so prevalent?" This experiment answers it cleanly.

**Setup:**
- Use Gemma Scope L12-16k (SAE weights already downloaded)
- Extract pre-trained decoder dictionary D (no model access needed — weight-only)
- For a set of 1,000 synthetic activations (sampled from standard Gaussian, matching d_model=2304):
  - Feedforward encoding: standard SAE encoder pass
  - OMP encoding: Orthogonal Matching Pursuit with same decoder D, K=16 (matched L0)
  - 2-pass encoding: feedforward pass → zero-out top-K → re-encode residual
- Measure absorption rate under each encoding for parent-child feature pairs
  - Parent-child pairs: identified from Phase 2 taxonomy (late-absorbed latents have known parent decoder column)
  - Absorption criterion: parent latent fails to activate when child is active

**Pilot:** 100 synthetic activations, seed=42, timeout=900s
**Pass criteria:**
- OMP absorption rate vs. feedforward absorption rate comparison computed
- If OMP << feedforward: encoder (amortization gap) is a major cause of early absorption
- If OMP ≈ feedforward: sparsity landscape is the dominant cause

**Note:** This requires only Gemma Scope SAE weights (accessible). No model access needed.
**Estimated cost:** 1-2 GPU-hours

---

### Phase R5-D: EDA Label Quality Sensitivity Analysis (Supplementary)

**Goal:** Characterize how sensitive EDA AUROC estimates are to label quality, addressing the
R4 finding that Gemma proxy labels cannot be validated with direct labels (model gated).

**Setup (weight-only, CPU-friendly):**
- Create a synthetic label noise model: start from GPT-2 L6 direct labels (ground truth),
  progressively add random label flips at rates 0%, 5%, 10%, 20%, 30%, 50%
- Compute EDA AUROC under each noise rate (10,000-bootstrap CI)
- Identify the minimum label quality needed for AUROC >= 0.65
- Apply this model to estimate the Gemma proxy-label quality required to explain
  L12-65k AUROC = 0.468 (already fully explained by class imbalance, but document explicitly)

**Pilot:** GPT-2 L6 data (18 positives), 5 noise rates, seed=42, timeout=300s
**Pass criteria:** Sensitivity curve generated; minimum label quality for AUROC >= 0.65 estimated

**Estimated cost:** <30 minutes (CPU-only, no GPU needed)

---

## Baseline Methods

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Decoder cosine similarity (negated) | -cos(enc_j, dec_j) = EDA by formula | EDA identity validation (confirmed in R4) |
| Random shuffled EDA | EDA with shuffled encoder-decoder pairings | Null distribution |
| OMP encoding | Orthogonal Matching Pursuit on same dictionary | R5-C mechanistic control |
| 2-pass encoding | Feedforward → zero top-K → re-encode residual | R5-C mechanistic control |
| No ITAC correction | Raw SAE output | R5-A ITAC baseline |

---

## Evaluation Metrics Summary

| Phase | Primary Metrics | Targets |
|-------|----------------|---------|
| R5-A (ITAC real) | FN rate before/after ITAC on late-absorbed (real text) | > 10% = restore as minor contribution; < 5% = confirms H5 falsification |
| R5-B (taxonomy robustness) | Early fraction at tau=0.3; KW test across configs | Early > 50% in >= 3/4 configs |
| R5-C (amortization gap) | OMP vs. feedforward absorption rate ratio | < 50% ratio = encoder implicated; > 80% ratio = sparsity landscape dominant |
| R5-D (label sensitivity) | AUROC vs. label noise rate curve | Sensitivity curve; minimum quality threshold for AUROC >= 0.65 |

---

## Expected Visualizations (Paper-Ready after R5)

- **Table 1 (main results):** EDA AUROC across all 8 tested configs (Gemma + GPT-2), with direct/proxy label notation, regime annotation (pass/fail, layer, width).
- **Figure 1 (EDA distributions):** Violin plots of EDA by absorption status (absorbed vs. non-absorbed) at L12-16k (best regime) and L12-65k (failure regime). Confirmed from R3; labels updated with R4 direct-label results.
- **Table 2 (three-subtype taxonomy):** Early/late/diffuse fractions across 4 SAE configs (after R5-B); threshold sensitivity shown at tau=0.2, 0.3, 0.4.
- **Figure 2 (taxonomy EDA ordering):** Violin/swarm plots of EDA by subtype (early/late/diffuse); KW test p-value annotated. Confirmed from R3.
- **Figure 3 (controlled dictionary experiment):** Bar chart comparing absorption rates under feedforward, OMP, and 2-pass encoding on the same dictionary (R5-C supplementary figure).
- **Appendix: Label quality sensitivity curve** (R5-D) — AUROC vs. label noise rate.
- **Appendix: ITAC real activation results** (R5-A) — FN rate before/after per latent.

### In methodology.md, include a section:
## Expected Visualizations
- Table 1: EDA AUROC by regime (model × layer × width × label_type) — main results
- Figure 1: EDA violin plots absorbed vs. non-absorbed (L12-16k and L12-65k)
- Table 2: Taxonomy fractions across 4 SAE configs (tau sensitivity table)
- Figure 2: Taxonomy EDA ordering violin/swarm plot
- Figure 3 (supplementary): Controlled dictionary experiment (R5-C)
- Appendix A: Label quality sensitivity curve (R5-D)
- Appendix B: ITAC real activation table (R5-A)

---

## Software Architecture (additions to existing R1-R4 codebase)

All Round 5 code extends existing modules in `exp/code/`:

```python
# exp/code/r5a_itac_real.py          — R5-A: ITAC on real text activations (GPT-2 L6)
# exp/code/r5b_taxonomy_robustness.py — R5-B: taxonomy on L5-16k and L19-16k
# exp/code/r5c_amortization_gap.py    — R5-C: OMP vs feedforward absorption comparison
# exp/code/r5d_label_sensitivity.py   — R5-D: synthetic label noise sensitivity analysis
# exp/code/utils.py                   — shared utilities (bootstrapping, logging, seed=42)
```

All outputs are machine-readable JSON with companion markdown summaries, resumable, seed=42.
