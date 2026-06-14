# Experimental Methodology (Iteration 9 -- Full-Mode Evidence-Grounded Plan)

## Overview

This iteration transitions from PILOT to FULL mode. The paper has been restructured from
THREE to TWO primary contributions based on pilot evidence:

- **C1 (Primary)**: First cross-domain absorption characterization -- semantic hierarchies
  show MORE absorption than first-letter spelling, inverting conventional wisdom.
- **C2 (Secondary)**: Causal mechanism confirmation via activation patching + tightened hedging
  classification.

**Demoted to appendix (pilot negative results):**
- GAS unsupervised detector: rho=0.12 (target 0.6). FAILED.
- CMI theoretical pillar: rho=0.044, p=0.83. NOT SUPPORTED.
- Absorption Tax quantitative predictions: rho=0.08 (absorption-MSE), rho=0.16 (R_pc). NOT SUPPORTED.

**Three critical gaps addressed in this iteration:**
1. **Probe quality** (CRITICAL BLOCKER): Best RAVEL probe F1=0.83 at layer 12 only.
   Must test layers 6, 18, 24 where factual knowledge is typically encoded in 2B models.
2. **Activation patching statistical significance**: Pilot n=7 words, Wilcoxon p=1.0.
   Expand to n >= 20 words with 200+ contexts each.
3. **Multi-layer, multi-SAE absorption measurement**: Pilot tested only L12-16k.
   Full mode covers layers 6, 12, 18, 24 x widths 16k, 65k.

All experiments are **training-free** (inference-only on pre-trained SAEs).

---

## Environment and Dependencies

```bash
pip install sae-lens transformer-lens datasets scikit-learn scipy statsmodels
pip install sae-spelling   # Chanin et al. first-letter absorption benchmark
```

**Models:**
- Gemma 2 2B (`unsloth/gemma-2-2b`, local cache, no HF token needed -- confirmed iter_008)
- GPT-2 Small (fallback only)

**SAEs:**
- Gemma Scope: `gemma-scope-2b-pt-res-canonical` at layers 6, 12, 18, 24 x widths 16k, 65k
- SAEBench: BatchTopK 16k, Matryoshka 32k at layer 12

**Hardware:**
- GPU: NVIDIA RTX PRO 6000 Blackwell, 95GB VRAM (confirmed available)
- Max parallel tasks: 4 GPUs (config)
- All tasks are inference-only (no training)

---

## Phase 0: Blocking Experiments (COMPLETED in iter_008 pilot -- scale up)

All three blocking experiments from 3 consecutive reviews are DONE in pilot. This iteration
scales up for statistical rigor and reports the threshold sensitivity data.

### Exp 0.1: Activation Patching at Scale (SCALE UP from n=7 to n>=20)

**Pilot result:** 14.3% recovery vs. 0.5% control. Wilcoxon p=1.0 (underpowered, n=7).

**Full-mode changes:**
- Expand from 7 words to >= 20 words (identify additional absorption pairs via integrated gradients)
- Increase contexts per word from 100 to 200
- Extend to cross-domain absorption pairs (city-continent, city-language) -- not just first-letter
- Wilcoxon signed-rank target: p < 0.01
- Bootstrap 95% CI on recovery rate difference

### Exp 0.2: Tightened Hedging (DONE -- extend to cross-domain)

**Pilot result:** Strict 7.4% vs loose 92.6%. 85.3% compensatory resolution. Definitive.

**Full-mode changes:**
- Apply same tightened classification to cross-domain hierarchies (not just first-letter)
- Run on all SAE configs (8 Gemma Scope + SAEBench)
- Test L0 sensitivity: thresholds at L0=22, 44, 88, 176

### Exp 0.3: CMI at L0=22 (DONE -- negative result, report as appendix)

**Pilot result:** rho=0.044, p=0.83. NOT SUPPORTED. No additional GPU work needed.
Report in appendix with documented analysis of why binary CMI formulation fails.

### Exp 0.4: Threshold Sensitivity Reporting (DATA EXISTS from iter_001)

**Status:** 141KB JSON (5x4 grid) exists at `iter_001/exp/results/full/ablation_threshold_sensitivity.json`.
Never reported in any paper version. CPU-only analysis to generate summary table and determine
whether control failure is threshold-dependent or structural.

---

## Phase 1: Cross-Domain Absorption Characterization (Primary Contribution)

### Step 1.1: Probe Training at Multiple Layers (CRITICAL)

**Pilot gap:** Only layer 12 tested. Best F1=0.83 (city-continent). All below 0.90 quality gate.

**Full-mode protocol:**
- Train probes at layers 6, 12, 18, 24 for all 4 hierarchies (16 probes total)
- Architecture: L2-regularized logistic regression (C=1.0, sklearn)
- Data: Full RAVEL dataset (~2041 cities), 80/20 stratified split, seed=42
- Frequency-balanced sampling for imbalanced hierarchies (city-country: 80 classes)
- Alternative prompt templates if ICL fails: fill-in-the-blank, multiple-choice
- For first-letter: sae_spelling pipeline (confirmed F1=1.0 in pilot)
- **Quality gate:** F1 >= 0.90 (strict), 0.85 (relaxed with documented rationale)
- Select best layer per hierarchy for downstream experiments

**Key prediction:** Layers 18-24 should yield better RAVEL probes (factual knowledge at later layers).

### Step 1.2: First-Letter Absorption Baseline (REPLICATION)

- Replicate Chanin et al. absorption measurement across 8 Gemma Scope SAE configs
- Use sae_spelling pipeline (F1=1.0 confirmed) for first-letter probes
- Full data: 1000+ letter-containing tokens (up from 282 in pilot)
- Bootstrap 95% CI (10k resamples, seed=42)
- Per-letter breakdown for R_pc analysis
- This is the reference baseline for all cross-domain comparisons

### Step 1.3: Cross-Domain Absorption Measurement (PRIMARY RESULT)

For each hierarchy with passing probe (F1 >= 0.85) x each SAE config:
- Cache residual stream activations on 500+ entity-containing tokens
- Encode through SAE, apply probe to SAE output
- Identify false negatives, run integrated-gradients attribution (10 steps)
- Compute absorption rate with bootstrap 95% CI
- Paired permutation test vs first-letter baseline (10k permutations)
- Cohen's d effect size per comparison

**Expected outcome (from pilot):**
- City-language: 10.4% vs first-letter 3.9% (p=0.005) -- most robust comparison
- City-continent: 53.4% pilot rate is inflated by probe quality; expect reduction with better probes
- City-country: included if probe quality passes gate

### Step 1.4: Hedging Decomposition per Hierarchy

Apply tightened hedging classification to all cross-domain absorption:
- Classify: strict hedging / compensatory resolution / persistent
- Per hierarchy type and SAE config
- Verify pilot finding: semantic hierarchies show proportionally more absorption (69-74%) vs hedging compared to first-letter (45%)

### Step 1.5: Architecture Comparison

- JumpReLU (Gemma Scope) vs. BatchTopK (SAEBench) vs. Matryoshka (SAEBench)
- On each hierarchy type with passing probe
- 2-way ANOVA: absorption ~ architecture * hierarchy_type
- Report L0 alongside absorption rates to control for sparsity differences
- Test H6: does JumpReLU maintain its advantage across hierarchy types?

---

## Phase 2: GAS Validation (APPENDIX -- Negative Result Documentation)

**Pilot verdict:** FAILED (rho=0.12, AUROC=0.59).

**Full-mode (fair evaluation):**
- Scale from 200 to 10k sequences for co-activation statistics
- Recompute GAS on full data to definitively confirm negative result
- Document failure mode analysis: geometric signals insufficient for functional absorption
- Report as negative result in appendix

---

## Phase 3: Absorption Tax (APPENDIX -- Qualitative Framework Only)

**Pilot verdict:** Quantitative predictions NOT SUPPORTED (rho=0.08, 0.16).

**Full-mode:**
- Recompute T(G) with improved probes
- Test qualitative direction: does T(G) predict cross-domain absorption RANKING?
  (even if not individual SAE rates)
- Report as qualitative theoretical framework only

---

## Baselines and Controls

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Random feature ablation | Zero random feature (not child) | Activation patching null |
| Shuffled hierarchy control | Randomly permute hierarchy labels | Absorption rate null (should be ~0) |
| Probe-only baseline | Measure FN rate without SAE (raw residual) | Should be ~0 for quality probes |
| Loose hedging classification | Current 98.6% method (any resolution) | Comparison for tightened hedging |
| First-letter absorption rates | Chanin et al. on first-letter | Cross-domain comparison baseline |

---

## Evaluation Metrics Summary

| Phase | Primary Metrics | Targets |
|-------|----------------|---------|
| 0.1 (activation patching) | Parent recovery rate, Wilcoxon p | p < 0.01 with n >= 20 |
| 0.2 (tightened hedging) | Strict vs. loose hedging rate | Strict << 92.6% (confirmed in pilot) |
| 1.1 (probes) | F1 per hierarchy x layer | >= 0.90 strict, 0.85 relaxed |
| 1.3 (cross-domain) | Absorption rate difference vs. first-letter | p < 0.01, Cohen's d > 0.8 |
| 1.5 (architecture) | Absorption rate per architecture x hierarchy | JumpReLU advantage persists |
| 2 (GAS) | Spearman rho(GAS, absorption) | < 0.3 (confirm negative, appendix) |
| 3 (absorption tax) | T(G) ranking prediction | Qualitative direction only |

---

## Expected Visualizations

- **Table 1**: Cross-domain absorption rates per hierarchy type x SAE config (primary result)
- **Table 2**: Probe quality heatmap: hierarchy x layer (quality gate pass/fail)
- **Table 3**: Activation patching results per word (recovery rate, child vs. control)
- **Table 4**: Hedging decomposition (strict / compensatory / persistent) per hierarchy
- **Table 5**: Architecture comparison: absorption rate per hierarchy x architecture
- **Figure 1**: Absorption rate bar chart across hierarchy types (grouped by architecture, with CI)
- **Figure 2**: Activation patching effect: violin/box plot of recovery rate (child vs. control)
- **Figure 3**: Hedging decomposition stacked bar chart per hierarchy
- **Figure 4**: Probe quality heatmap (hierarchy x layer, color = F1)
- **Figure A1** (appendix): Threshold sensitivity heatmap (cosine threshold x magnitude gap)
- **Figure A2** (appendix): GAS validation scatter (negative result)
- **Figure A3** (appendix): CMI at L0=22 scatter (negative result)
- **Figure A4** (appendix): Absorption Tax T(G) vs. observed cross-domain ranking

---

## Software Architecture

All code in `exp/code/`, adapted from iter_008 with full-mode sample sizes:

```
exp/code/phase0_activation_patching_full.py  -- Scaled activation patching (n>=20)
exp/code/phase0_tightened_hedging_full.py    -- Cross-domain tightened hedging
exp/code/phase0_threshold_reporting.py       -- CPU analysis of iter_001 data
exp/code/phase1_probe_training_full.py       -- 4 layers x 4 hierarchies
exp/code/phase1_absorption_firstletter.py    -- Replicate Chanin et al. (8 SAEs)
exp/code/phase1_absorption_crossdomain.py    -- Cross-domain (primary result)
exp/code/phase1_hedging_decomposition.py     -- Per-hierarchy decomposition
exp/code/phase2_gas_full.py                  -- GAS on 10k sequences (appendix)
exp/code/phase3_tax_empirical.py             -- Absorption Tax (appendix)
exp/code/phase4_architecture_comparison.py   -- Cross-hierarchy architecture comparison
exp/code/phase4_consolidation.py             -- Final synthesis
```

All outputs: machine-readable JSON with companion markdown summaries, resumable, seed=42.
