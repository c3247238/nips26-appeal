# Experiment Methodology: Rate-Distortion Theory of Feature Absorption in SAEs (iter_002)

## Overview

This methodology is revised based on comprehensive pilot evidence from iter_002. All experiments
build on the following validated facts and prune previously falsified directions.

**Primary models (in priority order):**
1. GPT-2 Small via SAELens pre-trained SAEs — open access; exact Chanin et al. labels available
2. Gemma 2 2B via Gemma Scope — gated access, used if available; fallback to GPT-2 Small
3. Llama-3.1-3B via SAELens — secondary fallback only

**Compute budget:** ~8–10 GPU-hours on single local GPU (RTX PRO 6000, 24+ GB VRAM).

**GPU available:** NVIDIA RTX PRO 6000 Blackwell Server Edition (confirmed from B2 pilot).

---

## Pilot Evidence Summary (What We Know)

### Validated Findings
- EDA (Encoder-Decoder Alignment) = 1 - cos(encoder_j, decoder_j): **AUROC=0.681 at GPT-2 L6,
  Cohen's d=+0.70**. This is the only working absorption detection metric validated so far.
- 4 hierarchy probes pass F1 >= 0.80 gate: first_letter (F1=0.820), city_country_binary (FAILED
  shuffle gate, EXCLUDED), noun_proper (F1=0.987), animate_inanimate (F1=1.0).
- EDA vs. 1/L0 (primary SAE suite): sigmoid fit beats linear (LRT p=0.027), inflection at L0_c~81.
  Directional support for H4a.
- SAELens has ≥11 SAE configurations for GPT-2 Small across layers 2–10 and widths 12k–98k.

### Falsified Directions (DO NOT REPEAT)
- ASI as currently formulated (cos^2 × freq_p/freq_c): AUROC=0.476 — FAILS. The frequency
  ratio is unstable at low base rates and the max-operation is noisy.
- RD threshold (lambda > sin^2(theta)) with lambda=1/L0: AUROC=0.410 — FAILS. lambda~0.02
  is too small to activate the threshold for any realistic decoder angle.
- cos^2(theta) between child and random parents as absorption predictor: INVERSE direction
  (Cohen's d=-0.48 at L6). Letter features have LOWER decoder cosine with parent features.
- Parent-latent-suppression as cross-domain absorption measurement: absorption_rate=0.0 for all
  clean hierarchies. Concept latents ARE reliably active — no absorption of the parent signal.
- Cross-model probe transfer (Qwen probes on Gemma): already excluded from iter_001.

### Design Revisions Required
1. **EDA replaces ASI and raw cos^2 as primary detection metric.** All Phase D experiments
   use EDA and its refined variants (encoder-only vs. decoder-only decompositions).
2. **H1 geometric mechanism needs restatement.** The key geometric signature of absorption is
   ENCODER-DECODER MISALIGNMENT (EDA), not parent-child decoder similarity. This is a richer
   finding than originally hypothesized.
3. **Cross-domain absorption (H2) needs correct measurement:** track child-feature suppression
   (child-specific SAE feature fails to fire because parent absorbs it), NOT parent-latent
   non-firing. The sae-spelling definition must be used as the operational model.
4. **ASI reformulation (if pursued):** Parent-perspective ASI — given identified (parent, child)
   pairs from EDA-ranked features, does ASI(p,c) = cos^2(encoder_p, decoder_c) × freq_p/freq_c
   predict which child-parent pairs actually exhibit absorption events?
5. **H4a phase transition:** sigmoid model is directionally supported on primary suite (5 configs,
   LRT p=0.027) but bootstrap CI for L0_c is extremely wide. Full Phase E will add width
   variation to tighten this.

---

## Phase B (Revised): Rate-Distortion Threshold and EDA Geometry

**Goal (revised):** Characterize the geometric signature of absorption via EDA, resolve the
H1 prediction failure, and establish what the decoder geometry says (even if it contradicts
the naive RD threshold).

### B.1-REV — EDA Geometric Decomposition (30 min)

**Setup:** GPT-2 Small, layers 6 and 10. Use EDA = 1 - cos(encoder_j, decoder_j) as primary
metric.

**Analysis:**
- Decompose EDA by feature type: letter features vs. non-letter features.
- Compute: mean_encoder_norm, mean_decoder_norm, cos(encoder, decoder) separately.
- Hypothesis (revised): Absorbed letter features have high EDA because their ENCODER direction
  has been pulled toward the parent while the DECODER remains specialized.
- Test: cos(encoder_j, parent_probe) vs. cos(decoder_j, parent_probe) for letter features.
  If encoder is closer to parent probe than decoder, this confirms the revised absorption model.
- Report: mean cos(encoder, letter_probe), cos(decoder, letter_probe) per feature class.

**Success metric:** Letter features have significantly higher cos(encoder_j, letter_probe) than
cos(decoder_j, letter_probe) (paired t-test p < 0.01), confirming encoder-decoder dissociation.

**Baseline:** Same analysis on non-letter features (should show no such dissociation).

**Output:** `exp/results/full/B1_eda_decomposition.json`

### B.1-RAVEN — Exact Chanin Labels with Pairwise EDA (45 min)

**Setup:** Load the Chanin et al. sae-spelling repo to get TRUE absorbed vs. non-absorbed
feature pairs (not just feature-level proxy labels). For each (parent p, child c) pair from
the first-letter task:
- Compute EDA_child = 1 - cos(encoder_c, decoder_c)
- Compute cos(encoder_c, decoder_p) — does child encoder align with parent decoder?
- Compute cos(encoder_p, decoder_c) — does parent encoder align with child decoder?

**Why this matters:** The pilot used proxy labels (probe_decoder_alignment) which give per-feature
labels, not per-pair absorption events. True pairwise labels from sae-spelling can distinguish
the absorbed child from the parent absorber.

**Success metric:** EDA of absorbed child features significantly higher than non-absorbed children.
AUROC >= 0.65 using pairwise EDA against sae-spelling ground truth.

**Output:** `exp/results/full/B1_pairwise_eda.json`

### B.2 — Sparsity-EDA Scaling (40 min)

**Data already collected from pilot:** 11 configs across layers 2–10 and widths 12k–98k.

**Full analysis (extend pilot data):**
- Separate analysis by SAE family (jb vs. ajt) — AJT SAEs show inverse EDA signal, likely
  due to different training regime (normalizing by decoder norm).
- Fit sigmoid to primary jb suite (5 configs): report LRT p, BIC comparison, L0_c with
  bootstrap CI. (Pilot: LRT p=0.027, L0_c~81).
- Width analysis: plot EDA_delta vs. width at matched L0. Test whether wider SAEs show
  higher EDA (more feature specialization → more misalignment for absorbed features).
- Report clear scope: EDA is layer-specific (works at L6, fails at L10); note in methods.

**Note:** No new experiment needed — reuse B2 pilot data and add proper analysis.

**Output:** `exp/results/full/B2_sparsity_scaling.json`

### B.3 — Cross-Architecture Comparison (30 min)

**Setup:** Compare jb SAEs (standard TopK) vs. ajt SAEs (alternative training regime) at
matched L0. Both available from SAELens at layer 6, width 46080 (ajt) and 24576 (jb).

**Finding to explain from pilot:** AJT SAEs show NEGATIVE EDA delta for letter features
(EDA_letter < EDA_nonletter). This suggests the AJT training changes the encoder-decoder
relationship systematically.

**Analysis:**
- Compute EDA distribution for letter vs. non-letter features in both architectures.
- Test: does AJT's lower EDA for letter features correspond to lower absorption rate?
  Use sae-spelling metric with ≥100 tokens per letter to measure true absorption rate.
- Theory prediction: AJT's modified training prevents the encoder-decoder dissociation
  that EDA measures, which may correspond to either (a) less absorption or (b) a different
  absorption geometry.

**Output:** `exp/results/full/B3_cross_arch.json`

---

## Phase C (Redesigned): Cross-Domain Absorption with Correct Measurement

**Critical redesign from pilot C2 findings:** The previous measurement tracked whether
concept-specific SAE latents (parent latents) fail to fire for concept words. This is
NOT the sae-spelling absorption definition. The sae-spelling definition tracks whether
the CHILD (specific token feature) fails to fire because the PARENT (category feature)
absorbs it.

**Correct operational definition:**
- Absorption event: token "elephant" has a specific SAE feature (child) that SHOULD fire
  (when "elephant" is isolated). But in context, a more general feature "starts-with-E" (parent)
  fires strongly AND the "elephant" child feature is suppressed.
- Measurement: Given a known parent SAE latent (identified by probe), check if child-specific
  SAE latents have significantly lower activation magnitude than expected.

### C.1 — Probe Training and Parent Latent Identification (carried over, 20 min)

**Status:** DONE from pilot. Passing hierarchies: first_letter, noun_proper, animate_inanimate.
city_country_binary excluded (failed shuffle gate).

**Additional work:**
- For each passing hierarchy, identify the top-5 parent SAE latents per parent category using
  empirical differential activation (v3 method from pilot).
- Record: parent_latent_idx, mean_concept_activation, mean_control_activation, differential.
- These are the "absorber" candidates for Phase C.2.

**Output (update):** `exp/results/full/C1_probe_training.json` (already exists, extend with
parent latent index lists)

### C.2-REDESIGN — Child-Feature-Suppression Absorption Measurement (60 min)

**Method:**
For each (hierarchy_type, parent_category):
1. Identify parent SAE latents (from C.1 empirical discovery, top-3 latents per category).
2. Collect "concept tokens" — tokens that belong to the parent category (e.g., tokens starting
   with "c" for the first_letter hierarchy).
3. For each concept token, find its "token-specific child latent": the SAE latent that fires
   most specifically for that token in isolation (max differential activation vs. unrelated tokens).
4. Check absorption: in context where the parent latent fires strongly (top-25% activations),
   does the child-specific latent fire at a lower rate than in contexts where the parent does
   NOT fire? Rate = P(child_active | parent_fired) vs. P(child_active | parent_not_fired).
5. Absorption rate = fraction of (parent, child) pairs where child activation is suppressed by
   parent presence.
6. Null control: permute parent-child assignments (100 permutations) to establish null
   absorption rate. Ratio-to-null >= 2.0 required for GO.

**Important:** n_concept_tokens >= 50 per parent category required. n_tokens per measurement
must be >= 100 (100-token sample from OpenWebText). Report n_pos, n_neg, and CI reliability.

**Success metric:** Ratio-to-null >= 2.0 for at least 2 hierarchy types, p < 0.05 after
Bonferroni correction over number of hierarchies tested. AUPRC reported alongside AUROC.

**Failure path:** If absorption_rate near zero for all hierarchies even with redesigned measurement,
report as scoped null result: EDA detects absorbed features but absorption may be primarily a
first-letter (orthographic) rather than semantic phenomenon.

**Output:** `exp/results/full/C2_child_suppression_absorption.json`

### C.3 — Hierarchy Property Correlation (30 min)

For each passing hierarchy, compute:
- Mean EDA of top parent latents (higher EDA → higher absorption susceptibility per revised model)
- Mean decoder cosine between parent and child latents (from pairwise B1 analysis)
- Spearman correlation between EDA and measured absorption rate per hierarchy

**Output:** `exp/results/full/C3_hierarchy_correlation.json`

---

## Phase D (Revised): EDA-Based Probe-Free Detection

**Revised from ASI (which failed) to EDA-based detection:**
EDA = 1 - cos(encoder_j, decoder_j) is the only validated predictor. Phase D validates
it rigorously and tests refined variants.

### D.1 — EDA Validation Against Exact Labels (30 min)

**Setup:** Load exact sae-spelling absorption labels for GPT-2 Small L6 (Chanin et al.).
This gives per-feature absorbed/not-absorbed ground truth on the first-letter task.

**Metrics:**
- AUROC (primary), AUPRC (required for imbalanced labels)
- Precision@k for k=50, 100, 500
- Precision-recall curve

**Baselines:**
- Random ranking (AUROC=0.5)
- cos(encoder_j, decoder_j) alone (without 1-transform)
- Feature activation frequency alone
- Decoder norm alone

**Run null:** 100 permutations of absorption labels. Confirm EDA exceeds null by ≥2 SD.

**Target:** EDA AUROC >= 0.65 on exact Chanin labels (pilot proxy labels gave AUROC=0.681,
exact labels should give cleaner signal).

**Output:** `exp/results/full/D1_eda_validation.json`

### D.2 — EDA Refined Variants (30 min)

Test whether improved EDA formulations outperform the basic cos(encoder, decoder):
- EDA-norm: (1 - cos(encoder_j, decoder_j)) × ||encoder_j|| / ||decoder_j||
- EDA-top5: use the top-5 most active features' encoder-decoder angles
- EDA-parent-aware: cos(encoder_j, parent_decoder) as child absorption predictor (given known
  parent from C.1 discovery)

Report AUROC for each variant. Select best for use in D.3.

**Output:** `exp/results/full/D2_eda_variants.json`

### D.3 — EDA Cross-Domain Validation (30 min)

For hierarchies from Phase C.2 with detected absorption:
- Compute mean EDA of empirically discovered parent latents.
- Spearman rho between mean EDA and absorption rate (across hierarchy types).
- Report as directional evidence (n=3-4 data points, not statistical test).

**Output:** `exp/results/full/D3_eda_cross_domain.json`

---

## Phase E: Phase Transition and Dynamics (Revised)

**Pilot finding:** Sigmoid fit to primary jb suite (5 layers) passes LRT p=0.027, L0_c~81.
However, only 5 data points from the same layer (different layers have confounded L0 and
representation quality). Need within-layer sparsity variation, not cross-layer comparison.

### E.1 — Within-Layer Sparsity Sweep (40 min)

**Problem with pilot data:** B2 pilot varied layers (2, 4, 6, 8, 10) at fixed width, not
L0 at fixed layer. Cross-layer comparison conflates absorption with representation maturity.

**Revised design:** Use AJT SAEs at fixed layer 6, which provide 3 distinct L0 values
(sce: L0~34, sle: L0~46, scl: L0~81) at width 46080. Plus jb at layer 6 (L0~51).
For jb feature-splitting suite at layer 8 (widths 12k, 49k, 98k, all at L0~50): controlled
width analysis at fixed L0.

**Analysis:**
- Compute EDA_delta (letter vs. non-letter) for AJT suite (within-layer, 3 L0 values).
  Problem: AJT shows NEGATIVE EDA_delta — may need architecture-specific analysis.
- Alternative: use sae-spelling absorption rate (not EDA) as the dependent variable for
  the phase transition analysis. Compute true absorption rate at each config.
- Fit sigmoid vs. linear to: absorption_rate vs. 1/L0 across AJT configs.
- Also test: jb suite (cross-layer, different L0) treated as observational data.

**Success metric:** Sigmoid BIC lower by >= 6 than linear BIC. If insufficient data, report
as underpowered (< 4 L0 variants with matching architecture).

**Output:** `exp/results/full/E1_phase_transition.json`

### E.2 — Hysteresis Test (60 min)

**Design unchanged from original plan.** Load GPT-2 Small SAE at high sparsity, fine-tune
500 steps with reduced sparsity, compare to scratch SAE.

**Prerequisites:** Must identify a SAELens SAE with L0 <= 25. From B2 pilot, lowest L0 is
18.5 at layer 2 (jb). Use layer 2 SAE as high-sparsity starting point.

**Skip condition:** If no accessible SAE has L0 <= 30, report H4b as untested.

**Output:** `exp/results/full/E2_hysteresis.json`

### E.3 — Width-EDA Analysis (30 min)

**Data already available from B2 pilot:** Feature-splitting suite at layer 8 (widths 12k, 49k,
98k, all L0~50). Plus primary suite.

**Analysis:**
- Is EDA_delta significantly larger for wider SAEs at matched L0?
- Theory: Wider SAEs have more capacity for specialization, so absorbed features will show
  greater encoder-decoder misalignment.
- Report: EDA_delta by log2(width), effect size, Spearman rho.

**Output:** `exp/results/full/E3_width_analysis.json`

---

## Phase F: Theoretical Framework and Architectural Mitigations

### F.1 — Formal Proof Revision (no GPU, 45 min)

**Critical change from iter_001:** The original H1 predicted higher decoder-decoder cosine
for absorbed pairs. Pilot shows the OPPOSITE: absorbed features (letter features) have LOWER
decoder cosine with candidate parents. The correct geometric signature is ENCODER-DECODER
MISALIGNMENT (EDA), not decoder-decoder similarity.

**Revised theoretical account:**
- Absorption mechanism: the SAE training pulls the ENCODER direction of the child feature
  toward the parent decoder direction (enabling the child feature to "piggyback" on parent
  activation), while the DECODER remains specialized to the child concept.
- Rate-distortion interpretation: EDA is the geometric fingerprint of this encoder-decoder
  dissociation, which is the absorbed state.
- Threshold restatement (revised): Absorption occurs when the encoder-decoder dissociation
  is energetically stable, i.e., when reducing the child's activation frequency saves more
  sparsity penalty than the decoder specialization costs in reconstruction.
- NEW proof target: derive from first principles why an absorbed feature should have high EDA
  (encoder pulled toward parent, decoder maintained). This requires analyzing the gradient
  flow for the child encoder separately from the child decoder.

**Output:** `exp/results/full/F1_theory_revised.md` with:
- Complete algebraic derivation of EDA as the geometric fingerprint of absorption
- Statement of what the RD threshold actually predicts (encoder-decoder angle, not
  decoder-decoder cosine)
- Revised impossibility theorem in terms of EDA

### F.2 — Architectural Mitigation Analysis (30 min)

**Setup:** If Matryoshka or OrtSAE available via SAEBench for GPT-2 or Gemma, compare EDA
distributions between architectures. Theory predicts: Matryoshka reduces EDA by providing
dedicated "slots" that keep encoder and decoder aligned for hierarchical features.

**Skip condition:** If no alternative architecture available, provide theoretical analysis only.

**Output:** `exp/results/full/F2_mitigation.json`

---

## Pre-Writing Audit (Mandatory)

Before any writing agent launches:
1. All Table 1 cells have source JSON files with traceable numbers.
2. The formal proof does NOT contain circular arguments.
3. Figure 1 is generated as a PDF (not text descriptor).
4. All n_pos >= 50 for AUROC computations; flag smaller samples explicitly.
5. AUPRC reported alongside AUROC for all imbalanced classification tasks.
6. Absorption rate 0.0 results from pilot C2 are reported explicitly in the paper.
7. All EDA AUROC values are for correct layer (L6); L10 failure is reported.

**Output:** `exp/results/full/audit_report.json`

---

## Statistical Analysis Protocol

- Bonferroni correction across hierarchy types in Phase C.
- Bootstrap CIs (1000 resamples) for all absorption rate estimates.
- Effect sizes: Cohen's d for continuous, rank-biserial r for Mann-Whitney.
- Null controls: shuffled-label (100 permutations) for absorption rate; permuted-score
  (100 permutations) for AUROC computations.
- Flag n_pos < 50 as insufficient for CI; present point estimate only.
- Separate AUROC and AUPRC tables — AUROC alone is misleading at extreme imbalance.

---

## Expected Visualizations

- **Figure 1 (PDF):** Method diagram. Two-solution comparison: absorbed vs. non-absorbed.
  Geometric illustration of encoder-decoder misalignment (EDA), showing how the encoder
  is pulled toward the parent direction while the decoder remains specialized. MUST be
  generated as a PDF.

- **Table 1:** Main results — EDA detection across configurations. Columns: Config | EDA AUROC
  | EDA AUPRC | Cohen's d | n_pos | Shuffled Null. Rows: GPT-2 L6, GPT-2 L10 (failure case),
  AJT L6 (failure case), plus any Gemma configs if available.

- **Figure 2:** EDA_delta vs. 1/L0 with sigmoid fit. One curve for primary jb suite (within
  fixed-width, cross-layer). Mark inflection point L0_c~81 with CI.

- **Figure 3:** Cross-domain absorption rates. Bar chart per hierarchy type with shuffled-null
  baseline. Error bars = 95% bootstrap CI. Requires non-zero signal from Phase C redesign.

- **Figure 4:** EDA precision-recall curve vs. baselines (cos(encoder, decoder), frequency,
  random). Requires exact Chanin labels from Phase D.1.

- **Figure 5 (optional):** Hysteresis if Phase E.2 is completed.

- **Appendix:** Full algebraic proof of EDA as geometric fingerprint of absorption.

---

## Dependency Map

```
F.1 (theory, no GPU) → runs concurrently with all phases
B.1-REV (EDA decomposition) → B.1-RAVEN (pairwise EDA) → D.1 (EDA validation)
B.2 data (already collected) → B.2 analysis → E.1, E.3
C.1 (probe training, already done) → C.2-REDESIGN → C.3
D.1 + D.2 → D.3 (requires C.2 data)
E.1, E.2 (parallel) → E.3
All full experiments → audit → writing
```

---

## Blocking Gates

Before Phase C.2-REDESIGN:
1. C.1 parent latent lists must be verified (n >= 3 per parent category).

Before Phase D.1:
1. Exact Chanin et al. labels must be loaded (n_pos >= 50).

Before writing:
1. Zero unresolved discrepancies in audit_report.json.
2. Table 1 has all cells with source JSON traceable numbers.
3. Figure 1 exists as a PDF.
4. F.1 proof does not contain circular arguments (explicitly noted if heuristic only).
