# Backup Ideas for Pivot (Synthesis Round 8 -- Iteration 11)

## Backup 1: Controlled Dictionary Experiment

**Title:** Controlled Dictionary Experiment: Separating Encoder from Dictionary in Absorption Causation

**Summary:** Hold the SAE decoder dictionary constant and vary the encoding method (feedforward vs. Orthogonal Matching Pursuit vs. 2-pass). This tests whether absorption is caused by the encoder's greedy activation selection or by the dictionary geometry itself. Strengthened by FULL-mode activation patching (d=0.75-1.50) confirming encoder-level competitive exclusion across all hierarchy types.

**Hypotheses:**
- If amortization gap dominates: OMP absorption < 50% of feedforward at matched L0
- If sparsity landscape dominates: absorption is similar across encoding methods
- 2-pass encoding shows intermediate absorption

**Pilot Focus:** Feedforward vs. OMP comparison on 1000 tokens, Gemma Scope L12-16k. 30 min.

**Novelty:** 8/10. First controlled dictionary experiment. MP-SAE (arXiv:2506.03093) uses different encoding but does not hold dictionary constant.

**Compute:** 1-2 GPU-hours.

**Pivot Trigger:** Activate if probe degradation ablation (H10) reveals cross-domain variation is entirely a probe artifact, undermining the primary contribution, or if spot-check (Phase 1.1) reveals patching data is unreliable.

---

## Backup 2: Feature Absorption as Competitive Exclusion -- Phase Transitions

**Title:** Feature Absorption as Competitive Exclusion: Phase Transitions and Limiting Similarity

**Summary:** Formalize absorption through ecological Lotka-Volterra competition and phase-transition theory. The universal causal mechanism (d=0.75-1.50 across hierarchies) is exactly the competitive exclusion principle from ecology. This provides a theoretical explanation for WHY cross-domain absorption rates differ: competition coefficients (decoder cosine similarity) vary by hierarchy type.

**Hypotheses:**
- Phase transition: absorption probability is sigmoid in decoder cosine similarity with transition width < 0.1
- Hierarchy-dependent theta*: first-letter < city-language < city-continent (matching absorption ordering)
- Lotka-Volterra scaling: theta* ~ sqrt(1/W) * (L0)^(-1/2)

**Pilot Focus:** Plot absorption probability vs. decoder cosine similarity using existing Phase 1 data. 15 min CPU.

**Novelty:** 9/10. Zero prior work on competitive exclusion ecology or phase transitions applied to SAE absorption. Entirely novel framing.

**Compute:** 0 additional GPU-hours (uses existing data).

**Pivot Trigger:** Activate as supplementary analysis when all Phase 0-2 work is complete. Provides the "why" to complement the "what" of cross-domain characterization. Can be added as Section 7 theory discussion or as a standalone follow-up paper.

---

## Backup 3: Absorption-Aware Post-Hoc Feature Correction

**Title:** Absorption-Aware Post-Hoc Feature Correction

**Summary:** Inference-time correction: when a child feature fires and the parent feature does not, re-activate the parent feature using the known absorption relationship. All existing absorption mitigations (Matryoshka, OrtSAE, ATM, masked regularization) are training-time interventions. This is the first evaluation of inference-time correction.

**Hypotheses:**
- Post-hoc correction recovers >= 50% of SAE-vs-probe gap on downstream tasks
- OR: correction has negligible effect because absorption permanently alters decoder geometry

**Pilot Focus:** Apply correction to first-letter absorption using known absorption patterns. 30 min.

**Novelty:** 8/10. First evaluation of inference-time absorption correction. Directly addresses reviewer demand for prescriptive guidance.

**Compute:** 1-2 GPU-hours.

**Pivot Trigger:** Activate if reviewers ask "so what can we DO about absorption?" -- the characterization paper lacks prescriptive recommendations.

---

## Backup 4: Within-Hierarchy Variation Analysis

**Title:** Within-Hierarchy Variation: Why Europe Absorbs at 90% While Africa Absorbs at 4%

**Summary:** Depth alternative to breadth. Rather than comparing across hierarchies, analyze per-class absorption variation WITHIN a single hierarchy. Why do some parent classes (e.g., Europe) show extreme absorption while others (e.g., Africa) are nearly immune? Predictable from class size, training frequency, and decoder similarity.

**Hypotheses:**
- Per-class absorption rate predictable from class size + log(frequency) + cos_sim (R^2 > 0.5)
- Extreme class imbalance in semantic hierarchies explains why they show higher absorption than uniform first-letter

**Pilot Focus:** Per-continent absorption rates from existing data. 30 min CPU.

**Novelty:** 7/10. Addresses depth critique from iter_010 review. Natural extension of existing work.

**Compute:** 0 additional GPU-hours (uses existing data).

**Pivot Trigger:** Activate if reviewers demand deeper mechanistic analysis within a single hierarchy rather than breadth across hierarchies.

---

## Dropped Ideas (with reasons)

| Candidate | Reason Dropped |
|-----------|---------------|
| GAS as Primary Detector | REFUTED: rho=0.12 (target 0.6) |
| CMI Taxonomy | NOT SUPPORTED: rho=0.044 (p=0.83) |
| Absorption Tax Quantitative | rho=0.08; qualitative framework only |
| EDA Universal Detector | Failed for 75% early absorption; replaced by GAS which also failed |
| ITAC Primary Detection | Core unsupervised hypothesis refuted by GAS failure |
| Hierarchy-Coherent SAE Loss | Crowded by HSAE, KronSAE; requires training (violates constraint) |
| Scaling Laws (standalone) | Incorporated into rate-distortion analysis |
| PAC-Bayes Generalization | Bound improvement negligibly small |
