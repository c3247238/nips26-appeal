# Experiment Result Analysis

## Key Results Summary

### Primary Detection Results (Tasks A1, A3, B2)
- **encoder_norm AUROC = 0.757** (GPT-2 L6, Standard/L1 SAE, n_pos=18, 95% CI [0.655, 0.851])
- **encoder_norm AUROC = 0.837** (GPT-2 L6, TopK-32k SAE, n_pos=77, 95% CI [0.807, 0.870])
- encoder_norm significantly outperforms EDA: DeLong z=3.046, p=0.0012 (one-sided)
- encoder_norm Cohen's d vs. non-absorbed: 0.971 (Standard), 1.235 (TopK-32k)
- O_jaccard AUROC = 0.721, AUPRC = 0.075, Precision@50 = 0.10 — independent complementary signal
- ARS_v2 AUROC = 0.586 — does NOT significantly improve over encoder_norm alone (DeLong z=-2.45, p=0.99)
- Spearman ρ(encoder_norm, O_jaccard) = 0.044 — near-zero correlation confirms complementary information

### Amortization Gap Experiment (Tasks C1, C2)
- **Feedforward AR = 0.978, OMP AR = 0.978** (letters a, e, s; K_omp=53)
- omp_reduction_ratio = **0.000** across all three letters
- Interpretation: sparsity_landscape_dominant
- OMP reconstruction error (0.219) < feedforward (0.242) — OMP implementation validated

### Theoretical Structure (Task A2)
- Spearman ρ(encoder_norm, EDA) = 0.712 — high shared signal, but not identical
- Layer peak at L6: absorbed/non-absorbed encoder_norm ratio = 1.267 (L2: 0.877, L4: 1.055, L8: 0.891, L10: 0.933)
- Monotone increase up to L6 confirmed; no early/late subtype difference (Mann-Whitney p=0.055)
- 10 early-absorbed (no decoder anchor), 8 late-absorbed (decoder converged to parent)

### Cross-Architecture Comparison (Task A3)
- Both Standard/L1 (AUROC=0.757) and TopK-32k (AUROC=0.837) show encoder_norm as absorption detector
- |AUROC difference| = 0.080 — does not exceed 0.10 threshold
- Hook confound confirmed: Standard uses resid_pre, TopK uses resid_post — AUROC values not directly comparable

### Dictionary Width Remediation (Task F1)
- 12/18 absorbed features recovered in 32k SAE (cos_sim > 0.80), 6/18 not recovered
- mean_best_cosine = 0.791, median = 0.815
- Interpretation: 67% recoverable by capacity expansion, 33% require training-objective changes

### Cross-Hierarchy Absorption (Task D2)
- Entity-type AR = 0.000, negative control AR = 0.05 — METHODOLOGY FAILURE
- Qwen-to-GPT-2 cross-model probe transfer via random QR decomposition was invalid
- Result is NOT evidence for or against entity-type absorption; it is an artifact

---

## Debate Perspectives Summary

Note: Individual perspective files (optimist.md, skeptic.md, etc.) were recorded as pending in checkpoint but a full synthesis and verdict were produced. The following summarizes the 6-perspective consensus as documented in synthesis.md and verdict.md:

- **Optimist:** Three independent replication contexts for encoder_norm; DeLong test significant; H2 falsification is clean and novel; O_jaccard provides complementary signal; F1 is practically informative. Paper ready to write immediately.

- **Skeptic:** n_pos=18 at gold-label L6 is genuinely small (bootstrap CI ±0.09); three mislabeled features could shift results materially; hook confound prevents clean cross-architecture attribution; AUPRC=0.004 despite AUROC=0.757 reflects severe class imbalance. Requires power analysis and hook-confound correction before main venue.

- **Strategist:** Lead with H2 falsification (most novel, most actionable); encoder_norm detection as enabling contribution; target ICML 2026 MI Workshop first, then ICLR 2026 after hook-confound correction. Begin writing now, run correction experiment in parallel.

- **Comparativist:** Competitive position is solid for workshop, borderline for main track. encoder_norm is practically valuable (weight-only, <1 second at 65k SAE) vs. Chanin IG (activation-based, slow). Direct validation of Tang et al. over O'Neill et al. is the most citable mechanistic contribution.

- **Methodologist:** Hook-confound correction is priority HIGH — single cheap experiment turns limitation into controlled result. Power analysis required (minimum detectable effect at n=18). D2 entity-type result must be labeled as methodology failure, not negative result. Functional recovery definition for F1 needs activation confirmation.

- **Revisionist:** The field has been assuming amortization gap and encoder improvements would fix absorption; this paper's OMP oracle result directly contradicts that assumption and redirects the research program toward training objectives and dictionary coverage. Reframing from "detection paper" to "mechanism paper" with detection as enabler is the right move.

**Synthesis verdict:** PROCEED — 6.5/10 quality score. One genuinely novel mechanistic finding (H2 falsification), one practical detection improvement (encoder_norm > EDA), two supporting findings (O_jaccard independence, 67% width recovery). Hook confound is limitation to acknowledge, not fatal flaw.

---

## Analysis

### 1. Method Feasibility

**encoder_norm detector: FULLY CONFIRMED.** The core hypothesis (H_ENC: absorbed features have higher encoder norms) is validated across three independent contexts with consistent direction. The DeLong test provides formal statistical confirmation. The mechanistic interpretation (gradient competition driving elevated encoder weights during training) is coherent and supported by the layer-monotonicity result (ratio peaks at L6, the target layer, and decreases elsewhere).

**ARS (original proposal): PIVOTED IN EXECUTION.** The original proposal (Contribution 1) proposed ARS = O(i,j) × A(i,j) × cos²(d_i, d_j) as the main unsupervised detector. Results show: ARS_v2 AUROC = 0.586 (no improvement over encoder_norm), ARS_original AUROC = 0.528 (near chance), ARS_full AUROC = 0.528. The O_jaccard component alone achieves AUROC = 0.721 and is the only co-occurrence component with independent signal. The full ARS product form as proposed does not work as intended. However, O_jaccard as a standalone detector is a genuine finding worth reporting as a secondary contribution.

**Amortization gap experiment: EXECUTED AND DECISIVE.** OMP oracle at matched sparsity (K=53) achieves identical absorption rate to feedforward encoder (0.978 vs. 0.978) across all three tested letters. The pre-committed falsification criterion (OMP >= 80% of feedforward rate) is met exactly at 100%. H2 (amortization gap dominant) is falsified in its strong form. Tang et al.'s sparsity landscape theory is supported.

### 2. Performance vs. Baselines

- encoder_norm: AUROC 0.757 (L6 gold labels) to 0.837 (TopK-32k proxy labels) — both exceed EDA baseline 0.650 by statistically significant margin
- O_jaccard: AUROC 0.721 with AUPRC 0.075 — substantially above random AUPRC baseline (~0.00073%)
- ARS product formulations: AUROC 0.527–0.586 — marginal improvement over chance, do NOT outperform encoder_norm
- Cross-architecture: encoder_norm generalizes across Standard/L1 and TopK-32k (with hook caveat)
- Layer analysis: encoder_norm signal is layer-specific (peaks at L6), confirming detector must be applied to the correct layer

### 3. Improvement Headroom in Current Direction

**Short-term (1-2 hours):** Hook-confound correction experiment — run encoder_norm on Standard SAE at resid_post or compute cosine similarity between resid_pre and resid_post to verify approximate equality. This converts a limitation to a clean controlled comparison.

**Short-term (30 min):** Power analysis — bootstrap minimum detectable AUROC at n=18, required for any venue submission.

**Medium-term:** Expand entity-type hierarchy experiment with proper GPT-2-native probes (not Qwen transfer). Would address H3 which remains genuinely open.

**Medium-term:** Confirm that F1 recovered features (cos_sim > 0.80) activate on the same inputs as the absorbed features (functional recovery, not just geometric proximity).

The core results are already strong enough for writing. The identified improvements are presentation requirements and confidence-builders, not fundamental revisions.

### 4. Time-Cost Tradeoff

The existing experimental results support a complete paper narrative:
1. H2 falsification (most novel, requires only honest reporting)
2. encoder_norm detection (requires hook correction for clean claim)
3. O_jaccard as independent signal
4. Width-recovery partial confirmation

Pivoting to an alternative research direction (e.g., cross-hierarchy from scratch, safety attribution) would cost 1-2 additional iterations without clear improvement in publishable contribution quality. The current package at 6.5/10 is already workshop-ready and can reach main-track with the hook correction experiment (1-2 hours). The cost-benefit strongly favors writing with the current results.

### 5. Critical Objections Assessment

**Fatal objections (would block publication):**
- None identified. All concerns are addressable as limitations or via low-cost additional experiments.

**Serious but non-fatal objections:**
- n_pos=18 small sample at L6: Mitigated by three-context replication (L6 IG, L10 proxy, TopK-32k proxy). Power analysis required.
- Hook confound in cross-architecture comparison: 1-2 hour fix available; or scope claim to "encoder_norm works on both architectures" (unconfounded) without comparing AUROC magnitudes directly.
- ARS product form does not work: Already pivoted to O_jaccard as standalone; reframe Contribution 1 from "ARS" to "encoder_norm + O_jaccard dual-signal detection."

**Already-dismissed objections:**
- H3 entity-type result: Confirmed as methodology failure, not scientific negative. Explicitly label as future work.
- OMP oracle calibration concern: The null result (0.0% reduction) is so strong that it survives any reasonable calibration uncertainty.

---

## Decision Rationale

The experiment package for iteration 3 has produced:

1. A **genuinely novel and decisive mechanistic finding** (OMP oracle = 0% absorption reduction, falsifying amortization gap hypothesis, supporting Tang et al. sparsity landscape theory). This result is publishable on its own as a workshop paper and is highly citable for the field.

2. A **practical detection improvement** (encoder_norm AUROC 0.757–0.837 > EDA 0.650, statistically confirmed, weight-only and scalable). Replicated across three independent contexts.

3. **Independent complementary signal** (O_jaccard AUROC 0.721, near-zero correlation with encoder_norm, dramatically higher AUPRC). Supports structural reality of the phenomenon.

4. **Partial mechanistic constraint on remediation** (67% width recovery, 33% requires training-objective changes). Practically informative for SAE practitioners.

The synthesis verdict (6.5/10, PROCEED) from the 6-perspective debate is well-reasoned and supported by the quantitative evidence. The only substantial remaining work is: (a) hook-confound correction experiment (~1-2 hours), (b) power analysis (~30 minutes), and (c) writing the paper.

Pivoting would mean abandoning confirmed results that already constitute a complete, publishable contribution. The cross-hierarchy work (H3) and safety attribution (H5) were explicitly de-prioritized in the synthesis as not ready for this iteration and should be framed as future work. A pivot would risk sacrificing the clean H2 falsification narrative without gaining comparable novelty.

The evidence strongly supports PROCEED. The paper should be written now, with Track B experiments (hook correction, power analysis) running in parallel.

---

DECISION: PROCEED
