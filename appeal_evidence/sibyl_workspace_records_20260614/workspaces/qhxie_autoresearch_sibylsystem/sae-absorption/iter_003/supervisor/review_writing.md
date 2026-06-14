# Supervisor Review: SAE Feature Absorption Paper
**Date:** 2026-04-14
**Score:** 5.5 / 10
**Verdict:** continue

---

## Executive Summary

The paper pursues a genuinely interesting mechanistic question — adjudicating amortization gap vs. sparsity landscape as the dominant cause of SAE feature absorption — and the OMP oracle experimental design is conceptually clean and novel as applied to pre-trained SAEs. The encoder weight norm heuristic is empirically correlated with gold absorption labels. Negative results are reported with exemplary scientific discipline.

However, the paper cannot be submitted in current form because the headline "decisive falsification" rests on an experiment with near-zero statistical power (97.8% ceiling absorption rate, 3 letters, 30 tokens each), the cross-architecture detection replication uses labels that E1 confirms are not actual absorbed features, and the mechanistic narrative for EncNorm is contradicted by the signal inverting direction at 3 of 5 measured layers. All core experiments are in PILOT mode at 30-100x smaller scale than specified in the methodology.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Novelty | 7/10 | OMP oracle applied to SAE absorption is genuinely new; EncNorm as absorption heuristic is empirically found but not theoretically novel; O_jaccard framing original |
| Technical Soundness | 4/10 | Three critical soundness issues: ceiling-effect null result presented as definitive, proxy labels used as absorption labels, mechanistic narrative contradicted by own data |
| Experimental Rigor | 4/10 | All experiments in PILOT mode; OMP ceiling effect makes the null result uninterpretable; hook confound uncontrolled; D2 cross-hierarchy experiment uses wrong probe for control |
| Reproducibility | 5/10 | Strong statistical toolkit, pre-registered criteria, code structure apparent; but GPT-2 Small only (single 2019 model), sae_spelling library version not pinned, figures exist but cross-verification needed |

---

## Critical Issues (would cause rejection on their own)

### 1. OMP Experiment: Ceiling Effect Makes Null Result Uninterpretable

The OMP oracle experiment (C2) operates at feedforward absorption rate = 0.978 across 3 letters × 30 tokens each. The standard error on a proportion difference at this baseline is approximately 0.037 per letter. The pre-committed falsification criterion — "OMP >= 80% of feedforward absorption rate implies H2 falsified" — is satisfied trivially at 1.000 regardless of any true effect: at 97.8% baseline, both encoder methods produce near-identical rates within sampling noise.

The paper's statement "the result is unambiguous" and "decisively falsifies the amortization gap hypothesis" is scientifically indefensible. A null result at near-zero statistical power cannot constitute a falsification. No confidence interval on (AR_OMP - AR_FF) is reported. No power analysis is presented.

**Required fix**: Run C2 at the specification in the methodology — 10,000 tokens, 26 letters, K_omp matched to feedforward mean L0. Report 95% bootstrap CI on the null result per letter. Include a power analysis. If the null holds at full scale with tight CIs, the falsification claim becomes defensible. The current pilot result does not establish it.

### 2. TopK-32k AUROC = 0.837 Measures Decoder Geometry, Not Absorption

E1 and A3 both confirm absorption_rate = 0.0 under IG measurement for letters a, e, o, s, t in the TopK-32k SAE. The n_pos = 77 "labels" come from decoder cosine alignment to letter probes (cosine >= 0.30), not from the Chanin et al. IG absorption detection pipeline. The TopK-32k experiment measures which SAE features have decoder directions that point toward letter-probe directions — a property of the weight matrix geometry, not a measurement of which features undergo the absorption failure mode.

The paper reports this AUROC=0.837 as a "cross-architecture replication" of EncNorm's absorption detection capability and includes it as a headline result in the abstract. This is inaccurate. The only valid absorption detection claim is AUROC=0.757 at n_pos=18 gold IG labels.

**Required fix**: Reframe the TopK-32k AUROC as "EncNorm predicts decoder-letter-probe alignment" explicitly, or remove from main results. Do not present proxy-label results as cross-architecture replication of absorption detection.

### 3. EncNorm Mechanistic Narrative Contradicted by Own Layer Data

A2 layer-monotonicity results:
- L2: ratio (absorbed/non-absorbed EncNorm) = 0.877 (absorbed features have *lower* norm)
- L4: ratio = 1.055
- L6: ratio = 1.267 (claimed mechanism)
- L8: ratio = 0.891 (absorbed features have *lower* norm again)
- L10: ratio = 0.933

The paper claims: "when child c fires on token t in place of parent j, the encoder gradient pushes ||w_enc,j||_2 upward." But at L2, L8, and L10 — 3 of 5 measured layers — absorbed features have *lower* encoder norms, which is the opposite of the stated mechanism. The paper reports "EncNorm peaks at L6" as a positive finding but does not address the signal inversion at other layers.

The AUROC=0.757 at L6 is a real empirical correlation that should be reported. The post-hoc mechanistic narrative is not supported by the full layer data and should be removed or substantially qualified.

---

## Major Issues (significantly weaken the paper)

### 4. All Experiments in PILOT Mode

Every core result JSON carries mode='PILOT'. The experimental plan specified 10,000 tokens for C2, 200,000 tokens for B1, 200 tokens for D2 — the actual runs used 90, 10,000, and 100 respectively. The paper presents these pilot-scale results as final published findings without acknowledgment. This is systematic misrepresentation of experimental completeness and is a submission blocker.

### 5. Cross-Hierarchy Experiment (H3) is Invalid and Absent

D2 entity-type absorption = 0.0% and the experiment notes explicitly acknowledge: "Used entity-type probe on control sentences for comparison." The negative control absorption rate is an artifact of applying the wrong probe. H3 was not validly tested. The paper correctly notes probe quality issues, but then substitutes the width recovery analysis (F1) as the third contribution without updating the stated contribution list. This substitution is fine, but the paper should not mention H3 generalization in future work framing without acknowledging the probe failure.

### 6. Width Recovery (F1) Hook Confound is Unquantified

Standard-24k SAE uses resid_pre; TopK-32k uses resid_post. The 67% geometric recovery figure is presented with a caveat but without quantification. Two absorbed features (2406, 11270) share the same best_matching_32k_idx (16435), and two others (24154, 7371) share index 29309 — these duplicate matches suggest possible false positives in the recovery claim. The hook-confound effect must be quantified to interpret whether the 67% figure is meaningful.

### 7. EncNorm vs. EDA Marginal Contribution Not Established

Spearman r(EncNorm, EDA) = 0.712 across all 24,576 latents. With ~51% shared variance, the AUROC improvement (0.757 vs. 0.650) may be explained by EncNorm's distributional shape being more favorable for 18-vs-24558 class imbalance rather than capturing independent absorption signal. A partial logistic regression analysis is needed to establish marginal contribution.

### 8. No Cross-Model Validation

All results are from GPT-2 Small (117M, 2019). Gemma Scope returned 401 Unauthorized. NeurIPS main track reviewers will ask why no results on models released in the past 6 years. This is not a fatal flaw for a workshop paper, but it is for a main-track submission.

### 9. MP-SAE Not Discussed

Costa et al. (arXiv:2506.03093) uses iterative matching pursuit during joint encoder+decoder training. The paper's Section 5.1 practical guidance ("Do not expect OMP or iterative encoders to help") conflates post-hoc OMP with fixed decoder (what was tested) with jointly trained iterative encoders (not tested). This distinction is essential for accurate practical guidance.

---

## What Works Well

- **OMP oracle concept**: Testing whether fixing the decoder and varying only the encoder can reduce absorption is a well-designed mechanistic experiment. The pre-committed falsification criterion is the right approach. If the experiment is run at full scale and the null holds with tight CIs, this is a publishable finding.

- **Exemplary negative result reporting**: Pre-registered falsification criteria for H2, honest reporting of D2 failure, acknowledgment of hook confound in F1, and explicit limitation section on n_pos=18. This is scientific writing at its best.

- **Statistical toolkit**: Bootstrap CIs, DeLong test for paired AUROC comparison, Cohen's d, layer-monotonicity analysis — all correctly applied.

- **O_jaccard as independent complementary signal**: AUROC=0.721, Spearman r=0.044 with EncNorm, AUPRC 75x random baseline — the structural independence from EncNorm is a genuine finding. The coverage limitation (9 of 18 absorbed features have O_jaccard=0 by construction) must be documented.

- **GPT-2 Small anchor**: The gold IG label set (n_pos=18) from Chanin et al. provides the cleanest, most reproducible evidence in the paper. The AUROC=0.757 result at this anchor is the strongest claim the paper can make.

---

## Path to Acceptance

**To reach score 7.0 (weak accept threshold):**
1. Run OMP experiment at full scale (10,000 tokens, 26 letters) — this is the single most important fix
2. Report bootstrap CI on (AR_OMP - AR_FF) for each letter; include power analysis
3. Reframe TopK-32k AUROC=0.837 accurately (decoder alignment, not absorption detection)
4. Scope EncNorm to "L6-specific heuristic on GPT-2 Small Standard/L1 SAEs"

**To reach score 7.5 (borderline accept):**
5. Quantify hook-confound baseline for F1 width recovery
6. Run partial logistic regression to establish EncNorm marginal contribution over EDA
7. Report O_jaccard AUROC decomposed by letter-set membership

**To reach score 8.0 (accept):**
8. Results on at least one additional model/SAE beyond GPT-2 Small
9. Accept Gemma Scope license and run primary experiments on Gemma-2-2B, or use Pythia SAEs

---

## Summary Verdict

The research direction is sound and the experimental concept is novel. The paper has the ingredients for a 7.5-8.0 paper — a clean mechanistic oracle experiment, an empirically discovered weight-only detection heuristic, and exemplary scientific rigor in negative result reporting. The current gap is not in ideas but in execution: pilot-scale experiments presented as final results, a critical null result at statistical power near zero, and a cross-architecture replication that measures decoder geometry rather than absorption. These are fixable in 1-2 iterations.

**Score: 5.5/10 — CONTINUE**

The quality gate threshold of 8.0 requires demonstrating that the OMP null result holds at full scale and that EncNorm's absorption detection claim is properly scoped and validated. Proceed with full-scale experiment runs.
