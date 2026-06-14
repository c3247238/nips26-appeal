# Statistical Audit Summary — task_STAT_audit

**Generated:** 2026-04-13T22:52  
**Mode:** PILOT  
**Output:** `exp/results/full/audit_report.json`

---

## Audit Outcome: WRITING BLOCKED

**79 claims audited. 0 discrepancies. 3 CRITICAL flags block writing.**

---

## Critical Flags (Block Writing)

### 1. B1_7_direction_check — H1 FALSIFIED (direction reversed)
- **Source:** `full/B1_decoder_geometry.json`
- **Finding:** Absorbed (letter) features have LOWER cos^2(theta) than non-absorbed features (pos_mean=0.052 < neg_mean=0.127)
- **Impact:** Rate-distortion threshold (lambda > sin^2(theta)) predicts AUROC < 0.5. H1 is falsified at the geometric level.
- **Action required:** Paper must not claim RD threshold predicts absorption in the correct direction. Either reframe H1 or remove it.

### 2. C1_4_city_country_shuffle — Probe quality gate failure
- **Source:** `full/C1_probe_training.json`
- **Finding:** city_country_binary probe shuffled F1=0.621 > 0.60 threshold. Dataset artifact likely (US vs. non-US binary with imbalanced geography in RAVEL).
- **Impact:** city_country_binary must NOT be included in C2 analysis.
- **Action required:** Remove city_country_binary from any cross-domain claims.

### 3. D2_10_asi_fails — H3 FALSIFIED (ASI below random)
- **Source:** `full/D2_ASI_validation.json`
- **Finding:** ASI AUROC=0.4215 (null mean=0.4973). ASI is anti-correlated with absorption labels.
- **Impact:** H3 (ASI predicts absorption) is falsified. Best detector is EDA (AUROC=0.6810).
- **Action required:** Paper must not claim ASI as primary contribution. EDA is the empirically supported detector.

---

## Missing Results (Not Blocking, but Incomplete)

| ID | Description |
|---|---|
| C2_MISSING_FULL | Full C2 cross-domain absorption not computed. Pilot shows absorption_rate=0.0 (H2 falsified). |
| C3 | Hierarchy property correlation — depends on C2 |
| D3 | ASI cross-domain validity — depends on C2 |
| TABLE1_GPT2_L10_missing | No GPT-2 L10 AUROC/AUPRC values in D2 |
| TABLE1_Gemma_missing | No Gemma Scope results (not tested) |

---

## What IS Available (Table 1 Partial)

| Detector | GPT-2 L6 AUROC | GPT-2 L6 AUPRC | L10 | Gemma |
|---|---|---|---|---|
| EDA | 0.6810 | 0.00590 | N/A | N/A |
| freq_ratio | 0.6125 | 0.00546 | N/A | N/A |
| ASI_combined | 0.4215 | 0.00284 | N/A | N/A |
| cos2_alone | 0.2059 | 0.00170 | N/A | N/A |
| Random | 0.5000 | 0.00289 | N/A | N/A |
| Shuffled null (ASI) | 0.4973±0.036 | — | N/A | N/A |

---

## Hypotheses Status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: RD threshold predicts absorption | **FALSIFIED** | Absorbed features have LOWER cos^2, not higher. AUROC < 0.5. |
| H2: Cross-domain absorption exists | **NOT SUPPORTED** | C2 pilot all NO_GO; absorption_rate=0.0 for all hierarchies. |
| H3: ASI detects absorption | **FALSIFIED** | ASI AUROC=0.4215 < null (0.497). |
| H4a: Phase transition in sparsity | **NOT SUPPORTED** | LRT p=0.456; Spearman rho=0.191 (p=0.574). |
| H4b: Hysteresis | **NOT CONFIRMED** | Saturation regime; all L0 values show ~96% absorption. |

---

## What Works

- **EDA (1 - cos(enc_j, dec_j)) AUROC=0.681** on GPT-2 L6 — best available detector, consistent with iter_001 result (0.629 reference)
- **F1_theory_analysis.md** contains complete non-circular algebraic derivation of Theorem 1
- **Figure 1 PDF exists** at `full/fig1_method.pdf`
- **Probe training (C1)** successful: first_letter F1=0.820, noun_proper F1=0.987, animate_inanimate F1=1.0
- **All pipeline scripts ran without errors**, 11 SAE configurations tested

---

## Recommended Path Forward

Since all 4 primary hypotheses are either falsified or unsupported, the paper framing needs major revision:

1. **Reframe around EDA as empirical detector**: EDA is the valid finding. Characterize when it works (L6 > L10).
2. **Report falsification results honestly**: H1, H3 direction reversal is a meaningful empirical finding — absorbed features have *lower* decoder cosine similarity to non-absorbed features.
3. **Narrow scope**: Drop ASI, drop RD threshold prediction, drop cross-domain claim. Focus on EDA characterization + absorption scaling observation.
4. **Or pivot**: Begin new iteration with different hypothesis about what predicts absorption.
