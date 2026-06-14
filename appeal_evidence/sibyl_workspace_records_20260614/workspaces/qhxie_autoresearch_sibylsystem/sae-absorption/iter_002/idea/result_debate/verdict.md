# Executive Verdict — SAE Feature Absorption iter_002

**Date:** 2026-04-13  
**Decision:** PROCEED (mandatory scope reduction)  
**Result Quality Score: 5 / 10**

---

## Overall Score: 5 / 10

Four of five primary hypotheses are falsified or unsupported. Three surviving signals are real but narrow in scope. The work is publishable at a workshop level now and at a full venue level contingent on Gemma Scope replication.

---

## Key Conclusions (What We Actually Learned)

1. **H1, H3 are cleanly falsified.** The rate-distortion threshold predicts the *wrong geometric direction* (absorbed features have LOWER decoder cosine to parents, not higher; AUROC=0.318). ASI is anti-correlated with absorption (AUROC=0.422 < null=0.497). Neither result is borderline — both are robust statistical falsifications.

2. **Encoder norm is the strongest absorption detector.** enc_norm AUROC=0.757 against exact Chanin labels (n_pos=18, GPT-2 Small L6 JB SAE) — the first probe-free detector to exceed chance on exact ground-truth labels in the literature. It was discovered post-hoc and must be validated cross-model.

3. **EDA is real but narrow.** EDA AUROC=0.650 (exact labels), z=+2.49 above null permutation. Works only for the JB Standard/ReLU SAE at L6; fails at L10 (AUROC=0.337 reversed) and all AJT variants. Not a general detector in current form.

4. **L1 SAEs show dramatically more absorption signal than TopK.** EDA gap=0.200 (z=−10.4; p=7.3e-23). 100% of L1 letter features exceed EDA>0.50 vs. 25.4% for TopK. This architectural comparison is the cleanest, most statistically robust result in the study.

5. **Absorption is saturated at ~96% across all tested configurations.** Eleven SAE configurations spanning 3.5x L0 variation all show 85–99% absorption. Phase transition and hysteresis hypotheses are untestable in this regime, not falsified.

---

## Action Plan (Priority Order)

| Priority | Experiment | Time | Decision Gate |
|----------|-----------|------|---------------|
| **P1** | Validate cos(enc_p, dec_c) on exact Chanin labels (GPT-2 L6) | ~30 min | AUROC ≥ 0.70 → primary contribution; < 0.60 → drop |
| **P2** | EDA-norm + enc_norm layer profile across GPT-2 Small L2–L10 | ~1h | Produces honest layer-specificity figure for paper |
| **P3 (GO/NO-GO)** | Gemma Scope replication: EDA + enc_norm vs. ground-truth absorption labels | ~2–4h | AUROC ≥ 0.65 → EMNLP/ICLR target; < 0.55 → workshop only |

**What to drop:** Phase C2 full cross-domain expansion, Phase E (phase transition training), ASI variants, H1 quantitative validation. None of these have tractable paths to positive results.

---

## Paper Reframing

**Old (unsupported):** "Feature absorption is rate-distortion optimal behavior; ASI predicts risk; absorption generalizes across semantic hierarchies."

**New (evidence-based):** "Absorbed features in trained SAEs exhibit distinctive encoder-decoder geometry — specifically elevated encoder norms and encoder-decoder dissociation — that enables probe-free detection (AUROC 0.65–0.76 against exact ground-truth labels). This is the first such validated post-hoc detection signal in the literature. L1-trained SAEs exhibit this signature far more strongly than TopK SAEs, suggesting absorption is architecturally mediated."

**Target venue:**
- **Now (current data):** Workshop paper — BlackboxNLP, MI@ICML, or Alignment Forum extended abstract. 6–8 pages.
- **After P3 (Gemma replication):** Full paper — EMNLP 2026 or ICLR 2027, if AUROC ≥ 0.65 cross-model.

---

## What Must NOT Be Claimed

- That EDA is a general absorption detector (it fails on 5+ of 6 configurations tested).
- That Proposition 2 provides theoretical support for large EDA in absorbed features (it predicts the wrong direction).
- That ASI has any merit as a detector (it is anti-predictive).
- That cross-domain absorption was observed (all semantic hierarchies returned null signal; C2 protocol calibration is also in question).
- That a phase transition in absorption was found (saturation regime; no variation to fit).

---

## Bottom Line

The experiments did not confirm the original theory. They found something different and arguably more interesting: absorbed features have a distinctive encoder-based geometric fingerprint — not the decoder-proximity signature the theory predicted, but an encoder isolation signature (high enc_norm, encoder-decoder dissociation). This is a real finding. It is narrow, post-hoc, and limited to one model-layer-architecture combination. Three experiments (P1, P2, P3) costing 4–6 GPU hours can establish whether this finding generalizes. Run them before writing.
