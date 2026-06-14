# Skeptic Analysis — Feature Absorption / EDA Detector
**Date:** 2026-04-13  
**Evidence base:** iter_002 pilot results in `exp/results/full/`  
**Status:** All four primary hypotheses falsified or unsupported by audit report.

---

## 1. Statistical Risk Inventory (Top 3 Risks)

### Risk 1 — Critical: Proxy Label Contamination in All Absorption Metrics

**Specific entry of concern:** `D2_ASI_validation.json`, labels section: `n_pos=71`, `label_method: "probe_decoder_alignment"`, `label_threshold: 0.32`.

The absorption labels used for ALL of B1, D1, D2 are **proxy labels** derived from decoder cosine similarity with letter probes, not from the `FeatureAbsorptionCalculator` run on actual model outputs. When D1 uses the *exact* Chanin et al. labels (18 absorbed features from iter_001 R4), it gets EDA AUROC = 0.650. When D2 uses the proxy labels (71 "letter features"), it gets EDA AUROC = 0.681.

These are measuring different things: EDA predicts "features that decode toward a letter direction" (proxy) versus "features that fail to fire when they should" (ground truth). The AUROC=0.681 result may be detecting decoder-direction alignment, not actual absorption behavior. The two label sets are n=18 vs n=71 for the same SAE — a 4x disagreement in how many features are "absorbed."

**Why it is unreliable:** Any model that predicts "this feature has a letter-like decoder" will trivially separate the proxy-labeled positives from negatives, because the proxy labels were constructed from decoder similarity in the first place. The apparent EDA signal may be a measurement-construction artifact.

---

### Risk 2 — Serious: n=11 Data Points for Phase Transition Analysis (H4a)

**Specific entry:** `B2_sparsity_analysis.json`: `n_points=5` for the primary sigmoid analysis (cross-layer JB suite); `n=3` for AJT within-layer suite. Bootstrap CI for inflection L0_c = [1.0, 790 million] — spanning nine orders of magnitude.

The LRT p=0.027 favoring sigmoid over linear is computed with 5 data points from different layers. This is not within-layer L0 variation — it confounds sparsity with "representation maturity at different depths." The spearman correlation of inv_L0 vs EDA_delta = rho=0.10, p=0.87 is plainly nonsignificant. The authors themselves flag: "Cross-layer comparison conflates absorption with representation maturity." The sigmoid BIC advantage (delta=2.42) is negligible with n=5 and uninterpretable when two models have 4 and 5 parameters respectively.

**Why it is unreliable:** With n=5 cross-layer points, the sigmoid fit has 4 parameters (L, k, x0, b) for 5 observations — 1 residual degree of freedom. This fit is nearly memorization. Any nonlinearity in the data (including layer effects unrelated to sparsity) will favor a flexible sigmoid. The LRT p-value is not reliable at n=5.

---

### Risk 3 — Serious: EDA Failure in AJT Suite Undermines Generalizability

**Specific entries:** `B2_sparsity_analysis.json` (AJT section) and `B3_cross_arch.json`: AJT variants at L6 show EDA_delta = negative (−0.204 to −0.217), meaning letter features have LOWER EDA than non-letter features. EDA AUROC for AJT variants = 0.154, 0.354, 0.158 — all below chance for the letter-feature signal.

EDA AUROC = 0.681 holds only for the JB (standard L1) SAE at L6. For L10 it is 0.337 (reversed). For all three AJT architectures (different activation functions, same layer) it inverts completely. There are 4 SAE configurations where EDA works (1 JB-L6) and at least 5 where it does not (L10, AJT×3, TopK shows much lower letter EDA). The claim that EDA is an absorption detector is supported by exactly one SAE release × layer combination out of the tested set.

**Why it is unreliable:** A detector with AUROCs ranging from 0.154 to 0.681 depending on which SAE you plug in is not a robust detector — it is a SAE-configuration-specific artifact. No theory predicts why EDA should work only for JB L1 at L6 and fail everywhere else tested. The EDA signal may be specific to a peculiarity of the JB SAE's training procedure, not a general feature of absorption.

---

## 2. Alternative Explanations for Claimed Improvements

### Claim: "EDA (AUROC=0.681) is a valid probe-free absorption detector."

**Alternative 1 — Decoder-normalized training artifact.** The JB SAE enforces unit-norm decoder columns (`dec_norms_are_unit: true`). Under this constraint, features with large encoder norms will automatically show high EDA (1 - cos(enc, dec)) if their encoder direction is forced by L1 pressure away from the decoder. The EDA-norm variant (AUROC=0.737) explicitly shows that `EDA_base × ||enc_j||` is a better predictor than EDA_base alone. This means EDA is partially capturing encoder magnitude, not specifically absorption. Letter features may simply have atypically large encoder norms in this SAE — a training artifact, not a structural absorption signal.

**Alternative 2 — Letter features are semantically unusual, not absorbed.** Letter features ("starts with E") encode a syntactic/token-level property that is orthogonal to most semantic content. They may have systematically different encoder-decoder alignment than semantic features for reasons unrelated to absorption: e.g., the SAE may represent them as broad detectors (large encoder coverage, narrow decoder) because the token-level trigger is low-dimensional but the reconstruction contribution is diffuse. This would produce high EDA without any absorption mechanism.

**Alternative 3 — Circular proxy labels.** As noted above, the D2 labels (n=71) are constructed from decoder cosine similarity. EDA includes a function of decoder direction. Any correlation between EDA and decoder-direction-based labels is potentially confounded.

---

### Claim: "H1 is falsified (absorbed features have LOWER decoder cosine, not higher), but this is explained by decoder drift post-training."

**Alternative explanation — H1 never applied to the tested features.** The rate-distortion threshold λ > sin²(θ_{p,c}) was derived for a parent-child pair at the *moment of absorption*. The decoder angle measured in B1 is between *letter features* (child) and *high-frequency non-letter features* (parent candidates). There is no verified mapping from any specific letter feature to a verified parent feature in this SAE. The comparison may be entirely unrelated to the theoretical setup. The "drift" story is post-hoc.

**Alternative explanation — Measurement design error.** B1 samples 500 random "parent candidates" from all non-letter features and measures max/mean cos²(θ_{letter, random_parent}). This procedure cannot identify actual hierarchically-related parent features. The lower cos² for letter features vs non-letter features simply says letter features are geometrically more isolated in decoder space — consistent with them encoding rare, specific syntactic properties. No causal absorption mechanism is implied.

---

### Claim: "Cross-domain absorption exists (H2 partially supported via C2 v9 first-letter rate=0.008)."

**Alternative — The C2 measurement primarily detects tokenization artifacts.** The `newline-safe filter` in C2 v9 (`len(tok.encode('\n'+word))==2`) drastically restricts the test vocabulary. With n_events=120 and absorption_rate=0.008 (1 out of 120 events), the claimed "absorption" in the first-letter hierarchy is a single observed failure from letter 'h'. The 95% CI is [0.0, 0.029]. The other hierarchies (animate_inanimate) show absorption_rate = 10^-8 (numerical zero). This is consistent with no absorption at all in the C2 measurement framework, with one false positive from the 'h' letter.

---

## 3. Proxy Metric Audit

### EDA vs. actual absorption

**What is measured:** 1 - cos(encoder direction of feature j, decoder direction of feature j). This captures intra-feature encoder-decoder dissociation.

**What is claimed:** EDA detects which SAE features are absorbed — i.e., which features fail to fire when they should because another feature has absorbed their representation.

**Gap:** EDA is a static weight-space property computed once per feature. Actual absorption is a behavioral property: the feature fails to activate on inputs that should trigger it. The connection is theoretical (Proposition 2 in F1_theory_revised.md, explicitly labeled "Mechanistic Conjecture"). No direct behavioral validation of EDA as an absorption predictor has been performed. The Chanin et al. exact-label validation (D1, n=18) gives AUROC=0.650, which is only moderately above chance and with AUPRC = 0.0015 (2x base rate), precision@500 = 0.0. A detector with precision 0 at the top 500 ranked features is not practically useful.

**Verdict:** Measured metric (EDA) is an unvalidated proxy for the claimed metric (behavioral absorption). The theoretical causal chain connecting them is a conjecture.

---

### freq_ratio as standalone detector (AUROC=0.612)

**What is measured:** freq_p / freq_c, computed as the ratio of activation frequencies between high-frequency and letter features.

**What is claimed:** A component of the proposed ASI metric.

**Gap:** freq_ratio alone achieving AUROC=0.612 indicates that letter features are simply rarer than the "parent" candidates used for comparison. This is definitionally true: letter features are specific (encode one of 26 letters) and should fire less often than general high-frequency features. This is not evidence of absorption — it is evidence that the label set (letter features) has lower activation frequency than the comparison set (random non-letter features), which is trivially true by construction.

---

### H4a phase transition claim

**What is measured:** BIC comparison between sigmoid and linear fits on 5 cross-layer data points for EDA_delta vs. inv_L0.

**What is claimed:** Absorption exhibits a sigmoid phase transition in sparsity.

**Gap:** The 5 data points span different layers, not different L0 values at the same layer. The sigmoid R² = 0.50 (linear R² = 0.039) with n=5 and 4 free parameters is not informative. The Spearman rho for EDA_delta vs inv_L0 = 0.10, p=0.87 is not significant. BIC delta of 2.42 is below the conventional threshold of 6 for "positive evidence" in favor of the more complex model.

---

## 4. Severity Classification

| Issue | Claim affected | Severity |
|---|---|---|
| Proxy labels (n=71 proxy vs n=18 exact) contaminate all AUROC metrics | All AUROC results for D1/D2/B1 | **FATAL** |
| EDA fails in 5/6 SAE configurations tested (JB L6 only works) | EDA as general detector | **FATAL** |
| C2 cross-domain absorption rate = ~0 across all non-spelling hierarchies | H2 (cross-domain absorption) | **FATAL** |
| ASI AUROC=0.421 < null mean=0.497 | H3 (ASI detects absorption) | **FATAL** (already confirmed) |
| H1 direction reversal (absorbed features have LOWER not higher decoder cos²) | H1 (RD threshold predicts absorption) | **FATAL** (already confirmed) |
| n=5 data points for phase transition claim; LRT barely significant | H4a (phase transition) | **SERIOUS** |
| Hysteresis untestable (all L0 values show ~96% absorption in saturation regime) | H4b (hysteresis) | **SERIOUS** |
| B3 cross-arch comparison uses mismatched hook points (resid_pre vs resid_post) and unmatched L0 | B3 architectural comparison conclusion | **SERIOUS** |
| EDA-norm AUROC=0.737 uses encoder magnitude — potentially conflates encoder scale with absorption | EDA-norm improvement claim | **MINOR** (but should not be presented as EDA improvement without controlling encoder norms) |
| First-letter probe passes F1 gate on only 17/24 letters (71%) | C1 probe quality | **MINOR** (7 letters excluded) |

---

## 5. Concrete Remediation

### Fatal Flaw 1: Proxy Label Contamination

**Proposed experiment:** Run `FeatureAbsorptionCalculator` (the exact Chanin et al. 2024 behavioral measurement) on all 26 letters for GPT-2 Small L6, not just the 8 letters available in iter_001 R4. This gives exact behavioral labels (feature fires or doesn't on relevant inputs) rather than proxy decoder-alignment labels.

- Dataset: same OWT token cache already in `exp/owt_tokens_cache.pt`
- Expected outcome: n_pos grows from 18 to ~50-80 verified absorbed features
- Then re-run EDA AUROC with exact labels across all letters
- Success criterion: EDA AUROC ≥ 0.65 with exact labels on ≥ 15 letters

**Expected duration:** 30-45 minutes. This is the single most important validation to run before any writing.

---

### Fatal Flaw 2: EDA Generalizability Failure (AJT + L10 + TopK)

**Proposed experiment:** Characterize the conditions under which EDA works. For each SAE configuration, measure:
1. Whether letter features have higher or lower EDA than non-letter features (direction test)
2. EDA AUROC against proxy labels (rank test)
3. SAE training details: was decoder normalization enforced? What is the typical encoder norm ratio?

**Expected outcome:** If EDA works only for L1-penalty SAEs with unit-norm decoders but not for TopK or activation-function variants, this must be reported as a serious scope limitation, not a general finding. The paper claim should be "EDA predicts absorption in L1-trained SAEs" not "EDA is a probe-free absorption detector."

**Success criterion for scope claim:** EDA AUROC ≥ 0.60 in at least 3 of 5 tested configurations, or a principled explanation (with predictive power) for why it fails in others.

---

### Fatal Flaw 3: C2 Cross-Domain Absorption Rate = 0

**Proposed experiment:** The C2 measurement method (FeatureAbsorptionCalculator with IG-based attribution) may not work for non-spelling hierarchies because the parent-child feature pairs for animate/inanimate or city/country are not aligned with how sae_spelling's measurement approach operates. The appropriate remedy is NOT to run more hierarchies with the same tool — it is to validate whether the measurement tool detects absorption in non-spelling hierarchies at all.

Proposed validation:
- For the `noun_proper` hierarchy (F1=0.987, clearly reliable probe), manually identify the top 5 SAE features that encode "proper noun" and "common noun" by probing decoder directions
- For 50 test sentences, check whether the proper-noun feature fires when it should (behavioral check)
- Compare to the EDA distribution of these 5 features

If the noun_proper hierarchy shows absorption_rate = 0 by both behavioral check AND C2 IG-based measure, then either: (a) absorption genuinely does not occur for noun hierarchies in GPT-2 Small, or (b) the IG-based absorption calculator is not detecting it. Only (a) is a publishable claim; (b) requires methodology fix.

---

## 6. Unresolved Empirical-Theory Tension (Not in Original Proposal)

The F1_theory_revised.md explicitly acknowledges in Part III: the formal EDA magnitude prediction from Proposition 2 says EDA(c) ≈ 1 - cos(θ_{p,c}), which should be SMALL if absorbed pairs have small θ_{p,c}. But the pilot finds letter features have large EDA (~0.45). F1 proposes that "the encoder overshoots" but labels this interpretation as "(a) plausible but unverified" vs. "(b) label set issues." 

**This tension is not a minor caveat.** If the theory predicts small EDA for absorbed features and the data shows large EDA, then either the theory is wrong about what EDA measures, or the labels are wrong. A paper cannot simultaneously:
- Claim Proposition 2 as theoretical support for EDA
- Acknowledge that Proposition 2 predicts the wrong direction for EDA magnitude

This must be resolved before writing. Either: (1) derive a corrected theory that predicts large EDA from absorption, with algebraic support; or (2) abandon Proposition 2 and present EDA as an empirical finding without theoretical justification.

---

## Summary Verdict

The strongest result in this dataset is EDA AUROC=0.681 for letter features in GPT-2 Small L6 with the JB SAE. This is a real finding (Cohen's d=0.70, Wilcoxon p=1.3×10^-7). However:

1. It holds in exactly 1 of ≥5 SAE configurations tested.
2. It uses proxy labels, not behavioral absorption labels.
3. The theoretical justification for it (Proposition 2) predicts the wrong direction for EDA magnitudes.
4. Practical utility is low: EDA precision@500 = 0.008 (8 positives per 500 top-ranked features, base rate 0.0007).

The path to a publishable paper exists, but requires: (a) exact behavioral labels, (b) honest scope limitation to specific SAE configurations, and (c) resolution of the Proposition 2 magnitude tension. The current result set cannot support the broad claims in the original proposal.
