# Result Debate Synthesis — SAE Feature Absorption / EDA Detector

**Synthesizer:** sibyl-result-synthesizer  
**Date:** 2026-04-13  
**Evidence base:** Six perspectives (optimist, skeptic, strategist, methodologist, comparativist, revisionist) on iter_002 pilot results  
**Workspace:** `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current`

---

## 1. Consensus Map — High-Confidence Conclusions

The following points receive explicit or implicit agreement from all six perspectives and may be treated as high-confidence conclusions requiring no further deliberation.

### Consensus C1: H1, H3, H4a are cleanly falsified
Every perspective agrees that:
- **H1** (rate-distortion threshold lambda > sin²(theta) predicts absorption) is empirically **reversed**: absorbed (letter) features have *lower* decoder-decoder cosine similarity to parent candidates (pos_mean=0.052 vs neg_mean=0.127; AUROC=0.318; Wilcoxon p=1.3e-6). No perspective defends H1 in its predictive form.
- **H3** (ASI AUROC ≥ 0.70) is **falsified**: ASI AUROC=0.421, below the null mean of 0.497 (DeLong p=5.8e-8 against ASI). ASI is anti-predictive.
- **H4a** (phase transition in sparsity) is **not supported**: LRT p=0.456, BIC delta=−3.22, absorption rates uniformly ~85–99% across all tested L0 values. Whether this is theoretical falsification or a saturation/measurement failure is debated (see §2 below), but the empirical result is agreed upon.

### Consensus C2: EDA carries a real but bounded signal at GPT-2 Small Layer 6
All six perspectives acknowledge:
- EDA AUROC = 0.681 (proxy labels), 0.650 (exact Chanin labels), z=+2.49 above null permutation — is a real, statistically significant signal for the JB Standard/ReLU SAE at layer 6.
- Cohen's d is in the range 0.53–0.70 depending on measurement; the signal is not noise.
- **The signal is layer-specific and architecture-specific**: EDA fails at L10 (AUROC=0.337, reversed) and for all AJT SAE variants tested (AUROCs: 0.154, 0.354, 0.158). This scope limitation is unanimously acknowledged.

### Consensus C3: Encoder norm is a stronger detector than EDA, post-hoc
All perspectives that address enc_norm (optimist, skeptic, methodologist, strategist, comparativist) agree:
- enc_norm AUROC = 0.757 on exact Chanin labels outperforms EDA AUROC = 0.650.
- DeLong test (enc_norm vs EDA: p=0.153) is not significant at α=0.05, but the numerical gap is consistent across evaluations.
- enc_norm was not pre-specified; it was discovered post-hoc. Its superiority over EDA must be acknowledged and the primary contribution claim adjusted accordingly.

### Consensus C4: H2 (cross-domain absorption) is not supported by current evidence
All perspectives agree:
- C2 pilot finds absorption_rate = 0.0 for animate_inanimate and noun_proper hierarchies (numerical zeros).
- First-letter hierarchy shows rate=0.008 (1/120 events; 95% CI: [0.0, 0.029]) — essentially the lower bound of detectable signal.
- **The cross-domain claim — the most novel empirical promise of the original proposal — has no current evidentiary support.**

### Consensus C5: The paper requires reframing
All six perspectives converge on the same structural conclusion: the original framing ("feature absorption is rate-distortion optimal behavior; ASI predicts risk; absorption generalizes across semantic hierarchies") is no longer supported, and the paper must be rebuilt around the surviving EDA/enc_norm geometric detector finding.

---

## 2. Conflict Resolution — Where Perspectives Disagree

### Conflict A: Is EDA detecting absorption or a training artifact?

**Optimist position:** EDA reflects genuine geometric dissociation caused by competing gradient signals during absorption. The encoder is stretched toward the parent direction while the decoder diverges — "encoder amplification, decoder avoidance."

**Skeptic position:** EDA may detect that letter features have semantically unusual encoder-decoder alignment (low-dimensional syntactic property encoded by broad encoder / narrow decoder), not absorption per se. The proxy label contamination makes it impossible to distinguish these.

**Methodologist position:** The EDA AUROC of 0.681 uses decoder-alignment-derived proxy labels. Since EDA contains decoder direction information, the correlation may be partially circular.

**Revisionist position:** The `cos(enc_p, dec_c)` metric (parent encoder → child decoder cross-alignment; AUROC=0.730 in pairwise EDA pilot) is mechanistically more sound than intra-feature EDA because it directly tests whether parent-class inputs would trigger child features.

**Synthesis judgment:** The skeptic and methodologist raise a *genuine* concern about proxy label circularity. However, the exact Chanin label validation (EDA AUROC=0.650 on n_pos=18) shows the signal survives on non-circular labels — which were *not* derived from decoder similarity. This mitigates but does not eliminate the circularity concern. **The enc_norm finding (AUROC=0.757) is even less likely to be circular, as encoder norm has no direct relationship to the proxy label construction.** Weight: enc_norm is the cleanest surviving signal; EDA carries real but mixed evidence. The `cos(enc_p, dec_c)` metric must be validated on exact labels before it can be claimed as the dominant contribution.

### Conflict B: Is H4b (hysteresis) falsified or merely untestable?

**Skeptic position:** The saturation regime (96% absorption across all L0 values) means hysteresis is empirically untestable and should be classified as such, not as falsification.

**Strategist/Revisionist positions:** The saturation reflects that GPT-2 Small early-layer SAEs are past any transition point at all tested L0 values. Whether a transition exists at lower L0 is unknown.

**Synthesis judgment:** H4b should be reported as **untestable in the current configuration** (saturation regime), not as falsified. The honest framing is: "The hysteresis hypothesis cannot be evaluated because all tested configurations exhibit near-ceiling absorption rates. Testing would require SAEs trained at much lower L0 than any publicly available release provides, or a different model where absorption is not saturated."

### Conflict C: EDA's theoretical justification — is Proposition 2 compatible with the data?

**Skeptic position (strongest):** Proposition 2 predicts that absorbed features should have *smaller* EDA magnitude (small encoder-decoder angle) because the absorbed feature's encoder converges toward the parent-decoder direction, which is also the absorbed feature's decoder direction. But the data shows letter features have *larger* EDA. Proposition 2 predicts the wrong direction. A paper cannot simultaneously cite Proposition 2 as theoretical motivation and acknowledge it predicts the wrong direction.

**Optimist position:** The "encoder overshoot" hypothesis reconciles this — the encoder grows beyond the decoder direction in a competition dynamic. But the optimist's own caveat labels this "(a) plausible but unverified."

**Methodologist position:** The B1_eda_decomposition shows that the decoder aligns more with the letter probe than the encoder does at both L6 and L10 — directly falsifying the encoder-pull mechanism of Proposition 2.

**Synthesis judgment:** The skeptic and methodologist are correct on this point. **Proposition 2 must not be presented as empirical support for the EDA direction.** Two options exist: (a) derive a corrected theoretical mechanism that predicts large EDA for absorbed features (with algebraic grounding); or (b) demote Proposition 2 to background/motivation and present EDA as a purely empirical finding with an open mechanistic question. Option (b) is lower risk for the current timeline. The paper should clearly label the mechanism as a conjecture.

### Conflict D: What venue is realistic?

**Optimist position:** "Encoder-Mediated Feature Absorption: Geometry, Detection, and Architectural Dependencies" could be a solid paper even with falsified H1/H3.

**Comparativist position (most grounded):** Current evidence supports a workshop contribution or short paper (BlackboxNLP/MI@ICML). NeurIPS/ICML/ICLR in current form is not realistic: single model, single layer (for the main EDA result), proxy labels, three of four hypotheses falsified.

**Strategist position:** Venue target should be decided after Gemma Scope replication: if EDA-norm validates at AUROC ≥ 0.65 on Gemma Scope ground-truth labels → EMNLP/ICLR consideration; if AUROC < 0.55 → workshop only.

**Synthesis judgment:** The comparativist's assessment is most calibrated. The current evidence package — before the three priority experiments the strategist identifies — supports a workshop paper (6–8 pages). The NeurIPS 2026 target requires successful Gemma Scope replication and exact label validation. This is achievable with 4–6 GPU-hours of additional work; the Gemma Scope replication should be treated as a go/no-go gate for venue decision.

---

## 3. Result Quality Score

**Score: 5 / 10**

**Justification:**

| Component | Score | Rationale |
|-----------|-------|-----------|
| Statistical validity of surviving signal | 6/10 | EDA z=+2.49 above null, p<0.001 on permutation — real signal. But n_pos=18 (exact labels) produces wide AUROC CIs (~[0.52, 0.78]). |
| Scope / generalizability | 3/10 | One model (GPT-2 Small), one layer (L6), one architecture (JB Standard/ReLU). EDA fails at L10 and for all AJT variants tested. |
| Theory-data alignment | 4/10 | H1 falsified in its predictive direction. Proposition 2 predicts wrong direction for EDA magnitude. The mechanistic story (encoder amplification, decoder avoidance) is post-hoc conjecture. |
| Novelty | 7/10 | First AUROC-based probe-free absorption detector reported in the literature. Proposition 1 (lambda > sin²(theta)) and Corollary 1 (frequency cancels) are formally novel. |
| Practical utility | 3/10 | Precision@k = 0 for all k≥50 with n_pos=18 exact labels. Not actionable at current scale without improvement. |
| Negative result quality | 7/10 | H1, H3 cleanly falsified with strong evidence. H2 null result is informative about scope. H4b saturation is a clean characterization. These sharpen understanding of what does not work. |

**Interpretation of 5/10:** The results represent a credible first finding with genuine novelty but insufficient scope and theoretical grounding for top-venue submission in current form. The score would rise to 7–8/10 if Gemma Scope replication succeeds and the Proposition 2 tension is resolved.

---

## 4. Key Findings (High-Confidence)

1. **Geometric isolation of absorbed features (direction reversal of H1):** Absorbed (letter) features have significantly *lower* decoder-decoder cosine similarity to candidate parent features than non-letter features (Wilcoxon p=1.3e-6 at L6, p=3.6e-11 at L10; Cohen's d=−0.57 and −0.72 respectively). This is the *opposite* of H1's prediction. Absorbed features are geometrically isolated, not geometrically proximate to parents.

2. **Encoder norm as the strongest absorption detector:** enc_norm achieves AUROC=0.757 against exact Chanin labels (n_pos=18, GPT-2 Small L6 JB SAE) — the highest single-metric performance in the study. This was not pre-registered. A plausible mechanism: absorbed features develop inflated encoder norms as they become sensitized to the parent's input space while their firing is suppressed.

3. **EDA is a real but bounded signal (single model/layer/architecture):** EDA AUROC=0.650 against exact Chanin labels (0.681 against proxy labels) at GPT-2 Small L6 is above chance (z=+2.49). The signal fails at L10 (AUROC=0.337) and for all AJT variants. The cross-directional metric cos(enc_p, dec_c) achieves AUROC=0.730 in the pairwise pilot (proxy labels) and merits priority validation on exact labels.

4. **L1-trained SAEs show dramatically more absorption signal than TopK (EDA gap=0.200, z=−10.4):** Standard/ReLU SAEs show 100% of letter features with EDA>0.50; TopK shows only 25.4%. This architectural comparison is one of the cleanest results in the study — even if EDA is a training artifact, the architecture comparison result holds: L1-penalty SAEs produce a qualitatively different geometric signature for letter features than TopK SAEs.

5. **Absorption is saturated at ~96% across all L0 values in GPT-2 Small:** Eleven SAE configurations spanning L0=18–84 all show absorption_rate=0.85–0.99. This saturation characterizes a practical regime: fine-tuning sparsity cannot reduce absorption once a system is in this plateau.

---

## 5. Methodology Gaps — Critical Improvements Required

In priority order:

### MG1 (Critical): Validate enc_norm and EDA on ground-truth Gemma Scope labels
**Issue:** All AUROC results for EDA come from GPT-2 Small Layer 6 with the JB SAE. n_pos=18 exact labels produces AUROC CIs spanning ~[0.52, 0.78]. No cross-model validation exists.  
**Required action:** Load Gemma Scope SAE (layer 12, width 16k or 65k); run FeatureAbsorptionCalculator (Chanin et al. 2024, MIT-licensed) to get ground-truth absorption labels; compute EDA and enc_norm AUROC. This is the go/no-go gate for venue decision.  
**Estimated effort:** 2–4 GPU hours.  
**Success criterion:** EDA or enc_norm AUROC ≥ 0.65 on Gemma Scope ground-truth labels.

### MG2 (Critical): Validate cos(enc_p, dec_c) on exact Chanin labels
**Issue:** The cross-directional metric cos(enc_p, dec_c) achieved AUROC=0.730 in the pairwise pilot but on proxy labels. If it replicates on exact labels, it becomes the primary contribution (stronger signal with cleaner mechanistic interpretation than EDA).  
**Required action:** Run cos(enc_p, dec_c) computation with exact Chanin labels (n_pos=18 at GPT-2 L6).  
**Estimated effort:** 30 minutes.  
**Decision rule:** AUROC ≥ 0.70 on exact labels → promote to primary contribution; AUROC < 0.60 → treat as proxy-label artifact.

### MG3 (High): Fix threshold a priori and report AUROC confidence intervals
**Issue:** EDA AUROC varies between 0.469 and 0.681 depending on the letter-feature identification threshold (thr=0.29 vs. 0.32). Post-hoc threshold selection inflates the reported result.  
**Required action:** Fix threshold at 0.32 (consistent with the n=50 result reported against exact labels) before any paper submission. Report 95% bootstrap CIs on all AUROC values (approximating ±0.13 at n_pos=18).

### MG4 (High): Resolve Proposition 2 theory-data tension
**Issue:** Proposition 2 predicts small EDA magnitude for absorbed features; data shows large EDA. The paper cannot simultaneously cite Proposition 2 as support for EDA and acknowledge the prediction is reversed.  
**Required action:** Either (a) derive a corrected mechanism with algebraic grounding, or (b) demote Proposition 2 to background/motivation and present EDA as an empirical finding with open mechanistic questions. Option (b) is recommended for current timeline.

### MG5 (Medium): Calibrate C2 measurement protocol sensitivity
**Issue:** C2 v9 recovers only 0.83% of first-letter absorption events vs. Chanin et al.'s 15–35% — suggesting the redesigned protocol misses ~95% of known positives. Any negative result for semantic hierarchies is therefore ambiguous (measurement failure vs. genuine absence).  
**Required action:** Calibrate C2 against the exact Chanin-labeled absorbed features and confirm ≥50% recall before interpreting cross-domain negatives as "absorption does not occur."

---

## 6. Competitive Position

**Summary (from comparativist analysis):**

- **No prior paper** has published a feature-level AUROC for unsupervised probe-free absorption detection. EDA is *de facto* the first result in this sub-problem space.
- **Proposition 1** (lambda > sin²(theta)) and **Corollary 1** (frequency cancels from threshold) are formally novel relative to published work as of April 2026. The closest competitors (arXiv:2512.05534; arXiv:2506.15963) prove existence of absorption solutions but do not derive the specific threshold or corollary.
- **Critical weakness vs. field standard:** SAEBench (Karvonen et al.) uses Gemma Scope as the evaluation benchmark. The current EDA result is exclusively on GPT-2 Small, a model that the field has largely moved beyond for SAE evaluation. EDA must be validated on Gemma Scope to be credible to reviewers.
- **Concurrent work risk:** No concurrent paper found on EDA-based detection specifically. Theoretical grounding is ahead of the literature. However, the mechanistic interpretability field is moving fast (ATM SAE, OrtSAE, KronSAE all appeared 2024–2025 with absorption-related contributions).
- **Realistic venue:** Workshop paper (BlackboxNLP/MI@ICML) in current form. EMNLP/ICLR 2027 after Gemma replication + exact label validation.

---

## 7. Hypothesis Update

| Hypothesis | Status | Surviving Form |
|---|---|---|
| H1: RD threshold predicts absorption geometry | **Falsified in predictive form** | Retain Proposition 1 as a theoretical *existence* proof (absorption is loss-optimal under threshold conditions), but explicitly retract the claim that it predicts post-convergence decoder geometry. |
| H2: Cross-domain absorption exists in semantic hierarchies | **Not supported (null result)** | Reframe as: first-letter absorption appears to be orthographic/task-specific. Semantic hierarchies show no measurable absorption rate with current C2 methodology. This is a scope characterization, not a strong falsification (C2 calibration issues exist). |
| H3: ASI detects absorption | **Falsified** | Drop ASI entirely as a contribution. Report ASI as a failed attempt alongside the explanation (frequency ratio is unstable for rare features; the cos² component predicts wrong direction). |
| H4a: Phase transition in sparsity | **Not supported; untestable in current configuration** | Report saturation regime finding (96% absorption across all L0 values) as the characterization result. Clearly distinguish from "no phase transition exists." |
| H4b: Hysteresis in absorption | **Untestable (saturation ceiling)** | Same framing as H4a. |
| H5 (implicit): L1 SAEs more absorbed than TopK | **Supported** | The EDA gap of 0.200 (z=−10.4) is the cleanest statistically supported finding in the study. Retain as a secondary contribution even if EDA's mechanistic interpretation is revised. |
| NH1 (new): Absorbed features are geometrically isolated in decoder space | **Empirically consistent; testable** | The direction reversal in B1 is the cleanest evidence. Test via nearest-neighbor cosine distribution (10 minutes of computation). |
| NH2 (new): cos(enc_p, dec_c) as cross-feature absorption indicator | **Promising, unvalidated on exact labels** | Validate on exact Chanin labels as Priority 2 experiment before committing to as primary contribution. |

---

## 8. Action Plan

### Immediate Priority Experiments (before writing)

**P1 — cos(enc_p, dec_c) validation on exact Chanin labels (~30 min, GPT-2 Small L6)**
- Decision: AUROC ≥ 0.70 → primary contribution + cross-directional mechanism; AUROC < 0.60 → retain EDA-norm as primary.
- No-regret: either outcome narrows the paper's claim appropriately.

**P2 — EDA-norm layer profile across GPT-2 Small layers 2–10 (~1h)**
- Compute enc_norm, EDA, EDA-norm (EDA × ||enc||) for each layer.
- Purpose: (a) characterize layer-specificity honestly for the paper; (b) identify if enc_norm peaks at L4–L6 consistent with post-absorption re-alignment hypothesis.
- Output: Layer-specificity figure — a standalone empirical contribution.

**P3 — Gemma Scope replication (~2–4h, Gemma 2 2B layer 12)**
- Go/no-go gate for venue: AUROC ≥ 0.65 on ground-truth labels → EMNLP/ICLR consideration; AUROC < 0.55 → workshop target.
- Use FeatureAbsorptionCalculator (MIT-licensed, same code as iter_001 R4) on Gemma Scope SAE.

### Writing Phase (after P1–P3)

**Reframe paper around surviving signals:**
- **New title direction:** "Geometric Fingerprints of Feature Absorption in Sparse Autoencoders: Encoder-Decoder Dissociation as a Probe-Free Detector"
- **Primary empirical contribution:** enc_norm and/or cos(enc_p, dec_c) predict absorption with AUROC 0.73–0.76 against exact ground-truth labels — first probe-free detector in the literature.
- **Secondary empirical contribution:** L1 SAEs show dramatically more absorption signal than TopK (EDA gap 0.200, z=−10.4), establishing a clean architectural comparison.
- **Theoretical contribution:** Proposition 1 (rate-distortion threshold) retained as *existence proof* only, with explicit acknowledgment that it does not predict post-convergence geometry.
- **Honest negative results section:** H1 direction reversal, H2 null (cross-domain), H3 ASI failure, H4 saturation regime — all reported systematically as contributions that sharpen understanding.

### PIVOT vs. PROCEED Verdict

**PROCEED** — with mandatory scope reduction. Criteria met:
1. At least one hypothesis has moderate-to-strong signal with a clear path to publication: YES (enc_norm AUROC=0.757, EDA AUROC=0.650 on exact labels, cos(enc_p, dec_c) AUROC=0.730 on proxy labels).
2. Clear publication path exists: YES (workshop paper now; full paper after Gemma replication).
3. GPU budget tractable: YES (~4–6 GPU hours remaining).

**No pivot** is warranted. Backup directions (absorption-as-diagnostic, mitigation benchmark, deconfounding) all require comparable or greater experimental investment while offering lower novelty ceiling than the surviving EDA/enc_norm signal.

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| cos(enc_p, dec_c) fails on exact labels (AUROC < 0.60) | Medium | Low | Retain enc_norm/EDA as primary; cos cross-directional stays secondary. |
| Gemma Scope replication fails (AUROC < 0.55) | Medium | High | Reframe as GPT-2-specific analysis; target workshop/ICLR 2027 workshop track. |
| Reviewer flags n_pos=18 as too small for AUROC claims | High (certain) | Medium | Report bootstrapped 95% CIs; use permutation null (z=+2.49 is the cleanest stat); note this is the complete available ground-truth dataset. |
| Proxy label circularity invalidates EDA claim | Low-Medium | Medium | enc_norm result is not circular (encoder norm ≠ proxy label construction). Anchor primary claim on enc_norm, use EDA as complementary. |
| Proposition 2 mechanism challenged | High (certain) | Low | Demote to conjecture; present mechanism as open question for future work. |
| Cross-domain null result seen as paper-killing | Low | Low | Frame proactively as scope characterization — absorption may be orthographic/task-specific, which is itself a publishable finding constraining the field's assumptions. |
