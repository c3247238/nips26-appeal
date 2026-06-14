# Revisionist Analysis — Feature Absorption in SAEs
**Date:** 2026-04-13
**Evidence base:** iter_002 pilot results, `exp/results/full/` and `exp/results/pilots/`
**Analyst:** sibyl-revisionist

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| H1: RD threshold (lambda > sin²(theta)) predicts absorption | **Refuted** | B1: absorbed features have LOWER cos²(theta) (pos_mean=0.052 vs neg_mean=0.127), AUROC=0.318 — opposite direction. | High |
| H1 corollary: co-occurrence frequency cancels | **Inconclusive** | Cannot test corollary when the base threshold is reversed; freq_ratio alone (AUROC=0.612) outperforms ASI (0.421), suggesting frequency is not irrelevant. | Low |
| H2: Cross-domain absorption exists (≥3 of 4 hierarchy types) | **Not Supported** | C2 v9 first_letter absorption_rate=0.008 (1 event of 120); animate_inanimate and noun_proper absorption_rate=10⁻⁸ (numerical zero). All C2 hierarchies returned NO_GO. | High |
| H3: ASI (AUROC ≥ 0.70) as probe-free detector | **Refuted** | D2: ASI AUROC=0.4215 (below null mean 0.4973). ASI is anti-correlated with absorption labels. DeLong test vs. EDA: delta=−0.260, p=5.8×10⁻⁸. | High |
| H4a: Phase transition in sparsity (sigmoid > linear) | **Not Supported** | E1: LRT p=0.456, BIC diff=−3.22. Spearman rho(inv_L0, absorption_rate)=0.191, p=0.574. | High |
| H4b: Hysteresis (absorption metastable) | **Not Confirmed** | E2: fine-tuned absorption=0.960 (baseline 0.959, scratch 0.964). SATURATION: all tested L0 values show ~96% absorption. Hysteresis untestable in this regime. | Medium |
| H5: Mitigations increase decoder angles | **Partially Supported** | F2: Standard/ReLU letter-pair cos²=0.021 vs TopK=0.015; EDA lower in TopK (0.476 vs 0.677). But direction is opposite to H1 prediction: TopK absorbs *less* despite similar low cos². | Medium |
| H6: ASI cross-domain correlation | **Not Testable** | C2 failed to produce nonzero cross-domain absorption rates; correlation undefined. | N/A |

---

## 2. Surprise Analysis

### Surprise 1: H1 direction is completely reversed (deviation: AUROC 0.318 vs expected ≥0.70)

**Finding:** The absorbed (letter) features have cos²(theta)=0.052, while non-absorbed features have cos²(theta)=0.127. The theoretical threshold predicts the *opposite* — that absorbed pairs should be geometrically closer. This is statistically robust: Wilcoxon p=1.3×10⁻⁶, Cohen's d=−0.57, and the direction reversal replicates at layer 10 (d=−0.72).

**Wrong assumption traced:** The theory assumes that the absorber (child feature) rotates its decoder toward the parent to "absorb" the parent's gradient. We assumed this would produce high decoder cosine similarity between the absorbed pair. But the measurement compares *letter features against random non-letter features*, not *specific absorber-absorbee pairs*. Letter features (which represent syntactic, token-level properties) appear to be geometrically *isolated* in decoder space — they have low cosine similarity to everything, which is consistent with them representing rare, orthogonal properties rather than absorbing anything. We conflated "features that undergo absorption" with "features that share decoder directions with something else."

More fundamentally: the rate-distortion theory describes absorption as a mechanism that *produces* high cos²(theta), but by the time we measure post-training, the absorbed parent feature may have already been suppressed or merged, leaving the child feature isolated. The geometry observed is the *result* of absorption, not its precondition.

**Mental model revision required:** The decoder geometry signature of absorption may not be "absorbed pairs are geometrically close" but rather "absorbed features are geometrically isolated." This would explain the direction reversal: absorption makes child features lonely in decoder space because their parent has been merged into them and no other features compete at the same direction.

---

### Surprise 2: EDA works but for theoretically unexpected reasons (deviation: AUROC 0.681 vs. null 0.500, but fails at L10 and in AJT suites)

**Finding:** EDA (1 − cos(enc_j, dec_j)) achieves AUROC=0.681 at GPT-2 layer 6 with Standard/ReLU (JB) SAE. This was the one significant positive finding. But EDA fails at layer 10 (AUROC=0.337, reversed), and at all AJT architectures. The `cos(enc_p, dec_c)` metric (projection of parent encoder onto child decoder) achieves AUROC=0.730 in the pairwise EDA pilot — higher than EDA — with Cohen's d=0.55 and z=6.4 above null.

**Wrong assumption traced:** We assumed EDA would be a general absorption detector based on the theoretical argument that absorbed features should have encoder-decoder dissociation. This is likely not the mechanism explaining the AUROC=0.681. The more parsimonious explanation is that letter features in the JB SAE are trained with an encoder that points away from the decoder because they encode a low-dimensional syntactic property that cannot be well-reconstructed by a single decoder direction. This is a training artifact of the L1-penalty SAE, not an absorption signal per se. The fact that `cos(enc_p, dec_c)` — projection of the *parent* encoder onto the *child* decoder — achieves the highest AUROC (0.730) is more theoretically grounded: it measures whether a parent-class encoder direction points toward a child decoder direction, which would indicate that the child feature would fire when parent-class inputs arrive, i.e., absorption is occurring.

**Revised understanding:** The EDA signal at JB L6 may be detecting letter features as semantically unusual, not as absorbed. The `cos(enc_p, dec_c)` metric is the mechanistically correct absorption indicator but was underemphasized in the original proposal.

---

### Surprise 3: Cross-domain absorption is undetectable across all hierarchy types (deviation: absorption_rate ≈ 0 vs. expected [5%, 50%])

**Finding:** The C2 measurement pipeline found near-zero absorption across all non-spelling hierarchy types. Even the first-letter hierarchy returned absorption_rate=0.008 (1 event of 120) in v9, and the other hierarchies returned numerical zeros. This is dramatically different from the expectation of [5%, 50%] absorption rates.

**Wrong assumption traced:** We assumed that absorption — as a general SAE failure mode — would manifest measurably in any hierarchy where parent-child structure exists. The C2 result exposes a critical methodological problem: the "absorption" signal for first-letter features that the Chanin et al. pipeline detects may be specific to the *spelling task format* and the *IG-based absorption calculator*, not a general property of SAE representations. The absorption measurement method was designed around a specific task (first-letter classification with in-context examples) and imports assumptions about how the task is performed that may not generalize.

There is also a deeper issue: the C2 v9 measurement uses "child suppression" logic — checking whether parent latents are suppressed when child tokens appear. This tests the wrong direction compared to the Chanin et al. method (which checks if letter latents fail to fire). The C3 analysis notes: "C2 v4 should implement traditional absorption measurement: for each child token, find child-specific SAE latents, check if they fail to fire while parent-class latents are active."

---

### Surprise 4: Absorption is near-total regardless of L0 (deviation: H4b saturation regime, absorption 95–96% across all configurations)

**Finding:** E1 measured 11 different SAE configurations and found absorption rates of 85–99% across all of them. E2 confirmed this: baseline=0.959, fine-tuned=0.960, scratch (different layer)=0.964. Varying L0 from ~18 to ~84 moves absorption rate from 0.96 to 0.96 — essentially no variation.

**Wrong assumption traced:** We assumed absorption rate would vary meaningfully with L0, allowing us to observe a phase transition. The actual picture is that absorption is already in a saturation regime: ALL letter features show near-total absorption in GPT-2 Small at every tested L0. There is no pre-transition regime to sample from. The phase transition question ("does absorption onset look like a sigmoid?") cannot be answered because we observe only the post-transition plateau.

This implies that if a phase transition exists, it occurs at much lower sparsity levels than any pre-trained SAE provides access to. Or, more radically, that the 96% absorption rate is not meaningfully different from a ceiling artifact of the measurement method.

---

## 3. Mental Model Revision

We went in believing: "Feature absorption is a graded failure mode, controlled by decoder geometry (specifically cos²(theta)), with severity predictable from a closed-form threshold. It is a cross-domain phenomenon affecting all hierarchically structured features."

The data requires revising this to: "Feature absorption as measured by the Chanin et al. IG-based method may be near-universal in the spelling task (96%+), with no measurable variation with L0 over the tested range. The decoder geometry signature of absorbed features is the *opposite* of what the rate-distortion theory predicts: absorbed features (letter features) are geometrically isolated, not geometrically similar to parents. The best available predictor of absorption labels is neither ASI nor raw decoder cosine, but rather the encoder-decoder relationship *across* feature pairs — specifically how much a parent feature's encoder direction points toward a child feature's decoder. Cross-domain absorption at a measurable rate may require a different measurement method than C2 used, or may genuinely not exist in the non-spelling domain."

---

## 4. Reframing Test

**Original research question:** *"Is feature absorption the rate-distortion optimal behavior under flat sparsity penalties, and what geometric threshold predicts it?"*

**Would we frame it the same way today?** No. The rate-distortion framing produces a falsifiable prediction (absorbed pairs have high cos²(theta)) that is empirically reversed. The "rate-distortion optimal" narrative requires a different operationalization of the theory — one that accounts for post-absorption geometry rather than pre-absorption geometry.

**Revised research question:** *"What is the geometric and structural signature of absorption in pre-trained SAEs, and can that signature be exploited to detect unverified absorbed features without task-specific probes?"*

This reframing:
1. Drops the causal theory-first approach and starts from the empirical geometry signal
2. Acknowledges that the `cos(enc_p, dec_c)` metric (AUROC=0.73) is the most promising detector
3. Shifts from "predict which pairs will be absorbed" to "characterize what absorption looks like in the weight space after it has occurred"
4. Does not depend on falsified H1 or the cross-domain claim

---

## 5. New Testable Hypotheses from Surprising Results

### NH1: Absorbed features are geometrically isolated in decoder space

**Statement:** Features that undergo absorption have lower average decoder cosine similarity to all other features than non-absorbed features. Specifically, the 10th-percentile nearest-neighbor decoder cosine similarity for absorbed features is lower than for non-absorbed features (one-tailed Wilcoxon p < 0.01).

**Mechanistic basis:** If a parent feature is absorbed into a child feature, the child's decoder points toward a synthesis of parent and child directions. This is a singular direction with no other features nearby. The geometrically-isolated signature already observed (pos cos²=0.052 vs neg cos²=0.127) is consistent with this.

**Concrete experiment:** For GPT-2 Small layer 6 JB SAE (n_pos=71 letter features), compute nearest-neighbor decoder cosine similarity (k=10) for each letter feature and each of 500 random non-letter features. Compare distributions. Estimated runtime: 10 minutes.

**Falsification:** No significant difference in nearest-neighbor cosine similarity between letter and non-letter features (p > 0.05).

---

### NH2: `cos(enc_p, dec_c)` is a generalizable cross-feature absorption indicator

**Statement:** For any two features p (parent class) and c (child class) in a pre-trained SAE, high `cos(enc_p, dec_c)` — the cosine similarity between the encoder of p and the decoder of c — indicates that c absorbs p's information. This metric generalizes across SAE architectures and layers.

**Mechanistic basis:** If feature c absorbs feature p, then activations that would trigger p also trigger c (because c's decoder direction has moved toward p's). This means p's encoder direction should point toward c's decoder direction. The pairwise EDA experiment found this metric achieves AUROC=0.730 with z=6.4 above null for proxy-labeled absorption in GPT-2 L6.

**Concrete experiment:** Compute `cos(enc_p, dec_c)` for all pairs (p ∈ high-frequency features, c ∈ letter features) and validate AUROC against exact Chanin et al. labels (n=67). Replicate at L10. If AUROC ≥ 0.65 at both layers, submit as a probe-free detector. Estimated runtime: 30 minutes.

**Falsification:** AUROC < 0.55 at L10 (replicate fails), or AUROC < 0.60 even with exact Chanin labels.

---

### NH3: The Chanin et al. absorption measurement is task-specific, not a general SAE property

**Statement:** The absorption rate measured by IG-based first-letter task analysis reflects the task format and vocabulary (short, common English nouns starting with specific letters in in-context examples) rather than a general SAE representation failure. When the measurement is applied to semantic hierarchies using semantically analogous prompts and a ground-truth behavioral test (not IG attribution), absorption rates are indistinguishable from zero.

**Mechanistic basis:** The v9 C2 pilot found 0% absorption for animate_inanimate and noun_proper hierarchies. C2's design (checking parent latent suppression) is methodologically different from Chanin's (checking child-feature non-firing via IG). Neither the null result in semantic hierarchies nor the positive result in first-letter spelling is unambiguous evidence about the SAE — they may both reflect the measurement method.

**Concrete experiment:** Implement the traditional absorption measurement for semantic hierarchies (identify child-specific SAE latents; measure whether they fail to fire while parent-class latents are active, on a behavioral test set). Compare to the IG-based first-letter rate. If the first-letter behavioral test also returns ~0% with proper attribution, the Chanin metric may be detecting something other than absorption. Estimated runtime: 1 hour.

**Falsification:** The traditional absorption measurement finds significant (>5%) semantic hierarchy absorption in at least 2 of 4 hierarchy types.

---

## Appendix: Key Numbers Reference

| Metric | Value | Source |
|---|---|---|
| Absorbed (letter) features cos²(theta) | 0.052 ± 0.039 | B1_decoder_geometry, L6 |
| Non-absorbed features cos²(theta) | 0.127 ± 0.181 | B1_decoder_geometry, L6 |
| EDA AUROC (JB L6) | 0.681 | D2, B1 |
| EDA AUROC (JB L10) | 0.337 | B1 (reversed) |
| cos(enc_p, dec_c) AUROC (letter vs non-letter) | 0.730 | B1_pairwise_eda |
| ASI AUROC | 0.421 (null=0.497) | D2 |
| freq_ratio AUROC | 0.612 | D2 |
| cos²_alone AUROC | 0.206 | D2 |
| C2 first_letter absorption_rate (v9) | 0.008 (n=1/120) | C2_child_suppression |
| C2 semantic hierarchy absorption_rate | ~0 (10⁻⁸) | C2_child_suppression |
| E1 absorption rates across all L0 values | 0.85–0.99 (saturation) | E1_phase_transition |
| E2 baseline vs fine-tuned absorption | 0.959 vs 0.960 | E2_hysteresis |
| Standard/ReLU EDA mean (letter features) | 0.677 | F2 |
| TopK EDA mean (letter features) | 0.476 | F2 |
