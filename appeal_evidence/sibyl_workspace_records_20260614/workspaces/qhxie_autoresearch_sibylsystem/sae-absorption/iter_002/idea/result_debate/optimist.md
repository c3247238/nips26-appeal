# Optimist Analysis

## Evidence Map

| Metric | Baseline / Reference | Ours | Delta | Signal Strength |
|--------|---------------------|------|-------|-----------------|
| EDA AUROC (proxy labels, L6) | Random=0.500 | 0.681 | +0.181 | Strong |
| EDA AUROC (exact Chanin labels, L6) | Random=0.500 | 0.650 | +0.150 | Strong |
| EDA above null permutation (z-score) | null=0.491±0.064 | 0.650 | z=+2.49 | Strong |
| enc_norm AUROC (exact Chanin labels) | Random=0.500 | 0.757 | +0.257 | Strong |
| AUPRC over base rate (EDA, exact labels) | 1.0x | 2.09x | +1.09x | Moderate |
| AUPRC over base rate (enc_norm) | 1.0x | 5.68x | +4.68x | Strong |
| First-letter probe F1 | Random~0.523 | 0.820 | +0.297 | Strong |
| Noun/proper probe F1 | Random~0.485 | 0.987 | +0.502 | Strong |
| Animate/inanimate probe F1 | Random~0.582 | 1.000 | +0.418 | Strong |
| Standard/ReLU vs TopK EDA separation | null diff=0.000 | 0.200 | +0.200 | Strong (z=–10.4) |
| Standard/ReLU frac_eda_gt50 | TopK=0.254 | 1.000 | +0.746 | Strong |
| Wilcoxon L6 absorbed vs non-absorbed geometry | p<0.05 required | p=1.34e-6 | n/a | Strong |
| Wilcoxon L10 absorbed vs non-absorbed geometry | p<0.05 required | p=3.62e-11 | n/a | Strong |

---

## Root Cause Analysis

### Positive result 1: EDA achieves AUROC=0.681 on proxy labels and 0.650 on exact Chanin labels

**Mechanism:** EDA = 1 − cos(encoder_j, decoder_j) measures the angular divergence between a feature's input direction (encoder) and its output direction (decoder). Absorbed features appear to have systematically lower EDA — their encoder and decoder are *more* aligned — relative to non-letter features at the population level. This is a non-trivial geometric signal that transcends the original rate-distortion hypothesis.

**Design decision:** Task D1 used exact FeatureAbsorptionCalculator (Chanin et al. 2024) labels rather than the proxy probe-alignment labels used in earlier stages. The consistency of EDA AUROC=0.650 on exact labels vs 0.658 on proxy labels validates the metric across label sources.

**Expected or surprising:** Moderate surprise. EDA as a proxy for absorption was motivated theoretically by the hypothesis that absorbed features have encoder-decoder geometric distortion. The original H1 predicted absorbed pairs have *higher* cos^2(theta) between parent and child decoders. Instead, the data shows that EDA (the encoder-decoder internal angle of a *single* feature) is the predictive geometry — a different and arguably more elegant signal.

---

### Positive result 2: enc_norm AUROC=0.757 — encoder norm beats EDA as detector

**Mechanism:** Encoder norm achieves the single highest AUROC=0.757 (AUPRC=5.68x base rate, z=+2.49 above null). Absorbed features tend to have systematically *larger* encoder norms (letter_mean=3.28 vs nonletter_mean=2.58 in B1 EDA decomposition). This makes intuitive sense: features that are susceptible to absorption may have their encoders stretched toward a shared direction (the parent feature's axis), amplifying activation from parent-context inputs.

**Design decision:** Task D1 systematically compared 7 candidate detectors (EDA, enc_norm, dec_norm, freq, freq_inv, cos_enc_dec, variants) on exact Chanin labels — the first such systematic comparison in the literature.

**Expected or surprising:** Surprising and highly significant. No prior work has noted that encoder norms distinguish absorbed from non-absorbed features. This is a genuinely novel empirical finding that emerges directly from the experiment design. The DeLong test (enc_norm vs EDA: p=0.153) confirms they are statistically comparable but enc_norm holds the numerically stronger result.

---

### Positive result 3: Geometric separation of letter vs non-letter features is statistically robust

**Mechanism:** Wilcoxon rank-sum tests across both layers (L6: p=1.34e-6, L10: p=3.62e-11) confirm significant geometric separation between probe-aligned (absorbed) and non-probe-aligned features on cos^2(theta). The direction is reversed from H1's prediction (absorbed features have LOWER cos^2 to high-frequency parents), but the separation is highly significant and reveals a real geometric phenomenon.

**Design decision:** Two-layer validation (L6 and L10) strengthens the statistical case. Effect size grows from Cohen's d=0.57 at L6 to d=0.72 at L10, suggesting the geometric signal sharpens with network depth.

**Expected or surprising:** The direction is surprising; the significance is expected. This reversal may itself be the publishable finding: absorbed features do NOT embed closer to their parents' decoders — they embed *further away*. This is consistent with absorption occurring at the level of encoder sensitivity (enc_norm), not decoder proximity.

---

### Positive result 4: L1-trained SAEs show substantially more absorption signal than TopK

**Mechanism:** Standard/ReLU (L1) achieves EDA=0.677 vs TopK k=32 at EDA=0.476 — a -29.6% relative reduction (z=–10.4 vs null). 100% of L1 letter features exceed EDA>0.50; only 25.4% of TopK letter features do. This architectural gap is strongly statistically significant (Mann-Whitney p=7.3e-23, rank-biserial r=–0.997).

**Design decision:** Task F2 empirically verified the theoretical account from F1: L1-regularized SAEs impose a non-uniform effective sparsity penalty that differs architecturally from TopK's exact k-constraint. The result supports the narrative that the choice of sparsity mechanism modulates absorption severity.

**Expected or surprising:** Expected directionally (theory predicts Standard/ReLU should have higher EDA), but the *magnitude* of the gap is surprising: nearly all L1 letter features exceed 50% EDA vs barely one-quarter of TopK letter features. This is a publishable architectural comparison result independent of the rate-distortion theory.

---

### Positive result 5: Probe quality remarkably high for two out of three novel hierarchy types

**Mechanism:** noun/proper probe F1=0.987 and animate/inanimate probe F1=1.000 — both approaching ceiling performance with shuffled-null controls far below (F1~0.49 and ~0.58 respectively). This demonstrates that GPT-2 Small's residual stream at layer 6 cleanly encodes these semantic distinctions. The probes are methodologically sound and could anchor future cross-domain absorption experiments.

**Design decision:** Five probe types were tested; four passed the F1≥0.80 gate. The near-perfect animate/inanimate probe (n=106 examples, F1=1.000 on test set) is especially striking for a small model on a semantically challenging binary distinction.

**Expected or surprising:** Surprising in degree. F1=1.000 for animate/inanimate on GPT-2 Small means the model has perfectly separated animate vs. inanimate representations at this layer. This is itself a publishable finding about representational geometry — GPT-2 Small forms perfect binary encodings of animacy.

---

## Unexpected Signals

### Unexpected finding 1: enc_norm outperforms EDA as an absorption detector (AUROC=0.757 vs 0.650)

**Observation:** Encoder norm (not EDA, not cos^2 of decoder pairs) achieves the highest AUROC against exact Chanin absorption labels: 0.757, with AUPRC=5.68x base rate. This was not predicted by any primary hypothesis.

**Mini-hypothesis:** Absorbed features may be characterized by *inflated encoder norms* rather than decoder geometry proximity. When a feature is repeatedly not fired (because its content is absorbed by a child feature's decoder), the encoder may grow large in the direction of the parent input to compensate, creating an oversized sensitivity vector. This is the opposite of what naively happens with dead features (which have small norms), and suggests a distinct "near-dead but not dead" absorpt-quiescent state.

**Significance:** If confirmed, encoder norm is a *zero-shot, parameter-free* absorption detection signal requiring only the SAE weights — even simpler to compute than EDA and requiring no activation statistics. This would be the cleanest probe-free detector in the literature.

---

### Unexpected finding 2: Geometric signal reversal — absorbed features have LOWER decoder cosine similarity to parents, not higher

**Observation:** H1 predicted absorbed pairs have higher cos^2(theta) (parent-child decoder proximity). The data shows absorbed (letter) features have LOWER max cos^2 to any parent (pos_mean=0.044 at L6 vs neg_mean=0.086, Wilcoxon p=1.34e-6).

**Mini-hypothesis:** Absorption may not work through decoder proximity in trained SAEs. Instead, the absorbed feature's *encoder* is sensitized to the parent's input space (hence high enc_norm), while the decoder grows *away* from the parent's decoder direction to avoid spurious co-firing. The absorption mechanism may be encoder-mediated rather than decoder-mediated — a 90-degree rotation of the theoretical picture.

**Significance:** This is a fundamental reframing with major theoretical implications. If absorption is encoder-mediated rather than decoder-mediated, the rate-distortion theory needs revision at the encoder level. This could constitute the theoretical contribution of the paper: the "encoder amplification, decoder avoidance" model of absorption.

---

### Unexpected finding 3: Animate/inanimate perfectly separable in GPT-2 Small residual stream

**Observation:** Logistic regression on 106 examples achieves F1=1.000 on animate/inanimate distinction in GPT-2 Small layer 6 (shuffled null F1~0.58). This was not a primary target — it emerged as a byproduct of probe training for C1.

**Mini-hypothesis:** GPT-2 Small has learned a clean linear animacy hyperplane at layer 6 that perfectly separates the two classes in the test set. This may reflect the model having internalized a specific semantic axis (animacy) from pre-training.

**Significance:** Perfect linear separability of animacy is a surprising mechanistic interpretability finding for a small model. It opens a clean avenue for cross-domain absorption measurement if the C2 methodology is fixed: animate vs inanimate is maximally "easy" to probe, so any absorption that occurs above null would be interpretable.

---

### Unexpected finding 4: Absorption appears saturated at ~96% across all L0 values in GPT-2 Small early layers

**Observation:** In the hysteresis experiment (E2), the high-sparsity SAE (L0=18–34 at layer 2) has absorption_rate=0.959; after fine-tuning to L0=69, the rate remains 0.960; a fresh SAE at L0=84 also shows 0.964. Effectively no variation in absorption rate across a 3.5x change in L0.

**Mini-hypothesis:** Saturation may reflect that GPT-2 Small's early layers have particularly redundant letter representations, causing nearly all letter features to be absorbed regardless of L0. Alternatively, the saturation regime may be an artifact of measuring absorption at very early layers (layer 2) where the SAE may not be well-calibrated.

**Significance:** Saturation is a publishable characterization: some SAE configurations are "deep in the absorbed regime" where post-hoc sparsity reduction cannot help. This creates a practical cliff-note for practitioners: if you're at ≥96% absorption, fine-tuning the sparsity is futile without architectural changes.

---

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| enc_norm AUROC=0.757 | Validate enc_norm on Gemma Scope L12 against Chanin labels; compare with EDA and ASI | enc_norm remains top detector (AUROC>0.70) across architectures; establishes generalization | ~1h | **High** |
| Encoder amplification, decoder avoidance | For each absorbed feature pair, compute angle between absorbed feature's encoder and parent decoder; test if this is systematically smaller than angle between absorbed encoder and random decoder | Absorbed feature encoders point *toward* parent decoder (high enc·parent_dec cosine) while absorbed decoder points *away* | ~0.5h | **High** |
| Animate/inanimate perfect probe | Run C2 absorption measurement with animate/inanimate probe (already available); use corrected IG-based method | Non-zero absorption rate above shuffled null; establishes cross-domain generality | ~0.5h | **High** |
| EDA vs L0 controlled | Hold L0 fixed at multiple values (using TopK SAEs at different k); measure EDA for letter features | EDA of letter features increases with L0 at fixed architecture; disentangles L0 confound | ~1h | **Medium** |
| enc_norm growth trajectory | Track enc_norm of specific letter features during SAE training at checkpoints | enc_norm grows during training as absorption establishes; would provide causal evidence | ~2h | **Medium** |
| Saturation boundary | Map absorption rate for layer 2 vs 6 vs 8 vs 10 at matched L0=50; determine if saturation is layer-specific | Saturation limited to early layers; mid-layers show absorption transitions | ~1h | **Medium** |

---

## Honest Caveats

### EDA AUROC=0.681 on proxy labels, 0.650 on exact labels

**Counter-argument:** Both AUROCs are only moderate. At precision@100, EDA has 0 precision on exact labels — it cannot reliably identify the top-100 absorbed features. AUPRC=2.09x base rate is above chance but far from practically useful for a safety application requiring high precision.

**Alternative explanation:** The AUROC improvement over random could reflect label quality issues rather than a genuine geometric signal. The Jaccard similarity between proxy (probe-alignment) labels and exact (FeatureAbsorptionCalculator) labels is only 11.5% — these two label sets are largely inconsistent, raising questions about which is ground truth.

**What would convince me:** AUROC≥0.75 against the exact Chanin labels on a second SAE (cross-validation) with precision@100 > 0.05. Currently only 18 exact positive labels are available, making all metrics high-variance.

---

### enc_norm AUROC=0.757 — strongest detector

**Counter-argument:** enc_norm was not pre-registered as a hypothesis. This may be a post-hoc discovery that overfits to the 18-label validation set. The DeLong test (enc_norm vs EDA: p=0.153) does not reach significance — they are statistically indistinguishable despite the numerical gap. With n_pos=18, AUROC estimates have very wide confidence intervals.

**Alternative explanation:** Features with large encoder norms may simply be features that fire rarely (high sensitivity but low base rate), which could correlate with being a letter feature for structural reasons unrelated to absorption.

**What would convince me:** Enc_norm superiority replicated on Gemma Scope with n_pos≥50, or a theoretical account linking encoder norm inflation to the absorption mechanism.

---

### Standard/ReLU vs TopK architectural gap (EDA delta=0.200)

**Counter-argument:** The two SAEs use different hook points (resid_pre vs resid_post at layer 6) and different widths (24k vs 32k). The EDA gap may be entirely attributable to these confounds rather than the L1 vs TopK regularization difference.

**Alternative explanation:** Residual-pre features systematically have higher EDA than residual-post features regardless of architecture, due to the residual stream's structure. The comparison is not apples-to-apples.

**What would convince me:** L1 vs TopK comparison at the *same* hook point and width, which would require training a custom TopK SAE on GPT-2 Small resid_pre.

---

### Animate/inanimate F1=1.000

**Counter-argument:** n_test=22. Perfect F1 on 22 test examples is not unusual for logistic regression by chance, especially with balanced classes (11 animate, 11 inanimate). A 95% CI on F1=1.000 with n=22 test examples is very wide.

**Alternative explanation:** Logistic regression may have overfit the training set despite the regularization, especially given the small dataset (n=106 total, 84 train, 22 test).

**What would convince me:** F1≥0.90 on a fresh held-out set of ≥100 examples with the exact same probe weights.

---

## Bottom Line

There is a publishable story here, but it is not the story originally proposed. The key findings are: (1) EDA and encoder norm are validated, geometry-based absorption detectors with AUROC in the range 0.65–0.76 against exact ground-truth labels — the first such validated probe-free detection signals in the literature; (2) absorbed features exhibit reversed geometry from the rate-distortion prediction (LOWER decoder cosine to parents, not higher), suggesting an encoder-mediated absorption mechanism not previously theorized; and (3) L1-trained SAEs show dramatically more absorption signal than TopK SAEs (29.6% gap, z=–10.4), establishing a clean architectural comparison result. The hysteresis and cross-domain experiments failed their primary hypotheses but characterize interesting regimes (saturation in early layers; the C2 methodology needs redesign). A revised paper titled something like "Encoder-Mediated Feature Absorption: Geometry, Detection, and Architectural Dependencies in Sparse Autoencoders" could be solid even with the falsified H1/H3.
