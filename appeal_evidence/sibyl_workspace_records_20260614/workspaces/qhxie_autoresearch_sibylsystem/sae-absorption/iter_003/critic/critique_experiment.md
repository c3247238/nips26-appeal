# Experiment Critique (Updated by Critic — Iter 003)

## Summary

Six critical experimental problems are identified. All core experiments are in PILOT mode with 10-30x smaller scale than the methodology specifies. The OMP experiment has near-zero statistical power (n=90 observations at 97.8% baseline). The cross-architecture result uses non-absorption proxy labels. The cross-hierarchy experiment is methodologically invalid. All are submission blockers.

---

## Critical Issues

### 1. OMP Ceiling Effect — The Experiment Cannot Detect What It Claims to Test

The core experimental claim — that OMP achieves 0% absorption reduction, decisively falsifying the amortization gap hypothesis — is undermined by a ceiling effect that makes the null result uninterpretable.

**The numbers:**
- Feedforward AR: 0.967 (letter a), 1.000 (letter e), 0.967 (letter s)
- OMP AR: 0.967 (letter a), 1.000 (letter e), 0.967 (letter s)
- Mean: both = 0.978

When 97-100% of tokens show absorption under both conditions, there is no room to observe a difference. The absorption measurement method ("whether Chanin IG main features fire") checks whether identified main features for a letter are present in the SAE's active set. With 30 tokens per letter and near-100% baseline, the standard error on the difference is approximately sqrt(2 * 0.978 * 0.022 / 30) ≈ 0.037. The pre-committed criterion of ≥80% ratio is satisfied trivially at 100% regardless of whether there is a real effect or not.

**Statistical power:** With baseline AR = 0.978, the maximum detectable difference at n=30 is approximately ±0.037 (1 standard deviation). The pre-committed criterion was checking whether ratio ≥ 0.80, i.e., whether the reduction is ≤ 20%. A 20% reduction from 0.978 would be AR_OMP = 0.782 — a change of 0.196. This is 5.3 standard deviations, meaning it is easily detectable. But the experiment cannot show that there is genuinely 0% reduction versus, say, a 2-3% reduction (which would be a ratio of 0.98-0.99). The paper claims "unambiguous" and "decisive" but has not estimated uncertainty on the null result.

**Measurement design flaw:** The absorption measurement checks whether specific known main features fire. But if the SAE has learned a training-time partial minimum (as the sparsity landscape account predicts), then any inference-time code obtained from the fixed dictionary will still not select the absorbed features, because the dictionary geometry no longer represents them correctly. Under this interpretation, the OMP result is tautologically expected from the hypothesis being tested — it is not a discriminating test.

**Fix:** Re-run with n ≥ 500 tokens per letter and all 26 letters. Report bootstrap 95% CI on (AR_OMP - AR_FF). Test at L2 or L10 where baseline AR is lower (L2 ratio = 0.877, meaning some recovery happens at lower layers). Consider modifying the measurement to capture whether OMP selects the absorbed feature at any point during the iterative pursuit (not just whether it appears in the final k-sparse code).

### 2. OMP Test Scope — Only 3 Letters, 90 Observations Total

The "full" experiment (labeled C2) uses letters {a, e, s} and 30 tokens each for a total of 90 observations. The proposal specified 26 letters and 10,000 tokens. The results labeled "PILOT" in the metadata appear to be the final data — the experiment was never run at full scale. This is a critical discrepancy between the stated experimental plan and what was actually executed.

### 3. Cross-Architecture "Replication" Uses Non-Absorption Labels

The E1 results show zero absorption in the TopK-32k SAE under the IG measurement pipeline:
```
"a": absorption_rate: 0.0
"e": absorption_rate: 0.0
"o": absorption_rate: 0.0
"s": absorption_rate: 0.0
"t": absorption_rate: 0.0
```

The n_pos=77 labels used for the AUROC=0.837 result come from decoder-alignment (cosine ≥ 0.30 to letter probes), not from the absorption pipeline. These 77 features are features that happen to align with letter probes in decoder space — not features that are functionally absorbed. The AUROC=0.837 result measures encoder norm's ability to predict decoder-letter alignment, which is a completely different quantity from absorption detection. This result should not be presented as replicating the AUROC=0.757 absorption detection finding.

### 4. Entity-Type Cross-Hierarchy: 0% Absorption, but Negative Control Shows 5%

The D2 result shows:
- Entity-type hierarchy (ANIMAL ⊃ dog, cat, bird, fish, horse): AR = 0.000 across all 3 seeds
- Negative control (weather + time of day co-occurrence): AR = 0.050 across all 3 seeds

The t-test is statistically significant (p = 9.6e-33) but the effect direction is OPPOSITE to H3: the negative control shows higher absorption than the positive hierarchy. This completely falsifies H3 (entity-type hierarchy shows absorption > negative control). The paper correctly labels this as a probe artifact, but the probe achieving 100% accuracy (D1) and yet showing 0% absorption suggests the probe is not measuring what is intended — it may be detecting surface lexical cues rather than semantic entity-type membership.

A deeper problem: the D2 notes state "Used entity-type probe on control sentences for comparison" — meaning the negative control absorption was measured using the entity-type probe, not a probe trained on the control concept. This is a methodological error. The negative control absorption should be measured with a probe trained on the control concept (weather/time), not with the entity-type probe applied to control sentences.

### 5. All Experiments in PILOT Mode with Small Token Counts

A review of the result file metadata reveals:
- A1: "mode": "PILOT"
- A3: "mode": "PILOT"
- B2: "mode": "PILOT"
- C2: "mode": "PILOT"
- D2: "mode": "PILOT"

All core experiments were run in pilot mode. The "full" results directory contains pilot-scale data, not full-scale experiments as described in the methodology. The paper presents these pilot results as if they are the final reported results, but the methodology specifies much larger scales (10,000 tokens for absorption measurement, 200,000 tokens for co-occurrence graph, 1,000 tokens for OMP experiment). This systematic underscaling is a major reproducibility concern.

## Major Issues

### 6. Gemma Scope Inaccessible — Results Are GPT-2 Small Only

The primary model from the proposal (Gemma-2-2B with Gemma Scope SAEs) was inaccessible (401 Unauthorized). All experimental results are from GPT-2 Small (117M parameters). GPT-2 Small is a small, 2019-vintage model. The paper's claims about SAE feature absorption in the context of modern safety-relevant interpretability work require results on at least one modern model (Gemma-2, Llama-3, Mistral). The paper as written overgeneralizes from a single, potentially non-representative model.

### 7. Layer 6 Specificity of EncNorm Signal

The A2 layer analysis shows encoder norm ratio (absorbed/non-absorbed) at GPT-2 layers:
- L2: 0.877 (absorbed features have LOWER encoder norm)
- L4: 1.055 (slightly higher)
- L6: 1.267 (peak — the validated layer)
- L8: 0.891 (absorbed features have LOWER encoder norm again)
- L10: 0.933

The signal inverts direction at L2, L8, and L10. The paper claims EncNorm is a general-purpose absorption detector, but it only works in the favorable direction at L4 and L6. The paper reports this as "peaks at L6" but does not address the inversion at other layers. A practitioner applying EncNorm to a non-L6 SAE would get incorrect predictions.

### 8. Co-occurrence Graph Built on Only 10,000 Tokens

The B1 co-occurrence graph uses 10,000 tokens, not the 200,000 specified in the proposal. With 24,576 features and 10,000 tokens, the average number of co-activation observations per feature pair is extremely small (approximately 10,000 * P(i fires) * P(j fires)). For rare features (activation frequency < 0.01), this yields fewer than 1 co-activation observation. The A_cooccur values of 0.333 for many features (the maximum under the f_i > 3f_j constraint) likely reflect very sparse co-occurrence statistics computed from insufficient data rather than a genuine signal.

---

## Fresh Analysis Notes (Iter 003 Critic Pass)

### 9. Iter-001 Labels Used Across Three Iterations — In-Sample Optimization Risk

The n_pos=18 labels (iter_001, R4, FeatureAbsorptionCalculator at GPT-2 L6) have been used to evaluate EDA in iter_001, EDA+ASI in iter_002, and EncNorm+ARS_v2+OMP in iter_003. The "discovery" of EncNorm in iter_002 was made on these 18 labels; the "validation" in iter_003 is on the same 18 labels. This is not independent validation — it is replication within the same dataset. Any metric that happened to correlate with these 18 specific features during iter_002 exploration will appear validated by iter_003. The AUROC=0.757 result must be treated as exploratory until validated on independently labeled absorption data.

### 10. O_jaccard AUROC Inflated by Structural Zero Coverage

Nine of 18 absorbed features have O_jaccard=0.000 by construction (not in the letter feature set). The B2 JSON explicitly lists these 9 features. The AUROC=0.721 conflates features that O_jaccard can potentially detect (9 features) with structural zeros that cannot be detected regardless of signal quality. The AUROC within the 9-feature detectable subset could be substantially higher or lower than 0.721; this is not reported. The claim that O_jaccard is a "complementary signal" requires this within-subset AUROC to be computed.

### 11. D2 Negative Control Absorption Is Measured with Wrong Probe

The D2 notes state: "Used entity-type probe on control sentences for comparison." This is a methodological error. The negative control (weather + time-of-day co-occurrence) was measured using the ANIMAL entity-type probe, not a probe trained for the weather/time concept. This makes the 5% negative control absorption rate uninterpretable — it measures how often the ANIMAL entity-type features fire on weather/time sentences, not whether a properly-trained weather/time probe shows absorption. The D2 result cannot support any conclusion about cross-hierarchy absorption generalization.
