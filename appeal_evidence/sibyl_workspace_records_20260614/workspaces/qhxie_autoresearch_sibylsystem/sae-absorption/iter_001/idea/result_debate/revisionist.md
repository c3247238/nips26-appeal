# Revisionist Analysis: Post-R4 Belief Update

**Agent**: revisionist
**Timestamp**: 2026-04-13
**Round**: R4 (Post-Experimental Final)
**Prior round**: R3 (Pre-R4 blocking experiments)
**Data sources**: Phase 1-5 full results, R4a-R4c results, ablation studies, threshold sensitivity

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence | Prior Belief (R3) | Belief Shift |
|------------|---------|--------------|------------|-------------------|--------------|
| H1 (EDA lower bound) | **Confirmed (regime-specific)** | SynthSAEBench AUROC=1.0; Gemma L12-16k AUROC=0.776 (p=6.4e-5, d=1.02); GPT-2 L6 direct labels AUROC=0.650 (p=0.027, d=0.53). 3/8 configs pass AUROC>=0.65. | High (0.85) | Confirmed | Narrowed: from "universal" to "regime-specific." The theorem is correct; the practical detector has a limited operating envelope. |
| H2 (D-EDA improvement) | **Falsified** | EDA = 1 - dec_cosine exactly (delta AUROC = 0.0 on GPT-2 direct labels). Pearson r=-1.0 on all 3 architectures (Gemma, GPT-2, Llama). D-EDA is mathematically redundant. | Very high (0.95) | Partially confirmed | Collapsed: D-EDA is not a separate metric. The directional decomposition idea was built on a mathematical misunderstanding of the relationship between angular divergence and cosine similarity. |
| H3 (Cross-domain) | **Falsified** | R4b shuffled control: 0/9 SAE-hierarchy combinations show real > shuffled p95. Real/shuffled ratio range 0.89-1.43x. Probe quality gate: best 59.5% (gate 85%). All 3 hierarchies fail. | High (0.85) | Reframed (pending) | Collapsed: the R3 "3x random" signal was an artifact of bridge-model probe noise, not genuine cross-domain absorption. Intra-RAVEL rho=0.924 may be artifactual (trivially driven by absorption-rate scaling with SAE width). |
| H4 (Three subtypes) | **Confirmed (with major caveat)** | KW p=0.0002 at L12-65k (tau=0.3); late>early at all 5 thresholds; early dominance 72-75%. But: early% at tau=0.2 drops to 32.3% (L12-65k); at tau=0.4 rises to 95.4%. | Medium-high (0.75) | Partially confirmed | The late>early ordering is robust. But the "75% early dominance" headline is threshold-contingent, not a natural constant. |
| H5 (ITAC efficacy) | **Falsified** | 3.14% FN reduction vs 20% target. 0% at L12-16k. R4d (real activations) not completed. Only 1/10 targets showed notable improvement (18.9%). | Very high (0.95) | Falsified | No change from R3. ITAC repositioning as "proof-of-concept" is the honest framing. |
| H6 (Scaling sign reversal) | **Falsified** | Partial rho(width, absorption|L0) = +0.37. L0 range 65-72 across canonical SAEs (insufficient variation for confound test). | Very high (0.95) | Falsified | No change from R3. The test was underpowered by design (canonical SAEs fix L0 per layer). |

---

## 2. Surprise Analysis

### Surprise 1: EDA = 1 - Decoder Cosine Similarity (Delta AUROC = 0.0)
**Deviation from expectation:** 100%. We expected D-EDA (directional residual decomposition) to provide information beyond scalar EDA. Instead, EDA and the negated decoder-encoder cosine similarity are exactly the same quantity (correlation r=-1.0 on all three architectures: Gemma 2B, GPT-2 Small, Llama-3.1-8B).

**Wrong assumption traced:** We assumed the "directional residual decomposition" -- projecting the encoder-decoder residual onto the decoder dictionary -- would capture *which* features the residual points toward, beyond the scalar magnitude of misalignment. This was a mathematical error. EDA(j) = 1 - cos(w_enc_j, d_j) *is* the negated cosine similarity by definition. The D-EDA decomposition r_j = w_enc - projection(w_enc onto d_j) has norm = sin(theta) where theta = arccos(1 - EDA), so D-EDA's magnitude is a monotone transformation of EDA. Any rank-based metric (AUROC) must yield identical values.

The R3 analysis that reported D-EDA AUROC = 0.762 on GPT-2 L10 vs EDA AUROC = 0.336 was using D-EDA's *absorption indicator* (a more complex derived quantity involving sparse projection and cosine analysis), not the raw D-EDA norm. R4's direct-label validation confirmed: when comparing EDA and dec_cosine as raw predictors, delta AUROC = 0.0 exactly.

**Implication:** D-EDA must be dropped as a separate contribution entirely. The paper should state that EDA = 1 - dec_cosine and connect it to the existing SAEBench metric. The novelty claim shifts from "new metric" to "systematic regime characterization of an existing metric + formal lower bound theorem."

### Surprise 2: H3 Cross-Domain Absorption Completely Collapsed Under Shuffled Control
**Deviation from expectation:** >100%. R3 showed "all 18 RAVEL measurements above 3x random baseline" and "intra-RAVEL rho = 0.924." R4 shuffled control showed real rates are statistically indistinguishable from shuffled null (ratio range 0.89-1.43x, 0/9 combinations exceed shuffled p95).

**Wrong assumption traced:** We assumed the R3 "3x random baseline" signal reflected genuine hierarchical absorption in RAVEL entity-attribute domains. Instead, the signal was entirely a probe-quality artifact. Bridge-model probes (Qwen2.5-0.5B or GPT-2 Medium) projected to Gemma's activation space via random orthonormal matrices produce random directions in Gemma SAE space. Both real and shuffled hierarchy labels generate nearly identical absorption rates because neither the real nor the shuffled probe directions encode meaningful semantic structure in Gemma's residual stream.

The R3 "3x random baseline" was comparing bridge-model noise against *uniform* random noise, not against the actual null of "non-hierarchical but still structured" directions. Any structured noise (e.g., a direction in Gemma space that happens to align with high-norm decoder columns) will produce absorption-like false positives at rates above truly random baselines.

**Implication:** The cross-domain contribution -- previously presented as the "cleanest novelty claim" -- is now the weakest. The R3 intra-RAVEL rho=0.924 (n=6 configs) cannot be trusted either, because: (a) at R4's n=3 configs, both real and shuffled produce rho=1.0 (trivially driven by absorption-rate scaling with width), and (b) the R3 larger-n result used the same flawed bridge-model probes. The paper should present intra-RAVEL coherence as a suggestive observation pending validation with same-model probes, not as a confirmed finding.

### Surprise 3: "75% Early Dominance" is Threshold-Contingent (32.3% to 95.4%)
**Deviation from headline:** ~40%. The "75% early absorption" finding at tau=0.3 was presented as stable and robust. The threshold sensitivity analysis reveals it varies enormously:
- At tau=0.2 (L12-65k): early=32.3%, late=33.8%, partial=33.8% (approximately equal thirds)
- At tau=0.3 (L12-65k): early=72.3%, late=13.8%, partial=13.8% (the headline)
- At tau=0.35 (L12-65k): early=81.5%, late=9.2%, partial=9.2%
- At tau=0.4 (L12-65k): early=95.4%, late=3.1%, partial=1.5%

Note: L12-16k shows no threshold sensitivity at all (75% early at tau=0.2 through 0.35) because n=16 is too small to resolve distribution shifts.

**Wrong assumption traced:** We assumed the cosine threshold for classifying latents as having a "present decoder" (max cos(d_k, parent_probe) >= tau) had a natural, defensible scale. Instead, the threshold is a researcher degree of freedom that dominates the headline finding. The data-driven threshold (p95 of random cosine similarity in d=2304) is approximately 0.049, far below all tested fixed thresholds. Any threshold substantially above random noise classifies most latents as "decoder-absent" simply because random unit vector pairs in high-dimensional spaces have near-zero cosine similarity.

**Implication:** The "75% early dominance" cannot be presented as a natural constant. The paper must include the full threshold sensitivity curve and frame the finding as: "Under standard decoder-matching thresholds (tau=0.3), early absorption dominates; the exact prevalence is threshold-dependent (32-95%), but the qualitative finding that a substantial fraction of absorbed latents lack corresponding decoder directions is robust across all thresholds." The late>early EDA ordering holds at all 5 thresholds tested -- that is the robust result. The prevalence numbers are not.

### Surprise 4: GPT-2 L10 Shows Reversed EDA Direction (Absorbed < Non-Absorbed)
**Deviation from expectation:** ~200%. At GPT-2 L10, absorbed latents have *lower* mean EDA than non-absorbed (Cohen's d = -0.37, AUROC = 0.34). This is not merely "EDA fails to detect absorption" but "EDA actively points in the wrong direction."

**Wrong assumption traced:** The lower bound theorem states EDA >= f(delta) for absorbed latents and EDA = 0 for monosemantic non-absorbed latents at the global minimum. But the theorem assumes convergence to the global minimum for non-absorbed latents. In practice, many non-absorbed latents are themselves polysemantic and have high EDA for reasons unrelated to absorption. At GPT-2 L10, the polysemanticity-driven EDA baseline in non-absorbed latents apparently exceeds the absorption-driven EDA in absorbed latents.

The polysemanticity ablation at L12-16k supports this: EDA AUROC in the monosemantic feature-density stratum is 0.64 vs 0.92 in the polysemantic stratum. Polysemantic absorbed latents are the easiest to detect because they have both absorption and polysemanticity elevating EDA simultaneously. When absorbed latents happen to be monosemantic while the background is polysemantic, the EDA direction reverses.

**Implication:** EDA is not a pure absorption signal. It is a composite of absorption signal and polysemanticity noise, and the signal-to-noise ratio varies by layer. This is much weaker than "EDA detects absorption."

---

## 3. Mental Model Revision

**Before (pre-R4):** We understood SAE absorption as a geometric phenomenon (encoder-decoder misalignment) that is detectable weight-only via EDA, generalizable across feature hierarchies, correctable at inference time, and taxonomizable into three subtypes dominated by early absorption at ~75%.

**After (post-R4):** The data tells a fundamentally different story on three of these four fronts.

1. **We assumed EDA was a general-purpose detector.** The data shows it is a regime-specific diagnostic whose signal depends critically on the interaction between absorption subtype and background polysemanticity. In monosemantic strata, EDA AUROC drops to 0.47-0.64. EDA works when absorption and polysemanticity are correlated (producing a joint elevation). It fails when they are not. The 3/8 config pass rate is not "EDA partially works" -- it is "EDA works only in a narrow favorable regime."

2. **We assumed cross-domain absorption was the strongest novelty claim.** R4's shuffled control demonstrates it was the weakest. The entire cross-domain signal collapsed when proper controls were applied. The field's reliance on the first-letter task is not because other tasks are hard to study -- it is because the measurement infrastructure (probes trained on the same model) is a hard prerequisite that we could not satisfy due to model access restrictions.

3. **We assumed the 75% early-dominance was a stable finding.** It is threshold-contingent. What IS stable is the qualitative finding that many absorbed latents lack a corresponding decoder direction, and the ordering late-EDA > early-EDA. Whether "many" means "a third" or "three-quarters" depends on definitional choices.

What held up:
- The lower bound theorem is validated (synthetic AUROC=1.0, real group separation at favorable configs)
- The late>early EDA ordering is robust across all thresholds and configs with adequate n
- The observation that absorption subtypes exist and have distinct geometric signatures
- The EDA=1-dec_cosine identity, while not novel, provides a clean theoretical connection

**New understanding:** The paper's contribution is not a "tool" (EDA as detector) but a "characterization" (mapping the landscape of SAE failure modes and their geometric signatures). The strongest scientific finding is that absorption is not one phenomenon but (at least) two geometrically distinct failure modes: decoder-absent (the SAE never learned the feature) and decoder-present-encoder-suppressed (the SAE learned it but the encoder drifted). This distinction has direct implications for mitigation: the first requires retraining, the second is potentially correctable at inference time.

---

## 4. Reframing Test

**Original research question (paraphrased):** "Can we detect, characterize, and correct SAE absorption using weight-based metrics?"

**Would we frame it the same way today?** No. The three-verb structure (detect, characterize, correct) promised too much. The detection story is regime-specific and confounded by polysemanticity. The correction story (ITAC) failed at 3% vs 20% target. Only the characterization story holds.

**Revised research question:** "What is the geometric structure of SAE feature absorption, and what does it reveal about why SAEs fail to learn hierarchical features?"

This reframing shifts the contribution from a practical tool (EDA detector + ITAC correction) to a scientific characterization (absorption subtypes and their implications for SAE training). The strongest findings -- the three-subtype taxonomy, the observation that many absorbed latents correspond to never-learned features, and the regime-dependence of weight-based detection -- are all characterization results. They tell the field something about *why* absorption happens, not just *where* it happens.

The revised question also removes the implicit promise of cross-domain generalization, which collapsed. It focuses on what the geometry tells us about the optimization dynamics of SAE training.

---

## 5. New Hypothesis Generation

### New H1: Polysemanticity-Absorption Interaction Hypothesis
**Statement:** EDA's discriminative power for absorption is mediated by the background polysemanticity level. EDA AUROC conditional on polysemanticity stratum should be consistently higher in polysemantic strata (where EDA elevation from absorption is additive with polysemanticity elevation) than in monosemantic strata.

**Evidence basis:** L12-16k feature-density stratification shows EDA AUROC = 0.64 in monosemantic half vs 0.92 in polysemantic half. But L12-16k decoder-neighbor stratification shows the *reverse*: monosemantic AUROC = 0.90 vs polysemantic AUROC = 0.66. These contradictory results (depending on polysemanticity proxy) suggest the interaction is real but its direction depends on how polysemanticity is operationalized.

**Falsification experiment:** Compute EDA AUROC by polysemanticity quartile using both proxies (feature density, decoder neighbor count) across all 8 SAE configs with labels. If the same proxy consistently shows the same direction of interaction across configs, the hypothesis is supported with that proxy. If both proxies show inconsistent patterns across configs, the interaction is an artifact of proxy choice. Estimated cost: 2 GPU-hours using cached data.

### New H2: Threshold-Independent Taxonomy via Rank-Based Partition
**Statement:** The three-subtype taxonomy can be defined in a threshold-free manner using the *rank* of max(cos(d_k, parent_probe)) within the absorbed-latent population rather than a fixed cosine threshold. A rank-based partition (e.g., bottom quartile = early, top quartile = late, middle = partial) would produce prevalence estimates stable across SAE configs and consistently significant KW statistics.

**Evidence basis:** The current threshold sensitivity shows early% varies from 32% to 95% across tau=0.2-0.4. The rank ordering of absorbed latents by decoder-parent cosine similarity is presumably stable. A rank-based partition anchors the taxonomy to the empirical distribution rather than to an arbitrary dimensional scale.

**Falsification experiment:** For each SAE config, compute percentile ranks of max decoder-parent cosine similarity. Partition at 25th/75th percentiles. Test KW significance across configs. If KW p < 0.01 for all configs with n >= 30, the rank-based taxonomy is robust. Estimated cost: 0.5 GPU-hours (analytical, no new data needed).

### New H3: Width-Dependent Early Absorption Prevalence
**Statement:** If early absorption is truly dictionary coverage failure, its prevalence should decrease monotonically with SAE width at matched layers. Wider dictionaries have more capacity to represent parent features, so fewer absorbed latents should be decoder-absent.

**Evidence basis:** This is the prescriptive implication of the early-dominance finding. The H6 scaling analysis found wider SAEs have MORE total absorption (rho = +0.37), but it did not separate by subtype. It is possible that wider SAEs have less *early* but more *late* absorption (more parent features learned creates more opportunities for encoder suppression). At tau=0.3: L12-16k early=75% (n=12/16) vs L12-65k early=72.3% (n=47/65) -- nearly identical, but L12-16k has too few observations. At tau=0.2: L12-16k early=75% vs L12-65k early=32.3% -- dramatically different, but this may reflect the threshold artifact rather than a width effect.

**Falsification experiment:** Run the taxonomy pipeline on all 6 Gemma configs (currently only L12-16k and L12-65k have taxonomy data). Compare early% across 16k vs 65k at matched layers (L5, L12, L19). If early% decreases from 16k to 65k at all three layers, the dictionary coverage mechanism is supported. If early% is constant or increases, the early-dominance interpretation as "dictionary coverage failure" is undermined. Estimated cost: ~1 GPU-hour with cached labels.

---

## 6. Summary of Belief Updates

### What we should stop claiming:
1. EDA is a "reliable absorption detector" -- it is a regime-specific diagnostic (3/8 configs pass, one config shows reversed direction)
2. D-EDA provides "complementary signal" -- it is mathematically identical to EDA (EDA = 1 - dec_cosine)
3. Cross-domain absorption is "confirmed" or even "partially confirmed" -- it is unvalidated (shuffled control: 0/9 pass)
4. "75% of absorption is early-type" as a stable finding -- it is threshold-contingent (32-95% depending on tau)
5. ITAC is a practical correction method -- it achieves 3% vs 20% target on synthetic activations

### What we should start claiming:
1. The EDA lower bound theorem is validated (synthetic AUROC=1.0, real group separation at favorable configs)
2. EDA works in a specific regime (16k-width, mid-layers, L5-L12) and its regime-dependence is itself scientifically informative
3. The late>early EDA ordering is robust across all thresholds and configs with adequate sample size
4. Many absorbed latents lack corresponding decoder directions; the fraction is definitionally sensitive but the phenomenon is real
5. EDA = 1 - dec_cosine (connecting absorption geometry to existing SAEBench infrastructure, validated on 3 architectures)
6. ITAC's failure confirms that the majority of absorption is not amenable to inference-time encoder correction

### What remains genuinely open:
1. Whether cross-domain absorption exists at meaningful levels beyond first-letter (requires same-model probes for Gemma 2B or Llama)
2. Whether early-absorption prevalence decreases with SAE width (requires taxonomy on all 6 configs)
3. Whether the polysemanticity-EDA interaction is causal or an artifact of proxy choice
4. Whether a threshold-independent taxonomy definition produces more stable prevalence estimates
5. Whether D-EDA's "absorption indicator" (the complex derived quantity, not the raw norm) captures genuine directional signal at deep layers

---

## 7. Honest Assessment for the Writing Phase

The paper has two genuinely solid results and several honestly reported negative results.

**Solid Result 1: EDA Lower Bound Theorem + Regime Characterization.** The theorem is mathematically correct and empirically validated on synthetic data (AUROC=1.0). The practical detector achieves AUROC 0.65-0.78 in the favorable regime (16k-width, mid-layers) across two model families (Gemma 2B proxy labels, GPT-2 Small direct labels). The regime-dependence -- EDA's failure at 65k width and deep layers -- is itself informative about absorption geometry.

**Solid Result 2: Three-Subtype Taxonomy + Early Absorption Observation.** The taxonomy reveals that absorbed latents are geometrically heterogeneous. The late>early EDA ordering holds robustly across all thresholds (5/5) and configs with adequate n. The observation that many latents lack corresponding decoder directions -- regardless of the exact percentage -- reframes absorption as partly a training coverage problem. This is conceptually the most important result.

**Critical risks for writing:**
- **EDA=1-dec_cosine identity:** The "new metric" is actually an existing metric re-derived from first principles. A reviewer may object that formalizing something already known is not a contribution. The defense requires showing that no one has previously (a) proved a lower bound, (b) characterized the regime-dependence, or (c) connected it to subtypes. This defense requires careful framing.
- **Cross-domain collapse:** The paper loses what was presented as its strongest novelty claim. The two-contribution structure (EDA characterization + taxonomy) must be sufficient for the target venue.
- **Sample sizes:** L12-16k has only n=16 absorbed latents. All taxonomy and ITAC statistics on this config are severely underpowered. L12-65k (n=65) is the only config with adequate statistical power for taxonomy claims.
- **Threshold sensitivity:** If the threshold sensitivity figure is prominent, it may cause reviewers to question whether the taxonomy is "real" or an artifact. The defense is the threshold-robust ordering (late>early at all thresholds), but the prevalence numbers -- which drive the "early dominance" narrative -- are not robust.
