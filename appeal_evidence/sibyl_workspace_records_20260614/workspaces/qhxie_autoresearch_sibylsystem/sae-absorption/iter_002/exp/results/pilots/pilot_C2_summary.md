# Task C.2: Cross-Domain Absorption Measurement — Pilot Summary

**Date:** 2026-04-13
**Mode:** PILOT
**Status:** NO_GO (for existence of absorption in non-spelling hierarchies)
**Version:** v3 (empirical concept latent discovery)

## Key Findings

### 1. Concept Latents Are Reliably Identified and Active

For all tested hierarchies, empirically-discovered concept latents (via differential activation)
fire reliably for their respective concept words:

| Hierarchy | Concept Latents Found | Mean Max Activation (Test) | Absorption Rate | Random Null |
|---|---|---|---|---|
| first_letter ('c') | 20 | 21.9 | 0.00 | 1.00 |
| animate_inanimate | 20 | 21.6 | 0.00 | 1.00 |
| noun_proper | 20 | 16.2 | 0.00 | 1.00 |
| city_country (FLAGGED) | 20 | ~8.0 | 0.70 | 1.00 |

**Interpretation:** absorption_rate=0.0 means concept latents ALWAYS activate for concept words.
This is NOT absorption — it is proper encoding. Random latents never activate (absorption_rate=1.0).

### 2. No Cross-Domain Absorption Detected

The ratio_to_null for all clean hierarchies = 0.0 (concept latents fire MORE than random).
This means: in GPT-2 Small L6, the animate, proper_noun, and letter-'c' SAE latents
reliably encode their respective semantic concepts. No absorption of child features into parent is observed.

### 3. City_Country Binary Shows Some Non-Encoding (FLAGGED)

US-city latents fail to activate for 70% of US test cities (vs. 100% for random latents).
However: this hierarchy FAILED the shuffled-label control in C.1 (shuffled F1 > 0.6).
Results are non-inferential.

## Methodological Notes

### v1/v2 Failure: Decoder Cosine Alignment
- Initial approach found latents via decoder cosine similarity with probe direction
- These latents had cos_sim ~0.25 but mean activation ~0.000
- They are NOT the latents that actually encode the semantic concept
- **Lesson**: Decoder cosine alignment is insufficient for cross-domain concept latent identification

### v3 Success: Empirical Differential Activation
- Find latents by differential activation: concept words vs. control words
- These latents have mean activation 2-21 for concept words
- Correctly identifies the semantic latents

### What C.2 Is Actually Measuring
The measurement is: "do parent concept latents fail to fire for child tokens?"
This is NOT the traditional Chanin et al. absorption definition, which is:
"does the child-specific feature fail to fire because it's absorbed into the parent?"

The correct C.2 measurement would require:
1. Identify a specific child SAE feature (e.g., the 'dog' SAE feature)
2. Check if that feature fails to fire when 'dog' tokens appear, because it's
   "absorbed" into the parent 'animate' feature
3. This requires identifying specific child features, which is a harder problem

## Pilot Pass Criteria Assessment

**Pass criterion:** ratio_to_null >= 1.5 for at least 1 non-spelling hierarchy

- noun_proper: ratio = 0.00 — FAIL (concept latents DO fire, no absorption)
- animate_inanimate: ratio = 0.00 — FAIL (concept latents DO fire, no absorption)

**Verdict:** NO_GO (existence proof of cross-domain absorption not established with this approach)

## Honest Reporting

This is a genuine negative result with scientific value:
- GPT-2 Small L6 SAE has reliably-activating semantic latents for animate/proper_noun hierarchies
- These latents do NOT show absorption behavior (they fire consistently)
- This contrasts with the first-letter finding where absorption WAS found in iter_001

**Recommendation for full experiment:**
1. Adopt the correct absorption measurement (child-feature suppression, not parent-latent failure)
2. This requires identifying specific child features per word (lexical-level SAE features)
3. Then measuring if those child features fail to fire for words where the parent latent is active
4. This is a more complex but scientifically sound measurement
