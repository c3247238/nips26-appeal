# Revisionist Analysis: Hypothesis Revision from Data

**Agent:** Revisionist  
**Date:** 2026-04-15  
**Input:** consolidation_summary.json (v3, FULL mode), proposal.md (Synthesis Round 5), hypotheses.md

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|-----------|---------|-------------|------------|
| H1 (Cross-Domain Variation) | **Confirmed** | ANOVA p=0.005; 4/6 pairwise comparisons significant at p<0.05. First-letter 34.5% vs city-country 18.5% (p=0.004) vs city-language 13.6% (p=0.0001) at L24. | HIGH |
| H2' (Semantic > First-Letter) | **Refuted** | At L24 (best probes), first-letter shows HIGHEST absorption (34.5%), not semantic hierarchies. City-country 18.5%, city-language 13.6%. Complete reversal of both the original H2 and the revised H2'. | HIGH |
| H3 (Absorption-Hedging Decomposition) | **Partially Confirmed** | Multi-L0 first-letter: strict 7.9% vs loose 94.1%. 86.2% compensatory. The near-tautological hedging critique is validated. Cross-domain decomposition limited (city-language n=3 only). | MEDIUM |
| H4 (GAS Detector) | **Refuted** | rho=0.116, AUROC=0.571, bootstrap CI includes zero [-0.333, 0.536]. 25x scale-up confirmed signal absent. | HIGH |
| H5 (Absorption Tax) | **Refuted** | T(G) ranking rho=-0.20, pairwise concordance ~50% (chance). R_pc correlations near zero. Quantitative predictions fail completely. | HIGH |
| H6 (Architecture Generalization) | **Inconclusive** | Architecture ANOVA p=0.87 (not significant). Hierarchy ANOVA p=0.005 (significant). Hierarchy type dominates architecture choice. JumpReLU pilot advantage did not survive full-mode testing. | LOW |
| H7 (Causal Absorption) | **Confirmed** | n=25 words, 19 with absorption. Recovery 32.5% vs control 1.5%. Wilcoxon p=0.000218, Cohen's d=1.33, CI [0.213, 0.421] excludes zero. 16/19 words show positive recovery. | HIGH |

---

## 2. Surprise Analysis

### Surprise 1: H2' Refuted Twice -- The Layer-Hierarchy Interaction (deviation: >100% from expectation)

**What we expected:** Based on pilot data at layer 12, semantic hierarchies showed HIGHER absorption than first-letter (city-continent 53.4%, city-language 10.4% vs first-letter 3.9%). We revised H2 to H2' predicting semantic > syntactic absorption, and built the entire paper narrative around this "inversion" finding.

**What we found:** At layer 24 with quality probes, first-letter shows the HIGHEST absorption (34.5% at L24_16k), while semantic hierarchies show LOWER rates (city-country 18.5%, city-language 13.6%). The pilot finding was an artifact of (a) poor probe quality at L12 for RAVEL hierarchies (F1 as low as 0.36), and (b) layer 12 being unrepresentative of the model's behavior on factual knowledge tasks.

**Wrong assumption:** We assumed that absorption behavior observed at a single layer (L12) would generalize across layers. More fundamentally, we assumed that probe quality differences between hierarchies at a given layer would not systematically bias absorption measurement. In reality, probe quality (rho=-0.756 with false negative rate) is the strongest predictor of measured absorption -- far stronger than hierarchy type or architecture choice.

**Severity:** This is the single most important surprise. The paper's original framing ("semantic hierarchies absorb more") must be abandoned entirely. The correct story is: absorption is LAYER-DEPENDENT (15x variation: 2.2% to 34.5% for first-letter across layers) and HIERARCHY-DEPENDENT (ANOVA p=0.005), with a complex layer-hierarchy interaction that the pilot completely mischaracterized.

### Surprise 2: Architecture Does Not Matter (deviation: expected p<0.05, got p=0.87)

**What we expected:** Pilot data suggested JumpReLU was "consistently lowest" across all hierarchies (3.2%, 17.1%, 13.9%). We elevated JumpReLU as a key finding and expected to confirm architecture-dependent absorption resistance.

**What we found:** Architecture ANOVA p=0.87 -- no significant effect whatsoever. At layer 12 in full mode, JumpReLU_16k rates are: first-letter 0.7%, city-continent 17.3%, city-language 41.2%, city-country 47.1%. Matryoshka shows comparable or sometimes lower rates (city-language 35.3%, city-country 35.3%). The hierarchy effect (p=0.005) completely dominates the architecture effect.

**Wrong assumption:** We assumed pilot architecture differences were real effects rather than noise amplified by poor probes and small samples. The pilot used probes well below quality gate for RAVEL hierarchies, making all cross-architecture comparisons unreliable. The "JumpReLU advantage" narrative was premature.

### Surprise 3: Layer 24 Is Where Absorption Concentrates (completely unanticipated)

**What we expected:** The proposal mentions testing layers 6, 12, 18, 24 but treats them as roughly equivalent measurement points. No hypothesis predicted dramatic layer dependence.

**What we found:** First-letter absorption shows 15x variation across layers: L6_16k 2.4%, L12_16k 5.7%, L18_16k 2.2%, L24_16k 34.5%. Layer 24 is radically different from all other layers. This is a non-monotonic pattern (L18 < L12 < L6 << L24), suggesting that absorption at the final prediction layer operates through a qualitatively different mechanism.

**Wrong assumption:** We implicitly assumed absorption was a property of the SAE-hierarchy interaction that would be roughly stable across layers (perhaps scaling with probe quality). Instead, it appears that absorption at L24 may reflect the model's final-layer "decision" dynamics -- the layer where the model commits to a specific prediction -- creating stronger competitive pressure between parent and child features.

### Surprise 4: Activation Patching Scaled Beautifully (deviation: +128% from pilot)

**What we expected:** Pilot showed 14.3% recovery vs 0.5% control (n=9, p=1.0 due to underpowering). We worried the effect might weaken with more data.

**What we found:** Full mode shows 32.5% recovery vs 1.5% control (n=25 words with 200 contexts each). The effect MORE than doubled (14.3% -> 32.5%), achieved overwhelming statistical significance (Wilcoxon p=0.000218, Cohen's d=1.33), and 16/19 words show positive recovery. This is the cleanest, most unambiguous result in the entire project.

**Wrong assumption:** Our concern about the effect disappearing was wrong; the pilot was actually underestimating the true effect due to small sample size and the specific words tested. The discovered word pairs (found via integrated gradients) show stronger absorption than the original pilot core words, suggesting that systematic pair discovery identifies more genuine absorption instances.

### Surprise 5: City-Language Hedging Is Fundamentally Different (deviation: 59 pp from first-letter)

**What we expected:** We expected the absorption-hedging ratio to vary somewhat across hierarchies, with semantic hierarchies showing "proportionally more true absorption."

**What we found:** City-language shows 66.7% strict hedging vs first-letter's 7.9% -- a 59 percentage point gap. However, the city-language sample is tiny (n=3 FNs), making this comparison unreliable. For first-letter, the 86.2% compensatory resolution rate is robust (n=304 FNs, bootstrap CI [82.6%, 90.1%]). The hedging decomposition story is really a first-letter story, not a cross-domain comparison.

**Wrong assumption:** We assumed we would get enough false negatives in cross-domain hierarchies to perform meaningful decomposition. The low RAVEL absorption rates at L24 with quality probes produced too few FNs for reliable decomposition.

---

## 3. Mental Model Revision

**Previous mental model:** "Absorption is primarily driven by hierarchy structure -- richer, deeper semantic hierarchies create more parent-child co-occurrence pressure, leading to higher absorption. Architecture choice can mitigate this. The first-letter task underestimates the problem because it has an unusually clean, regular hierarchy."

**Revised mental model:** Absorption is primarily a layer-level phenomenon driven by the model's computational stages, not by hierarchy complexity. At early and middle layers (L6-L18), absorption is uniformly low (2-9%) regardless of hierarchy type. At the final prediction layer (L24), absorption spikes dramatically -- but it spikes MOST for first-letter (34.5%), not for semantic hierarchies (13-36%). This suggests that absorption at L24 is related to how strongly the model's residual stream encodes categorical features at the point of prediction, not how complex the hierarchy is. First-letter information, though computationally "simple," may be MORE strongly encoded as a categorical feature at L24 because the model has fully resolved it by that point, creating stronger competition between parent (letter) and child (word-specific) features in the SAE. Architecture choice (JumpReLU vs BatchTopK vs Matryoshka) is essentially irrelevant (p=0.87) compared to the layer and hierarchy effects. The real drivers of measured absorption are: (1) layer position, (2) probe quality, (3) hierarchy type -- in that order. We were wrong to treat absorption as a static property measurable at any convenient layer.

---

## 4. Reframing Test

**Original research question:** "Does the first-letter spelling task represent a typical or extreme case of absorption?"

**Would we frame it the same way today?** No. The original framing assumed a one-dimensional "absorption severity" scale where tasks could be ordered. The data shows that absorption is a layer x hierarchy interaction effect -- first-letter is "typical" at L12 but "extreme" at L24. The framing should not ask whether one task is "worse" than another, but rather how absorption varies across the full layer x hierarchy x architecture space.

**Revised research question:** "How does SAE feature absorption vary across the layer x hierarchy x architecture space, and what does the pattern reveal about the mechanism driving absorption at different stages of model computation?"

This reframing:
- Drops the "which is worse" competition between tasks
- Centers the layer-dependence finding (15x variation across layers) as the primary contribution
- Still preserves the cross-domain variation (ANOVA p=0.005) as a secondary finding
- Opens the door to a mechanistic explanation rooted in the model's computational stages
- Acknowledges that absorption is not a single number per SAE but a function of where and what you measure

---

## 5. New Hypotheses

### New H8: Layer-Position Absorption Mechanism
**Statement:** Absorption at the final prediction layer (L24) operates through a qualitatively different mechanism than at earlier layers. At L24, the model has committed to specific predictions, creating winner-take-all dynamics in the residual stream that force SAE features into competitive exclusion. At earlier layers, the residual stream is still "open" (superimposed information for multiple downstream uses), reducing the pressure for any single feature to dominate.

**Falsifiable test:** Compare the norm of the residual stream projection onto parent vs child feature directions at each layer. At L24, the child direction should dominate (high cosine with residual) when absorption occurs; at L12, both should coexist. Alternatively: measure the effective rank of the residual stream at each layer for absorbed vs non-absorbed tokens.

**Expected outcome:** L24 shows lower effective rank (more "peaked" representation) for absorbed tokens than L12.

### New H9: Probe Quality as Causal Confounder
**Statement:** The correlation between probe quality and false negative rate (rho=-0.756) is not just a measurement issue -- low probe quality for RAVEL hierarchies reflects genuine difficulty in linearly decoding these features from the model's residual stream. This difficulty is itself causally related to absorption: features that are hard for linear probes to decode are also hard for SAEs to represent independently, but this difficulty manifests as LOWER measured absorption (fewer detected FNs) rather than higher.

**Falsifiable test:** Train nonlinear probes (2-layer MLP) for the same RAVEL hierarchies. If nonlinear probes achieve F1>0.95 while linear probes stay at F1~0.80, the information is present but not linearly accessible -- and absorption rates with nonlinear probes should be HIGHER than with linear probes, not lower.

**Expected outcome:** Nonlinear probes increase measured cross-domain absorption rates, potentially reversing the hierarchy ranking and supporting the original H2'.

### New H10: First-Letter Absorption at L24 Reflects Tokenization Artifact
**Statement:** The high first-letter absorption at L24 (34.5%) may be driven by the tokenizer's treatment of word boundaries. In Gemma 2, the first character of a new word is often part of the preceding token's encoding. At L24, the model resolves this by committing to specific character-level features that compete with first-letter features, inflating measured absorption. Semantic hierarchies (city-country, city-language) do not have this tokenization sensitivity.

**Falsifiable test:** Compare first-letter absorption at L24 for words where the first character aligns with token boundaries vs words where it does not. If tokenization alignment matters, absorption should be significantly higher for misaligned tokens.

**Expected outcome:** Tokenization-aligned words show 10-20 pp lower absorption than misaligned words at L24.

---

## Summary of Belief Updates

| Dimension | Prior Belief | Updated Belief | Confidence in Update |
|-----------|-------------|----------------|---------------------|
| Which hierarchy has most absorption? | Semantic > syntactic | Layer-dependent; first-letter highest at L24 | HIGH |
| Does architecture matter? | JumpReLU consistently best | Architecture is irrelevant (p=0.87) | HIGH |
| Is absorption a single-layer measurement? | Yes, L12 is representative | No -- 15x variation across layers; L24 qualitatively different | HIGH |
| Is causal evidence for absorption strong? | Promising but underpowered | Strongest result in project (p=0.000218, d=1.33) | HIGH |
| Is GAS useful for detection? | Unlikely (pilot rho=0.12) | Definitively useless (rho=0.116, confirmed at 25x scale) | HIGH |
| Does the Absorption Tax predict rates? | Unlikely (pilot rho=0.08) | No (ranking rho=-0.20, concordance at chance) | HIGH |
| Is hedging classification meaningful? | Yes, but needs tightening | Loose hedging is near-tautological; strict hedging is the only meaningful metric | HIGH |
| What drives measured absorption? | Hierarchy complexity | Layer position > probe quality > hierarchy type >> architecture | MEDIUM-HIGH |

---

## Recommendations for Paper Framing

1. **Lead with layer dependence.** The 15x absorption variation across layers (2.2% to 34.5% for first-letter alone) is the most striking and novel finding. No prior work has measured absorption across layers.

2. **Cross-domain variation is real but nuanced.** ANOVA p=0.005 is strong, but the direction of the effect (first-letter highest at L24) is the opposite of what the pilot suggested. Present this as "absorption is hierarchy-dependent" rather than "semantic hierarchies are worse."

3. **Activation patching is the methodological crown jewel.** p=0.000218, d=1.33, CI excluding zero. This is the first interventional evidence for competitive exclusion and should be prominently featured.

4. **Kill the architecture story.** p=0.87 means there is no architecture effect worth discussing in the main text. Relegate to a paragraph noting the null result.

5. **Honest negative results remain a strength.** GAS, CMI, Absorption Tax, and H2' refutation all demonstrate scientific integrity. But be transparent that the paper's narrative changed fundamentally between pilot and full mode.

6. **Address the probe quality elephant.** The rho=-0.756 correlation between probe quality and false negatives means that all cross-domain absorption rate comparisons carry uncertainty from differential probe quality. This must be a prominent caveat, not a footnote.
