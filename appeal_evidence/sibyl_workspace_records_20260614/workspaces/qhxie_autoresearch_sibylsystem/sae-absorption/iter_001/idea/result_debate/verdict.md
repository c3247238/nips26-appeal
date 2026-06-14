# Result Debate Verdict (Post-R4 Update)

**Date**: 2026-04-13
**Decision**: PROCEED — run Backup A (~1–2 GPU-hours), then write
**Supersedes**: 2026-04-12 verdict (pre-R4 data)

---

## Overall Score: 5.0 / 10

The experimental program yielded two solid contributions and a definitive negative result landscape. Four of six original hypotheses are falsified, including the previously highlighted cross-domain contribution (H3) which is now unambiguously falsified by R4's shuffled control. The paper is publishable at workshop or mid-tier venue level, contingent on honest framing and targeted pre-writing additions.

---

## Key Conclusions (What We Actually Learned)

**1. EDA is not a novel metric — it is a formal theorem on an existing one.**
EDA = 1 − cos(w_enc, d_dec) is mathematically identical to the negated decoder cosine similarity already present in SAEBench, confirmed at Pearson r = −1.000 across all three model families. The theorem (EDA >= delta^2 * sin^2(theta) / (2 + delta^2)) is genuinely novel and provides the first theoretical justification for why decoder cosine similarity correlates with absorption. The contribution is the theorem plus a regime map, not a new metric formula.

**2. The decoder cosine metric (= EDA) detects absorption only in a specific, characterizable regime.**
3 of 8 tested configurations pass AUROC >= 0.65: Gemma L5-16k (0.698, proxy labels), Gemma L12-16k (0.776, proxy labels), GPT-2 L6 (0.650, direct labels from canonical FeatureAbsorptionCalculator). The passing configs share the profile: 16k-width dictionaries, layers 5–12. GPT-2 L10 shows reversed direction (AUROC = 0.344), revealing a boundary condition: the metric anti-predicts absorption when background polysemanticity among non-absorbed latents exceeds absorption-driven EDA elevation. The regime-dependence is itself a finding, not merely a limitation.

**3. H3 cross-domain absorption is definitively falsified.**
The R4 shuffled hierarchy control is unambiguous: real absorption rates in RAVEL entity-attribute hierarchies are statistically indistinguishable from shuffled null for all 9 domain-SAE combinations (0/9 exceed shuffled p95; real/shuffled ratio range 0.89–1.43). The R3 intra-RAVEL coherence (rho = 0.924) is artifactual — shuffled labels produce rho = 1.0 trivially. Cross-domain absorption characterization requires same-model probes on the target model (Gemma 2B or Llama-3.1-8B), which remain HF-gated. This was the largest downgrade from the 2026-04-12 synthesis.

**4. The three-subtype taxonomy reveals absorption heterogeneity — but the dominant percentage is threshold-contingent.**
At the canonical threshold (tau = 0.3), ~72–75% of absorbed latents on both tested configs are early-type (no corresponding decoder direction — dictionary coverage failure). The late > early EDA ordering is robust across all 5 thresholds. The 75% early-dominance figure varies from ~32% (tau = 0.2) to ~95% (tau = 0.4) on L12-65k; the robust finding is the ordering, not the prevalence. The practical implication holds at any threshold above ~0.22: most absorbed latents are structurally resistant to inference-time encoder corrections, pointing toward training-time dictionary solutions (Matryoshka SAE, wider dictionaries, hierarchically-aware training objectives).

**5. ITAC failure is confirmatory, not merely negative.**
ITAC achieves 2.69–3.14% mean FN reduction against a 20% target (H5 falsified). The failure is mechanistically explained by the taxonomy: ITAC targets encoder suppression and is structurally inapplicable to early-type absorbed latents (~75% of cases). The single notable success case (22.7% FN reduction on one late-type latent) confirms the mechanism works when the taxonomy says it should. ITAC and the taxonomy validate each other from orthogonal directions.

---

## Action Plan (Prioritized)

| Priority | Action | GPU Cost | Impact |
|----------|--------|----------|--------|
| **1 — PRE-WRITING** | Run Backup A (amortization gap experiment): fix Gemma L12-16k decoder; compare feedforward vs. OMP vs. 2-pass encoding absorption rates | 1–2 hrs | Potential third contribution; adjudicates Tang vs. O'Neill mechanism; either outcome publishable |
| **2 — PRE-WRITING** | Extend taxonomy to GPT-2 L6 and L10 configs (direct labels already available) | ~0.5 hrs | Strengthens taxonomy to cross-model; minimal additional cost |
| **3 — WRITING** | Write as two-contribution paper (+ Backup A as potential third) with honest negative-results section | 0 GPU hrs | Paper framing, credibility, venue fit |
| **4 — BACKGROUND** | Continue pursuing Gemma 2B and Llama-3.1-8B HF access | 0 GPU hrs | If granted: upgrade proxy-label AUROCs to direct; potential taxonomy extension; revision-stage |

**Total remaining compute before writing**: 1.5–2.5 GPU-hours.

---

## What the Paper Looks Like

**Title direction**: "Characterizing SAE Feature Absorption: A Formal Lower Bound and Geometric Subtype Taxonomy"

**Primary Contributions**:
1. **Theorem 1 (EDA Lower Bound)**: First formal proof that decoder cosine similarity provides a lower bound on absorption degree under biconvex optimization; synthetic validation (AUROC = 1.0, F1 = 0.974); real-data confirmation of group separation (Gemma L12-16k, Cohen's d = 1.02, p < 1e-4).
2. **Regime Characterization**: The decoder cosine metric achieves AUROC 0.65–0.78 in 16k-width, mid-layer (5–12) SAEs across Gemma 2B (proxy labels) and GPT-2 Small (direct labels). Regime failure at 65k-width and deep layers is mechanistically explained by early-type dominance. Boundary condition: metric reverses direction (AUROC < 0.5) when background polysemanticity exceeds absorption-driven misalignment.
3. **Three-Subtype Taxonomy**: First mechanistic classification of SAE absorbed latents into early (decoder-absent), late (decoder-present, encoder-suppressed), and partial subtypes. Late > early EDA ordering robust across all tested thresholds. At tau = 0.3, early-type dominates (~72–75% on both configs tested). Reframes absorption mitigation from encoder fixes to dictionary-coverage solutions.
4. **Backup A result** (contingent): First controlled experiment isolating the amortization gap vs. loss landscape contributions to absorption causation.

**Honest Negatives** (supplementary):
- H3 (cross-domain): falsified; the measurement infrastructure is established and the question remains open pending same-model probe access.
- H5 (ITAC): falsified; confirms early-dominance finding from a complementary angle.
- H6 (scaling): falsified; canonical SAEs provide insufficient L0 variation to test this hypothesis.
- H2 (D-EDA): mathematically redundant; the complex derived indicator may merit future investigation at deep layers.

**Venue Target**:
- NeurIPS 2026 MI Workshop: RECOMMENDED (strong fit with honest characterization work and negative results).
- EMNLP 2026 main: POSSIBLE with expanded taxonomy evidence (GPT-2 extension) and strong framing.
- NeurIPS/ICML/ICLR main: NOT recommended given current evidence base (thin taxonomy sample, unverifiable Gemma AUROCs).
