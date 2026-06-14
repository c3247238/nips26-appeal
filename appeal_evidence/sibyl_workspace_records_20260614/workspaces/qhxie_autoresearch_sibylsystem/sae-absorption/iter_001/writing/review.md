# Writing Quality Review

## Summary

This paper introduces Encoder-Decoder Alignment (EDA), a weight-only metric for detecting feature absorption in Sparse Autoencoders, validates it on Gemma Scope and GPT-2 SAEs, tests cross-domain generalization via RAVEL entity-attribute hierarchies (finding the null result attributable to probe-quality failure), and establishes a three-subtype absorption taxonomy (early/late/partial) with the central finding that 72–75% of absorbed latents are early-type (dictionary coverage gaps). The abstract and introduction accurately represent the experimental outcomes, including all falsified hypotheses. The writing is specific, numbers-first, and avoids most banned patterns. The paper's largest structural challenge is that Section 5 (cross-domain) must consistently frame the shuffled-control null result while also preserving the legitimate pre-shuffled intra-RAVEL coherence evidence, and the current text mostly succeeds but has one consistency gap requiring a clarifying sentence. Two appendices cited in the main text (Appendix B and C) are absent from paper.md. Figure 1 is missing as a rendered PDF. These three issues should be resolved before submission.

---

## Detailed Assessment

### Structural Coherence: 8/10

The argument flows logically from motivation through theory, validation, cross-domain, taxonomy, and discussion. The Introduction's three-gap framing maps cleanly onto three sections (3–4, 5, 6), and each section's transition sentence does useful work. The section-end transitions are functional ("With EDA validated as a screening tool in favorable regimes, we turn to cross-domain characterization"; "The null result and remaining open questions motivate our second contribution...").

The Introduction's Contribution 2 correctly states: "A proper shuffled hierarchy control shows real absorption rates were statistically indistinguishable from shuffled null rates across all 9 tested domain-SAE config combinations (H3 falsified)." Section 5.2 also correctly states the shuffled control result. Good.

One flow concern: Section 5.3 ("Conditional Intra-Domain Coherence") presents intra-RAVEL rho = 0.924 from the pre-shuffled analysis (phase3e_crossdomain_analysis.json, n=6 SAE configs) as "suggestive existence evidence conditional on replication." However, the shuffled control summary (r4b) was run on a smaller pilot subset (n=3 SAE configs) and notes its own rho = 1.0 is an artifact of small n. The paper does not explain that the intra-RAVEL rho = 0.924 comes from a different, larger experiment than the shuffled control pilot. A reviewer who notices both the n=9 shuffled control (3 hierarchies × 3 configs) and the n=6 configs rho analysis may question why the coherence claim is based on n=6 configs while the null result tested n=3. One clarifying sentence is needed.

The Discussion in Section 7.2 explains the first-letter vs. RAVEL correlation gap with three structural factors. This is clear and well-organized. The Conclusion is accurate and comprehensive without redundancy.

The section ordering (4.3 = Pilot-to-Full Discrepancy, 4.4 = Polysemanticity Stratification) is reversed relative to the outline (outline has 4.3 = polysemanticity, 4.4 = pilot discrepancy). The paper's ordering is more logical given the content — polysemanticity stratification belongs after the main results discussion, not before — so this is a paper improvement over the outline, not an error. However, the outline's section numbering does not match paper.md, which is a minor internal inconsistency.

### Notation & Terminology Consistency: 9/10

Cross-checking all symbols against notation.md:

- EDA(j), w_{e,j}, d_j, z_j, W_e, W_d, d_model, d_SAE, delta, theta_jc — all used exactly as defined. No deviations found.
- r_j (residual), beta (sparse coefficients), S_j (absorbing source set) — correctly defined and used consistently.
- tau = 0.3 for subtype classification — defined in Section 6.1, consistent with notation.md.
- ITAC formula in Equation (6): z_j^corr = max(0, d_j^T (e + z_abs * d_abs)) — matches notation.md exactly.

Cross-checking terms against glossary.md:

- "feature absorption" (lower-case): used correctly throughout after introduction.
- "biconvex SDL loss": introduced correctly on first use in Section 2.1 as "sparse dictionary learning (SDL) loss, which has a biconvex structure."
- "partial minimum": used consistently.
- "early absorption", "late absorption", "partial absorption": correct with correct adjectival forms ("early-type latent", "late-type latent").
- "SAEBench" (one word), "Gemma Scope" (two words, no hyphen): consistent.

One minor issue: Section 5.3 uses "$\rho$" without the $\rho_s$ subscript for Spearman rank correlations (three occurrences). The notation.md specifies $\rho_s$ for Spearman throughout. The abstract also uses "$\rho$ = 0.924" without subscript. The context is unambiguous but the inconsistency should be resolved.

One additional minor issue: The paper's Table 1 caption labels D-EDA AUROC as "AUROC (D-EDA)" while the main text and notation.md use "D-EDA AUROC" — minor formatting inconsistency in the column header.

### Claim-Evidence Integrity: 8/10

All major quantitative claims verified against source data:

- EDA AUROC = 0.776 at L12-16k [0.700, 0.863]: confirmed (phase1_summary: 0.7765, CI [0.700, 0.863]).
- EDA AUROC = 0.629 at GPT2-L6 [0.561, 0.692]: confirmed (phase5: 0.629, CI [0.561, 0.692]).
- DeLong diff at L5-16k = +0.396, p ≈ 0: confirmed (phase1: diff = 0.3963, p = 0.0).
- DeLong diff at L12-16k = +0.553, p ≈ 0: confirmed (phase1: diff = 0.5529, p = 0.0).
- Cohen's d = 1.019 at L12-16k: confirmed (phase1: 1.0187).
- Cohen's d = −0.152 at L12-65k: confirmed (phase1: −0.1523).
- Taxonomy L12-65k: Early 72.3%, Late 13.8%, Partial 13.8%: confirmed (phase2a).
- KW p = 0.0002 at L12-65k: confirmed (phase2a).
- ITAC mean FN reduction 3.14% at L12-65k: confirmed (phase2b: 3.143%).
- Best individual ITAC case j=61217, 18.9%: confirmed (phase2b: 18.859%).
- FVU change −4.23% at L12-65k: confirmed (phase2b: −0.04232 = −4.23%).
- Polysemantic AUROC = 0.922 at L12-16k [0.842, 0.979]: confirmed (ablation_polysemanticity: 0.9225, CI [0.842, 0.979]).
- Monosemantic AUROC = 0.643 at L12-16k [0.518, 0.763]: confirmed (ablation_polysemanticity: 0.6433, CI [0.518, 0.763]).
- Intra-RAVEL rho = 0.943, 0.886, 0.943; mean = 0.924: confirmed (phase3e_crossdomain_analysis.json).
- Shuffled control 0/9 combinations pass: confirmed (r4b_shuffled_control_summary.md: all 9 fail).

One unsupported claim: Section 7.1 states "focusing on the top 5% of latents by EDA contains approximately 56% of absorbed latents (Prec@50 = 0.0035), reducing the supervised evaluation budget by 20×." The Prec@50 = 0.0035 is confirmed, but the "56% of absorbed latents" and "20×" figures are not explicitly sourced in any result file. These can be derived from the data (5% × 16,384 = 819 candidate latents; 819 × Prec@50 ≈ 2.9 absorbed; 2.9 / (16 total positive) ≈ 18%; the "56%" does not follow easily from this). This claim should be either sourced explicitly or restated with the numbers that do derive from Prec@50.

One minor precision note: The paper states "Pearson $r > 0.999$" for the SAEBench cross-validation. The actual values range from 0.9997 to 0.9999. Writing "$r > 0.9997$" would be more informative.

### Visual Communication: 7/10

Seven figures and four tables. Figures 2–7 are referenced before they appear. All seven figure PDFs are present in the `writing/figures/` directory (confirmed via file listing: fig2–fig7 all have PDFs). Tables 1–4 are inline and properly sequenced.

**Critical issue: Figure 1 is missing as a rendered PDF.** Figure 1 is referenced in the Introduction ("Figure 1a, see end-of-paper Figures section") and in Section 3.1 ("As illustrated in Figure 1b"). The file `writing/figures/fig1_absorption_mechanism_desc.md` is a textual descriptor only — no corresponding PDF was generated. At NeurIPS, a missing method diagram for the paper's core concept (the absorption mechanism and EDA geometry) is a significant gap. The "(see end-of-paper Figures section)" parenthetical is non-standard and will likely be flagged by reviewers.

**Issue: Tables lack standalone captions.** All four tables (1–4) use bold labels as headers (e.g., "Table 1: EDA Detection Performance Across SAE Configurations") but have no descriptive caption sentence. NeurIPS convention requires table captions to include a brief description of what the reader should observe. This applies to all four inline tables.

**Issue: Figure 5 and 6 purpose alignment.** After the H3 null result, Figures 5 (cross-domain rates) and 6 (intra-RAVEL coherence) need updated captions that are consistent with the null result framing. If the paper now presents the intra-RAVEL coherence as "suggestive but conditional," Figure 6's caption should not imply the correlations are conclusive. Visual audit noted these figures were generated, but captions should be reviewed for alignment with the final null-result framing.

All figures are referenced before they appear — the visual sequence (theory → synthetic → real SAE → cross-domain → taxonomy) supports the paper's argument structure effectively.

### Writing Quality: 8/10

The prose is specific and numbers-first throughout. Abstract leads with the problem and three gaps. Sections 3–6 consistently open with a claim or finding, then provide evidence.

**Banned patterns checked:**

- "In recent years...": none found.
- "to the best of our knowledge...": none found.
- "Furthermore" / "Moreover" / "It is worth noting": none found.
- "groundbreaking", "revolutionary", "game-changing", "novel" as hype term: none found.
- Vague "significantly outperforms" without numbers: none found. All comparative claims cite specific deltas.

**One clarity gap (Major):** Section 3.2's proof sketch is insufficiently connected. The sketch states "yields a residual with magnitude ≥ δ sin(θ_{jc}). Converting to cosine distance gives the quadratic form in Equation (3)." The algebraic step from a perpendicular residual magnitude ≥ δ sin(θ_{jc}) to the quadratic form δ²sin²(θ_{jc}) / (2 + δ²) is not shown. The paper promises "Appendix B" for D-EDA details but no appendix covers the theorem proof. Since the theorem is Contribution 1's theoretical foundation and is presented as a formal result with a proof sketch, an appendix proof sketch or pointer to the arXiv extended version is needed.

**One redundancy (Minor):** Section 6.3's opening sentence ("Early absorption constitutes 72–75% of all absorbed latents at both tested configurations") repeats what Section 6.2 already established. Remove or rephrase as a transitional recap.

**One ambiguous modifier (Minor):** Section 7.3, third paragraph: "The partial Spearman ρ(width, absorption | L₀) = 0.368, remaining positive and non-significant (p = 0.133)." "Remaining" has no clear referent — remaining compared to what? Revise to: "The partial Spearman ρ(width, absorption | L₀) = 0.368 (positive, p = 0.133, not significant)."

**One sentence that could confuse non-expert reviewers (Minor):** Section 4.3 states "The pilot used a capped subset of 100 latents with enriched positive prevalence; the full 65,536-latent evaluation dilutes positives to 0.024% of the population (n_pos = 16)." The 0.024% = 16/65,536 is correct but the jump from pilot to full is made without explaining what "enriched positive prevalence" means operationally. Add: "...enriched positive prevalence (all positives included, 100 negatives sampled uniformly)..."

---

## Issues for the Editor

1. **[Critical] Figure 1 PDF is absent**: Location — Introduction line "Figure 1a, see end-of-paper Figures section" and Section 3.1 "As illustrated in Figure 1b." The `fig1_absorption_mechanism_desc.md` descriptor exists but no rendered PDF was generated. **Fix**: Generate the tikz diagram (or matplotlib reproduction) to produce `fig1_absorption_mechanism.pdf`. If generation is not feasible before submission, remove the two forward references to Figure 1a/1b and replace with verbal descriptions. The parenthetical "(Figure 1a, see end-of-paper Figures section)" in particular looks non-standard and should be removed if no figure exists.

2. **[Critical] Appendix B and C are absent from paper.md**: The main text references "Appendix B" twice for D-EDA conditioning details (Sections 3.3 and 7.1). The outline describes Appendix C for RAVEL probe details. Neither appendix appears in paper.md. **Fix**: Write a brief Appendix B (1–2 paragraphs on ill-conditioning of the sparse projection when d_SAE >> d_model; the qualitative argument is already in Section 3.3 and needs only formalization) and Appendix C (probe accuracy table for city-continent/city-country/city-language; shuffled control full 9-row table from r4b results).

3. **[Major] Section 5.3 needs one clarifying sentence**: Location — Opening of Section 5.3. The intra-RAVEL rho = 0.924 comes from phase3e (n=6 SAE configs, the full cross-domain experiment) while the shuffled control was run on a pilot subset of n=3 configs. Reviewers who read both Section 5.2 (shuffled control invalidated the absolute rates) and Section 5.3 (rho = 0.924 presented as conditional evidence) may question why the coherence evidence is not equally invalidated. **Fix**: Add the sentence "The intra-RAVEL coherence reported below derives from the full cross-domain analysis (n=6 SAE configurations; phase3e), while the shuffled control was run on a pilot subset of 3 configurations; the coherence result is therefore not directly tested by the shuffled control." This preempts the reviewer question.

4. **[Minor] Add descriptive captions to Tables 1–4**: Tables 1–4 have bold-label headers but no descriptive caption text. **Fix**: Add a 1–2 sentence caption below each table following the convention: describe what the table reports and what the reader should notice. Example for Table 1: "EDA AUROC and group separation statistics across all 8 SAE configurations. Mid-layer narrow SAEs (L5-16k, L12-16k) pass the AUROC ≥ 0.65 threshold; wider and deeper configurations fall to chance-level discrimination."

5. **[Minor] Unsourced "56%" and "20×" in Section 7.1**: Location — "focusing on the top 5% of latents by EDA contains approximately 56% of absorbed latents (Prec@50 = 0.0035), reducing the supervised evaluation budget by 20×." The 56% is not traceable to source data from the experimental results. **Fix**: Either derive the figure explicitly in-text from Prec@50 and population sizes, or restate as: "focusing on the top 5% of latents by EDA (819 candidates in a 16k SAE) reduces the evaluation surface to 5% of the total while the Prec@50 = 0.0035 confirms absorption enrichment in this subset."

---

## What Works Well

1. **Honest negative result handling throughout**: The paper reports three falsified hypotheses (H3, H5, H6) clearly and contextualizes each mechanistically. Section 7.3 explains why D-EDA, ITAC, and H6 failures are predictable from the taxonomy rather than incidental. The null results strengthen the paper's credibility rather than weakening it, and the text treats them this way.

2. **Theorem 1 and its empirical grounding**: The formal lower bound (Equation 3) is stated with appropriate conditions (partial minimum, non-degenerate θ_{jc}) and consistently qualified as necessary-but-not-sufficient. The polysemanticity confound is introduced and explained correctly. The SynthSAEBench validation (AUROC = 1.000, F1 = 0.974) provides isolated empirical evidence for the bound before real-SAE results, following correct scientific practice.

3. **Table 1 structure**: Table 1 includes EDA, D-EDA, decoder cosine baseline, and shuffled null AUROC for all 8 configurations, plus Cohen's d and pass/fail markers. The shuffled null column (near-chance values confirming AUROC is not an artifact of label imbalance) is methodologically exemplary. The DeLong test note below the table is concise and correctly formatted.

SCORE: 7
