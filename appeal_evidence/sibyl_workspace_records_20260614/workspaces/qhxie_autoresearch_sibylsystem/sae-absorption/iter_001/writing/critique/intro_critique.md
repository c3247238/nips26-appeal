# Critique: Introduction

## Summary Assessment
The introduction is well-structured and dense with evidence. It establishes three clear gaps, states three matching contributions with specific numbers, and avoids the usual hand-waving. The writing is direct and largely free of filler. However, the contribution list is overloaded with results that belong in the abstract or body, the paper organization paragraph adds nothing, and several claims need tighter evidence linkage or terminology fixes.

## Score: 7/10
**Justification**: A solid 7 that could reach 8 with targeted revisions. The gap-contribution structure is clean, the technical claims are backed by numbers, and the section avoids banned patterns. It loses points for: (a) contribution bullets that read like compressed results sections rather than contribution statements, (b) a fourth "contribution" paragraph that blurs the three-contribution framing, (c) the EDA lower bound formula appearing with undefined symbols, (d) no disclosure of the Neuronpedia proxy-label limitation for Gemma AUROC, and (e) a road-map paragraph that wastes ~60 words in a page-constrained venue.

## Critical Issues

### Issue 1: Contribution 1 cites a DeLong comparison to a baseline not introduced until Section 3.5
- **Location**: Paragraph beginning "1. **Encoder-Decoder Alignment (EDA)**", final clause.
- **Quote**: "outperforms the decoder cosine baseline by +0.396 AUROC (DeLong $p \approx 0$)"
- **Problem**: The "decoder cosine baseline" is defined for the first time in Section 3.5 of the method section (method.md, lines 69--73). At this point in the introduction, the reader has no idea what this baseline is or why it matters. A comparison to an undefined baseline is an orphaned data point. The DeLong p-value compounds the issue by introducing a statistical test the reader cannot evaluate without seeing both AUROC curves.
- **Fix**: Either (a) insert a one-clause definition inline ("the strongest non-EDA baseline, which uses decoder-column isolation rather than encoder-decoder alignment") or (b) remove this comparison from the introduction entirely and let Section 3.5 carry it. Option (b) is cleaner: the introduction's job is to convey *what* EDA does and *why* it matters, not to pre-report every statistical comparison.

### Issue 2: AUROC numbers for EDA are stated without disclosing the proxy-label limitation
- **Location**: Contribution 1, "AUROC = 0.776 at L12-16k (Gemma Scope), AUROC = 0.629 at GPT2-L6"
- **Problem**: The introduction never mentions that the Gemma Scope labels are *Neuronpedia proxy labels*, not exact Chanin et al. labels. The outline discloses this prominently (outline line 17: "Neuronpedia proxy for Gemma --- limitation disclosed upfront"). An AUROC of 0.776 against proxy labels and 0.629 against exact labels are qualitatively different quantities. A reviewer who reads the introduction, forms expectations, and then discovers the proxy-label caveat in Section 4.4 will feel misled---especially since the pilot-to-full collapse at L12-65k (0.853 to 0.468) was attributed to proxy label instability.
- **Fix**: Add a parenthetical after the Gemma AUROC: "AUROC = 0.776 at L12-16k (Gemma Scope, proxy labels)" and note GPT-2 as "AUROC = 0.629 at GPT-2 L6 (exact Chanin et al. labels)". This is one clause and prevents a trust violation later.

## Major Issues

### Issue 3: Full EDA lower bound formula appears with undefined symbols
- **Location**: Contribution 1, sentence beginning "We derive a formal lower bound..."
- **Quote**: "$\text{EDA}(j) \geq \delta^2 \sin^2(\theta_{jc}) / (2 + \delta^2)$"
- **Problem**: Presenting the full inequality forces the reader to process four undefined symbols ($j$, $\delta$, $\theta_{jc}$, $c$) before any of them have been introduced. These are formally defined in notation.md and in Section 3.2. The introduction should convey *that* a lower bound exists and *what it implies* (stronger absorption produces larger EDA), not the formula itself.
- **Fix**: Replace the formula with: "We derive a formal lower bound (Theorem 1, Section 3) showing that EDA is monotonically increasing in absorption degree." Defer the equation to Section 3.

### Issue 4: The "additional" paragraph breaks the three-gap, three-contribution symmetry
- **Location**: Paragraph beginning "We additionally report Inference-Time Absorption Correction (ITAC)..." (line 23)
- **Problem**: After carefully setting up three gaps and three contributions, the introduction tacks on ITAC (3.14% FN reduction) and H6 falsification as an extra paragraph at the same visual level. This structurally weakens the paper's framing: the reader asks "are there three contributions or five?" The proposal explicitly repositions ITAC as a proof-of-concept and H6 as supplementary (proposal.md lines 140--148). The introduction should reflect this demotion.
- **Fix**: Either (a) fold the ITAC and scaling results into a subordinate clause within Contribution 3 ("...redirecting remediation strategy from encoder correction toward dictionary allocation. A proof-of-concept inference-time correction confirms this framing: it reduces false negatives by 3.14% mean for the late-absorption minority.") or (b) label this paragraph explicitly as "Supplementary results" to visually demote it.

### Issue 5: Contribution 2 cites "18 measurements" before the reader knows what that number means
- **Location**: Paragraph starting with "2. **First cross-domain absorption characterization.**"
- **Quote**: "All 18 SAE-hierarchy measurements (6 configs x 3 hierarchies) exceed the 3x random baseline"
- **Problem**: The reader encounters "18 SAE-hierarchy measurements" without having been told there are 6 SAE configs and 3 hierarchies. The parenthetical "(6 configs x 3 hierarchies)" partially helps but comes after the claim, forcing a re-read. The significance of "all 18 exceed 3x random baseline" is diluted by the unexplained structure.
- **Fix**: Restructure as: "We measure absorption across 3 RAVEL entity-attribute hierarchies (city-continent, city-country, city-language) on 6 Gemma Scope SAE configurations. All 18 measurements exceed the 3x random baseline; intra-domain coherence (Spearman rho = 0.924) confirms stable absorption rankings across hierarchies."

### Issue 6: Gap 3 previews only two subtypes but Contribution 3 introduces three
- **Location**: Gap 3 paragraph (line 13)
- **Quote**: "A latent whose parent concept was never allocated in the dictionary (a training-time coverage failure) cannot be fixed by the same means as a latent whose parent decoder direction exists but whose encoder was trained away from it (an inference-time encoder suppression)."
- **Problem**: This previews early and late absorption subtypes but omits partial absorption entirely. Contribution 3 then introduces all three subtypes. The gap-to-contribution foreshadowing is incomplete.
- **Fix**: Add a clause for partial: "...nor by the same means as a latent whose parent direction fires in some contexts but not others (a context-dependent selective failure)."

### Issue 7: Section roadmap is mechanical boilerplate consuming ~60 words
- **Location**: Final paragraph, "Section 2 establishes the theoretical framework..." (line 25)
- **Problem**: Five sentences, each of the form "Section N does X." This repeats the table of contents without conveying the *logic* connecting sections. At NeurIPS (9-page limit), this wastes page budget. The outline allocates ~700 words to the intro; the current section is ~731 words, making every word count.
- **Fix**: Compress to a single sentence: "Section 2 provides background; Sections 3--4 derive and validate EDA; Sections 5--6 present the cross-domain characterization and taxonomy; Section 7 consolidates findings and limitations." Or delete entirely---the three-contribution list already previews the paper's structure.

## Minor Issues
- **Paragraph 1, sentence 1**: "Sparse Autoencoders (SAEs)" (plural). The glossary says preferred first-use form is "Sparse Autoencoder (SAE)" (singular). Use "A Sparse Autoencoder (SAE) decomposes..." or keep plural but match glossary convention.
- **Paragraph 1, sentence 2**: "SAEBench confirms that dictionary learning produces interpretable features" -- This makes SAEBench sound like an endorsement of SAE quality. The glossary defines it as "The benchmark suite for evaluating SAE quality." Rephrase: "SAEBench demonstrates that dictionary learning produces interpretable features..."
- **Paragraph 3**: "OrtSAE, Matryoshka SAE, KronSAE, and masked regularization (Narayanaswamy et al., 2026; Costa et al., 2025)" -- Four names are listed before any citation. Move citations closer: "OrtSAE (Narayanaswamy et al., 2026), Matryoshka SAE, KronSAE, and masked regularization (Costa et al., 2025)." This also clarifies which citation goes with which method.
- **Paragraph 2**: "biconvex sparse dictionary learning (SDL) loss" -- The glossary specifies first-use phrasing as "sparse dictionary learning (SDL) loss, which has a biconvex structure." The intro inverts this. Use the glossary's phrasing.
- **Contribution 1**: "(Theorem 1)" -- The reader does not know which section contains Theorem 1. Add "(Section 3, Theorem 1)."
- **Contribution 1**: "(Gemma Scope)" after "L12-16k" is ambiguous -- does it modify the config or cite a source? Write "on Gemma Scope L12-16k" instead.
- **Contribution 1**: "GPT2-L6" -- Glossary writing conventions specify "GPT-2 Small" with hyphen. Write "GPT-2 L6" at minimum, or "GPT-2 Small (L6)" on first mention.
- **Contribution 3**: "~72--75%" -- Tilde + range is redundant (the range already communicates approximation). Write "72--75%".
- **Contribution 3**: "Kruskal-Wallis $p$ = 0.0002 at L12-65k" -- This level of statistical detail belongs in Section 6, not the introduction. Replace with the plain implication: "statistically robust across configurations."
- **Paragraph 2**: "The practical consequence is severe" -- "Severe" is a subjective judgment. The sentence continues with the actual consequence ("any feature-based causal analysis...will silently produce incorrect conclusions"), which is stronger. Cut "is severe ---" and connect directly.
- **Paragraph 3**: "Despite sustained attention" -- Vague filler. Start directly with "Architectural responses such as OrtSAE..."
- **Additional paragraph**: "FVU change: $-4.23$%" -- Negative FVU change means reconstruction improved, but the sign convention is confusing in context. Either rephrase as "without degrading reconstruction quality" (simpler) or omit from the intro (better).
- **Line 25, Section roadmap**: Section 8 (Conclusion) from the outline is absent from the roadmap. Either add it or note that Section 7 also serves as the conclusion.
- **Em-dash spacing**: The intro uses "---" with spaces ("parent latent --- one encoding"), while the method section uses em-dashes without spaces. Pick one convention.

## Visual Element Assessment
- [x] Figure 1 referenced before appearance (panels a and b both cited in paragraphs 2 and contribution 1)
- [ ] Figure 1 references lack inline descriptions of what each panel shows -- "(Figure 1a)" is parenthetical with no preview of content. Add brief descriptors: "(Figure 1a, parent-child suppression mechanism)" and "(Figure 1b, encoder-decoder angular geometry)."
- [x] Figures match outline plan (Figure 1 = two-panel conceptual diagram as planned)
- [ ] Verify that `fig1_absorption_mechanism_desc.md` has been converted to an actual figure (tikz/matplotlib/PDF) before submission. Currently only a description file exists.
- [x] No text-heavy sections that need additional visual support (appropriate for a ~700-word introduction)

## What Works Well
1. **The concrete "starts-with-A" absorption example (paragraph 2)** immediately grounds the abstract concept. "A latent trained to detect 'starts-with-A words' (parent) fails to fire whenever a 'starts-with-A proper nouns' latent (child) is active" gives the reader the exact mental model in one sentence. This example carries through the entire paper as the running illustration.
2. **The three-gap framing is sharp and falsifiable.** Each gap is stated as a specific claim about the current state of the field: "detection requires foreknowledge," "generalizability is assumed, not tested," "no actionable taxonomy." The bolded labels make the structure scannable, and the contribution list maps one-to-one onto the gaps.
3. **Honest acknowledgment of negative results directly in the introduction.** The ITAC paragraph reports "3.14% mean false-negative reduction" (well below the 20% target) and labels it a "proof-of-concept." This signals methodological honesty and pre-empts reviewer concerns about overclaiming. A reviewer will notice this candor favorably.
