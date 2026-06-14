# Writing Quality Review

## Summary

This paper presents the first systematic study linking feature absorption detection (via the Chanin et al. differential correlation metric) to downstream interpretability task performance (steering and sparse probing) in GPT-2 Small SAEs. The authors measure absorption rates for 26 first-letter features across layers 0, 4, 8, and 10, then test steering effectiveness and sparse probing F1 at layers 4 and 8. The results are consistently null: no significant correlation is found between absorption rate and either task metric, and the relationship is inconsistent across layers (opposite-sign slopes for H1). The paper argues that the standard differential correlation metric may not predict task degradation in this model family.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clear logical arc: problem statement (Section 1) → background (Section 2) → hypotheses (Section 3) → methodology (Section 4) → results (Section 5) → discussion (Section 6) → limitations (Section 7) → conclusion (Section 8). Each section flows naturally into the next, and the abstract accurately represents the paper's content and results. The argument structure is explicit: problem (absorption may degrade tasks) → approach (systematic correlation study) → evidence (null results) → conclusion (absorption metric may not predict task degradation in this setting).

One minor structural issue: Section 6.3 "Comparison with Pilot" interrupts the flow of the Discussion. It would fit better as a subsection of the methodology or as a brief paragraph within Section 6.1, where statistical power is already discussed. As a standalone subsection, it feels like an afterthought.

### Notation & Terminology Consistency: 8/10

Notation is generally consistent and well-defined. Key symbols are introduced before first use: $A(f)$ for absorption rate (Section 4.3), $S(f, s)$ for steering success (Section 4.4), F1$(f, k)$ for probing F1 (Section 4.5), and $\beta$ for regression slopes (Section 4.6). The SAE reconstruction formula in Section 2.1 uses standard notation.

One inconsistency: the paper uses both "F1$(f, 5)$" (Section 3.2) and "F1$(f, k)$" (Section 4.5) without clarifying that $k=5$ is the primary analysis point. This is mentioned in Section 4.5 but should be cross-referenced in Section 3.2.

Another issue: Section 5.4 refers to "$\text{F1}_{\text{full}}$" (defined in Section 4.5) but never reports its values. Either report the numbers or remove the mention.

### Claim-Evidence Integrity: 7/10

Most claims are well-supported with specific numbers. The abstract, results, and conclusion all cite the exact Pearson $r$ and $p$-values. Table 1 provides a clean summary of hypothesis tests. Table 2 gives concrete examples of high-absorption features with their task performance.

However, several claims lack supporting evidence:

1. **"Most features achieving $S(f, 50) \geq 0.70$"** (Section 5.2): No data is provided to support this. The mean success rates for HIGH and LOW categories are given, but not the distribution.

2. **"Full-activation probing consistently outperforms k-sparse probing"** (Section 5.3): No numbers are provided. What is the mean F1$_{\text{full}}$? By how much does it outperform k-sparse?

3. **"Feature X achieves F1 = 1.00 at layer 4 with zero absorption"** (Section 5.3): This is presented as evidence that high and low absorption features span the same range, but only one example is given. A more systematic comparison (e.g., range of F1 for HIGH vs. LOW categories) would strengthen the claim.

4. **The pilot comparison in Section 6.3** cites $r = -0.153$, $p = 0.456$ for the pilot and $r = -0.301$, $p = 0.136$ for the full experiment, but these numbers do not appear in any table or figure. They should be in a table or at minimum cross-referenced to where they were first reported.

### Visual Communication: 8/10

The paper has 5 figures and 3 tables, meeting the minimum requirements. All figures are referenced before they appear. The visual audit (already performed) confirmed that figure numbering is now consistent. Captions are self-explanatory and include takeaways.

The figure plan is well-executed:
- Figure 1 (pipeline): Provides methodological context.
- Figure 2 (absorption rates): Shows the empirical distribution.
- Figure 3 (dose-response): Illustrates the steering strength effect.
- Figures 4-5 (scatter plots): Directly test the hypotheses.

One gap: the paper would benefit from a figure or table showing the full-activation probing results (F1$_{\text{full}}$) compared to k-sparse, since this claim is made in the text but unsupported by data.

### Writing Quality: 7/10

The writing is generally clear and direct. The paper avoids most banned patterns. However, several issues remain:

**Banned patterns found:**

1. **"Moreover"** (Section 2.3, line 65): "Moreover, neither steering nor probing has been systematically correlated with absorption rates." Replace with "Similarly," or restructure.

2. **"Furthermore"** (Section 2.4, line 69): "Furthermore, all three approaches target absorption reduction as a primary objective." Replace with "All three" or "In addition,".

3. **"It is worth noting that"** (Section 5.2, line 220): "It is worth noting that even the most absorbed feature..." Replace with "Even the most absorbed feature..." (lead with the concrete observation).

**Unclear or awkward sentences:**

1. Section 4.2: "Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network where feature abstraction increases." This is grammatically odd --- "sample" is not the right verb. Suggest: "layers 4, 8, and 10 span the mid-to-late network."

2. Section 4.3: "A child $c$ is flagged as absorbing if $\rho(f, c)$ exceeds the Chanin et al. threshold, indicating that the child's activation reliably coincides with the parent's absence." This is confusing: the correlation is computed "conditioned on the parent concept being present" (previous sentence), but the threshold flagging says the child activates when the parent is absent. Clarify the conditioning logic.

3. Section 6.1, "Metric sensitivity" paragraph: "It may not capture other forms of feature failure that would more strongly degrade downstream tasks, such as complete suppression without child compensation, or decoder direction corruption that preserves latent activation patterns." The phrase "preserves latent activation patterns" is ambiguous --- does it mean the latent still fires but the decoder direction is corrupted? Clarify.

4. Section 8.3: "These conclusions are subject to the limitations discussed in Section 7, including the single-model scope and narrow feature set." This sentence weakens the closing paragraph. The limitations have already been acknowledged throughout; restating them in the final sentence undermines the paper's confidence. Either integrate limitations more naturally or let the closing thought stand on its own.

**Passive voice overuse:**
- "has been identified" (abstract, Section 1.2)
- "has been detected" (Section 2.2)
- "has been validated" (Section 2.2)
- "has been systematically correlated" (Section 2.3)

While some passive voice is acceptable in academic writing, the opening of the abstract and Section 1.2 rely heavily on it. Consider leading with active constructions where the subject is clear.

## Issues for the Editor

1. **[Major] Missing full-activation probing data**: Section 5.3 claims "Full-activation probing consistently outperforms k-sparse probing" but provides no numbers. Either add a small table with F1$_{\text{full}}$ values or remove the claim. **Fix**: Add a table comparing F1$(f, 5)$ and F1$_{\text{full}}$ for all 26 features, or report means in the text.

2. **[Major] "Moreover" and "Furthermore" survive in Sections 2.3 and 2.4**: These are banned filler transitions. **Fix**: Replace "Moreover, neither steering..." with "Neither steering..." (Section 2.3). Replace "Furthermore, all three..." with "All three..." (Section 2.4).

3. **[Major] "It is worth noting that" in Section 5.2**: Banned pattern. **Fix**: Delete the phrase and start with "Even the most absorbed feature..."

4. **[Minor] Pilot data not in any table**: Section 6.3 cites pilot numbers that appear nowhere else. **Fix**: Add a small comparison table or footnote in Section 5.4 or 6.3.

5. **[Minor] Section 8.3 closing sentence weakens the conclusion**: The final sentence restates limitations that have already been discussed. **Fix**: Remove the sentence or rephrase to integrate more naturally (e.g., "While these conclusions are bounded by the limitations in Section 7, the methodology itself is generalizable...").

## What Works Well

1. **The abstract is a model of clarity** (paragraph 1): It states the problem, the gap, the methodology, the key results with exact numbers, and the implication --- all in one paragraph. Every sentence earns its place.

2. **The "What Would Change Our Conclusion?" section (6.4)** is an excellent addition that anticipates reviewer objections. It transforms a null-result paper from "we found nothing" to "here are the exact conditions under which our conclusion might not hold." This is sophisticated scientific writing.

3. **The visual audit has already fixed the figure numbering issue**, and the current ordering (Figure 3 = dose-response, Figures 4-5 = scatter plots) is logical and reader-friendly.

SCORE: 7
