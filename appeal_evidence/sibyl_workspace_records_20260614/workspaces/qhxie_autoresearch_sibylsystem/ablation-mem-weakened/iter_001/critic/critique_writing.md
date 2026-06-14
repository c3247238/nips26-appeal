# Writing Critique

## Overall Assessment

The paper is well-structured and clearly written, with a logical arc from problem statement to null results to implications. The abstract is a model of clarity. However, several writing issues undermine the paper's credibility and readability.

## Major Issues

### 1. Claims Stronger Than Evidence (Critical)

The abstract states: "These null results suggest that feature absorption... may not be a critical failure mode for steering and probing in GPT-2 Small SAEs." This framing presents the null result as informative evidence against absorption being critical. However, given severe power limitations (n=26, restricted variance, ~65% power for |r|>=0.50), the study cannot distinguish between a true zero effect and a small-to-medium true effect that the sample size failed to detect. The word "suggest" softens the claim, but the overall framing is still too strong.

**Fix:** Replace "may not be a critical failure mode" with "we find no detectable relationship in this setting, but power constraints and model scope limit the strength of this conclusion." Emphasize the methodological contribution over the substantive conclusion.

### 2. Banned Transition Words Survive (Major)

The writing review flagged these but they were not fixed:
- "Moreover" (Section 2.3): "Moreover, neither steering nor probing has been systematically correlated..."
- "Furthermore" (Section 2.4): "Furthermore, all three approaches target absorption reduction..."
- "It is worth noting that" (Section 5.2): "It is worth noting that even the most absorbed feature..."

**Fix:** Replace "Moreover, neither steering..." with "Neither steering..." (Section 2.3). Replace "Furthermore, all three..." with "All three..." (Section 2.4). Delete "It is worth noting that" and start with "Even the most absorbed feature..." (Section 5.2).

### 3. Missing Data for Claimed Results (Major)

Section 5.3 claims "Full-activation probing consistently outperforms k-sparse probing" but provides no numbers. Section 4.5 defines F1_full but it is never reported in the results. This is a claim without evidence.

**Fix:** Either add a small table comparing F1(f,5) and F1_full for all 26 features, or remove the claim from Section 5.3.

### 4. Passive Voice Overuse (Minor)

The abstract and Section 1.2 rely heavily on passive voice:
- "has been identified" (abstract, Section 1.2)
- "has been detected" (Section 2.2)
- "has been validated" (Section 2.2)
- "has been systematically correlated" (Section 2.3)

While some passive voice is acceptable in academic writing, the opening relies too heavily on it. Consider leading with active constructions where the subject is clear.

### 5. Awkward Sentences (Minor)

- Section 4.2: "layers 4, 8, and 10 sample the mid-to-late network" -- "sample" is not the right verb. Suggest: "layers 4, 8, and 10 span the mid-to-late network."
- Section 4.3: The conditioning logic for differential correlation is confusing. The correlation is computed "conditioned on the parent concept being present" but the threshold flagging says the child activates when the parent is absent. Clarify.
- Section 6.1, "Metric sensitivity": "preserves latent activation patterns" is ambiguous. Does it mean the latent still fires but the decoder direction is corrupted?
- Section 8.3: The closing sentence "These conclusions are subject to the limitations discussed in Section 7..." weakens the closing paragraph. The limitations have already been acknowledged throughout.

### 6. Pilot Data Not in Any Table (Minor)

Section 6.3 cites pilot numbers (r=-0.153, p=0.456) that appear nowhere else. These should be in a comparison table or footnote.

## What Works Well

1. **The abstract is excellent**: It states the problem, gap, methodology, key results with exact numbers, and implication---all in one paragraph. Every sentence earns its place.
2. **Section 6.4 "What Would Change Our Conclusion?"** is sophisticated scientific writing that anticipates reviewer objections.
3. **The visual audit fixes** (figure renumbering, Table 2 cleanup) were well-executed.
4. **Honest reporting of negative results** is the paper's strongest aspect.

## Score: 7/10

The writing is clear and professional, but the claim-evidence mismatch and surviving banned patterns prevent a higher score.
