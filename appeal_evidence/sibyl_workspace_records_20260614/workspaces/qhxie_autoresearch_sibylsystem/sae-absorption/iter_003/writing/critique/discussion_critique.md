# Critique: Discussion

## Summary Assessment

The Discussion section is tightly argued and technically honest — its limitations subsection is unusually candid and well-specified, and the H2 falsification interpretation in §5.1 delivers the paper's main punch with appropriate directness. The main weaknesses are: (1) a missing synthesis element promised in the outline (Table 3 and a "gradient competition interpretation of EncNorm" paragraph); (2) a terminology inconsistency in §5.3 using "F1 result" as a label that never appeared in Experiments; and (3) a logical gap in §5.1's "practical guidance" that leaves a key constraint unstated. The section is more a list of isolated sub-points than a unified mechanistic synthesis.

## Score: 7/10

**Justification:** The section earns its score through evidence-based argumentation, specific numbers, and candid risk reporting. Loses points because: the outline promised a unifying mechanistic synthesis paragraph (gradient competition driving EncNorm elevation) and a Table 3 (practitioner implications matrix) that do not appear; the section lacks any cross-reference to EncNorm's detection results when making practical guidance claims; and §5.3 introduces "F1" as a result label that will confuse readers who saw Table 3 (width recovery) labeled differently in Experiments. To reach 8/10: add the EncNorm gradient-competition mechanistic paragraph, write or reference the Table 3 implications matrix, and fix the "F1 result" terminology.

---

## Critical Issues

### Issue 1: Missing unifying synthesis (outline promise unmet)

- **Location**: §5.1, §5.3 — the transition zone between them
- **Quote**: The outline specifies: "Gradient competition interpretation of encoder_norm: during training, absorbed features receive elevated gradient signal from loss terms where absorbing children interfere; this drives ||W_enc_j|| upward" as required content of §6 Discussion.
- **Problem**: This paragraph is entirely absent from the section. The Discussion explains the H2 falsification mechanistically but never connects it back to why EncNorm works — the two contributions are left unjoined. A reader who skips from OMP result to EncNorm AUROC will not understand why a training-time attractor would elevate encoder weight norms. This is the paper's central mechanistic synthesis and it is missing.
- **Fix**: Add a paragraph between §5.1 and §5.2 (or at the end of §5.1): "The same training-time gradient competition that creates absorption attractors also inflates the encoder weight norms of absorbed latents. When child feature $c$ fires on token $t$ in place of parent $j$, the reconstruction residual is attributed to $j$'s encoder direction; the encoder gradient pushes $\|\mathbf{w}_{e,j}\|_2$ upward as $j$ struggles to compete. EncNorm therefore measures the accumulated evidence of gradient competition across training — a weight-only fingerprint of the training-time phenomenon, not of inference-time behavior. This explains why EncNorm achieves AUROC $= 0.757$–$0.837$ without requiring any activation data, and why it peaks at the layer where absorption is most prevalent (L6, ratio $= 1.267$)."

### Issue 2: Missing Table 3 / practitioner implications matrix

- **Location**: §5.1, §5.3 (end of section)
- **Quote**: The outline specifies: "Table 3: Implications for mitigation approaches — what each result says for practitioners (which fixes help, which don't)."
- **Problem**: No such table exists in the section. The practical guidance is scattered across §5.1 (three bullet points) and §5.3 (two sentences about "early" vs. "late" absorption). A unified table would make the practitioner guidance actionable and would directly satisfy a NeurIPS workshop audience seeking concrete takeaways.
- **Fix**: Add a compact table (3-4 rows, 3 columns) before or after §5.3, e.g.:

| Intervention | Addresses early absorption? | Addresses late absorption? | Evidence |
|---|---|---|---|
| Iterative encoder (OMP, ISTA) | No | No | H2 falsification (0% reduction) |
| Wider dictionary | Yes (67% recovery) | No (33% unrecovered) | F1 result |
| Masked regularization | Plausible | Yes | Tang et al. + H2 support |
| EncNorm screening | Detection only | Detection only | AUROC 0.757–0.837 |

---

## Major Issues

### Issue 3: "F1 result" label is undefined in reader context

- **Location**: §5.3, sentence 1
- **Quote**: "The F1 result (67% recovery, 33% non-recovery in wider SAE) provides the clearest practical guidance"
- **Problem**: In the Experiments section, the width recovery analysis appears as §4.4 ("Wider SAE Feature Recovery (F1)") and Table 3. But in the Discussion, calling it "F1 result" treats an internal experiment label as if it were a term of art — readers who did not track the internal experiment naming scheme will not know what "F1" refers to. This also conflicts with the potential confusion with "F1 score" (precision-recall harmonic mean), a standard metric in ML.
- **Fix**: Replace "The F1 result" with "The width recovery analysis (§4.4, Table 3)". This is unambiguous and uses the actual section reference.

### Issue 4: Practical guidance in §5.1 is incomplete — the "EncNorm as screen" recommendation lacks a critical qualifier

- **Location**: §5.1, "Practical guidance" bullet (3)
- **Quote**: "Use EncNorm as a fast first-pass screen — latents with elevated EncNorm are candidates for targeted regularization during training."
- **Problem**: This recommendation implies EncNorm can guide targeted regularization of specific latents, but the paper never demonstrates that EncNorm distinguishes which absorbed features *will benefit from* targeted intervention — indeed §4.4 explicitly shows EncNorm does NOT distinguish recovered from non-recovered features ($3.289$ vs. $3.212$, $p = 0.73$). The recommendation as written overstates what EncNorm can do. A practitioner following this guidance might apply targeted regularization to EncNorm-flagged latents and find it ineffective for 33% of them with no prior warning.
- **Fix**: Add qualifier: "Use EncNorm as a fast first-pass screen to identify absorbed-candidate latents (AUROC $= 0.757$), but note that EncNorm does not distinguish which candidates will benefit from dictionary width expansion vs. which require training-objective changes (§4.4). Structural analysis of the absorbing child's decoder direction is needed to make that distinction."

### Issue 5: §5.2 hook confound limitation and §5.3 width recovery analysis are not cross-connected

- **Location**: §5.2 (hook confound) and §5.3 (F1 result)
- **Quote**: §5.2: "A matched experiment using two SAEs at the same hook would resolve this confound." §5.3 passively notes "Hook confound note" inline without explicitly linking to §5.2.
- **Problem**: Both §5.2 and §5.3 independently raise the hook confound, but they treat it as two separate issues rather than recognizing it as the same underlying problem undermining both comparisons. The reader is asked to track the same caveat twice without a clear synthesis. More importantly, §5.3 says "The Standard-24k and TopK-32k SAEs differ in both dictionary size **and** hook point" — this is a critical threat to validity of the 67%/33% recovery claim that deserves its own prominence, not a parenthetical note.
- **Fix**: Consolidate hook confound discussion. Move the parenthetical "hook confound note" from §5.3 into the main body of §5.3 with a forward reference: "These recovery rates are subject to the hook confound identified in §5.2: the two SAEs operate on different activation spaces (resid\_pre vs.\ resid\_post), which may inflate measured cosine similarity between decoder directions in different representational geometries. The 67\% recovery figure should be interpreted as an upper bound until a matched-hook experiment is conducted."

---

## Minor Issues

- **§5.1, para 2**: "The mechanism is consistent with the sparsity landscape account \citep{tang2025partial}: absorbed features exist at partial minima of the SDL training loss." — The citation here is correct, but the sentence would benefit from flagging that this is an interpretation consistent with the data, not proven by it. The OMP experiment falsifies the amortization gap; it cannot affirmatively *prove* the sparsity landscape account without ruling out other accounts. Change "consistent with" to "supports, and provides the first direct empirical evidence for" or alternatively: "The OMP result is most naturally explained by the sparsity landscape account \citep{tang2025partial}..." to be epistemically precise.

- **§5.2, first sentence**: "**Small positive-class sample.** Gold IG labels provide $n_\text{pos} = 18$ absorbed features at GPT-2 L6." — The word "provide" is passive and weak. The key concern here is that the DeLong test p=0.0012 sounds strong but may be unreliable at n=18. Rewrite: "The gold IG label set contains only $n_\text{pos} = 18$ absorbed features; at this sample size, bootstrap CIs span $\approx 0.13$ AUROC and the DeLong $p$-value should be treated as approximate." The current version contains this content but buries the p-value caveat mid-sentence.

- **§5.3, sentence 3**: "\citeauthor{eda2025}" — This shorthand appears twice in §5.3. The paper's references use "karvonen2025saebench", "tang2025partial", "narayanaswamy2026masked" as citation keys. "\citeauthor{eda2025}" does not correspond to any citation key in the paper (not in glossary or elsewhere). This appears to be either a placeholder or a citation to an unpublished internal result. If this refers to an internal iteration finding (the early/late absorption taxonomy from iter_001/002), it should cite the current paper itself or specify the source. If it is a real external citation, provide the full key.

- **§5.1, "Practical guidance" item (1)**: "Do not expect iterative solvers or improved inference-time encoders to help." — "iterative solvers" is informal. The glossary uses "OMP" as the specific term; "Orthogonal Matching Pursuit and other iterative encoders" would be more precise and consistent.

- **Terminology check**: The section uses "encoder direction $\mathbf{w}_{e,j}$" (§5.1, para 2), which matches notation.md's `W_enc_j`. However, the section also uses "displaced from the decoder direction $\mathbf{d}_j$" — notation.md uses `d_j` for this. Both are correct and consistent. No action needed.

- **Glossary compliance**: The section uses "child features" (§5.1, para 2, "high-frequency child features") where the glossary prefers "child latents" or "absorbing latents". Minor: change "child features" to "child latents" for consistency.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan — **FAIL**: The outline specifies "Table 3: Implications for mitigation approaches" for §6 Discussion. No table appears.
- [x] All visuals referenced before appearance — N/A (no visuals in section)
- [x] Captions are self-explanatory — N/A
- [ ] No text-heavy sections that need visual support — **PARTIAL**: §5.3 would benefit greatly from a 4-row practitioners' table (see Issue 2). Without it, the practical guidance is scattered across three bullet points in §5.1 and two sentences in §5.3, making it hard to act on.

---

## What Works Well

1. **§5.2 Limitations is exemplary**: The limitations section is the best in the paper. Each limitation is: named precisely ("Hook confound in cross-architecture comparison"), quantified where possible ("AUROC difference $0.757$ vs.\ $0.837$ cannot be attributed to architecture"), explained mechanistically ("different activation spaces with different representational geometry"), and given a specific resolution ("A matched experiment using two SAEs at the same hook would resolve this confound"). This is exactly what top-venue reviewers expect — it pre-empts the most obvious reviewer objections.

2. **§5.1's OMP interpretation is tight**: The paragraph explaining why OMP cannot help — "At a partial minimum, the encoder direction $\mathbf{w}_{e,j}$ has been displaced from the decoder direction $\mathbf{d}_j$ by persistent gradient competition during training. This displacement is a property of the trained weight matrices, not of the inference procedure" — is the paper's clearest and most mechanistically precise sentence. It correctly generalizes from the OMP result to all inference-time encoder variants, which is the logical reach needed for the claim.

3. **§5.4 OMP limitation is appropriately scoped**: "Our experiment cannot rule out this joint hypothesis. We view this as an important target for future work." — This is honest about the scope of the falsification without overclaiming. Many papers would omit this caveat; its inclusion improves credibility with reviewers familiar with the theoretical literature.
