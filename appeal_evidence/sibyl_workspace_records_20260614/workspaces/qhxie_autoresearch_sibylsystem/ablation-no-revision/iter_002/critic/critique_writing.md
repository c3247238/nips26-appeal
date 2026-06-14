# Writing Critique

## Summary Assessment: 5/10

The paper is technically rigorous and honest about negative results, which is its strongest quality. However, four critical writing issues undermine the paper's credibility: (1) H4 conclusion overclaims — the experiment tests dictionary completeness, not whether absorption predicts importance; (2) H1 is labeled "falsified" but layer 4 (49.3%) actually CONFIRMS the hypothesis — this is an internal contradiction; (3) H2 was not tested despite layer 4 having 260x more data — the justification is post-hoc rationalization; (4) 8 perfect-score latents (A_f=1.0, each firing on exactly 100 tokens) are positional artifacts being treated as open questions. The paper's framing understates its predominantly negative results.

---

## Critical Issues

### Issue 1: H4 Conclusion Is Not Supported by the Experiment

**Location**: Abstract, Section 5.3, Conclusion

**Problem**: The paper concludes "absorption level does not predict which SAE latents are causally important for circuit tracing" based on the H4 experiment. This is not supported. Both bottom-10% and top-10% absorption subsets yielded 0.0 faithfulness. The comparison between full SAE (0.289) and subsets (0.000) tests whether dictionary completeness affects patching fidelity — and it does. But it does NOT test whether absorption level predicts circuit importance.

**The correct experiment** (comparing full SAE patching at layer 4 at 49.3% absorption vs layer 8 at 20.9% absorption) was never conducted. The paper cannot claim a causal conclusion about absorption predicting importance when the causal variable was never isolated.

**Current text** (Abstract):
> "the H4 circuit faithfulness experiment was uninformative due to a critical design flaw — both absorption subsets yielded 0.0 faithfulness, preventing any conclusion about whether absorption level predicts circuit importance"

This is good and correct. But the paper then adds:
> "suggesting that dictionary completeness — not absorption level — drives patching fidelity"

This "suggests" conclusion is drawn from a comparison that tests dictionary completeness, not absorption quality. The framing in Section 5.3 and Conclusion must be updated to remove any implied conclusion about absorption not predicting importance.

**Fix**: Remove all language implying that H4 shows absorption level does or does not predict importance. Report H4 as inconclusive: "Both absorption subsets yielded 0.0 faithfulness, preventing any conclusion about whether absorption level predicts circuit-level causal importance. The full SAE (0.289) outperforms subsets (0.000), but this tests reconstruction capacity, not absorption quality. The correct experiment comparing full SAE representations at layers with different absorption profiles was not conducted."

---

### Issue 2: H1 Hypothesis Verdict Is Internally Inconsistent

**Location**: Abstract, Section 5.1, Table 1

**Problem**: The hypothesis predicts ">20% of latents at layers 4-10" have A_f > 0.5. At layer 8: 0.19% (falsified). At layer 4: 49.3% (confirms the hypothesis). The paper labels H1 as "falsified" in Table 1, but layer 4 (49.3%) is NOT falsified — it exceeds the >20% threshold.

**Current text** (Abstract):
> "at layer 8 (the pre-registered test layer under the <10% falsification criterion), only 0.19% do, falsifying the hypothesis at that layer; at layer 4 (exploratory), 49.3% have A_f > 0.5, exceeding the >20% threshold"

**Problem with this framing**: "falsifying the hypothesis at that layer" implies H1 is falsified, but then "at layer 4... exceeding the >20% threshold" shows one layer confirms it. The hypothesis cannot be cleanly labeled as "falsified" when one layer confirms and another contradicts.

**Fix**: Restate precisely in Abstract, Section 5.1, and Table 1:
- "H1 is falsified at layer 8 (0.19%, below 10% threshold) but not at layer 4 (49.3%, exceeding >20% threshold)"
- "Results show strong layer-dependence requiring explanation"
- "The hypothesis cannot be cleanly labeled as falsified when one layer confirms and another contradicts"
- Table 1 should have separate rows: "H1 (layer 8): falsified" and "H1 (layer 4): not falsified"

---

### Issue 3: H2 Justification Is Post-Hoc Rationalization

**Location**: Abstract, Section 5.5

**Problem**: The paper states H2 was not tested due to "insufficient absorption variance at layer 8" (46/24,576 latents). This is correct. But layer 4 has 49.3% absorption (~12,000 latents) — 260x more data — and was never attempted. The paper's rationale is therefore self-defeating: it correctly identifies that layer 8 lacks variance, but fails to execute the alternative (layer 4) that has ample variance.

The paper also states the rationale as "early termination after H1/H3/H4 falsification" (Section 4.3) — this is post-hoc rationalization, not the correct justification.

**Fix**: Either run H2 at layer 4 (49.3% absorption, ~12,000 latents provides ample statistical power for Spearman correlation), or formally retire H2 with explicit justification:
> "H2 was not tested at layer 8 due to insufficient absorption variance (46/24,576 latents with A_f > 0.5). Layer 4 (49.3% absorption, ~12,000 latents) was identified as the appropriate alternative but the experiment was not conducted. H2 is formally retired: insufficient statistical power even at the highest-absorption layer tested in this study."

---

### Issue 4: 8 Perfect-Score Latents Are Positional Artifacts, Not Open Questions

**Location**: Section 5.1, Section 6.5

**Problem**: Eight latents have A_f = 1.0, each firing on exactly 100 tokens. This equals the number of sequences in the pilot corpus. The paper acknowledges this is "suspicious" and discusses it in Section 6.5 as an "open question" — but the pattern is too regular to be an open question.

If these latents fired on semantically meaningful tokens, token counts would vary across latents. The fact that all 8 fire on exactly 100 tokens (one per sequence) is strong evidence of positional artifact: these latents likely fire at a consistent position in each sequence (e.g., token 42 in each of the 100 sequences), not at semantically meaningful tokens.

**Fix**: Investigate immediately and document:
1. Compute token position consistency across sequences for each perfect-score latent
2. If all fire at the same position across all 100 sequences, confirm as positional artifact and exclude from primary analysis
3. Document exclusion in Appendix with quantified impact on absorption statistics
4. Do NOT leave as open question — the regularity is dispositive

---

## Major Issues

### Issue 5: Terminology Inconsistencies Persist

**Location**: Throughout manuscript

Three terminology issues from prior review remain:

1. **"layer-wise"** in Section 5.2 — the glossary specifies "per-layer" as preferred
2. **"activations"** as singular noun in Section 3.1 — should be "activation" (not "activations" as singular)
3. **"significant"** without "statistical" qualifier in Section 5.2 — prefer "statistical significance"

**Fix**: Global search-and-replace across the full manuscript.

---

### Issue 6: Section 6.4 Percentage Accuracy

**Location**: Section 6.4, paragraph 1

**Problem**: "Approximately 79% of latents have A_f <= 0.5" at layer 8. From Table 2: 20.9% have A_f > 0.5 and 76.8% have A_f = 0.0. That accounts for 97.7% of latents. The 79% figure (100 - 20.9 = 79.1) conflates "not clearly absorbed" with "completely independent" — these are distinct categories.

**Fix**: State explicitly: "At layer 8, 20.9% of latents show A_f > 0.5 (absorbed), 76.8% show A_f = 0.0 (completely independent), and 2.3% show 0 < A_f <= 0.5 (partial absorption). The two clean categories account for 97.7% of latents."

---

### Issue 7: Figure 2 Description Mismatch

**Location**: Section 5.2

**Problem**: The text says "Figure 2 shows the detailed absorption score distribution for layer 4" but Figure 2 per the figure plan and actual files is "fig1_layer_absorption.pdf" showing the inverted-U across all 6 layers. The layer-4 histogram is Figure 4.

**Fix**: Correct the figure reference to Figure 4, or update the text to match what Figure 2 actually shows.

---

### Issue 8: H5 Framing Overstates Practical Significance

**Location**: Abstract, Section 5.4

**Problem**: "10-fold reduction" (2.25% to 0.19%) overstates the effect. The absolute numbers are what matter: 97.75% non-absorbed at 2K vs 99.81% at 24K. Both dictionary sizes have negligible absorption rates.

**Fix**: Lead with absolute numbers: "97.75% non-absorbed at 2K vs 99.81% at 24K" and note the effect explains <1% of variance.

---

## Minor Issues

### Issue 9: "Shows That" vs "Suggests That"

**Location**: Section 5.3, Analysis paragraph

**Problem**: "The full-vs-subset comparison shows that dictionary completeness — not absorption level — drives patching fidelity." This was flagged in prior review and remains unfixed. The evidence is consistent with this interpretation but does not definitively prove it.

**Fix**: Change "shows that" to "suggests that."

---

## What Works Well

1. **The abstract** is detailed and honest — key numbers (0.19%, 49.3%, 11pp, 10-fold reduction), all five hypothesis outcomes listed. After fixing the critical H1 framing issue, this will be a strong abstract.

2. **Section 3 (The Absorption Score)** is the paper's strongest technical writing. The five-step computation is unambiguous, the design rationale is justified empirically against random dictionary controls, and the validation establishes the metric as detecting genuine structure.

3. **The H4 failure analysis** (Section 6.2) is analytically sharp. It correctly identifies that dictionary completeness drives the full-vs-subset contrast, and honestly acknowledges the design flaw without overclaiming.

4. **Figure integration** has improved from the prior review. All five figures exist and are referenced before appearance. Figure captions are self-explanatory.

5. **Negative results reporting** is excellent — the paper consistently reports specific expected vs. observed values with clear explanations.

---

## Priority Fix Order

1. Fix H4 conclusion framing — remove all language implying absorption does/does not predict importance
2. Fix H1 verdict — state as layer-dependent, not uniformly falsified
3. Fix H2 justification — explicitly retire with correct rationale or run at layer 4
4. Investigate 8 perfect-score latents as positional artifacts
5. Global terminology fixes
6. Figure reference fix
7. "shows that" -> "suggests that"
8. Section 6.4 percentage clarification

---

## Specific Line-By-Line Issues

| Location | Current | Suggested |
|----------|---------|-----------|
| Abstract | "falsifying the hypothesis at that layer; at layer 4, 49.3%" | "falsified at layer 8; however, at layer 4 (exploratory), 49.3% exceeds the >20% threshold" |
| Abstract | Remove | "the correct experiment... was not conducted" — move to Discussion |
| Abstract | "suggesting that dictionary completeness — not absorption level — drives patching fidelity" | Delete this conclusion — H4 does not support it |
| Section 5.1 | "H1 is falsified" | "H1 is falsified at layer 8 (0.19%, below 10% threshold) but not at layer 4 (49.3%, exceeding >20% threshold)" |
| Section 5.3 | "shows that dictionary completeness" | "suggests that dictionary completeness" |
| Section 5.3 | Remove | Any conclusion about absorption level predicting circuit importance — H4 does not support this |
| Section 5.5 | "early termination after H1/H3/H4 falsification" | Replace with explicit retirement justification |
| Section 6.2 | "shows that" | "suggests that" |
| Section 6.4 | "approximately 79% of latents" | "20.9% (absorbed) + 76.8% (completely independent) = 97.7%; the remaining 2.3% have 0 < Af <= 0.5" |
| Section 5.2 | "Figure 2 shows the detailed absorption score distribution for layer 4" | "Figure 4 shows the detailed absorption score distribution for layer 4" |
| Section 3.1 | "activations exceed 1%" | "activation exceeds 1%" |
| Section 5.2 | "layer-wise" | "per-layer" |
| Section 5.2 | "$p = 0.872$" | "$p = 0.872$ (no statistical significance)" |
| Section 5.4 | "10-fold reduction" | "97.75% non-absorbed at 2K vs 99.81% at 24K" |