# Writing Critique

## Summary Assessment

The paper is technically rigorous and honest about negative results, which is its strongest quality. However, two critical writing issues undermine the abstract and create internal contradictions: (1) the H1 falsification framing is contradictory (binary falsification vs layer-dependent finding), and (2) the abstract contains an admission about an unconducted experiment that belongs in the Discussion. Additionally, H4 is framed as a "falsified" hypothesis when it is actually an unconducted experiment — the design flaw means the hypothesis was never tested. Three terminology deviations from the glossary persist, and one interpretive overstatement ("shows that" vs "suggests that") remains unfixed.

---

## Critical Issues

### Issue 1: Abstract Internal Contradiction (H1 Framing)

**Location**: Abstract, lines 6-8

**Problem**: The abstract states H1 is "falsified at layer 8 but not at layer 4" — contradiction since falsification is binary, not layer-dependent. The pre-registered hypothesis predicted >20% of latents in mid-to-deep layers (layers 4-10 collectively). Falsification criterion was <10% prevalence across layers 4-10, not per-layer. The paper cannot simultaneously falsify at layer 8 and confirm at layer 4.

**Current text**:
> "at layer 8 (the pre-registered test layer), only 0.19% do, falsifying the hypothesis at that layer; however, at layer 4, 49.3% have $A_f > 0.5$, exceeding the >20% threshold at that exploratory layer"

**Fix**: Restructure to be precise. H1 is falsified at the pre-registered test layer (layer 8, 0.19% < 10% falsification threshold). Layer 4 (49.3%) is an exploratory finding, not "not falsified at that layer." Example: "At layer 8 (pre-registered test layer), only 0.19% have Af > 0.5, below the <10% falsification threshold, falsifying H1 at that layer; however, at layer 4 (exploratory), 49.3% have Af > 0.5, a finding that requires further investigation given the collective hypothesis framing."

---

### Issue 2: Abstract Admits Unconducted Experiment

**Location**: Abstract

**Problem**: The abstract states "the correct experiment (comparing full SAE representations at layers with different absorption profiles) was not conducted" — this is a limitation that belongs in the Discussion/Limitations section, not as a parenthetical qualifier on results in the abstract.

**Fix**: Remove entirely from abstract. The H4 design flaw is properly addressed in Section 5.3 and Section 6.2. The abstract should report what was done and what was found, not preemptively caveat findings with design flaws already discussed at length in the body. State the uninformative result clearly: "Both absorption subsets yielded 0.0 faithfulness, preventing any conclusion about whether absorption level predicts circuit-level causal importance."

---

### Issue 3: H4 Framed as Falsified When It Was Never Tested

**Location**: Abstract, Section 5.3, Section 6

**Problem**: The paper states H4 is "falsified" but the experiment does not test the hypothesis because the causal variable was not isolated. The key comparison (low-absorption vs. high-absorption subsets) is uninterpretable because both yielded 0.0 — not because absorption didn't affect faithfulness, but because the subset selection method (corpus-wide absorption scores) selected latents irrelevant to the circuit, not latents differing in circuit-relevant absorption.

**Fix**: Reframe H4 as genuinely untested, not falsified. The correct experiment (full SAE at layer 4 vs. layer 8) was not run. The paper should state explicitly: "H4 remains an unconducted experiment — the pilot results do not support any conclusion about whether absorption level predicts circuit-level causal importance."

---

## Major Issues

### Issue 4: Terminology Deviations Persist

Three issues from the prior review remain unfixed:

1. **"layer-wise"** in Section 5.2 — the glossary specifies "per-layer" as preferred
2. **"activations"** as singular noun in Section 3.1 — the glossary says "activation" (not "activations" as singular)
3. **"significant"** without "statistical" qualifier in Section 5.2 — prefer "statistical significance"

**Fix**: Global search-and-replace across the full manuscript.

---

### Issue 5: Section 6.4 Percentage Accuracy

**Location**: Section 6.4, paragraph 1

**Problem**: "Approximately 79% of latents have $A_f \leq 0.5$" at layer 8. From Table 2: 20.9% have $A_f > 0.5$ and 76.8% have $A_f = 0.0$. That accounts for 97.7% of latents, leaving only 2.3% unaccounted. The 79% figure appears to be 100 - 20.9 = 79.1, but the derivation is not stated, and 76.8% with $A_f = 0.0$ is a separate category from $A_f \leq 0.5$ but not $> 0.5$.

**Fix**: Clarify the derivation explicitly: "approximately 79.1% (100 - 20.9) have neither high absorption (Af > 0.5) nor complete independence (Af = 0.0) — they fall in the middle range. Of these, 76.8% have Af = 0.0 exactly and approximately 2.3% have 0 < Af <= 0.5."

---

### Issue 6: Figure 2 Description Mismatch

**Location**: Section 5.2

**Problem**: The text says "Figure 2 shows the detailed absorption score distribution for layer 4, decomposing the peak into its constituent latents" but Figure 2 per the figure plan and actual files is "fig1_layer_absorption.pdf" showing the inverted-U across all 6 layers. The layer-4 histogram is Figure 4.

**Fix**: Correct the figure reference to Figure 4, or update the text to match what Figure 2 actually shows (inverted-U across all layers).

---

## Minor Issues

### Issue 7: Interpretive Claim Overstates Evidence

**Location**: Section 5.3, Analysis paragraph

**Problem**: "The full-vs-subset comparison shows that dictionary completeness — not absorption level — drives patching fidelity." This was flagged in the prior review (suggested "suggests that" instead of "shows that") and remains unfixed.

**Fix**: Change "shows that" to "suggests that" throughout. The evidence is consistent with the interpretation but does not definitively prove it given the design flaw in subset selection.

---

### Issue 8: Section 3.3 Isotropic Explanation Unclear

**Location**: Section 3.3

**Problem**: "This is guaranteed because random decoder columns are isotropic in d-dimensional space; the top-5 co-firers span at most 6 dimensions, explaining a vanishing fraction of the 768-dimensional residual stream variance." The logic is implied but not stated. Does isotropic mean the variance is evenly distributed, hence a 6-dimensional subset explains little? Why does isotropic guarantee this?

**Fix**: Add the missing step: isotropic Gaussian columns in 768-dim space have expected squared cosine similarity of 1/768 with any fixed direction. A 6-dimensional subspace captures expected 6/768 = 0.78% of variance, confirming that random decoder co-firers cannot reconstruct substantial variance.

---

## What Works Well

1. **The abstract** is detailed and honest — key numbers (0.19%, 49.3%, 11pp, 10-fold reduction), all five hypothesis outcomes listed. After fixing the critical H1 framing issue, removing the unconducted-experiment admission, and reframing H4 as untested, this will be a strong abstract.

2. **Section 3 (The Absorption Score)** is the paper's strongest technical writing. The five-step computation is unambiguous, the design rationale is justified empirically against random dictionary controls, and the validation establishes the metric as detecting genuine structure.

3. **The H4 failure analysis** (Section 6.2) is analytically sharp. It correctly identifies that dictionary completeness — not absorption level — drives the full-vs-subset contrast, and honestly acknowledges the design flaw without overclaiming. However, it should explicitly state that H4 is untested, not falsified.

4. **Figure integration** has improved from the prior review. All five figures exist and are referenced before appearance. Figure captions are self-explanatory.

5. **Negative results reporting** is excellent — the paper consistently reports specific expected vs. observed values with clear explanations. This is the paper's strongest aspect across all reviews.

---

## Specific Line-By-Line Issues

| Location | Current | Suggested |
|----------|---------|-----------|
| Abstract | "falsifying the hypothesis at that layer; however, at layer 4, 49.3%" | "falsified at layer 8 (pre-registered); however, at layer 4 (exploratory), 49.3%" |
| Abstract | "the correct experiment... was not conducted" | DELETE — move to Discussion |
| Abstract | "H4 is uninformative" | "H4 remains untested — the experiment design did not isolate absorption as the causal variable" |
| Section 3.1 | "a latent's activations exceed 1%" | "a latent's activation exceeds 1%" |
| Section 5.2 | "layer-wise" | "per-layer" |
| Section 5.2 | "near-zero correlation, $p = 0.872$" | "near-zero correlation, $p = 0.872$ (no statistical significance)" |
| Section 5.3 | "shows that dictionary completeness" | "suggests that dictionary completeness" |
| Section 6.4 | "approximately 79% of latents" | "approximately 79.1% of latents (100 - 20.9)" with explicit derivation |
| Section 5.2 | "Figure 2 shows the detailed absorption score distribution for layer 4" | "Figure 4 shows the detailed absorption score distribution for layer 4" |

---

## Recommendation

The writing quality is salvageable with targeted fixes. Priority order:
1. Fix abstract H1 framing (Critical)
2. Remove unconducted-experiment admission from abstract (Critical)
3. Reframe H4 as untested, not falsified (Critical)
4. Global terminology fixes (Major)
5. Figure reference fix (Minor)
6. "shows that" -> "suggests that" (Minor)

After these fixes, the paper's writing will be consistent with its technical rigor.