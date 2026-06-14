# Critique: Experiments

## Summary Assessment

The Experiments section is technically sound and reports results with appropriate specificity. The section correctly uses random dictionary controls to validate the metric, acknowledges H2 as untested (not falsified), and honestly flags the H4 design flaw. However, pervasive structural and referential inconsistencies undermine navigability: figure numbers don't match the outline's Figure & Table Plan, table numbers are swapped relative to the outline, verdict language violates glossary conventions, and the section numbering contradicts the outline roadmap. The writing quality is otherwise strong but needs consistency fixes.

## Score: 6/10

**Justification**: The experimental content is solid -- hypotheses are clearly stated, falsification criteria are explicit, and results are reported with specific numbers. However, the section has pervasive structural and referential inconsistencies that would confuse a reviewer trying to replicate or navigate the paper: figure numbers don't match the outline's Figure & Table Plan, table numbers are swapped relative to the outline, verdict language uses "falsified" where the glossary prescribes "uninformative", and the section's own numbering (5.1-5.10) doesn't align with the outline's roadmap (which says Section 6 is Discussion but the actual discussion file is "10. Discussion"). Reaching the next level requires fixing these inconsistencies so the paper is navigable and internally coherent.

---

## Critical Issues

### Issue 1: Figure Reference Mismatch in H3 Section

- **Location**: Section 5.2, line 25: "Figure 1 illustrates the inverted-U pattern."
- **Quote**: "Figure 1 illustrates the inverted-U pattern."
- **Problem**: The text references Figure 1 for the inverted-U pattern, but according to the outline's Figure & Table Plan, the per-layer absorption chart (inverted-U, fig1_layer_absorption.pdf) is **Figure 3**, appearing in Section 5.2 (H1+H3). Figure 1 per the plan is the experimental pipeline diagram (gen_pipeline.pdf) in the Method section. The figure reference is off by at least two numbers.
- **Fix**: Change "Figure 1" to "Figure 3" to match the outline's Figure & Table Plan, or update the outline's figure numbering to match what the text actually shows.

### Issue 2: Table Numbers Don't Match Outline's Figure & Table Plan

- **Location**: Sections 5.8, 5.9 and the outline's Figure & Table Plan
- **Quote**: Tables in the section: Table 1 (Hypothesis Summary), Table 2 (Layer L0 and Absorption), Table 3 (H5 Dictionary Size), Table 4 (H4 Circuit Faithfulness)
- **Problem**: The outline's Figure & Table Plan specifies:
  - Table 1: Hypothesis Results Summary (Section 5.6) -- matches
  - Table 2: Per-Layer L0 and Absorption Statistics (Section 5.1) -- matches
  - Table 3: Circuit Faithfulness Details (Section 5.3/H4) -- WRONG: section has this as Table 4
  - Table 4: Dictionary Size Comparison (Section 5.4/H5) -- WRONG: section has this as Table 3
  The tables for H4 (Faithfulness) and H5 (Dictionary Size) are swapped relative to the outline plan.
- **Fix**: Renumber Table 3 and Table 4 in the section to match the outline, or update the outline's table numbering to match the section's actual order. Whichever direction, the numbering must be consistent.

### Issue 3: Section Numbering Inconsistent with Outline Roadmap

- **Location**: The experiments section (5.x) and discussion.md ("10. Discussion")
- **Problem**: The outline says:
  - Section 5: Experiments (5.1-5.5 for H1-H5, 5.6 Summary, 5.7-5.10 for supplementary tables and resources)
  - Section 6: Discussion
  - Section 7: Conclusion

  But the actual section files use:
  - experiments.md: Section 5 (correct)
  - discussion.md: Section 10 (inconsistent with outline)
  - conclusion.md: presumably Section 11 (if following the actual numbering)

  The intro.md (line 25) says: "Sections 5 through 8 present results for H1, H3, H4, and H5 respectively. Section 9 notes that H2 remains untested. Section 10 discusses the findings." This is internally consistent with the section file numbering but inconsistent with the outline which says Section 6 is Discussion.
- **Fix**: Decide on one numbering scheme (either 5-10 with Conclusion at 11, or 5-7 with Discussion at 6 and Conclusion at 7) and update all section files and internal cross-references to be consistent.

### Issue 4: H4 Verdict Uses "Falsified" Where Glossary Prescribes "Uninformative"

- **Location**: Section 5.3, line 43
- **Quote**: "**Verdict**: H4 is falsified as an uninformative experiment."
- **Problem**: The glossary (Preferred Terminology table) explicitly states: "inconclusive: 'falsified' when experiment design was flawed" and "uninformative: 'falsified' for H4 (experiment design flaw, not null result)". The verdict sentence tries to have it both ways -- calling it "falsified as an uninformative experiment" -- but the correct verdict is simply "uninformative". The experiment was not falsified; it was never capable of testing the hypothesis due to a design flaw.
- **Fix**: Change the verdict to: "**Verdict**: H4 is uninformative. The hypothesis cannot be tested with the current design because both latent subsets fail entirely." Remove "falsified" from the H4 verdict.

### Issue 5: H3 Says "Sparsest Layer" When Layer 8 Has Highest L0 (Least Sparse)

- **Location**: Section 5.2, line 27
- **Quote**: "The sparsest layer in our sample is layer 8 (L0 = 71.9)"
- **Problem**: The glossary defines L0 as "number of non-zero activations" and states: "L0 is a property of activation patterns on data, not a direct measure of the training L1 penalty." A higher L0 means more non-zero activations, which means LESS sparsity. Layer 8 has the highest L0 (71.9) in the dataset, making it the LEAST sparse layer, not the sparsest. This is the opposite of what the text claims.
- **Fix**: Change to "The layer with the highest L0 in our sample is layer 8 (L0 = 71.9, the least sparse)" throughout the paper. This correction applies to the intro (line 15: "the sparsest layer (layer 8, L0 = 71.9)") and discussion section 10.1 as well.

### Issue 6: H1 Title and Body Focus Only on Layer 8, Burying Layer 4's Central Finding

- **Location**: Section 5.1 title and opening paragraphs
- **Quote**: "H1: Absorption Prevalence Is Extremely Low" / "Only 46 of 24,576 latents (0.19%) have $A_f > 0.5$"
- **Problem**: The title and body report only the layer 8 result (0.19%). The central finding -- that absorption peaks at layer 4 (49.3%) -- is buried in Table 2 (Section 5.7). The outline specifies the title should be "H1: Absorption Is Extremely Rare at Layer 8 but Prevalent at Layer 4" -- the current title drops the layer 4 part entirely. The 260x difference between layers is the paper's central empirical finding.
- **Fix**: Rename section title to match the outline: "H1: Absorption Is Extremely Rare at Layer 8 but Prevalent at Layer 4". Add a paragraph after the layer 8 falsification: "In contrast, layer 4 shows 49.3% of latents with $A_f > 0.5$ -- exceeding the >20% threshold and revealing a 260x layer-dependence in absorption prevalence (Table 2)."

### Issue 7: "8 Latents" Paragraph Is Layer-8-Specific, Not Layer 4

- **Location**: Section 5.1, paragraph 3: "Eight latents achieve the maximum score of $A_f = 1.0$. Each fires on exactly 100 tokens..."
- **Problem**: The "8 latents" observation comes from the H1 pilot at layer 8 (46 absorbed latents). At layer 4, the proposal correctly states that **6,170 latents** (25.1%) have Af=1.0. The current paragraph implies layer 4 also has 8 perfect-score latents. This is a factual error.
- **Fix**: The "Eight latents" paragraph is appropriate in the H1 layer 8 context. Add a separate note when discussing layer 4 results: "At layer 4, 6,170 latents (25.1%) score $A_f = 1.0$, exhibiting bimodal distribution with 34.2% scoring exactly 0.0." Ensure discussion.md line 7 also uses 6,170, not 8.

---

## Major Issues

### Issue 8: H4 Section Uses "Shows That" Instead of "Suggests That"

- **Location**: Section 5.3, final paragraph
- **Quote**: "Keeping only 10% of latents by any criterion destroys the reconstruction capacity needed for patching."
- **Problem**: Given the H4 experiment's design flaw (task-agnostic subset selection), any conclusion from it should be hedged with "suggests that" not "shows that". The outline's Critical Correction #16 flags: "Section 5.3 says 'shows that dictionary completeness' but should be 'suggests that' given the design flaw in subset selection."
- **Fix**: Replace "shows that" with "suggests that" in any H4 causal conclusions. The conclusion that "dictionary completeness drives patching fidelity" is a reasonable interpretation but cannot be "shown" by an experiment with this design flaw.

### Issue 9: "Layer-Wise" Should Be "Per-Layer" Per Glossary

- **Location**: Section 5.2 or any text using "layer-wise" terminology
- **Problem**: The glossary (Preferred Terminology table) states: "per-layer: layer-wise (when describing across layers)". The experiments section should use "per-layer" consistently.
- **Fix**: Replace any occurrence of "layer-wise" with "per-layer" in the section. Verify across the entire section.

### Issue 10: Table 4 "Half Restored" Column Is Ambiguous

- **Location**: Table 4 (Section 5.9), column header "Half Restored"
- **Quote**: Table 4 header: "Patching Method | Faithfulness | Half Restored"
- **Problem**: "Half Restored" is not defined in the table, the section text, or the glossary. The notation.md defines $\Delta_{\text{logit}}$ and faithfulness, but not "Half Restored." The column appears to show the baseline logit difference (0.400), not "half restored."
- **Fix**: Rename to "Baseline $\Delta_{\text{logit}}$" or "Clean-to-Corrupted $\Delta$" to match notation.md's definition.

### Issue 11: H2 Section Is a Placeholder, Not a Proper Writeup

- **Location**: Section 5.5
- **Quote**: "No pilot was run for H2. Early termination after H1, H3, and H4 falsification determined that the full H2 experiment would not be informative given the near-zero overall absorption rates."
- **Problem**: This reads as an excuse rather than a proper deferred-experiment acknowledgment. The proposal explicitly marks H2 as "CRITICAL PATH" and notes it must be analyzed at layer 4 (12,000 absorbed latents, not layer 8's 46). The current section does not convey urgency or position the pending analysis as the critical path item it represents.
- **Fix**: Restructure as: "H2: Token Frequency and Absorption Correlation (Deferred). Hypothesis: [restate]. Why deferred: [H1/H3/H4 made layer-8 pilot uninformative; layer 4 has 260x more absorbed latents]. Analysis plan: [pre-registered protocol from Section 4]." Give it roughly the same structure and length as the tested hypotheses.

---

## Minor Issues

- **Section 5.7 header "H5: Dictionary Size Breakdown"**: Mislabeled -- appears to be Table 2 data (layer statistics). Section 5.8 "H5: Dictionary Size Breakdown" is similarly mislabeled. → Renumber: 5.7 → "H1/H3: Layer Sweep Statistics" (Table 2), 5.8 → "H5: Dictionary Size Statistics" (Table 3), 5.9 → "H4: Circuit Faithfulness Details" (Table 4).

- **Figure 4 and Figure 5 not referenced in experiments section**: The outline lists Figure 4 (layer 4 histogram) and Figure 5 (mean absorption vs dict size) as experimental figures, but experiments.md only references Figures 1, 2, 3. → Add references to Figure 4 in the H1 layer 4 discussion and Figure 5 in the H5 section.

- **Computational resources section (5.10)**: Describes what was run (runtimes, memory footprint), not what was found. If kept, move to Section 4 (Setup) as it belongs with experimental design, not results.

- **"Early termination" framing for H2**: Sounds like a post-hoc rationalization. The proposal notes H2 was deferred for 11 iterations. Frame it as a deliberate pending analysis with a pre-registered protocol.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan -- NO, figure numbers and table numbers have mismatches
- [ ] All visuals referenced before appearance -- PARTIAL: Figure 1 is referenced but the number is wrong per the outline
- [ ] Captions are self-explanatory -- Cannot verify from text provided
- [ ] No text-heavy sections that need visual support -- Tables provide good density; confirm figures add beyond what tables show

---

## What Works Well

1. **H5 reporting is exemplary (Section 5.4)**: Correctly notes the subsample bias (upper-bound estimate), includes random dictionary controls at all three sizes, and states practical insignificance honestly ("even at 2K, 97.75% of latents are not absorbed"). This is the cleanest subsection.

2. **Table 2 (Section 5.7) is well-designed**: The full layer sweep with both mean Af and % Af > 0.5, plus % Af = 0.0, gives the reader everything needed to verify the inverted-U claim without referencing external files.

3. **H4 conclusion is appropriately qualified**: "The difference is 0.000, precluding any conclusion about absorption level and downstream causal validity" -- this is the correct interpretation of an uninformative result.

4. **Random dictionary controls throughout**: H1, H3, H5 all include random controls. This validates that the metric detects learned structure and the near-zero rates are not threshold artifacts.

5. **Explicit acknowledgment of H2 as untested**: "The pre-registered falsification criterion (Spearman $r \geq 0$) cannot be evaluated without data; we report the null result honestly and note H2 as pending" is intellectually honest and distinguishes between "untested" and "falsified."

---

## Cross-Section Consistency Check

| Consistency Issue | Location | Severity |
|---|---|
| H1 title drops layer 4 finding (should be "Extremely Rare at Layer 8 but Prevalent at Layer 4") | Section 5.1 title | Critical |
| "Sparsest layer" is factually inverted -- layer 8 has highest L0 (least sparse) | Section 5.2 | Critical |
| H4 "falsified" vs. glossary "uninformative" for design-flaw experiments | Section 5.3 verdict | Critical |
| Table 3/4 numbering swapped relative to outline Figure & Table Plan | Sections 5.8, 5.9 | Critical |
| Figure 1 reference for inverted-U should be Figure 3 per outline | Section 5.2 | Major |
| Section numbering (10 for Discussion) inconsistent with outline (6 for Discussion) | discussion.md | Major |
| "Shows that" should be "suggests that" for H4 given design flaw | Section 5.3 | Major |
| Layer 4 cluster uses "8 latents" instead of 6,170 | discussion.md line 7 | Major |
| "Layer-wise" should be "per-layer" per glossary | Section 5.2 | Minor |
| Table 4 "Half Restored" column ambiguous -- rename to "Baseline $\Delta_{\text{logit}}$" | Section 5.9 | Minor |
| H2 section reads as excuse, not critical-path pending analysis | Section 5.5 | Minor |
| Figure 4 and 5 not referenced in section | Sections 5.2, 5.4 | Minor |