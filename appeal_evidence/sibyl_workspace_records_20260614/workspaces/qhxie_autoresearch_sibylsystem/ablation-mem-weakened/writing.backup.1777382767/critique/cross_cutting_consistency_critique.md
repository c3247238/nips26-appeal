# Cross-Cutting Consistency Critique

## Scope
This critique focuses on issues that span multiple sections of the paper and persist despite the individual section critiques. These are "meta" issues that only become visible when reading the entire manuscript holistically.

## Summary Assessment

The paper has undergone substantial revision since the first round of section critiques, with many critical issues (model size, H3 notation, figure numbering, Table 2 layer-10 data) successfully resolved. However, several cross-cutting inconsistencies remain that undermine the paper's coherence. The most significant is a persistent mismatch between the proposal's planned controls and what was actually implemented, combined with a CV definition that changed between the Method and Results sections without explanation.

## Persistent Cross-Cutting Issues

### Critical Issue 1: Mismatch Between Proposal and Actual Controls

- **Location**: Proposal (Controls paragraph) vs. Method Section 4.4
- **Proposal stated**: "Match HIGH and LOW absorption features for base activation strength; Include random feature baseline; Include null steering baseline"
- **Method states**: Only null steering baseline ($s = 0$) is included. Random feature baseline and activation strength matching are omitted.
- **Problem**: This is not just a minor deviation---the proposal explicitly planned three controls and the paper delivers one. The justification given ("the primary comparison is between HIGH and LOW absorption features within the same model and layer") does not address why a random baseline wouldn't strengthen the design. For a null-result paper, strong controls are especially important because the burden of proof is higher.
- **Cross-section impact**: The Results section makes claims like "steering is generally effective" without a random baseline to establish what "effective" means relative to chance. The Discussion (6.1) claims steering is "robust" but this robustness is measured against no-steering, not against random directions.
- **Fix**: Either (a) add the missing controls retroactively if data exists, (b) explicitly justify each omission with a principled argument, or (c) reframe the paper as preliminary/exploratory and acknowledge the limited control structure.

### Critical Issue 2: CV Definition Inconsistency (Method vs. Results)

- **Location**: Method Section 4.6 vs. Results Section 5.4
- **Method states**: "CV = sigma / |mu|" (with absolute value on mu)
- **Results states**: "CV = sigma / |mu| = 0.932" (same formula, but the notation.md defines CV = sigma / mu without absolute value)
- **Problem**: The notation.md defines CV = sigma / mu (no absolute value), which would produce negative CV when mu is negative. The Method added |mu| to fix this, but the notation.md was not updated to match. This creates a three-way inconsistency: notation.md (no abs), Method (with abs), and the original critique which flagged the negative CV issue.
- **Cross-section impact**: Readers consulting the notation table will see a different formula than the one used in the paper. The glossary also defines CV = sigma / mu without the absolute value.
- **Fix**: Update notation.md line 75 and glossary line 77 to match the Method: "CV = sigma / |mu|". Add a footnote explaining why the absolute value is necessary (slopes can have opposite signs, making raw CV negative and uninterpretable).

### Critical Issue 3: Figure Filename Mismatch

- **Location**: experiments.md figure references vs. figures/ directory
- **Text references**: `figures/fig2_absorption_rates.pdf`, `figures/fig3_dose_response.pdf`, `figures/fig4_absorption_vs_steering.pdf`, `figures/fig5_absorption_vs_probing.pdf`
- **Actual files**: `fig2_absorption_rates.pdf`, `fig3_absorption_vs_steering.pdf`, `fig4_absorption_vs_probing.pdf`, `fig5_dose_response.pdf`
- **Problem**: The figure numbering in the text does NOT match the filenames. In the text: Figure 3 = dose-response, Figure 4 = absorption vs steering, Figure 5 = absorption vs probing. In the files: fig3 = absorption vs steering, fig4 = absorption vs probing, fig5 = dose-response.
- **Cross-section impact**: This is a silent error that will break the LaTeX compilation. The text references `fig3_dose_response.pdf` but the actual file is `fig5_dose_response.pdf`. Similarly, text references `fig4_absorption_vs_steering.pdf` but actual file is `fig3_absorption_vs_steering.pdf`.
- **Fix**: Either rename the files to match the text references, or update the text references to match the filenames. The text order (dose-response in 5.2, scatter plots in 5.4) suggests Figure 3 should be dose-response, which means the filenames need renaming: fig3->fig5, fig4->fig3, fig5->fig4. Alternatively, keep filenames and update text references.

### Major Issue 4: Proposal vs. Paper Scope Divergence

- **Location**: Proposal (Method section) vs. entire paper
- **Proposal planned**: Gemma-2-2B (primary), Pythia-160M (validation), Llama-3.1-8B (optional); layers 8, 12, 16; dictionary sizes 16K and 65K; steering strengths [1.0, 2.0, 5.0, 10.0]
- **Paper delivers**: GPT-2 Small only; layers 0, 4, 8, 10; single dictionary size (24K); steering strengths [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- **Problem**: The paper's scope is dramatically narrower than the proposal. Only one model family, only two layers with full task data, no cross-dictionary comparison. The Introduction's "first systematic study" claim is weakened by this narrow scope---the systematicity is in the methodology, not the breadth of models tested.
- **Cross-section impact**: The Discussion (6.4) acknowledges this as a limitation, but the Introduction (1.3) and Conclusion (8.2) still frame the work as a comprehensive systematic study. The gap between promise and delivery creates a credibility issue.
- **Fix**: Tone down the "systematic" framing in the Intro and Conclusion. Emphasize the methodology as systematic (four-phase pipeline, pre-registered hypotheses) while acknowledging the scope is limited to GPT-2 Small. Replace "first systematic study" with "first systematic methodology applied to GPT-2 Small."

### Major Issue 5: H3 Criterion Changed Without Documentation

- **Location**: Method Section 4.6
- **Original criterion (from proposal)**: "CV(k) < 0.5" (undefined k)
- **First critique fixed to**: "CV = sigma / mu" with CV < 0.5
- **Current Method states**: "CV = sigma / |mu|" with CV < 0.5, plus "slopes have opposite signs" as an additional criterion
- **Problem**: The H3 criterion evolved from a single threshold (CV < 0.5) to a compound criterion (opposite signs OR CV > 0.5). This evolution is not documented anywhere. The Results section reports "slopes have opposite signs" for H1 and "CV = 0.932" for H2, using different rejection reasons for the two hypotheses. This is confusing---why not report both criteria for both hypotheses?
- **Cross-section impact**: Table 1 reports H3 as a single row with "Slopes: opposite signs (H1); CV = 0.932 (H2)" which is asymmetric and hard to parse. The reader cannot tell whether H3 is rejected because of opposite signs, because of CV, or both.
- **Fix**: Restructure H3 reporting. For each hypothesis pair (H1, H2), report: (a) slope signs, (b) slope magnitudes, (c) CV, (d) overall consistency verdict. Use a dedicated sub-table or bullet points for clarity.

### Major Issue 6: Absorption Classification Threshold Still Inconsistent

- **Location**: notation.md vs. Method vs. outline
- **notation.md line 85**: HIGH: A(f) > 0.10
- **Method Section 4.3**: HIGH: A(f) > 0.10, MEDIUM: 0.05 <= A(f) <= 0.10, LOW: A(f) < 0.05
- **outline.md**: HIGH (>50%), MEDIUM (10-50%), LOW (<10%)
- **Problem**: The outline still has the old 50% threshold, which was never updated after the paper switched to 10%. The notation.md and Method now agree (10%), but the outline---which serves as the paper's structural blueprint---is inconsistent.
- **Cross-section impact**: Any reader (or reviewer) who checks the outline against the paper will find a major discrepancy. The outline's Figure & Table Plan also references the 10% threshold in its captions, suggesting the outline was partially updated but the classification section was missed.
- **Fix**: Update outline.md Section 4.3 to match the paper: HIGH > 10%, MEDIUM 5-10%, LOW < 5%. Also update the outline's Table 3 description.

### Major Issue 7: "Not Supported" vs. "Rejected" Inconsistency

- **Location**: Glossary vs. Results vs. Conclusion
- **Glossary (line 127)**: "not supported" (not "rejected" or "falsified" for hypotheses)
- **Results (Table 1)**: "Not supported" (correct)
- **Conclusion (8.1)**: "None of the three hypotheses are supported by the data" (correct)
- **But**: The Results text (5.4) says "directly rejecting consistency" for H3. The Discussion (6.1) says "The relationship... is therefore not consistent across layers." These use stronger language than "not supported."
- **Problem**: The glossary establishes "not supported" as the preferred term, but the Results and Discussion sections use "rejecting" and "not consistent" which are stronger. For a null-result paper, precise language matters.
- **Fix**: Standardize on "not supported" throughout. Change "directly rejecting consistency" to "directly inconsistent with the consistency hypothesis" or "fails the consistency criterion."

### Minor Issue 8: Missing Figure 1 PDF

- **Location**: Method Section 4.1
- **Text references**: `figures/fig1_pipeline.pdf`
- **Actual file**: `figures/fig1_pipeline_desc.md` (a markdown description, not a PDF)
- **Problem**: Figure 1 is referenced as a PDF but only a TikZ description exists. This will break LaTeX compilation.
- **Fix**: Either generate the PDF from the TikZ description or change the reference to point to the .md file (if the pipeline supports markdown figure descriptions).

### Minor Issue 9: Section 3 Missing from Draft

- **Location**: outline.md vs. paper.md/sections/
- **Outline plans**: Section 3 "Research Questions and Hypotheses" with formal RQs, H1-H3, and falsification criteria
- **Actual draft**: No Section 3 file exists. The hypotheses appear only in the Introduction (1.3) and Method (4.6).
- **Problem**: The outline explicitly plans a dedicated hypothesis section, but it was never written. The hypotheses are split between Intro and Method, which fragments the paper's logical structure.
- **Fix**: Either write a brief Section 3 (recommended) or update the outline to remove Section 3 and redistribute its content to Intro and Method.

### Minor Issue 10: Terminology Inconsistencies

- **"GPT-2 Small" parameter count**: Intro says "124M parameters, 85M non-embedding" (fixed from first critique). Method says "124M parameters, 85M non-embedding, 12 layers, d = 768". Good---these now match.
- **"res-jb" vs. "gpt2-small-res-jb"**: Intro uses "res-jb, 24,576 latents". Method uses "gpt2-small-res-jb SAE release". Standardize on one name.
- **"first-letter" vs. "first letter"**: Glossary says "first-letter" (hyphenated). The paper uses both "first-letter features" and "first letter features" inconsistently.
- **"steering strength" values**: Method says $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$. Results mentions $s = 50$ as primary but does not clearly state that all six strengths were tested for all features.

## Visual Element Cross-Check

| Figure/Table | Referenced In | File Exists | Filename Match | Status |
|-------------|---------------|-------------|----------------|--------|
| Figure 1 | Method 4.1 | No (only .md) | N/A | BROKEN |
| Figure 2 | Results 5.1 | Yes | fig2_absorption_rates.pdf | OK |
| Figure 3 | Results 5.2 | Yes | fig5_dose_response.pdf | MISMATCH |
| Figure 4 | Results 5.4 | Yes | fig3_absorption_vs_steering.pdf | MISMATCH |
| Figure 5 | Results 5.4 | Yes | fig4_absorption_vs_probing.pdf | MISMATCH |
| Table 1 | Results 5.4 | Inline | N/A | OK |
| Table 2 | Results 5.4 | Inline | N/A | OK |
| Table 3 | Results 5.1 | Inline | N/A | OK |

## What Works Well (Cross-Section)

1. **Model size consistency**: The Intro and Method now both correctly state "124M parameters, 85M non-embedding"---this was fixed from the first critique.
2. **Hypothesis language**: Table 1 consistently uses "Not supported" and the Conclusion follows suit.
3. **Figure captions**: All figure captions are self-explanatory and include the key takeaway.
4. **Results numbers**: The correlation values are consistent across Intro, Results, and Conclusion.

## Priority Fix Order

1. **Fix figure filename mismatches** (Critical Issue 3)---this will break compilation
2. **Generate or fix Figure 1 PDF reference** (Minor Issue 8)
3. **Update outline.md classification thresholds** (Major Issue 6)
4. **Fix CV definition in notation.md and glossary** (Critical Issue 2)
5. **Standardize H3 reporting** (Major Issue 5)
6. **Address missing controls** (Critical Issue 1)
7. **Tone down scope claims** (Major Issue 4)
8. **Write Section 3 or update outline** (Minor Issue 9)
