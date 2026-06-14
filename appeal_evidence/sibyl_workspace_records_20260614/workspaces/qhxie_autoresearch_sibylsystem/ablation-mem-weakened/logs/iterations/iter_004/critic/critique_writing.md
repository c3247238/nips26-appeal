# Writing Critique: Competitive Suppression in Sparse Autoencoders

## Overall Assessment

The paper is well-written, structurally coherent, and intellectually honest about null results. However, there is a persistent tension between the paper's confident mechanistic framing and the empirical reality of predominantly null results. The writing often slips from "consistent with" to "explains" without adequate evidentiary support.

## Structural Issues

### 1. Title Overstates Empirical Contribution

The title "Competitive Suppression in Sparse Autoencoders" presents competitive suppression as an established fact. But the primary predictive hypothesis (H6) was falsified, and the precision-recall asymmetry (H7) is consistent with multiple explanations. A more accurate title would be tentative: "Testing the Locally Competitive Algorithm Hypothesis for SAE Feature Absorption" or "Decoder Correlations and Feature Absorption: A Null-Result Study."

### 2. Abstract Accurately Reports but Still Overinterprets

The abstract honestly reports H6 falsification (precision@20 = 0.0) but then claims "the mechanistic framework is strongly supported" by the precision-recall asymmetry. This is an overinterpretation. The asymmetry is consistent with competitive suppression but does not strongly support it. The abstract should use more tentative language: "consistent with" rather than "strongly supported."

### 3. Section 5.1: "Exact, Not Metaphorical"

The claim that "the LCA-SAE structural correspondence is exact, not metaphorical" is technically true only for tied-weight SAEs. For the untied SAE analyzed in the paper, it is an approximation. The text acknowledges this two paragraphs later but the "exact" claim has already been made. This creates a misleading first impression.

**Fix:** Lead with the limitation: "For tied-weight SAEs, the correspondence is exact. For the untied SAEs analyzed here, it holds as a first-order approximation."

### 4. Section 5.2: Explaining Null Results as Evidence

The discussion of why prior work found null results (Section 5.2) subtly reframes null results as supporting evidence: "The inhibition framework explains why." This is a logical problem: a framework that "explains" null results by predicting them is unfalsifiable. If absorption had correlated with steering, the framework would also "explain" that (via competitive suppression reducing steering effectiveness). The framework is compatible with any outcome.

**Fix:** Acknowledge that the null results are consistent with multiple explanations, including the simple absence of any absorption-downstream correlation.

## Claim-Evidence Integrity

### 5. H1b Prominence vs. Correction Outcome

The paper prominently features H1b (delta-corrected steering at L8: r = -0.431, p = 0.028) as "the strongest signal in the dataset" and "the inhibition framework explains why delta correction is essential." However, this result does NOT survive multiple comparison correction (Bonferroni p = 0.334, BH-FDR q = 0.107). The correlation_report_full.json explicitly states zero significant results after correction.

This is a form of p-hacking by emphasis. The uncorrected p-value is highlighted; the corrected outcome is mentioned but not given equal prominence.

**Fix:** State prominently in the abstract and key results that ZERO hypotheses survive multiple comparison correction. Frame H1b as "a trend that does not survive correction" rather than "the strongest signal."

### 6. H8 Data Source Missing

Section 4.5 claims "Total incoming inhibition shows no reliable relationship with absorption rate (descriptive r = +0.12, p = 0.55)." No JSON file contains this statistic. The h6_inhibition_graph.json has top-k correlations but not per-feature total incoming inhibition.

**Fix:** Either generate the data file or remove specific numbers and use qualitative language.

### 7. Table 3 Values Unverified

The graph statistics in Table 3 (clustering coefficient, std edge weight) are not verifiable from available data files. The caption notes these are "descriptive statistics computed from the top-k neighbor subgraph" but no raw graph data is available.

**Fix:** Add a supplementary data file with the raw graph statistics or note more explicitly that these are approximate values.

## Methodological Writing

### 8. H9 Tautology Not Acknowledged in Paper

The H9 co-occurrence analysis (completed after the paper) revealed a perfect r = -1.0 correlation that is tautological by construction (p_11 + absorption_rate = 1.0). This operationalization flaw is not mentioned in the paper. If the paper is revised to include H9, this flaw must be acknowledged.

### 9. H10 Random SAE Not Integrated

The H10 random SAE baseline shows 8x higher absorption in random vs. trained SAEs (0.278 vs 0.034, p < 0.001). This fundamentally challenges the validity of the Chanin absorption metric but is not mentioned in the paper.

**Fix:** Integrate H10 as a methodological finding. The honest conclusion: the Chanin metric detects structural artifacts, not learned pathologies.

## Minor Writing Issues

### 10. Redundant LCA Equation

The LCA dynamics equation appears in both Section 2.3 (Background) and Section 3.1 (Framework) with nearly identical explanatory text. Some repetition is necessary for readability, but the duplication is excessive.

**Fix:** In Section 3.1, reference Section 2.3 and focus on the implications rather than re-deriving the equation.

### 11. "We Do Not Recommend" Contradicts Earlier Framing

Section 5.3 states "We do not recommend the local inhibition graph as a diagnostic tool in its current form." This is honest but contradicts the paper's title and abstract framing, which present the graph as a central contribution. The paper should not have titled itself around a tool it explicitly disavows.

**Fix:** This is a structural issue requiring retitling, not just a wording fix.

## Positive Aspects

1. **Intellectual honesty about H6/H8 falsification** is rare and commendable.
2. **Table 5 integration** of prior findings under the competitive suppression framework is well-structured.
3. **The four-step competitive suppression mechanism** (Section 3.2) is intuitive and clearly explained.
4. **Methodological contributions** (baseline correction, precision-recall decomposition, EC50) are clearly described and reusable.
