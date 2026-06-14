# Skeptic Analysis: Iter 9 FULL-Mode Results

**Agent**: Skeptic (Maximum Skepticism)
**Timestamp**: 2026-04-16
**Scope**: Consolidation of 11 completed + 1 failed task from iter_009

---

## 1. Statistical Risk Inventory (Top 3)

### Risk 1: Probe Quality Asymmetry Confounds the Entire Cross-Domain Comparison

The central claim of this paper -- that absorption varies dramatically across hierarchies (11.6% to 45.1% at L24_16k) -- rests on probes of wildly different quality:

| Hierarchy | Best F1 (L24) | Gate | N classes |
|-----------|---------------|------|-----------|
| First-letter | 0.960 (sae_spelling probe: 1.0) | STRICT PASS | 26 |
| City-continent | 0.871 | RELAXED PASS | 6 |
| City-language | 0.818 | BELOW GATE (included) | 23 |
| City-country | 0.726 | BELOW GATE (included) | 80 |

City-country's probe F1=0.726 is disastrously low for a 80-class problem (balanced accuracy only 0.565). This means roughly 27% of "probe-correct" predictions at the raw level are already on shaky ground. Any "false negative" detection (raw probe correct but SAE-reconstructed probe wrong) is unreliable when the raw probe itself misclassifies a quarter of instances. The reported 45.1% absorption rate for city-country is the HIGHEST in the paper, yet it comes from the WORST probe. This is the exact confound pattern flagged in evolution lessons: "Absorption rate correlates with probe F1 at rho=-0.67."

The consolidation summary itself acknowledges this ("city-country F1=0.73 (below gate, caveat)") but still prominently features the 45.1% rate as part of the "4x range" headline finding. A single parenthetical caveat does not adequately address a confound that could explain the entire between-hierarchy variance.

**Severity: FATAL FLAW** for the city-country result specifically; **SERIOUS CONCERN** for the overall cross-domain narrative.

### Risk 2: Benign/Pathological Experiment Tests Only ONE Hierarchy on ONE SAE at PILOT Scale

H8 is declared "FALSIFIED" with "HIGH confidence" based on:
- 50 entities, all from city-continent hierarchy, all from class "Europe"
- A single SAE (L24_16k)
- Mode: PILOT (not FULL, despite the consolidation labeling)
- All 1471 instances drawn from a single class label

The per-class summary confirms: the only class tested is Europe. This means the "100% pathological" finding could be an artifact of Europe-specific probe behavior rather than a general property of absorption. A continent probe predicting "Europe" vs. being ablated to predict something else will always produce a large logit change because Europe is the dominant class. The mean |logit change| of 3.98 nats means the ablation effectively flips the prediction from a high-confidence class to a random one -- this would happen for ANY sufficiently accurate probe regardless of whether the absorption is "pathological" in the intended sense.

Furthermore, the mean control |logit change| is only 0.004 -- but this control zeros random DIRECTIONS in the residual stream, not random SAE features. The comparison is between zeroing a semantically meaningful feature direction vs. zeroing a random direction. The 1000x ratio is unsurprising and does not specifically demonstrate that absorption is pathological rather than the trivially expected outcome of removing a semantically relevant direction from the reconstruction.

**Severity: SERIOUS CONCERN**. The experiment conflates "removing a meaningful feature direction causes logit change" with "this specific absorption instance is pathological." The former is a tautology; the latter requires comparing parent recovery to a baseline where the parent feature is simply absent (not absorbed).

### Risk 3: Cross-Domain Activation Patching Shows REVERSE Direction

The cross-domain patching (H7 extension) found child zeroing recovery of 0.05% vs. control recovery of 14.4%. This is not merely a null result -- it is a result in the OPPOSITE direction. The control (zeroing random features) recovers the parent probe prediction MORE than zeroing the putative child feature. This means either:

(a) The "child features" identified by the cross-domain absorption pipeline are not the actual features suppressing the parent, OR
(b) The absorption detected in cross-domain hierarchies operates through a fundamentally different mechanism than first-letter absorption, OR
(c) The cross-domain "absorption" is substantially a measurement artifact of low probe quality.

The Wilcoxon p=1.0 and Cohen's d=-0.91 (large, wrong direction) with n=93 entities make this a well-powered null. Yet the paper plans to present first-letter patching (p=0.000218, d=1.33) as "causal evidence for absorption" while hand-waving away the cross-domain failure as "suggesting distributed mechanism." This asymmetric treatment of evidence is concerning: positive results are trumpeted, negative results are explained away.

**Severity: SERIOUS CONCERN**. The causal mechanism claim is domain-limited. If the causal story does not replicate across domains, the cross-domain absorption rates may not measure the same phenomenon.

---

## 2. Alternative Explanations

### For H1 (Cross-Domain Variation, p=7.37e-66):

**Alternative**: Probe quality is the primary driver of apparent absorption rate differences. The probe quality ordering (first-letter F1=0.96 > city-continent F1=0.87 > city-language F1=0.82 > city-country F1=0.73) does NOT match the absorption ordering (city-country 45.1% > city-continent 31.4% > first-letter 27.1% > city-language 11.6%), which argues somewhat against a simple confound. However, city-country's extremely high absorption rate (45.1%) combined with its extremely low probe quality (F1=0.73) is suspicious. A 80-class probe with 0.73 F1 will have systematic misclassifications at the "raw" level that the pipeline counts as "correct" based on majority-class predictions. When the SAE reconstruction degrades these already-fragile predictions, the result is inflated false negative counts.

Additionally, the first-letter probes trained at position -6 achieve F1=1.0 (perfect), while the cross-domain probes at position -2 achieve F1=0.73-0.87. The absorption measurement methodology is fundamentally different: first-letter uses the sae_spelling pipeline with perfect probes at a fixed token position, while cross-domain uses a generic pipeline with imperfect probes. The 27.1% first-letter absorption at L24 vs. 34.5% in iter_008 shows methodology-dependent measurement even within the same hierarchy.

### For H3 (Hedging Decomposition, chi-square p=1.04e-19):

**Alternative**: The "strict vs. compensatory" classification ratio is a function of how many SAE features fire that are correlated with the parent concept. In the first-letter task, the sparse coding is clean (26 well-separated letter features), so when a false negative occurs, there is always a compensatory feature available. In semantic hierarchies with overlapping representations, compensatory features are rarer, making more false negatives classified as "strict." The chi-square test is testing probe quality and representation structure, not a meaningful mechanistic distinction.

### For H8 (Falsified -- 0% benign):

**Alternative**: Removing the parent direction from the decoder always causes large logit changes because the parent direction encodes SEMANTICALLY MEANINGFUL information. This is true whether or not absorption is occurring. The experiment does not have a proper "no-absorption" baseline: what is the logit change when you ablate the parent direction on instances where the parent feature IS firing? If that logit change is similarly large (~4 nats), then the experiment measures "parent direction is important" rather than "absorption is pathological."

### For Architecture Invariance (H6):

**Alternative**: The architecture comparison has extremely unbalanced designs -- JumpReLU at 16k and 65k (two widths), BatchTopK at 16k only, Matryoshka at 32k only. Comparing 16k JumpReLU to 32k Matryoshka is not "matched width." The non-significant architecture effect (p=0.754 at L12) could simply reflect insufficient statistical power from 4 architectures measured at different widths, not true invariance.

---

## 3. Proxy Metric Audit

### Gap 1: "Absorption Rate" != "True Feature Absorption"

The paper defines absorption as: raw probe predicts correctly, SAE-reconstructed probe predicts incorrectly. This is a PROXY metric. True absorption (as defined by Chanin et al.) involves a specific mechanism: child feature absorbs the parent's representation direction into its decoder. The proxy metric captures ALL reasons why SAE reconstruction degrades probe accuracy, including:

- Reconstruction loss (general quality degradation)
- Feature interference (unrelated features corrupting the reconstruction)
- Probe fragility (the probe relies on subtle directions that SAE smooths out)
- Genuine competitive exclusion (what we actually mean by "absorption")

The "strict absorption rate" (where the main feature is absent AND a probe-aligned absorber is identified) is MUCH lower: at L24_16k, strict rates are 0% for first-letter, 2.0% for city-continent, 3.7% for city-country, 22.6% for city-language. The headline "absorption rates" (27-45%) are dominated by compensatory cases where the mechanism is unclear.

### Gap 2: Kruskal-Wallis on Binary Observations

The Kruskal-Wallis test (p=7.37e-66, N=3566) treats each entity-level absorption (0/1) as an observation. With N=3566, even tiny real differences become "highly significant." The question is not whether the difference is statistically significant (with this N, any 2 pp difference would be) but whether the difference is PRACTICALLY meaningful after accounting for the probe quality confound.

### Gap 3: The Paper Claims "Cross-Domain" but Tests on a Single Model

All experiments use Gemma 2 2B with Gemma Scope SAEs. The "cross-domain" variation is across hierarchy types, not across models. The absorption rates could be entirely specific to Gemma 2 2B's representation structure. The claim should be "hierarchy-dependent absorption in Gemma 2 2B SAEs," not "cross-domain absorption characterization" in the general sense.

---

## 4. Severity Classification

### Fatal Flaws

**FF1: City-Country Probe Quality (F1=0.726) Invalidates Its 45.1% Absorption Rate.**
The headline "4x range" finding relies on the city-country number being the highest. With a probe that fails on ~27% of instances, the "absorption" measurement is unreliable. This does NOT invalidate the overall cross-domain finding (city-continent at F1=0.87 and first-letter at F1=0.96 still show a real difference), but the specific numbers and the "dramatic 4x variation" framing are misleading.

### Serious Concerns

**SC1: Benign/Pathological Experiment (H8) Lacks Proper Control Condition.**
Testing only Europe-class absorption on city-continent with a single SAE, then claiming "ALL absorption is pathological" globally, is an unsupported generalization. The experiment needs: (a) multiple hierarchies, (b) a baseline measuring logit change when parent is ablated on NON-absorbed instances, and (c) at minimum FULL mode execution (this was PILOT).

**SC2: Cross-Domain Patching Failure Undermines Mechanistic Coherence.**
If the causal mechanism (competitive exclusion) only works for first-letter but shows REVERSE effects for semantic hierarchies, the paper cannot claim to have identified THE mechanism of absorption. It may have identified a mechanism for one specific hierarchy type.

**SC3: Asymmetric First-Letter Methodology.**
First-letter absorption uses a completely different pipeline (sae_spelling, position -6, perfect probes F1=1.0) than cross-domain (generic pipeline, position -2, imperfect probes). Comparing absorption rates across these pipelines conflates methodology differences with hierarchy differences.

**SC4: Per-Class Absorption Variance in City-Continent Is Enormous.**
Europe: 90.2% absorption. Africa: 3.9%. Asia: 24.4%. The "31.4% aggregate" masks that Europe alone accounts for most of the absorption. This is likely because Europe is the most frequent class and the probe decision boundary is calibrated differently for it. The per-class variance dwarfs the between-hierarchy variance.

### Minor Caveats

**MC1: Multiple Comparisons.**
The consolidation reports 9 hypothesis verdicts but does not apply any overall correction. The individual tests use Bonferroni for pairwise comparisons, which is appropriate, but the meta-level question "how many of these 9 hypotheses would we expect to falsely support by chance?" is not addressed. At alpha=0.05 with 9 independent tests, ~0.45 false positives expected. The strongly significant results (H1 p=7.37e-66, H8 p=2.69e-242) will survive any correction; the marginal ones should be flagged.

**MC2: Rate-Distortion Predictors Show Opposite-Direction Correlations.**
The individual predictors (cos_sim, co_occur, r_parent) all correlate NEGATIVELY with absorption, opposite to theoretical predictions. The pilot (n=20) showed different directions. This is a textbook case of unstable small-sample correlations. The paper should note that the theoretical framework not only fails quantitatively but qualitatively predicts the wrong direction.

**MC3: City-Language Classes Include Messy Multi-Language Labels.**
Classes like "Aimar,Ketua,Spanish" and "Arabic,Kurdish,Turkish" suggest data quality issues. Multi-language labels for a single city create ambiguous class boundaries that degrade probe quality and potentially inflate absorption.

---

## 5. Concrete Remediation

### For FF1 (City-Country Probe Quality):

**Experiment**: Train an MLP probe (2-layer, hidden dim 256) for city-country instead of logistic regression. With 80 classes and 1504 training samples (only ~19 per class), logistic regression is underfitting. Alternatively, reduce the number of country classes to ~20 (top frequency) to achieve F1 > 0.85.

**Expected outcome**: If absorption rate drops substantially (e.g., from 45% to <35%) with a better probe, the confound is confirmed. If it remains high, the finding is strengthened.

**Minimum fix**: Remove city-country from the headline cross-domain comparison figure. Report it only in an appendix with explicit caveats. Lead the paper with first-letter (F1=0.96) vs. city-continent (F1=0.87) vs. city-language (F1=0.82) as the well-controlled comparison.

### For SC1 (Benign/Pathological Control):

**Experiment**: Run the ablation experiment on CORRECTLY-CLASSIFIED instances (where parent feature IS firing). Measure |logit change| when ablating the parent direction. If this is also ~4 nats, the experiment measures "parent direction is important" rather than "absorption-specific pathology."

**Expected outcome**: If correctly-classified instances show similarly large logit changes, the benign/pathological distinction is measuring feature importance, not absorption impact. If correctly-classified instances show smaller changes, absorption genuinely amplifies the downstream effect.

**Dataset**: Same 50 entities, same SAE, but select instances where sae_pred matches raw_pred (no absorption). Requires ~30 minutes of GPU time.

### For SC2 (Cross-Domain Patching):

**Experiment**: For the cross-domain patching, instead of zeroing the TOP-5 child features by cosine similarity, try: (a) zeroing ALL features with cosine similarity > 0.1 to the parent probe direction, (b) zeroing the top-1 most activated feature that is NOT the parent, (c) using a gradient-based feature selection (which features, when zeroed, maximally recover the parent?). The current child selection strategy (top-5 by cos_sim) may not identify the actual absorbing features in multi-feature hierarchies.

**Expected outcome**: If method (c) reveals that 20-50 features each contribute small amounts to absorption (rather than 1-5 dominating), the "distributed mechanism" hypothesis gains evidence. If even the gradient-based approach fails, the cross-domain absorption may not involve competitive exclusion at all.

### For SC3 (Methodology Asymmetry):

**Experiment**: Run the first-letter absorption measurement using the SAME generic pipeline as cross-domain (position -2, sklearn logistic regression probe). Compare this first-letter rate to the sae_spelling pipeline rate. Report both.

**Expected outcome**: If the generic-pipeline first-letter rate is substantially different from the sae_spelling rate (27.1%), the methodology difference is a real confound. The paper should then use the generic pipeline for ALL hierarchies in the main comparison.

### For SC4 (Per-Class Variance):

**Analysis**: Report per-class absorption rates for ALL hierarchies in a supplementary table. Run the Kruskal-Wallis test WITHIN hierarchies (e.g., is absorption uniform across continents?) to understand whether between-hierarchy or within-hierarchy variance dominates. If within-hierarchy variance exceeds between-hierarchy variance, the "hierarchy type" framing is misleading.

**Expected outcome**: With Europe at 90.2% and Africa at 3.9% within city-continent, the within-hierarchy variance (87 pp) far exceeds the between-hierarchy variance (34 pp). This suggests that CLASS-LEVEL properties (frequency, separability, probe confidence) matter more than hierarchy type.

---

## Summary Verdict

The paper has genuine positive findings -- the cross-domain comparison between first-letter (F1=0.96, absorption=27.1%) and city-continent (F1=0.87, absorption=31.4%) is the most trustworthy result, and the causal evidence for first-letter absorption (d=1.33, p=0.000218) is strong. The honest reporting of 9 negative results remains exemplary.

However, the paper systematically over-claims:
1. The "4x variation" headline includes the unreliable city-country number.
2. The "ALL absorption is pathological" claim rests on a single-class, single-hierarchy pilot.
3. The "causal mechanism" claim only holds for first-letter and actively fails for cross-domain.
4. The "architecture invariance" claim comes from an underpowered, unmatched design.

The central message should be hedged to: "Absorption rates differ across hierarchy types, with evidence from two well-controlled comparisons (first-letter vs. city-continent), causal confirmation for first-letter spelling, and multiple informative negative results about unsupervised detection."
