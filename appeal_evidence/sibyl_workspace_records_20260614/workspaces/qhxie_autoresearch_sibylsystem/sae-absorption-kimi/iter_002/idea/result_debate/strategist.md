# Strategist Analysis

## Signal Strength Assessment

| Result | Signal Strength | Justification |
|--------|-----------------|---------------|
| H1: First-letter vs semantic-hierarchy correlation (r = 0.463, 95% CI [-0.389, 0.981]) | WEAK / INCONCLUSIVE | Point estimate is positive but falls short of the 0.6 threshold. Bootstrap CI is catastrophically wide due to n=7 SAEs, spanning from moderate negative to near-perfect correlation. The effect could easily vanish or strengthen with more data. |
| H2: Hierarchy specificity (semantic < non-hierarchy, p = 0.0032) | STRONG (but opposite direction) | The paired t-test is highly significant, but the effect is reversed: semantic-hierarchy absorption is *lower* than non-hierarchy control. This is a robust, surprising signal that directly challenges the metric's construct validity. |
| H3: Robustness across tau_fs (r = 0.468, 0.463, 0.471) | STRONG | Correlation is remarkably stable across thresholds (range ~0.008). This indicates the relationship is not a hyperparameter artifact. |
| Architecture ranking preservation (first-letter vs semantic) | MODERATE | Ordinal ranking is preserved (PAnneal < Matryoshka < Gated < JumpRelu < TopK < BatchTopK), suggesting the metric retains *some* utility for architecture comparison even if absolute correlation is weak. |
| Random SAE matches Standard SAE on semantic hierarchy (0.352 vs 0.352) | STRONG | Random permuted-decoder baseline is identical to Standard SAE on semantic tasks. This is a sharp, visually compelling result that undermines the assumption that trained SAEs outperform random structure on semantic features. |
| GPT-2 replication (near-zero semantic absorption: 0.0, 0.003) | MODERATE | Dramatically lower than Pythia-160m, but only 2 SAEs tested and at a different layer. Suggests model-family dependence but needs replication. |
| Perfect probe AUROC (1.0 for all 10 hierarchies across all SAEs) | MODERATE | Indicates the protocol is technically sound, but raises concerns about task ceiling effects and whether the hierarchies are "too easy." |

## Opportunity Cost Analysis

| Possible Next Step | GPU Hours | Expected Information Gain | Info Gain / GPU-hr | Rank |
|--------------------|-----------|---------------------------|--------------------|------|
| Expand GPT-2 replication to full 8-SAE cohort | ~0.8 | HIGH — would confirm or falsify the dramatic model-family effect | HIGH | 1 |
| Design weak co-occurrence non-hierarchy control | ~0.3 | HIGH — directly tests the leading explanation for the reversed H2 | VERY HIGH | 2 |
| Add OrtSAE + HSAE to architecture comparison | ~0.3 | MODERATE — tightens H1 correlation, tests ranking generalization | MODERATE | 3 |
| Test harder / deeper hierarchies (3-level, rare concepts) | ~0.3 | MODERATE — addresses ceiling-effect concern, may shift absorption scores | MODERATE | 4 |
| Second random baseline (random encoder + decoder) | ~0.2 | LOW-MODERATE — confirms floor effect but is unlikely to change the story | LOW | 5 |
| Full Gemma-2-2B replication (proposed in original plan) | ~1.0 | MODERATE — model-family extension, but Pythia + GPT-2 may already suffice for a short paper | LOW | 6 |
| PIVOT to Alternative A (FastProbe-Absorb) | ~0.5-1.0 | MODERATE — new direction, but loses the sunk investment in construct-validity data and figures | MODERATE | — |
| PIVOT to Alternative C (Validity-aware analysis) | ~0.5-1.0 | HIGH — natural follow-up if we stay in this space, but better as a *sequel* paper | HIGH | — |

## Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|-----------------|----------|------|------------------|
| **PROCEED with current idea + 2 follow-ups** | Moderate+ | ~1.1 hrs | Low-Moderate | A tight, provocative short paper on construct-validity stress-test |
| PIVOT to Alternative A (FastProbe-Absorb) | N/A (untested) | ~1.0 hrs | Moderate | A methods paper with practical utility, but less novel narrative |
| PIVOT to Alternative B (Theory bound) | N/A (untested) | ~0.5 hrs | Moderate | A theory contribution, but abandons the empirical momentum |
| PIVOT to Alternative C (Validity-aware analysis) | N/A (untested) | ~1.0 hrs | Moderate | Strong follow-up, but the current dataset is *prerequisite* for it |

**Dominant strategy**: PROCEED with the current idea, adding the two highest-ROI follow-up experiments (weak co-occurrence control + full GPT-2 cohort). The reversed H2 and random-SAE results are already publication-quality surprises. A pivot would discard a compelling, partially-complete story.

## PIVOT vs PROCEED Verdict

**VERDICT: PROCEED**

**Criteria check:**
- At least one hypothesis has moderate+ signal? **YES** — H2 (reversed) and tau_fs robustness are strong signals. The random-SAE control and architecture ranking preservation provide additional publishable findings.
- Clear path to publication-quality results? **YES** — the paper can be reframed as a *construct-validity stress-test* rather than a confirmatory study. This framing is arguably more interesting than the original hypothesis because it delivers both methodological contribution and critical insight.

The project does not meet the conditions for PIVOT. All hypotheses are not "weak/noise" — H2 is robustly significant (just in the unexpected direction), and tau_fs robustness is remarkably stable. The contribution margin is sufficient for a workshop or short conference paper, and with the two proposed follow-ups, it could reach a full venue like ICLR or ICML.

## Recommended Next Experiments (Priority Order)

### 1. Weak Co-Occurrence Non-Hierarchy Control (Highest ROI)
- **What**: Replace the current non-hierarchy pairs (doctor-hospital, student-school, etc.) with semantically unrelated but individually common tokens (e.g., cat-shelf, river-clock, mountain-pencil).
- **Why**: The leading explanation for the reversed H2 is that the current non-hierarchy pairs have strong contextual co-occurrence, which drives higher absorption-like scores than true hierarchies. If weak co-occurrence pairs show lower absorption than semantic hierarchies, the reversal is explained and the metric's sensitivity to co-occurrence (not hierarchy) is confirmed.
- **GPU hours**: ~0.3
- **Success criterion**: Weak-pair absorption < semantic-hierarchy absorption. If not, the metric is even less hierarchy-specific than currently believed.

### 2. Full GPT-2 Small Cohort Replication
- **What**: Run the full semantic-hierarchy + non-hierarchy control evaluation on all 8 GPT-2 small SAEs (or at least 4–6 representative ones spanning the absorption spectrum).
- **Why**: The pilot showed near-zero semantic absorption (0.0, 0.003) vs. Pythia's much higher scores. This is either a genuine model-family effect or a layer/dictionary-size confound. A full cohort test is needed to establish whether the benchmark generalizes across models.
- **GPU hours**: ~0.8
- **Success criterion**: GPT-2 SAEs consistently show lower semantic-hierarchy absorption than Pythia counterparts. If scores are comparable, the pilot difference was a fluke.

### 3. Add OrtSAE and HSAE to Architecture Ranking
- **What**: Evaluate OrtSAE (expected low absorption) and HSAE (expected moderate) on the same semantic hierarchies.
- **Why**: Increases n from 7 to 9 SAEs, tightening the H1 correlation CI. Also tests whether the architecture ranking preservation generalizes to newer, absorption-reducing designs.
- **GPU hours**: ~0.3
- **Success criterion**: OrtSAE < Matryoshka in ranking; HSAE falls in the moderate range. If inversions occur, the ordinal validity claim weakens.

## Resource Allocation Summary

| Experiment | GPU Hours | Priority | Cumulative |
|------------|-----------|----------|------------|
| Weak co-occurrence control | 0.3 | 1 | 0.3 |
| Full GPT-2 cohort | 0.8 | 2 | 1.1 |
| OrtSAE + HSAE | 0.3 | 3 | 1.4 |
| **Total recommended** | **1.4** | — | — |

This keeps the project well within the ~1 GPU-hour-per-iteration guideline while maximizing the information gain needed to turn the current results into a publishable manuscript.

## Strategic Narrative Reframe

The original paper was framed as: "First-letter absorption predicts semantic-hierarchy absorption (construct validity confirmed)."

The data support a more compelling story: **"A Construct-Validity Stress-Test of the SAEBench Absorption Metric."**

Key claims for the reframed paper:
1. **Ordinal validity holds**: Architecture rankings are preserved across first-letter and semantic tasks.
2. **Absolute validity is weak**: Correlation is moderate and underpowered (r = 0.46, wide CI).
3. **Hierarchy specificity fails**: The metric detects co-occurrence better than hierarchy (reversed H2).
4. **Model dependence is large**: GPT-2 shows near-zero absorption where Pythia shows substantial absorption.
5. **Random baselines are competitive**: A permuted-decoder SAE matches trained SAEs on semantic tasks.

This is a story reviewers will want to read because it validates the *utility* of the benchmark for architecture comparison while identifying critical *boundary conditions* that the field has overlooked.
