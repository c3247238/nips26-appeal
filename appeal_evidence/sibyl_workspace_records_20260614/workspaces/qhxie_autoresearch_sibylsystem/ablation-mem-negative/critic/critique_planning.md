# Critique: Planning and Methodology

## Summary Assessment

The methodology is well-structured and the planning documents are thorough. The validation protocol with staged success criteria is appropriate. However, the plan overcommits to DFDA as a primary contribution despite its broken metric, and the cross-model validation plan was not executed due to access issues. The ablation and baseline plans exist in the methodology document but were not implemented.

## Score: 6/10

**Justification**: The planning is methodologically sound on paper but execution fell short. Key planned experiments (random baseline, ablations, cross-model validation) were not completed. The DFDA plan did not anticipate the metric artifact problem.

---

## Critical Issues

### Issue 1: DFDA Evaluation Protocol Was Flawed From the Start
- **Location**: Plan methodology, Section 3.4
- **Problem**: The DFDA evaluation protocol measures MSE improvement on child-dominant examples (where the parent is already suppressed). This protocol was destined to produce artifactual results because the MLP can trivially learn to predict near-zero values. The planning documents did not anticipate this metric failure mode.
- **Assessment**: The plan should have included a parent-positive validation protocol from the beginning: evaluate on examples where the parent should activate according to ground truth, regardless of child presence.
- **Fix**: For future iterations, rebuild DFDA evaluation around parent-positive examples. Add a metric caveat protocol to the planning template: "Any improvement >50% on a residual metric triggers automatic review for near-zero prediction artifacts."

### Issue 2: Cross-Model Validation Blocked But Not Adequately Substituted
- **Location**: Plan methodology, Section 3.1
- **Problem**: The plan mandated cross-model validation on Gemma-2B and Pythia-2.8B as a Go/No-Go gate (F1 >= 0.55). When access was blocked, the plan was not adequately substituted. Cross-layer validation (layers 4, 8, 10) was used as a proxy, but layer 4 failed (F1=0.432), meaning the "proxy" actually showed partial failure.
- **Assessment**: The Go/No-Go gate was effectively failed (layer 4 < 0.5) but the project proceeded anyway. This undermines the validation protocol's credibility.
- **Fix**: Either (a) enforce the Go/No-Go gate and pivot, or (b) redefine the gate to apply only to the primary layer (layer 8) and honestly report layer 4 as a limitation. Do not claim cross-layer validation when one layer fails.

---

## Major Issues

### Issue 3: Ablations Planned But Not Executed
- **Location**: Plan methodology, Section 4.1
- **Problem**: The plan includes a detailed ablation table (UAD: no clustering, no phi coefficient, no dead feature filtering, single-link clustering; DFDA: linear only, larger MLP, no residual). None of these ablations were executed.
- **Assessment**: Ablations are essential for justifying design choices. Their absence is a serious methodological gap.
- **Fix**: Execute the planned ablations. Each is computationally cheap (~10 seconds). Report results in the paper.

### Issue 4: Random Baseline Planned But Not Executed
- **Location**: Plan methodology, Section 5.2
- **Problem**: The plan lists "Random pair selection" as a baseline for UAD detection, but no random baseline was computed.
- **Assessment**: Without a random baseline, the F1=0.725 claim is unanchored.
- **Fix**: Compute random baseline as planned.

### Issue 5: E4 (End-to-End Pipeline) Was Planned But Not Executed
- **Location**: Plan methodology, Section 3.4
- **Problem**: E4 (UAD -> DFDA -> probe accuracy improvement) was planned as a full experiment but does not appear in the results. This explains the missing E4 in the paper.
- **Assessment**: The end-to-end pipeline was the most ambitious experiment and its absence is notable. Given DFDA's broken metric, executing E4 would have been futile anyway.
- **Fix**: Either execute E4 with a fixed DFDA metric, or explicitly state in the paper that E4 was deferred pending DFDA metric rebuild.

---

## Minor Issues

- **Sample size justification**: The plan uses 1000 samples but does not justify why this is sufficient for phi coefficient estimation.
- **Bootstrap CI**: Defined in notation.md but noted as "not yet implemented." Should have been implemented before paper writing.
- **Table numbering**: The plan's expected visualizations (Figure 1-6, Table 1-2) do not match the paper's actual numbering.

---

## What Works Well

1. **Staged validation protocol**: The pilot/full distinction with explicit success criteria is methodologically sound.
2. **Go/No-Go gate**: The F1 >= 0.55 threshold for cross-model validation is appropriately conservative.
3. **Risk mitigation table**: Identifies key risks (UAD F1 drop, Gemma loading failure, DFDA insignificance) with concrete mitigations.
4. **Training-free focus**: The decision to use pre-trained SAEs keeps experiments fast and reproducible.
5. **Honest reporting of blocked experiments**: The paper correctly notes that cross-model validation was blocked rather than omitting it.
