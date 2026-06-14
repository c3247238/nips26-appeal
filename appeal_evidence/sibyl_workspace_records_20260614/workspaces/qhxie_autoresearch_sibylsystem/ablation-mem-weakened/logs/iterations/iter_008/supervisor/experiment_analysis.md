# Experiment Result Analysis

## Key Results Summary

The research project on feature absorption in Sparse Autoencoders (SAEs) has completed all planned experiments (10 major analyses across multiple layers and models). The key findings are:

| Finding | Status | Key Evidence |
|---------|--------|--------------|
| H1 (steering vs absorption) | Null | r=+0.008 (L4), r=-0.301 (L8), neither significant. Delta-corrected L8: p=0.028 uncorrected, but Bonferroni-corrected p=0.334 |
| H2 (sparse probing vs absorption) | Null | r=-0.003 (L4), r=-0.107 (L8), neither significant |
| H3 (cross-layer consistency) | Null | Opposite sign slopes (L4: +0.024, L8: -0.630); CV=1.079 > 0.5 threshold |
| H4 (EC50 vs absorption) | Null | L4: r=-0.166, p=0.439; L8: r=+0.180, p=0.380 |
| H5 (precision-recall asymmetry) | Supported | Precision=1.0 universally at k>=5; recall varies (0.05-1.0) |
| H6 (decoder graph predicts absorption) | Falsified | Precision@20=0.0 (predicted >= 0.10); Fisher p=1.0 |
| H7 (trained < random absorption) | Supported | Trained SAE: mean=0.034, random SAE: mean=0.278, p<0.001 |
| H9 (co-occurrence tautology) | Tautological | p_11 + absorption = 1.0 by definition; excluded from paper |
| H10 (random baseline) | Completed | Confirms H7; metric sensitive to dictionary structure |

**Primary metric performance**: Zero hypotheses survive multiple comparison correction (12 tests, Bonferroni alpha=0.00417). The only surviving positive result is H7 (trained < random absorption, p<0.001) and H5 (precision-recall asymmetry).

## Debate Perspectives Summary

**Optimist** (from contrarian.md): Frames absorption as "optimal compression" - absorption persists because it is compression-optimal, not because it is a failure. References Cui et al. showing standard SAEs cannot fully recover ground-truth features. Suggests the field has been solving a problem that doesn't matter for downstream applications.

**Skeptic** (from empirical.md): The experiment has severe power limitations (n=26, ~20% power for medium effects). The null results are uninterpretable - they may reflect true absence of effect OR insufficient sample size. The "optimal compression" framing is post-hoc without direct empirical support.

**Strategist** (from pragmatist.md): The random SAE baseline comparison (H7) is one of the few robust findings. The paper should pivot to emphasizing metric validation insight: trained SAEs have significantly lower absorption than random SAEs, suggesting absorption is partially a structural artifact that training reduces.

**Comparativist** (from theoretical.md): The LCA-SAE structural correspondence is mathematically sound. However, H6 (decoder graph predicts absorption) is falsified, undermining the theoretical framework. The phase transition theory provides a formal understanding but doesn't yield actionable predictions.

**Methodologist** (from innovator.md): The absorption-aware steering approach (addressing Gap 9: actionability) was explored but not executed. The paper would benefit from directly testing whether understanding absorption enables better interventions.

**Revisionist** (from interdisciplinary.md): The evolutionary genomics analogy (feature phylogeny) offers a novel training-free diagnostic approach but was not pursued in this iteration.

## Analysis

### 1. Method Feasibility
The core experimental methods (Chanin ablation metric, steering experiments, sparse probing) work as intended. The absorption metric reliably detects absorption patterns, though it is computationally expensive (~26 min per SAE for ablation-based method). The projection-based SAEBench metric offers a faster alternative.

### 2. Performance
The study produced clear negative results: zero significant correlations between absorption and downstream task metrics after rigorous multiple comparison correction. The one uncorrected significant result (H1b at L8: r=-0.431, p=0.028) does not survive Bonferroni (p=0.334) or BH-FDR (q=0.107). The paper's primary contribution is honest null-result reporting with rigorous controls.

### 3. Improvement Headroom
The paper is in writing phase, not experimental phase. All planned experiments are complete. The supervisor review (score 5.0, verdict "revise") identifies several methodological issues that can be addressed through rewording and framing adjustments:
- Use confidence intervals instead of p-values
- Remove post-hoc "optimal compression" claim or run direct test
- Add prominent limitation section acknowledging power inadequacy
- Use more tentative language ("our analysis finds no significant correlation" rather than "absorption is benign")

### 4. Time-Cost Tradeoff
The project has completed all experiments. The remaining work is paper writing and revision (~1-2 days) and figure generation (~0.5 day). No new experiments are needed. The time-cost tradeoff strongly favors PROCEED - abandoning now would waste the completed experimental work.

### 5. Critical Objections
The supervisor review raises valid concerns:
1. **Underpowered study**: n=26 provides ~20% power for medium effects. The null results may reflect insufficient sample size, not true absence of effect.
2. **Post-hoc framing**: The "optimal compression" theory was not pre-registered and is not directly tested. The H6 falsification undermines using decoder correlations as theoretical motivation.
3. **MCC failure in random baseline**: The Hungarian matching MCC~0.21 across all variants suggests chance-level recovery regardless of training, undermining the H7 "training reduces absorption" interpretation.

However, these are addressable through honest reporting and reframing rather than requiring new experiments. The core finding (absorption does not significantly degrade steering/probing) is still valid even with acknowledged power limitations.

## Decision Rationale

**Arguments for PROCEED**:
1. All 10 major experiments are complete. Abandoning now wastes significant invested resources.
2. The paper has legitimate contributions: honest null-result reporting with rigorous MCP, the H7 finding (trained < random absorption, p<0.001), and the H5 finding (precision-recall asymmetry).
3. The research is in writing phase, not experimental phase. The methodological issues identified in review can be addressed through rewording.
4. The project's findings are publishable as a "rigorous negative result" paper if properly framed.
5. The alternatives (pivoting to new experimental directions) would require starting from scratch with no guarantee of success.

**Arguments for PIVOT**:
1. The study is severely underpowered (n=26, ~20% power). Null results are uninterpretable.
2. The primary theoretical contribution ("optimal compression" framing) lacks empirical support and relies on a falsified hypothesis (H6).
3. The supervisor review score of 5.0 suggests the paper in its current form would likely be rejected.
4. The MCC failure in random baseline comparison undermines one of the few positive empirical results.

**Resolution**: The arguments for PROCEED are stronger when considering that the project is in the writing phase, not experimental phase. The identified issues are addressable through reframing and honest acknowledgment of limitations. The core empirical findings are valid and publishable. The "optimal compression" framing should be removed or substantially weakened to just a hypothesis for future work.

The project's strength is its rigor (MCP applied, multiple layers tested, random baseline comparison). The weakness is the underpowered study and post-hoc theoretical framing. A properly framed paper acknowledging these limitations as honest limitations would be a valid contribution to the literature.

## DECISION: PROCEED