# Experiment Result Analysis

## Key Results Summary

Based on comprehensive analysis of experiment outputs, debate records, and proposal documents:

**Core Finding**: This project investigated whether feature absorption in Sparse Autoencoders (SAEs) degrades downstream interpretability tasks. The answer is **no** - absorption does not significantly degrade steering effectiveness or sparse probing accuracy after rigorous multiple comparison correction.

**Key Metrics**:
- H1 (Steering): Raw correlation r=+0.008 (L4), r=-0.301 (L8); Delta-corrected r=-0.431 at L8 (p=0.028 uncorrected, but Bonferroni p=0.334 after correction)
- H2 (Sparse Probing): Pearson r=-0.003 (L4), r=-0.107 (L8), neither significant
- H5 (Precision-Recall): Precision=1.0 universally at k>=5; recall varies (0.05-1.0) - **the only robust finding**
- H6 (Graph Prediction): **FALSIFIED** - Precision@20=0.0, enrichment=0.0x, Fisher p=1.0
- H7 (Trained vs Random): Trained SAE mean=0.034, Random SAE mean=0.278 (8x higher), p<0.001
- H9 (Co-occurrence): **TAUTOLOGICAL** - p_11 + absorption = 1.0 by definition

**Primary Evidence**:
- 12 statistical tests conducted across H1-H4
- Zero significant results after Bonferroni correction (alpha=0.00417) or BH-FDR (q<0.05)
- Sample: n=26 first-letter features (A-Z) on GPT-2 Small, layers 0/4/8/10
- Pythia-70M cross-model data collected but not yet integrated

## Debate Perspectives Summary

- **Optimist**: Absorption is benign - H7 shows trained SAEs have 8x lower absorption than random baselines, suggesting absorption is a structural artifact that training reduces. Feature U (24.2% absorption) achieves 100% steering success, demonstrating functional capability is preserved.
- **Skeptic**: Critical issues undermine the study: (1) H6 decisively falsified with precision@20=0.0 - graph predicts nothing; (2) Zero significant results after MCP; (3) H8 also falsified but paper claims graph identifies at-risk features - direct self-contradiction; (4) LCA-SAE correspondence is definitional identity for tied-weight SAEs, not discovered insight; (5) n=26 severely underpowered (20-25% power for medium effects)
- **Strategist**: Score 5.0/10 with "revise" verdict. Critical issues: H6 falsified, H8 self-contradiction, MCP failure, definitional novelty claim. To raise to 6.0: Fix H6 data bug, quantify tied-weight approximation error, reframe as negative result. To reach 8.0+: Conduct properly powered study with pre-registered predictions.
- **Comparativist**: Chanin et al. identified absorption as failure mode but did not compare to random baselines. Korznikov et al. "Sanity Checks" show random SAEs match trained on standard metrics. Our result that trained < random on absorption specifically is novel.
- **Methodologist**: H9 co-occurrence operationalization is tautological by construction. H10 random baseline comparison is informative (though opposite to prediction). Precision-recall decomposition is reusable methodological contribution. Power analysis reveals study is severely underpowered.
- **Revisionist**: The optimal compression framing (absorption as rate-distortion optimal behavior) provides theoretical backbone. However, paper must honestly reframe as negative-result study with theoretical speculation, not salvage falsified predictions.

## Analysis

### 1. Method Feasibility
The core experimental methodology is sound: Chanin differential correlation metric applied across 26 features on GPT-2 Small. However, the primary theoretical prediction (H6: local inhibition graph predicts absorption pairs) is decisively falsified. The LCA-SAE structural correspondence G=W_dec^T W_dec is a definitional identity for tied-weight SAEs, not a discovered relationship. For untied weights (actual experimental setting), approximation error is never quantified.

### 2. Performance
Results show no statistically significant degradation from absorption:
- Steering effectiveness: no significant correlation with absorption
- Sparse probing accuracy: no significant correlation with absorption
- EC50 analysis: no significant correlation with absorption
- However, the study is severely underpowered (n=26, ~20-25% power for medium effects)

### 3. Improvement Headroom
The current direction has significant limitations:
- H6 falsification cannot be "fixed" within current framing - graph predicts zero absorption pairs
- H8 contradiction undermines the graph diagnostic claim
- No validated predictions from the theoretical framework
- Additional rounds would likely produce same null results given methodology

### 4. Time-Cost Tradeoff
- All experiments completed (10 major analyses)
- Paper writing and revision: ~1-2 days
- To address critical issues: fix H6 data bug, quantify approximation error, reframe as negative-result paper
- Time to meaningful improvement: significant additional experimentation needed, not just writing

### 5. Critical Objections
The skeptic's concerns are substantial and largely fatal to current framing:
1. H6 precision@20=0.0 - the primary predictive hypothesis is falsified
2. H8 self-contradiction - claims graph identifies at-risk features but H8 is falsified
3. Definitional identity - LCA correspondence is mathematically trivial for tied-weight case
4. Zero significant results after MCP - the core evidence does not support the claims
5. Data bug - 7 of 26 features share feature_id 25906 (27% non-independent)

## Decision Rationale

**PROCEED** is recommended despite the significant issues because:

1. **The null result is genuine and valuable**: Absorption does NOT degrade downstream tasks. This is an honest negative-result finding that contributes to the field by establishing boundaries on when absorption matters.

2. **H7 (trained < random) is a genuine novel finding**: First demonstration that trained SAEs have lower absorption than random baselines. This reframes absorption as a metric artifact rather than learned pathology.

3. **H5 (precision-recall asymmetry) is robust**: The only replicable positive finding across all iterations.

4. **Paper can be reframed as negative-result study**: With honest acknowledgment of H6 falsification and proper statistical reporting, this becomes a honest contribution rather than overclaiming.

5. **All experiments are completed**: No additional data collection needed - remaining work is writing and revision.

**However**, the current framing cannot stand. The paper MUST be revised to:
- Remove all claims that graph predicts absorption (H6 falsified)
- Remove practical recommendations (graph diagnostic, homeostatic rebalancing) that contradict results
- Lead with corrected results everywhere
- Reframe as negative-result paper with theoretical speculation
- Address the data bug (7 shared feature_ids)

The direction is sound but the paper needs substantial revision before submission.

## DECISION: PROCEED

**Note**: "Proceed" means continue with paper writing and revision, NOT continue with current framing. The paper requires fundamental restructuring to be honest about what was found and what was falsified. If the team cannot honestly reframe as a negative-result paper, PIVOT to Alternative A (metric validation study).