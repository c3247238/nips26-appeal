# Supervisor Review: Feature Absorption as Optimal Compression

**Score: 5.5 / 10 (Borderline Reject)**
**Verdict: Revise**

---

## Executive Summary

The paper presents a theoretically-motivated connection between the Locally Competitive Algorithm (LCA) and SAE feature absorption, but the primary predictive tool fails decisively. The local inhibition graph predicts zero absorption pairs (H6: precision@20=0.0, p=1.0), and zero statistical results survive multiple comparison correction across 12 tests. The title, abstract, and contributions promise a graph-based diagnostic tool that predicts nothing. The LCA-theoretical framework is the only valid contribution but is obscured by failed predictive claims.

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Novelty & Significance | 5 | The LCA-SAE correspondence is a mathematical identity for tied-weight SAEs (G=W_dec^T W_dec). Novelty claim is weakened by: (1) exact only for architectures not used in practice (untied weights), (2) standard top-k neighbor construction, (3) competitive suppression as absorption explanation already hypothesized in Chanin et al. |
| Technical Soundness | 5 | The LCA structural correspondence is mathematically exact, but the local inhibition graph construction is too coarse. The primary hypothesis (H6) is decisively falsified with p=1.0. The theoretical framework explains the precision-recall asymmetry but does not yield a working predictive tool. |
| Experimental Rigor | 5 | Zero results survive MCP. The single uncorrected significant result (H1b L8: p=0.028) does not survive Bonferroni (p=0.334) or BH-FDR (q=0.107). H9 is a definitional tautology. OrtSAE ablation compares unmatched L0. Probe quality confound in CMI-absorption. Feature U (n=1) overgeneralized to conclude absorption is benign. |
| Reproducibility | 6 | n=26 features insufficient for correlation analysis (detectable r>0.5 only). Single model (GPT-2 Small). Single SAE variant (res-jb, 24K). H8 computation undocumented. Homeostatic rebalancing deferred. |

---

## Critical Issues

### 1. Zero Significant Results After MCP (Critical)

The paper claims H1b at layer 8 (Pearson p=0.028, Spearman p=0.009) as evidence of a real effect, but these are uncorrected p-values from pre-registered hypotheses. After Bonferroni correction for 12 tests, the lowest corrected p is 0.107 (BH-FDR). The paper incorrectly presents p=0.028 as evidence of statistical significance.

**Every p-value citation must include MCP context:** "p=0.028 uncorrected; Bonferroni-corrected p=0.334; BH-FDR q=0.107."

### 2. Title-Abstract-Contributions Mismatch (Critical)

The title "Competitive Suppression in Sparse Autoencoders" and Section 1.4 contribution "First local inhibition graph for SAE diagnostics" create the expectation of a predictive tool. The abstract explicitly claims the graph "predicts absorption pairs." Results show precision@20=0.0 (H6 falsified). Section 5.3 claims "the graph identifies latents with high total incoming inhibition" despite H8 finding r=+0.12, p=0.55. **This is a direct contradiction.**

If the contribution is the LCA-theoretical framework, retitle to emphasize theory, not graph predictions.

### 3. H9 is a Definitional Tautology (Critical)

H9 co-occurrence measurement: p_11 + absorption_rate = 1.0 by construction (if parent fires, child is not absorbing; if parent does not fire, child is counted as absorbed). This is a mathematical identity, not an empirical finding. The reported r=-1.0 is a definitional relationship. The paper references H9 in Section 4.8 integration table despite knowing this issue from iteration 1.

**Remove H9 from the paper entirely.**

---

## Major Issues

### 4. OrtSAE Ablation at Unmatched L0 (Major)

OrtSAE without penalty (L0~920) vs OrtSAE with penalty (L0~550). The paper criticizes others for unmatched comparisons but makes the identical mistake. The conclusion "orthogonality penalty does not appear to reduce absorption" is invalid because the confound explains any difference.

### 5. Probe Quality Confound in CMI-Absorption (Major)

Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1.

### 6. CMI-Absorption Dimensional Instability (Major)

rho=-0.383 (p=0.059) at d'=10 reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236 at d'=10. Sign reversal across dimensions is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension.

### 7. Post-Hoc Power Analysis (Major)

Section 3.6 claims "approximately 20% power to detect a medium effect size." Power should be computed before the experiment, not after observing null results to explain them away. Remove this entirely.

### 8. Feature U Overgeneralization (Major)

Feature U (24.2% absorption, 100% steering) is n=1 case evidence overgeneralized to conclude "absorption is benign." Other high-absorption features show much lower steering success: H (19.0%) at 0.55, S (16.0%) at 0.65. One feature's behavior does not establish a general principle.

### 9. MCC Pipeline at Chance Level (Major)

MCC~0.21 across ALL variants including Random control (MCC=0.197) suggests Hungarian matching yields chance-level recovery regardless of training. The paper acknowledges MCC failure but does not address the implication: absorption reduction may reflect sparsity-induced suppression rather than genuine hierarchical feature recovery.

### 10. H8 Computation Undocumented (Major)

H8 claim (r=+0.12, p=0.55 for total incoming inhibition vs absorption rate) has no verifiable data source. The h6_inhibition_graph.json does not contain per-feature total incoming inhibition statistics. Provide reproducible computation or remove specific numbers.

---

## What Would Raise the Score

The only robust finding is H7 (precision-recall asymmetry), which supports the theoretical framework but does not validate the predictive tool.

1. **Remove H9** (definitional tautology)
2. **Correctly report all MCP corrections** with zero significant results
3. **Reframe contributions** around the theoretical framework, not the failed graph predictor
4. **Remove Feature U overgeneralization** (n=1 case)
5. **Document H8 computation** or remove specific numbers
6. **Address MCC pipeline validity** before claiming matching-based comparisons

---

## Risks

- Delta-corrected steering isolates absorption-specific effects but the random baseline may not match the specific absorption phenomenon being corrected for
- Precision=1.0 universally could indicate the probing task is too easy rather than absorption not affecting selectivity
- The single uncorrected significant result (H1b p=0.028) appears only with delta-correction and only at layer 8 - this pattern suggests possible p-hacking across conditions (6 steering strengths x 2 layers)
- H6 precision@20=0.0 is reported as "informative" but provides no actionable predictive value
- The untied-weight approximation weakens the core theoretical claim
- The LCA-SAE structural correspondence is a mathematical identity for tied-weight SAEs - this is not novel, it is a restatement of Rozell et al. (2008) applied to SAEs

---

## Evidence Gaps

- n=26 features provides insufficient power for correlation analysis (detectable effect size is r>0.5 only)
- Single model (GPT-2 Small 124M) limits generalization
- Single SAE variant (res-jb, 24K latents) limits generalization to other architectures
- First-letter features are a shallow hierarchy - semantic hierarchies may exhibit different dynamics
- Homeostatic rebalancing (H10) was deferred - the proposed repair cannot be evaluated
- Random SAE baseline comparison used unmatched L0 values
- MCC pipeline is at chance level for Random SAEs - the matching procedure itself may be invalid
- Probe quality confound not controlled in CMI-absorption analysis
- Total incoming inhibition per-feature computation is not documented