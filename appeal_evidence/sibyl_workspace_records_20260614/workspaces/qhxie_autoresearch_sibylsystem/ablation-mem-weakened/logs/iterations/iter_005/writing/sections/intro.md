# 1. Introduction

## 1.1 The Absorption Problem in Sparse Autoencoders

Sparse autoencoders (SAEs) have become the dominant tool for mechanistic interpretability of large language models, decomposing high-dimensional activations into sparse, monosemantic feature representations (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024). However, a critical limitation known as **feature absorption** threatens the reliability of SAE-based interpretability. First systematically identified by Chanin et al. (2024), absorption occurs when hierarchical features co-occur and the sparsity objective incentivizes merging general (parent) feature directions into specific (child) latents, causing the parent feature to fail to fire even when its concept is present.

The dominant response in the literature has been architectural: Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and related approaches all retrain from scratch to reduce absorption. Yet this response rests on an unverified assumption: that absorption meaningfully degrades downstream interpretability tasks. No prior work has systematically tested whether absorption actually harms feature steering, sparse probing, or other practical applications with proper statistical controls.

## 1.2 Research Gap and Questions

**Gap**: The prevailing narrative treats absorption as a failure mode requiring mitigation, but no study has rigorously tested whether absorption actually degrades downstream SAE functionality with appropriate multiple comparison correction.

This paper addresses four research questions:
- **RQ1**: Does feature absorption significantly degrade steering effectiveness or sparse probing accuracy?
- **RQ2**: Does absorption affect recall (coverage) but not precision (selectivity)?
- **RQ3**: Does a decoder-correlation-based inhibition graph predict absorption pairs?
- **RQ4**: Do high-absorption features retain functional steering capability?

## 1.3 Approach

We conduct a systematic, multi-method investigation across 26 first-letter features (A-Z) at multiple layers of GPT-2 Small using the gpt2-small-res-jb SAE (24,576 latents). Our approach is training-free: we analyze existing pretrained SAEs rather than training new ones. We apply rigorous multiple comparison correction (12 tests, Bonferroni alpha = 0.00417) and include random baseline comparisons to isolate absorption-specific effects.

## 1.4 Key Findings

Our central findings are:

1. **Null results on downstream tasks (H1-H4)**: Zero hypotheses survive multiple comparison correction. Absorption does not significantly degrade steering effectiveness (r=+0.008, p=0.970 at L4; r=-0.301, p=0.136 at L8) or sparse probing accuracy (r=-0.003, p=0.987 at L4; r=-0.107, p=0.604 at L8).

2. **Precision-recall asymmetry (H5)**: Precision equals 1.0 universally at k >= 5 across 26 features, while recall varies widely (0.05-1.0). This is the one robust, replicable finding.

3. **Decoder correlation graph falsified (H6)**: The inhibition graph (constructed from decoder correlations $G = W_{\text{dec}}^T W_{\text{dec}}$) achieves precision@20 = 0.0, falsifying the hypothesis that decoder geometry captures absorption dynamics.

4. **Random SAE baseline comparison (H10)**: Random SAEs show 8x higher absorption (mean=0.278) than trained SAEs (mean=0.034, p<0.001), indicating the Chanin absorption metric is sensitive to structural artifacts that training reduces.

5. **High-absorption features retain steering capability**: Feature U (24.2% absorption at L8) achieves 100% steering success, demonstrating that absorption does not destroy functional utility.

## 1.5 Theoretical Reframing

These findings motivate a fundamental reframing: under hierarchical co-occurrence and sparsity constraints, absorption is the **optimal compression strategy** that minimizes rate (sparsity loss) while preserving decoder alignment (reconstruction fidelity). The precision-recall asymmetry is the signature of this behavior—decoder directions remain accurate even when encoder activation is suppressed. This reframing explains why absorption persists but is benign: it is not a pathology but a consequence of the rate-distortion trade-off in lossy compression.

## 1.6 Contributions

1. First systematic test of absorption-downstream correlation with multiple comparison correction (12 tests, Bonferroni alpha = 0.00417)
2. First falsification of decoder-correlation-based absorption prediction (precision@20 = 0.0)
3. First precision-recall decomposition for absorption analysis, establishing absorption as a coverage problem not a selectivity problem
4. First demonstration that random SAEs show significantly higher absorption than trained SAEs, reframing absorption as a structural artifact
5. Reusable methodological framework: baseline correction, precision-recall decomposition, EC50 dose-response analysis

## 1.7 Paper Structure

Section 2 provides background on SAEs and absorption. Section 3 describes our methodology. Section 4 reports experimental results. Section 5 discusses implications. Section 6 addresses limitations and future work. Section 7 concludes.

<!-- FIGURES
- None
-->