# 5. Discussion

## 5.1 UAD: A Genuine Methodological Advance

UAD is the first method to detect feature absorption without ground-truth parent features or supervised probe directions. The F1 = 0.725 with perfect recall on GPT-2 Small layer 8 demonstrates that co-occurrence clustering captures a real structural signature of absorption. The precision of 0.569 means 43% of same-cluster pairs are false positives---UAD is a detection tool requiring post-hoc filtering, not a finished solution.

The key advance is qualitative, not quantitative. Prior methods [Chanin et al., 2024; Karvonen et al., 2025] require knowing what to look for. UAD requires only unlabeled text. This eliminates the supervision bottleneck that has constrained all prior absorption detection to known concepts---precisely the domain where SAEs are least needed.

## 5.2 The Determinism-Robustness Distinction

Perfect multi-seed consistency (F1 = 0.725 across all seeds) reflects determinism, not robustness. UAD operates on a fixed SAE's co-occurrence matrix, and 1,000 samples suffice to stabilize phi coefficient statistics. The SAE weights are frozen; corpus sampling randomness does not perturb cluster assignments for the top 500 features.

This is a practical advantage---full reproducibility without variance---but it does NOT demonstrate robustness to SAE retraining, corpus change, or model change. A true robustness test would require (1) retraining the SAE with different initializations, (2) evaluating on different corpora, or (3) testing on different model architectures. These remain future work.

## 5.3 Cross-Layer Variation

UAD performance varies by layer (F1 = 0.432 to 0.725). Layer 8 is optimal, consistent with prior work showing mid-to-late layers contain the most structured feature hierarchies [Elhage et al., 2022]. Layer 4's lower F1 (precision = 0.276) may reflect weaker hierarchical structure in early layers, where features encode lower-level positional and syntactic patterns rather than semantic abstractions.

All layers maintain perfect recall, confirming that UAD does not miss collisions. The precision drop in early layers simply means more false positives---the clustering still captures absorption signatures, but with more noise from non-absorbed co-occurring features.

## 5.4 DFDA: Preliminary Status

DFDA's 99.5% improvement metric is artifactual. The MLP learns to predict near-zero parent values in child-dominant positions, not to recover absorbed activations. The residual compensation architecture---predicting a parent residual from child activation and adding it to the parent's SAE output---is conceptually sound. But the evaluation protocol is flawed: measuring MSE on child-dominant examples (where the parent is already near zero) conflates near-zero prediction with absorption recovery.

A valid evaluation would test on parent-positive examples: inputs where the parent feature should activate according to ground truth, regardless of child presence. If DFDA increases parent activation on these examples without degrading reconstruction, the method would be validated. This protocol is future work.

## 5.5 Limitations

1. **Single model.** Only GPT-2 Small is fully validated. Cross-model evaluation was blocked: Gemma-2B is gated on HuggingFace and Pythia-2.8B SAEs were unavailable via SAELens at experiment time.

2. **Single concept domain.** Ground truth uses first-letter features (a-z) only. Semantic hierarchies (WordNet) and more complex concept relationships are not validated.

3. **No random baseline.** F1 = 0.725 is unanchored---we do not know the F1 of random feature pair selection on the same SAE.

4. **No ablations.** The contributions of HAC (vs. other clustering methods), phi coefficient (vs. raw co-occurrence), and 2-layer MLP (vs. linear) are not isolated.

5. **English only.** All experiments use English text from OpenWebText.

6. **Single SAE configuration.** One SAE width ($d_{\text{SAE}} = 24{,}576$) and one sparsity level per model.

## 5.6 Future Work

- Cross-model validation (Gemma-2B, Pythia-2.8B) when access is resolved
- Semantic hierarchy validation (WordNet, concept hierarchies beyond first letters)
- Random baseline and ablation suite (clustering method, normalization, MLP depth)
- DFDA metric rebuild: parent-positive evaluation protocol
- End-to-end pipeline: UAD -> DFDA -> probe accuracy improvement
- Human evaluation of feature interpretability for detected pairs

<!-- FIGURES
- None
-->
