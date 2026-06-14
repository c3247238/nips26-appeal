# 7. Conclusion

We introduced the Phi Modulator Framework, the first unified mathematical abstraction for dynamic weight decay methods in deep learning. The framework expresses the weight decay update as $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, recovering Cautious Weight Decay, Scheduled Weight Decay, cosine schedules, Weight Norm Control, AlphaDecay, and all their compositions as special cases of a single modulation function $\varphi$ along four orthogonal axes: temporal, directional, spatial, and target-norm.

Three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---provide the first quantitative tools for characterizing weight decay behavior beyond final accuracy. BEM disentangles modulation strategy from total decay budget; CSI captures coupling stability between WD and optimizer dynamics; AIS measures whether gradient-weight alignment carries exploitable signal.

A systematic evaluation of 126 controlled experiments across 7 methods, 2 optimizers (AdamW, SGD), 2 architectures (ResNet-20, VGG-16-BN), 2 datasets (CIFAR-10, CIFAR-100), and 3 seeds reveals two complementary findings:

1. **Under AdamW, all dynamic weight decay variants are statistically equivalent to constant weight decay** ($p > 0.05$ for all paired comparisons), including the degenerate case of no weight decay. A ten-fold variation in effective WD budget produces less than 0.5% accuracy variation.

2. **Under SGD, weight decay magnitude matters.** Removing WD entirely drops accuracy by 0.92% on CIFAR-10 and 1.71% on CIFAR-100 (ResNet-20), with weight norms nearly doubling. Constant WD achieves the best SGD results.

The Phi Invariance Conjecture---that AdamW's adaptive per-parameter scaling subsumes any Phi modulator's effect---provides a mechanistic explanation supported by weight norm convergence analysis (1.2% variation under AdamW vs. 97% under SGD), alignment informativeness diagnostics (AIS invariant across methods), and the confirmed SGD boundary condition.

Practitioners using AdamW can safely rely on constant weight decay---the simplest strategy already captures the available benefit. Dynamic weight decay research should prioritize conditions where Phi Invariance breaks: SGD without batch normalization, ImageNet and LLM scale, and Vision Transformer architectures where the conjecture's boundary conditions are most likely violated.

<!-- FIGURES
- None
-->
