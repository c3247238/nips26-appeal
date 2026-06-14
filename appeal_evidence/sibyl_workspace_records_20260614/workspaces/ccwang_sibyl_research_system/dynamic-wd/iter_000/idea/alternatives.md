# Backup Ideas for Potential Pivot

## Alternative A: Pure Empirical Alignment Characterization Study

### Motivation
If the theoretical contributions (Theorems 3.1, 4.1, 5.1) prove too difficult to close, or if the alignment proxy turns out to be too noisy for practical use, we can pivot to a purely empirical study that provides the first systematic characterization of gradient-parameter alignment dynamics in deep learning.

### Core Contribution
- Systematic tracking of delta_t across multiple architectures (ResNet, VGG, ViT), datasets (CIFAR-10/100, ImageNet subset), optimizers (SGD, SGD+momentum, AdamW), and weight decay settings
- Phase-dependent alignment analysis: how delta_t evolves during warmup, stable training, and learning rate decay phases
- Per-layer alignment disaggregation: show that different layers (Conv, BN, FC) exhibit qualitatively different alignment trajectories
- Distribution analysis: bimodal vs. unimodal delta_t distributions, temporal autocorrelation
- Correlation with generalization: relate alignment statistics to train-test gap

### Target Venue
NeurIPS or ICML empirical/analysis track, or TMLR

### Risk Assessment
- Low risk: purely observational, no algorithm to fail
- Medium impact: provides foundation for future work but limited algorithmic contribution
- High feasibility: all experiments complete within ~8 hours on 4x RTX PRO 6000

### Differentiation from Main Proposal
This drops all theory and the dynamic WD algorithm, focusing entirely on the empirical phenomenon. It is the "safe harbor" if everything else fails.

---

## Alternative B: Alignment-Aware Weight Decay for LLM Pre-training (Applied Focus)

### Motivation
The contrarian correctly identifies that CIFAR experiments have limited impact in 2026. If our alignment-aware method shows promise on small-scale experiments, we can pivot to an applied study targeting LLM pre-training, which is the highest-impact application area.

### Core Contribution
- Implement per-layer alignment-aware WD for AdamW (the dominant LLM optimizer)
- Use NanoGPT or LitGPT to train small transformers (60M-350M parameters) on OpenWebText
- Compare against uniform WD, AlphaDecay (spectral-based), and CWD
- Show that alignment-aware WD provides better perplexity at equivalent compute
- Connect layer-level alignment to attention vs. FFN weight dynamics

### Target Venue
EMNLP, NAACL, or workshop papers at NeurIPS/ICML

### Risk Assessment
- Medium risk: transformer training is more complex, alignment behavior may differ from CNNs
- High impact if successful: directly relevant to LLM community
- Medium feasibility: NanoGPT experiments take 2-4 hours, but tuning takes longer

### Key Technical Differences from Main Proposal
- AdamW instead of SGD (requires re-deriving alignment interaction with adaptive preconditioning)
- Per-layer rather than global alignment (AlphaDecay-style granularity but dynamic)
- No convergence theory (purely empirical + heuristic justification)
- Drops theoretical contributions for practical relevance

### Differentiation from AlphaDecay
AlphaDecay uses **static** spectral analysis; we use **dynamic** alignment per step. AlphaDecay assigns decay once; we continuously adapt. The combination of AlphaDecay's module selection with our step-level dynamics could be a strong hybrid.

---

## Alternative C: Minimal Theory Extension (Theorem 3.1 Only + Strong Experiments)

### Motivation
If the cumulative contraction bound (Theorem 4.1) proves too hard to close (the Abel summation and non-linear coupling between trajectories are genuine technical barriers), we can submit a paper with only Theorem 3.1 (time-varying SGDW convergence) plus strong experiments.

### Core Contribution
- Theorem 3.1: Time-varying SGDW preserves O(T^{-1/2}) convergence under lambda_t = O(gamma_t^2)
  - This is already novel and has a clear proof path
- Comprehensive experimental comparison: AADWD vs. Fixed WD vs. CWD vs. AlphaDecay vs. Stagewise
- Ablation studies demonstrating alignment signal value
- Alignment trajectory visualization as empirical contribution

### Target Venue
AISTATS or COLT (theory-oriented) or NeurIPS (if experiments are strong enough)

### Risk Assessment
- Low risk: Theorem 3.1 has high feasibility (2-3 weeks, per theoretical perspective)
- Lower impact: one theorem instead of three
- High feasibility: the safest path to a publishable result

### How to Strengthen
- Add comparison theorem: formally show that dynamic WD with alignment-aware schedule dominates fixed WD under alignment variance condition (even if the full cumulative contraction bound is not closed)
- Add AdamW extension experiments (not theory) to broaden appeal

---

## Pivot Decision Framework

| Trigger | Pivot To |
|---------|----------|
| delta_hat_t noise makes proxy unreliable (Tier 0 fails) | Alternative A (empirical study) |
| Theory Theorems 4.1/5.1 cannot be closed in time | Alternative C (Theorem 3.1 + experiments) |
| CIFAR experiments show < 0.1% improvement | Alternative B (LLM focus) or reframe as robustness story |
| CWD dominates in all experiments | Alternative A + focus on "why CWD works" theoretical explanation |
| Everything works but reviewer pushback on scale | Extend to Alternative B as follow-up |
