# Backup Ideas for Potential Pivot

## Alternative 1: The Alignment Mirage

**Title**: The Alignment Mirage: When Gradient-Weight Geometry Misleads Dynamic Regularization

### Summary
Critical analysis demonstrating that in BN architectures, alignment-based WD is a noisy proxy for effective LR scheduling, not genuine alignment-aware regularization. First systematic measurement of alignment signal SNR.

### Core Hypothesis
In networks with batch normalization, gradient-weight alignment delta_t is a noisy proxy for effective LR dynamics, not a regularization signal. Alignment-aware WD in BN networks is an indirect, suboptimal way to schedule the effective learning rate.

### Evidence from iter_003
- Random mask (90.12%) matches CWD (90.06%) on CIFAR-10/ResNet-20
- CWD underperforms constant WD on both CIFAR-10 (-0.07%) and CIFAR-100 (-0.31%)
- The total spread across ALL methods on CIFAR-10 is only 0.25%

### Key Experiments
1. **Alignment Signal Quality Audit**: Measure SNR = Var(delta_t) / Var(delta_hat_t - delta_t) across training phases. Hypothesis: SNR < 1 for most of training.
2. **BN Confound Test**: Compare alignment-aware WD on ResNet-20 with BN vs without BN (fixup init). If alignment helps only without BN, the confound is confirmed.
3. **Effective-LR-Aware WD**: Simpler alternative computing effective LR per layer and scheduling WD to achieve target effective LR trajectory.
4. **Random Alignment Control**: Apply alignment-scaled WD with shuffled alignment values. If random performs comparably to true alignment, the signal is spurious.
5. **Matched effective-WD**: Adjust constant WD to match CWD's mean effective WD. Tests whether CWD simply finds a better average WD strength.

### Why This Works as Pivot
- Even negative results about alignment are publishable and valuable
- The BN confound has never been explicitly tested in the WD literature
- Aligns with the contrarian perspective's strongest argument
- Reusable: all diagnostic infrastructure from the main proposal carries over
- The "alignment mirage" framing is provocative and attracts attention
- Lower compute cost than main proposal

### Differentiation from prior work
Defazio (2025) and D'Angelo et al. (NeurIPS 2024) partially anticipate the core insight. Must add: systematic SNR measurement, formal BN vs non-BN comparison specifically for alignment-aware WD, and random mask control (never done).

### Novelty Score: 6/10
Sufficient for empirical contribution; needs SNR analysis to reach top venue.

### Resource Estimate
~80 GPU-hours

---

## Alternative 2: Architecture-Aware Weight Decay (Nuclear Norm Trap)

**Title**: Rethinking Dynamic Weight Decay: Parameter Structure Matters More Than Training Dynamics

### Summary
The performance variation across WD methods is dominated by WHICH parameters receive WD (parameter-type structure), not by HOW WD strength is modulated. Architecture-aware WD (different strategies for attention products, FFN weights, embeddings, normalization) provides larger gains than any dynamic WD method.

### Core Hypothesis
WD on multiplicatively-interacting parameters in attention layers (K-Q and V-P products) is equivalent to nuclear norm regularization (Kobayashi et al., NeurIPS 2024), causing potentially harmful low-rank bias. This structural effect is unaffected by any dynamic WD modulation and is the dominant source of performance variation.

### Key Experiments
1. **Factorial ablation**: {scheduling: fixed/cosine/SWD} x {alignment: none/CWD/continuous} x {parameter-type: uniform/no-bias-norm/architecture-aware}
2. **Architecture-aware WD**: no WD on K-Q/V-P attention products + standard WD on FFN + no WD on norms/biases
3. **Nuclear norm impact**: Measure attention matrix rank with and without WD on ViT architectures
4. **Comparison**: Standard AdamW vs CWD vs architecture-aware WD vs NaP-on-attention + WD-on-FFN

### Why This Works as Pivot
- Backed by rigorous theory (Kobayashi et al., NeurIPS 2024)
- Challenges the entire dynamic WD research agenda, not just one method
- Simple, actionable prescription (no WD on attention products)
- Relevant as transformers become dominant architecture
- If low-rank bias is beneficial (counter-finding), that is equally publishable

### Novelty Score: 7/10
Higher than alignment mirage because nuclear norm angle is less anticipated and more actionable.

### Resource Estimate
~60 GPU-hours (requires ViT experiments)

---

## Alternative 3: Comprehensive Diagnostic Benchmark

**Title**: Unifying Dynamic Weight Decay: When Does Adaptive Scheduling Actually Help?

### Summary
If theoretical contributions prove too hard to formalize rigorously, pivot to comprehensive empirical benchmark with standardized diagnostic metrics. Emphasis on revealing WHY methods converge in performance.

### Key Components
1. **Unified PyTorch WD Optimizer** (~200 LOC): All methods as pluggable presets
2. **Diagnostic Metrics**: CSI, AIS, BEM with validation against generalization gap
3. **Comprehensive Grid**: 7+ methods x 3 architectures x 2-3 datasets x 3-5 seeds
4. **Actionable Finding**: Show CSI predicts when WD scheduling helps

### Why This Works as Pivot
- No deep theorems required
- Builds on existing why-weight-decay codebase
- Reuses all iter_003 data
- "Revisiting" papers publish well when done comprehensively

### Novelty Score: 5/10
Sufficient for workshop or systems paper; needs ImageNet scale for top venue.

### Resource Estimate
~120 GPU-hours

---

## Pivot Decision Tree

```
Main proposal (Lyapunov certificate + cumulative alignment + PMP-WD)
  |
  +-- Certificate too conservative?
  |     +-- YES: Certificate predicts "constant WD is near-optimal" => Still publish with this as finding
  |     +-- YES but also boring: Pivot to Alternative 3 (Benchmark)
  |
  +-- Alignment uninformative?
  |     +-- YES, BN confound explains it: Pivot to Alternative 1 (Alignment Mirage)
  |     +-- YES, fundamentally uninformative: Pivot to Alternative 2 (Architecture-Aware)
  |
  +-- Proofs have fatal gaps?
  |     +-- YES: Pivot to Alternative 3 (Benchmark) with unified operator as primary contribution
  |
  +-- ViT results show dramatic parameter-type effects?
  |     +-- YES: Elevate Alternative 2 (Architecture-Aware) to front-runner
  |
  +-- Everything works: Publish main proposal as planned
```
