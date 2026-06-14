# Backup Ideas for Potential Pivot

## Alternative 1: Contrarian Mechanism Decomposition Paper

**Title**: "Rethinking Dynamic Weight Decay: Why Temporal Scheduling Is a Red Herring and Spatial Modulation Is All You Need"

**Core Claim**: The temporal dimension of dynamic WD is provably ineffective under budget equivalence. The spatial dimension (per-parameter, per-layer modulation) is the only axis where dynamic WD provides genuine benefit, and this benefit operates through effective learning rate modulation rather than regularization optimization.

**When to pivot**: If the unified framework's theoretical contributions are perceived as "just notation" or if proximal characterization of CWD is too crude. This alternative focuses purely on the contrarian empirical message with minimal theory.

**Key experiments**:
1. Budget equivalence verification at scale (5 temporal schedules, mean-matched constant WD)
2. CWD mechanism decomposition (effective-lambda, random mask, inverted mask, continuous alignment)
3. Effective LR equivalence test (CWD-LR variant with constant WD but equivalent per-parameter LR)
4. Critical period interaction (CWD only in first/last 10% of training)

**Novelty**: 7/10 -- Primarily empirical but addresses the most important practical question in WD research.

**Advantage**: Clean, high-impact message. Every experiment has a clear pass/fail criterion. Even if some hypotheses fail, the controlled comparison is valuable.

**Risk**: May face publication resistance as a "negative result" paper. Mitigated if at least one constructive insight survives (e.g., spatial modulation works for the right reasons).

---

## Alternative 2: Rigorous Empirical Assessment with Standardized Metrics

**Title**: "Measuring What Matters: A Rigorous Empirical Assessment of Dynamic Weight Decay Methods with Standardized Evaluation Metrics"

**Core Claim**: Under compute-controlled, equal-tuning-effort conditions, the performance differences between dynamic WD methods and optimally tuned constant WD are smaller than reported in individual papers. The proposed standardized metrics (BEM, CSI, AIS) provide predictive power for method rankings that raw accuracy alone does not capture.

**When to pivot**: If the theoretical framework proves difficult to complete rigorously (e.g., proximal characterization requires too many approximations) or if reviewers want a purely empirical contribution.

**Key experiments**:
- Full 7-method x 4-benchmark x 5-seed benchmark with equal HP tuning budget
- CWD falsification battery
- Metric validation (CSI/AIS correlation with accuracy)
- Cross-validation: train metrics on CIFAR, predict rankings on ImageNet

**Novelty**: 6/10 -- Benchmark + metrics is a well-established paper format. Novelty comes from being the first to do this for WD methods specifically.

**Advantage**: Low theoretical risk. The experiments are the contribution. Works regardless of which hypotheses are confirmed.

**Risk**: Moderate -- benchmark papers sometimes struggle at top venues unless the findings are surprising. Our budget equivalence result is surprising enough to carry the paper.

---

## Alternative 3: Thermodynamic-Allostatic WD Framework (High-Risk, High-Reward)

**Title**: "Allostatic Weight Decay: A Multi-Timescale Regulation Framework from Synaptic Homeostasis to Thermodynamic Process Optimization"

**Core Claim**: Weight decay in deep learning is the computational analogue of synaptic homeostasis. The four WD sub-approaches correspond to different levels of the biological regulation hierarchy. A multi-timescale allostatic controller (fast reactive + medium adaptive + slow predictive + periodic consolidation) outperforms all single-mechanism approaches.

**When to pivot**: If the proximal framework paper is scooped or if an ambitious interdisciplinary framing would be better received. This alternative is richer but riskier.

**Key experiments**:
1. Three-level allostatic WD controller ablation
2. Periodic WD boost ("sleep phase") with 10-20% duty cycle
3. Anticipatory WD at LR transitions
4. Thermodynamic state diagrams (WD vs norm vs loss)

**Novelty**: 9/10 -- Allostatic WD is genuinely novel (0 arXiv results for "allostatic weight decay"). The multi-timescale hierarchy and periodic WD boost have no precedent.

**Advantage**: Highest novelty and interdisciplinary appeal. Could attract attention from both ML and computational neuroscience communities.

**Risk**: High -- The allostatic controller has many hyperparameters (3 levels x 2-3 params each). May be perceived as over-engineered. The biological analogy may be seen as superficial by ML reviewers. The improvement margins may be small despite the framework's complexity.

**Recommended conditions for pivot**: Only if (a) the unified framework paper encounters a fatal theoretical flaw, (b) pilot experiments for the allostatic controller show >1% improvement on CIFAR-100, and (c) the periodic WD boost independently improves norm stability.
