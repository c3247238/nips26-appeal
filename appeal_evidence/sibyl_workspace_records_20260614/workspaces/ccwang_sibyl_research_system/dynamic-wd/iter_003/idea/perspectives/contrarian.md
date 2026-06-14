# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Dynamic/adaptive weight decay provides meaningful improvements over well-tuned constant weight decay.**
   - Evidence challenging it:
     - Wen et al. 2025, "Fantastic Pretraining Optimizers and Where to Find Them" (arXiv:2509.02046): Systematic comparison of 10 optimizers at 0.1B-1.2B scales shows speedups of novel optimizers over well-tuned AdamW **shrink from 1.4x at 0.1B to merely 1.1x at 1.2B**. If entire novel optimizer architectures only give 1.1x speedup at scale, what hope does tweaking the WD coefficient have?
     - SWD (Xie et al. 2020/2023, arXiv:2011.11152): The authors themselves admit their improvement is "marginal for complex loss landscapes such as ImageNet."
     - FAdam (Hwang 2024, arXiv:2405.12807): Attempted to correct WD using diagonal Fisher information -- the paper was **withdrawn from ICLR 2025** because the practical gains did not survive rigorous review.
     - Previous iteration findings from this project: "Budget Equivalence: mean(lambda_t) = fixed lambda yields identical performance." This directly contradicts the premise that scheduling WD matters.

2. **Assumption: Gradient-weight alignment is an informative signal for modulating weight decay.**
   - Evidence challenging it:
     - Previous iteration findings: "Alignment Signal Inapplicability: grad-weight alignment uninformative at nonconvex scale." This was already experimentally confirmed in earlier iterations of this project.
     - Draganov et al. 2024, "The Hidden Pitfalls of the Cosine Similarity Loss" (arXiv:2406.16468, ICML HiLD Workshop 2024): Proves that cosine similarity gradients go to zero when embeddings have large magnitude -- an unavoidable condition in practice. The cosine similarity between gradient and weight vectors (the foundation of alignment-aware WD methods like CWD) suffers from the same pathology: in high dimensions with large weight norms, the alignment signal degrades to noise.
     - Zhang et al. 2018, "Three Mechanisms of Weight Decay Regularization" (arXiv:1810.12281): Found that "almost all of the regularization effect of weight decay was due to applying it to layers with BN (for which weight decay is meaningless)" -- the mechanism is effective learning rate adjustment, not anything related to gradient-weight alignment.

3. **Assumption: Weight decay acts as regularization, justifying complex adaptive schemes to optimize its "regularization strength."**
   - Evidence challenging it:
     - D'Angelo et al. 2024, "Why Do We Need Weight Decay in Modern Deep Learning?" (NeurIPS 2024, arXiv:2310.04415): Landmark result -- "weight decay is **never useful as an explicit regularizer** but instead changes the training dynamics in a desirable way." WD's actual mechanisms are loss stabilization (SGD) and bias-variance tradeoff (LLMs). If WD is not regularization, then the entire premise of "optimizing regularization strength dynamically" is built on a misconception.
     - Hernandez-Garcia & Konig 2018, "Do Deep Nets Really Need Weight Decay and Dropout?" (arXiv:1802.07042): Systematic ablation shows weight decay and dropout are **unnecessary for object recognition if enough data augmentation is introduced.** With heavier augmentation, models without WD match or exceed models with WD.

4. **Assumption: A "unified framework" encompassing all WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) would be scientifically valuable.**
   - Evidence challenging it:
     - The four sub-approaches may address genuinely different phenomena that resist unification. Decoupled WD (AdamW) addresses the L2 vs. WD inequivalence in adaptive optimizers -- a **structural** property of the optimizer, not a property of weight decay itself. Alignment-aware WD (CWD) addresses coordinate-wise sign conflicts -- a **geometric** property. Norm-matched WD addresses target norm control -- an **equilibrium** property. Scheduling addresses temporal dynamics -- a **temporal** property. Forcing these into one formula lambda(t, w, g, tau) may produce a Frankenstein equation that is technically correct but scientifically vacuous, explaining nothing while claiming to explain everything.
     - Historically, "unified frameworks" in optimization that subsume many methods as special cases often become tautological. If every method is a special case, the framework's predictive power is inversely proportional to its generality.

5. **Assumption: The proposed standardized metrics (BEM, CSI, AIS) would be useful and adopted.**
   - Evidence challenging it:
     - BEM (Budget Equivalence Metric): Previous iteration already found budget equivalence holds trivially -- mean(lambda_t) = constant lambda gives identical performance. This means BEM might simply confirm that all methods are equivalent under compute normalization, which is a negative result for the field, not a useful metric.
     - CSI (Coupling Stability Index): No clear evidence that "coupling stability" is actually the bottleneck in WD methods. Zhang et al. 2018's finding that WD mainly works through effective learning rate adjustment suggests the coupling is already well-understood.
     - AIS (Alignment Informativeness Score): If alignment is uninformative (as the project's own pilot experiments showed), then a metric measuring "alignment informativeness" would consistently return low scores -- confirming that it should not be used, rather than providing useful differentiation between methods.

6. **Assumption: CWD (ICLR 2026) represents a genuine advance in alignment-aware WD.**
   - Evidence challenging it:
     - CWD uses a binary sign-alignment mask -- the crudest possible discretization of alignment. The theoretical justification involves a bilevel Pareto-optimal interpretation and sliding-mode behavior, but the paper acknowledges that "the ODEs and Lyapunov candidates are treated as smooth even though the dynamics include the discontinuous indicator function." The rigorous analysis via Filippov's framework is deferred as future work.
     - A community report noted that CWD **worsened performance when applied to the Conda optimizer**, contradicting the "optimizer-agnostic" claim.
     - The improvement margins are small: 0.1-0.6% accuracy improvement. At billion-parameter scale, this may well fall within the noise of different random seeds and hyperparameter configurations.

7. **Assumption: Weight decay's interaction with normalization layers (BN/LN) is well-understood and can be systematically exploited.**
   - Evidence challenging it:
     - Zhou et al. 2021, "FixNorm: Dissecting Weight Decay for Training Deep Neural Networks" (arXiv:2103.15345): Proposes discarding weight decay entirely and directly controlling its two mechanisms (effective LR and cross-boundary risk). Achieves 77.7% on ImageNet with EfficientNet-B0, outperforming the original baseline. The implication: weight decay is a blunt instrument that conflates multiple distinct effects, and directly controlling those effects is superior.
     - Zhang et al. 2018 (Three Mechanisms): The three mechanisms (effective LR, Jacobian regularization, damping reduction) are **architecture-dependent and optimizer-dependent.** There is no single "WD mechanism" to unify -- different combinations matter in different settings.

8. **Assumption: Visualization and diagnostic frameworks for WD would reveal actionable insights.**
   - Evidence challenging it:
     - Weight norm trajectories, gradient-weight alignment cosines, and spectral density plots are already standard in individual papers (CWD, SWD, AlphaDecay, OUI all include them). The marginal value of systematizing these into a "framework" may be low -- the problem is not lack of visualization tools but lack of consistent experimental methodology.
     - The risk is producing a "diagnostic dashboard" that generates many pretty plots but does not actually help practitioners choose better WD strategies.

### Landscape of Doubt

The weight decay research landscape is riddled with a fundamental tension: **the more we understand about WD, the less it looks like something worth optimizing dynamically.**

D'Angelo et al. (NeurIPS 2024) delivered the clearest verdict: WD is not regularization. It is a training dynamics modifier. Its primary effect in networks with normalization is controlling the effective learning rate (Zhang et al. 2018). Once you understand this, the entire edifice of "dynamic WD" looks like an elaborate way to indirectly adjust the learning rate -- something we already know how to do directly with LR scheduling.

The budget equivalence finding from this project's earlier iterations is devastating: if mean(lambda_t) = constant lambda gives the same result, then WD scheduling is a red herring. You can schedule WD however you like, as long as the average matches a well-tuned constant. This means the elaborate theoretical machinery of SWD, ADANA, and proposed scheduling approaches adds complexity without benefit.

The alignment signal finding is equally damaging: if gradient-weight alignment is uninformative at the nonconvex scale of real deep learning, then CWD's success cannot be attributed to alignment-awareness. Instead, CWD might succeed simply because it applies **less total weight decay** (by masking out some coordinates) -- a trivially equivalent effect to reducing the WD coefficient.

The optimizer scaling results from Wen et al. (2025) provide the killing blow: even radically different optimizers (Muon, SOAP) converge to within 1.1x of well-tuned AdamW at 1.2B scale. If an entirely new optimization algorithm cannot meaningfully outperform AdamW at scale, what hope does a minor WD modification have?

This landscape suggests that the "Unified Dynamic Weight Decay Framework" risks being a solution to a non-problem. The real finding may be the **null hypothesis**: constant, well-tuned weight decay is sufficient, and dynamic modifications provide marginal-to-zero benefit at practical scales.


## Phase 2: Initial Candidates

### Candidate A: The Null Hypothesis Paper -- "When Dynamic Weight Decay Doesn't Matter"

- **Challenged assumption**: Dynamic/adaptive/alignment-aware weight decay methods provide meaningful improvements over well-tuned constant weight decay.
- **Evidence against**: Budget equivalence (own pilot data), optimizer scaling diminishing returns (Wen et al. 2025), WD-as-dynamics-modifier not regularizer (D'Angelo et al. 2024), alignment signal uninformative (own pilot data), SWD marginal on ImageNet (Xie et al. 2020).
- **Contrarian hypothesis**: Under properly controlled conditions (equal compute budget, equal hyperparameter tuning effort, same architectures), the performance gap between constant WD and any dynamic WD method is statistically insignificant on standard benchmarks at practical scale. The reported improvements in individual papers are artifacts of unequal hyperparameter tuning, different compute budgets, or cherry-picked evaluation points.
- **Exploitation plan**: Run the definitive head-to-head comparison that nobody has done. All major WD methods (AdamW constant, CWD, SWD/AdamS, cosine WD schedule, AdamWN) on the same codebase, same compute budget, same hyperparameter search protocol (grid search with equal number of trials per method), same seeds, same architectures (ResNet-20/VGG-16-BN on CIFAR, ResNet-50/ViT on ImageNet). Report mean +/- std with proper statistical tests. The contribution is methodological rigor, not algorithmic novelty.
- **Novelty estimate**: 6/10 (the finding itself is contrarian; the methodology is straightforward but nobody has done it properly)

### Candidate B: The Effective Learning Rate Reduction -- "Dynamic Weight Decay Is Just Indirect Learning Rate Scheduling"

- **Challenged assumption**: Dynamic WD methods work because they optimize the regularization-generalization tradeoff. Their success mechanisms are fundamentally different from learning rate scheduling.
- **Evidence against**: Zhang et al. 2018 (WD = effective LR for BN layers), D'Angelo et al. 2024 (WD = dynamics modifier, not regularizer), Wan et al. 2020 (spherical motion dynamics = angular update governed by eta*lambda), FixNorm (Zhou et al. 2021: directly controlling effective LR + cross-boundary risk outperforms WD).
- **Contrarian hypothesis**: Every successful dynamic WD method can be equivalently expressed as a specific learning rate schedule. CWD is equivalent to a coordinate-wise LR increase (by reducing the norm-shrinking that lowers effective LR). SWD is equivalent to a gradient-norm-aware LR schedule. The "alignment" and "scheduling" aspects are epiphenomena -- the real mechanism is always effective learning rate modulation.
- **Exploitation plan**: For each dynamic WD method, derive the equivalent effective LR schedule analytically. Then implement the equivalent LR schedule with constant WD and show it matches the dynamic WD method's performance. This would be a clean, falsifiable claim: dynamic WD is LR scheduling in disguise.
- **Novelty estimate**: 7/10 (the analytical equivalences would be novel; the core intuition has been hinted at but never rigorously demonstrated)

### Candidate C: The Alignment Mirage -- "Why Gradient-Weight Alignment Is Uninformative and CWD Works for the Wrong Reasons"

- **Challenged assumption**: CWD's sign-alignment mask is the source of its improvement. Gradient-weight alignment is a useful signal for WD modulation.
- **Evidence against**: Project's own pilot finding (alignment signal uninformative at nonconvex scale), cosine similarity pitfalls in high dimensions (Draganov et al. 2024), CWD works but alignment itself is noisy (binary sign is the crudest discretization -- it's robust to noise precisely because it throws away all magnitude information).
- **Contrarian hypothesis**: CWD improves over AdamW not because of alignment awareness but because it applies **less total weight decay** by masking out ~50% of coordinates. This is equivalent to reducing lambda by ~50%. If we run AdamW with lambda/2, we should get similar performance to CWD. Furthermore, the sign-alignment mask acts as a form of **coordinate-wise dropout on the WD term**, which provides stochastic regularization benefits independent of any alignment signal.
- **Exploitation plan**: (1) Compare CWD to AdamW-with-half-lambda. (2) Compare CWD to AdamW-with-random-50%-mask (random coordinates, not alignment-based). (3) Measure the actual fraction of coordinates masked per layer per step -- if it is consistently ~50%, the alignment signal is indistinguishable from a coin flip. (4) Derive the expected masking fraction analytically for Gaussian gradient-weight distributions.
- **Novelty estimate**: 8/10 (this directly challenges the ICLR 2026 paper and would be highly impactful if correct)


## Phase 3: Self-Critique

### Against Candidate A (The Null Hypothesis)

- **Steelman the conventional view**: CWD (ICLR 2026) reports consistent improvements across 111M to 2B parameter scales, with the gap **stable or widening** with model size. This contradicts the "diminishing returns at scale" narrative from Wen et al. 2025 (which studied optimizers, not WD modifications). SWD was accepted at NeurIPS 2023 with clear improvements on Adam/CIFAR. Multiple independent groups have found dynamic WD benefits -- this is not a single-lab artifact.
- **Cherry-picking check**: I am selectively citing the budget equivalence finding (from one project's pilots) and the optimizer scaling result (about entire optimizers, not WD). The SWD "marginal on ImageNet" quote is the authors being cautious, not a definitive negative result. CWD's improvements at billion scale are hard to dismiss.
- **Confounding check**: The budget equivalence finding (mean lambda_t = constant lambda) may only hold for specific scheduling shapes. If a WD schedule front-loads decay (strong early, weak late), the trajectory effect may differ from constant WD even if the average matches. Golatkar et al. 2019 showed that regularization in the "critical period" matters more than total regularization budget.
- **Actionability check**: Even if the null hypothesis is correct, publishing it is valuable for the field -- preventing wasted research effort on dynamic WD. But a pure "nothing works" paper is hard to publish without also providing constructive insight.
- **Verdict**: MODERATE. The null hypothesis is plausible but the evidence is not yet strong enough to overcome the multiple positive results from CWD/SWD. The critical period argument (Golatkar et al. 2019) is a serious confound.

### Against Candidate B (Effective LR Reduction)

- **Steelman**: For networks with normalization (BN/LN), WD indeed operates primarily through effective learning rate. But not all parameters have normalization -- the final classification layer, embedding layers, and bias terms are typically excluded from normalization. For these parameters, WD operates through genuine norm shrinkage, and the effective LR equivalence breaks down. Furthermore, AdamO (arXiv:2602.05136) explicitly separates radial (norm) and tangential (direction) dynamics, showing they have distinct effects on optimization. The "WD = LR scheduling" claim over-simplifies.
- **Cherry-picking check**: I am heavily relying on Zhang et al. 2018's finding about BN layers, but this was specifically for SGD. For Adam/AdamW, the interaction is more complex because the adaptive learning rate per parameter is already a form of scale-free update. Zhuang et al. 2022 (arXiv:2202.00089) show AdamW's advantage is specifically due to scale-freeness, not just effective LR.
- **Confounding check**: Effective LR is one dimension of WD's effect. WD also induces low-rank bias (Galanti et al. 2022), neural collapse (Jacot et al. 2024), and rotational equilibrium (Kosson et al. 2023). These structural effects cannot be replicated by LR scheduling alone. The "reduction to LR" misses the geometry.
- **Actionability check**: If we can show the LR equivalence for each method, this provides genuine insight for practitioners. But the incomplete equivalence (parameters without normalization, structural effects) weakens the claim.
- **Verdict**: MODERATE. The core insight is partially correct (for BN/LN layers) but the full claim is too strong. The structural effects of WD (low-rank, neural collapse) are not reducible to LR scheduling.

### Against Candidate C (The Alignment Mirage)

- **Steelman**: CWD's ablation study (Table 3 in CWD paper) explicitly tested the "random mask" baseline -- AdamW with a random 50% mask performed **significantly worse** than CWD (loss 2.82 vs 2.56). This directly contradicts the hypothesis that CWD works by simply reducing total decay. The sign-alignment mask is doing something that a random mask does not.
- **Cherry-picking check**: I am proposing that alignment is noise, but CWD's ablation against random masking is strong evidence that it is not pure noise. The binary sign discretization is crude but may capture a real signal despite high-dimensional noise.
- **Confounding check**: Even if the alignment signal is noisy, the binary sign function acts as a **hard thresholding** that extracts the sign bit -- the most robust bit of any noisy signal. CWD might work precisely because it uses the most noise-robust possible discretization (sign), not despite its crudeness. The question is whether the sign bit of alignment contains meaningful information, not whether the full cosine similarity does.
- **Actionability check**: The "CWD = half-lambda" comparison is trivially falsifiable. If CWD consistently outperforms AdamW-with-half-lambda and AdamW-with-random-mask, the alignment mirage hypothesis is dead. But if it doesn't, this would be a significant finding.
- **Verdict**: WEAK for the naive version (CWD = half-lambda -- already disproved by their ablation). But MODERATE for the refined version: CWD works, but not because of alignment per se -- it works because the sign mask happens to correlate with a useful structural property (e.g., it preferentially decays parameters that are growing in the "wrong" direction relative to the loss surface, which is related to effective LR rather than regularization).


## Phase 4: Refinement

### Dropped

- **Candidate A (pure null hypothesis)**: Too risky as a standalone paper. The critical period argument (Golatkar et al. 2019) and CWD's consistent results at scale weaken the claim that dynamic WD never matters. However, the budget equivalence finding remains a valuable component of a larger argument.

### Strengthened: Candidate B+C Hybrid -- "Rethinking Dynamic Weight Decay: Mechanism Decomposition Reveals the Load-Bearing Components"

The strongest contrarian position combines insights from both Candidates B and C:

1. **WD operates through multiple mechanisms** (effective LR, norm control, structural effects) and these mechanisms are **architecture-dependent and optimizer-dependent** (Zhang et al. 2018). Any "unified framework" that treats WD as a single phenomenon will miss this multi-mechanism structure.

2. **Dynamic WD methods succeed when they improve the effective LR dynamics**, not when they optimize regularization strength. CWD works because the sign mask preferentially decays parameters whose growth is counterproductive for the loss -- this is effective LR modulation, not alignment-based regularization.

3. **The alignment signal itself is noisy and uninformative** (project's pilot data, cosine similarity pitfalls in high dimensions), but **the sign bit is robust** (CWD's success). This means the continuous alignment score proposed by other perspectives (AIS) is likely to be noise-dominated, while binary sign thresholding is near-optimal for extracting whatever signal exists.

4. **Budget equivalence holds for mean-preserving WD schedules**, meaning temporal WD scheduling is a red herring. The only WD modifications that matter are **coordinate-wise** (like CWD) or **layer-wise** (like AlphaDecay), which change the spatial distribution of decay rather than its temporal profile.

### Additional corroboration

- **Golatkar et al. 2019** ("Time Matters in Regularizing Deep Networks"): Proved that WD only matters in the "critical period" early in training. After the critical period, WD has no effect on the final solution. This supports the view that temporal WD scheduling has limited value -- what matters is the WD during the critical period, not its trajectory over the full training.
- **Kosson et al. 2023** (Rotational Equilibrium): WD induces balanced angular rotation across layers. The equilibrium is reached regardless of the WD value (only the equilibrium angular speed changes). This means WD scheduling cannot change the qualitative dynamics -- only the quantitative speed.
- **FixNorm (Zhou et al. 2021)**: Directly controlling WD's mechanisms (effective LR + cross-boundary risk) outperforms WD itself on ImageNet. This is the strongest evidence that WD is a proxy for its underlying mechanisms, and directly controlling those mechanisms is superior to elaborate WD modifications.

### Selected front-runner

**"Rethinking Dynamic Weight Decay: Why Temporal Scheduling Is a Red Herring and Spatial Modulation Is All You Need"**

Core claim: The temporal dimension of dynamic WD (scheduling lambda(t)) is provably ineffective under budget equivalence. The spatial dimension (modulating WD across coordinates or layers) is the only axis where dynamic WD provides genuine benefit, and this benefit operates through effective learning rate modulation rather than regularization optimization.


## Phase 5: Final Proposal

### Title
**Rethinking Dynamic Weight Decay: Mechanism Decomposition Reveals When and Why Temporal Scheduling Fails While Spatial Modulation Succeeds**

### Challenged Assumption
The widely-held belief that dynamic weight decay -- including temporal scheduling (SWD, ADANA, cosine WD), alignment-aware modulation (CWD), and norm-targeted control (AdamWN) -- provides improvements by optimizing the regularization-generalization tradeoff. This assumption underpins the entire "Unified Dynamic Weight Decay Framework" proposal and the three proposed evaluation metrics (BEM, CSI, AIS).

### Evidence

**For the mainstream view (steelman):**
- CWD (ICLR 2026) shows consistent 0.1-0.6% improvements across 111M-2B parameter scales
- SWD (NeurIPS 2023) improves Adam on CIFAR benchmarks
- Multiple independent theoretical justifications exist: bilevel Pareto optimality (CWD), gradient-norm bounding (SWD), target-norm control (AdamWN)
- AdamO (2026) provides a clean radial/tangential decomposition showing distinct effects

**Against the mainstream view (our position):**
- D'Angelo et al. (NeurIPS 2024): WD is never useful as explicit regularization in modern DL; it changes training dynamics
- Budget equivalence (project pilots, iterations 0-2): mean(lambda_t) = constant lambda gives identical performance, meaning temporal WD scheduling is a no-op under budget normalization
- Alignment signal uninformative at nonconvex scale (project pilots): the cosine similarity between gradients and weights is too noisy to be useful as a continuous signal
- Wen et al. (2025): Optimizer improvements shrink to 1.1x at 1.2B scale -- even radical optimizer changes yield diminishing returns, let alone WD tweaks
- Zhang et al. (2018, Three Mechanisms): WD's regularization effect is architecture-and-optimizer-dependent, operating through 3 distinct mechanisms rather than a single "regularization strength"
- Hernandez-Garcia & Konig (2018): WD is unnecessary with sufficient data augmentation
- FixNorm (Zhou et al. 2021): Directly controlling WD's mechanisms outperforms WD itself
- Golatkar et al. (2019): WD only matters in the critical period; temporal scheduling post-critical-period is provably irrelevant

### Hypothesis
Dynamic weight decay operates on two orthogonal axes: temporal (scheduling lambda over time) and spatial (modulating lambda across coordinates/layers). **Temporal scheduling is provably ineffective under budget equivalence** -- any time-varying schedule with the same mean as a constant achieves identical performance. **Spatial modulation (CWD, AlphaDecay) succeeds** not through regularization optimization but through **effective learning rate redistribution** across parameters. The "alignment" in CWD is a noisy proxy for a simpler underlying signal: whether a parameter is growing counterproductively relative to the loss landscape.

### Method

**Experiment 1: Budget Equivalence Verification (Temporal axis)**
- Implement 5 temporal WD schedules: constant, cosine, linear decay, inverse-sqrt, SWD
- For each, fix the mean WD to the same value (1e-4 for SGD, 0.01 for AdamW)
- Train ResNet-20 on CIFAR-10/100 and ResNet-50 on ImageNet, 3 seeds each
- Prediction: All schedules achieve statistically indistinguishable final accuracy
- This validates/falsifies budget equivalence at scale

**Experiment 2: Spatial Modulation Mechanism Decomposition (CWD)**
- Implement CWD, AdamW-with-half-lambda, AdamW-with-random-50%-mask, AdamW-with-gradient-magnitude-mask (top 50% by |g_i|)
- Compare all on CIFAR-10/100 and ImageNet, 3 seeds each
- Track: per-layer masking fraction, effective LR per layer, weight norm trajectory
- Prediction: CWD outperforms random mask (confirmed by their ablation) but the mechanism is effective LR redistribution, not alignment-based regularization

**Experiment 3: Effective LR Equivalence Test**
- For CWD, derive the per-parameter effective LR under the sign mask
- Implement a "CWD-LR" variant: use constant WD but apply the equivalent per-parameter LR multiplier
- If CWD-LR matches CWD's performance, the mechanism is purely effective LR modulation

**Experiment 4: Critical Period Interaction**
- Apply CWD only in first 10% of training (critical period), constant WD thereafter
- Apply CWD only after first 10% of training, constant WD in critical period
- Prediction: Critical-period-only CWD captures most/all of CWD's benefit (consistent with Golatkar et al. 2019)

**Experiment 5: AIS Metric Validation**
- Compute the proposed Alignment Informativeness Score during training
- Compare AIS values between CWD (alignment-aware) and constant WD (alignment-agnostic)
- Prediction: AIS will show that alignment is noisy (high variance, low signal-to-noise ratio) and that the difference between alignment-aware and alignment-agnostic methods is not captured by alignment informativeness

### Baselines (properly tuned, not strawman)
- AdamW with grid-searched constant lambda (5 values: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
- CWD with same lambda grid
- SWD/AdamS with default hyperparameters from NeurIPS 2023 paper
- Each baseline gets equal hyperparameter search budget (fairness principle)

### Experimental plan
- **Architectures**: ResNet-20 (CIFAR), VGG-16-BN (CIFAR), ResNet-50 (ImageNet), ViT-Small (CIFAR/ImageNet)
- **Datasets**: CIFAR-10, CIFAR-100, ImageNet
- **Seeds**: 42, 123, 456
- **Compute**: 8x RTX PRO 6000 Blackwell (98GB each)
- **Time budget**: CIFAR experiments ~15 min each, ImageNet experiments ~4-8 hours each
- **Statistical tests**: Paired t-test or Wilcoxon signed-rank test for pairwise comparisons, with Bonferroni correction for multiple comparisons

### Risk Assessment
**What if the mainstream view turns out to be correct?**
- If temporal scheduling shows significant improvements over constant WD (violating budget equivalence), we still contribute the mechanism decomposition framework as a diagnostic tool
- If CWD's benefit is genuinely alignment-driven (not reducible to effective LR), we contribute the precise characterization of what alignment information CWD exploits
- If AIS is informative, we contribute the first rigorous validation of alignment-based metrics for WD
- In all cases, the controlled experimental comparison is a contribution regardless of which hypothesis wins

**What if both hypotheses are partly correct?**
- Most likely scenario: temporal scheduling has near-zero effect, spatial modulation has small but real effect, mechanism is a combination of effective LR and some genuine alignment signal. This "nuanced negative" result is still publishable and valuable.

### Novelty Claim
1. **First rigorous test of budget equivalence** for temporal WD scheduling at scale (CIFAR + ImageNet)
2. **Mechanism decomposition** showing spatial vs. temporal axes of dynamic WD are not equal
3. **Effective LR equivalence test** for CWD -- first direct test of whether CWD's benefit is reducible to LR redistribution
4. **Critical period interaction analysis** connecting Golatkar et al. 2019's theory to modern dynamic WD methods
5. **Contrarian reinterpretation**: The "unified framework" may be solving a non-problem on the temporal axis, while the spatial axis (which CWD and AlphaDecay address) works for reasons different from what their authors claim

### Implications for the Unified Framework

If this contrarian analysis holds, it reshapes the unified framework in several ways:

1. **Temporal scheduling should be de-emphasized** or presented as a null result (budget equivalence). This challenges SWD and ADANA as meaningful contributions.
2. **Alignment-aware WD should be reinterpreted** as effective LR modulation, not regularization optimization. This challenges the AIS metric's premise.
3. **The framework should center on spatial modulation** (per-layer, per-parameter) as the only axis where dynamic WD adds value.
4. **BEM becomes a diagnostic for the null hypothesis** rather than a comparison metric: if all methods score equally under BEM, it confirms budget equivalence.
5. **CSI should measure effective LR stability** rather than "coupling stability" -- since WD's primary mechanism is effective LR, the relevant stability metric is about the LR dynamics, not the WD-optimizer coupling per se.
6. **AIS is likely to be a negative result** -- measuring alignment informativeness will reveal that alignment is uninformative, which is itself an important finding but argues against alignment-aware WD rather than for it.

This contrarian analysis does not argue against the paper itself -- a unified framework that honestly reports both positive and negative findings (temporal scheduling = null, alignment = noisy proxy, spatial modulation = real but operates through LR) would be stronger and more impactful than one that uncritically promotes all four sub-approaches as equally valuable. The most impactful unified framework would be one that tells practitioners: "Stop scheduling your weight decay temporally. Instead, focus on per-layer or per-parameter WD modulation, and understand that you're really adjusting the effective learning rate distribution."
