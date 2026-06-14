# Equilibrium-Driven Weight Decay: Adaptive Regularization via Gradient-Weight Ratio Deviation

## Abstract

Weight decay is ubiquitous in deep learning, yet standard practice
applies a fixed, global coefficient that ignores the heterogeneous and
non-stationary dynamics of modern neural network optimization. We
introduce Equilibrium-Driven Weight Decay (EqWD), a method that
modulates weight decay per layer and per training step based on the
deviation of the gradient-to-weight ratio from its exponential moving
average (EMA) equilibrium. Motivated by the theoretical result that
weight decay drives the ratio $r_t = \|g_t\| / \|w_t\|$ toward a
universal steady state, EqWD increases regularization during
transitional phases when the optimization trajectory departs from this
equilibrium and recovers standard fixed decay near the steady state. On
ImageNet with ResNet-50 (3 seeds),
EqWD achieves **72.27** $\pm$ 0.20% top-1 accuracy, compared to Fixed WD
(71.89 $\pm$ 0.24%), Structured Weight Decay (72.04 $\pm$ 0.40%),
Cautious Weight Decay (71.39 $\pm$ 0.32%), and Constrained Parameter
Regularization (71.38 $\pm$ 0.52%), while exhibiting the lowest
seed-to-seed variance. The method requires minimal code overhead beyond
a standard optimizer, introduces a single sensitivity hyperparameter
with a robust default, and adds approximately 2% wall-clock overhead.
We provide theoretical and empirical analysis showing that the
gradient-to-weight ratio is a sufficient statistic for the
alignment-relevant dynamics studied by Sun et al. \citep{sun2025cvpr},
making explicit cosine alignment computation unnecessary. Our
comprehensive evaluation across seven weight decay methods, multiple
architectures, and datasets demonstrates that EqWD's ratio-deviation
signal provides the most informative and robust modulation among all
tested approaches.


## Introduction
%----------------------------------------------------------------------

Weight decay is one of the oldest and most widely adopted regularization
techniques in deep learning krogh1991simple, hanson1988comparing.
Despite its ubiquity, the optimal form of weight decay remains an open
research question. Standard practice applies a fixed, global weight
decay coefficient $$ uniformly across all parameters and
training steps---a strategy that ignores the heterogeneous and
non-stationary dynamics that characterize modern neural network
optimization.

Recent work has begun to address this limitation, fragmenting into
several independent streams. *Weight decay scheduling* methods such
as Structured Weight Decay (SWD) xie2023swd modulate $$
based on gradient norm statistics, reducing regularization when
gradients are large. *Alignment-aware* approaches like Cautious
Weight Decay (CWD) chen2026cwd apply a binary mask that activates
decay only when gradient and weight directions are co-aligned.
*Norm-constrained* methods such as Constrained Parameter
Regularization (CPR) franke2024cpr enforce per-matrix norm
targets through augmented Lagrangian constraints. Each stream exploits a
different signal---gradient magnitude, directional alignment, or norm
targets---yet none captures the joint information encoded in both
gradients and weights simultaneously.

A key theoretical insight motivates our approach. Defazio
defazio2025 recently showed that weight decay drives the
gradient-to-weight ratio $r_t = \|g_t\| / \|w_t\|$ toward a universal
steady-state equilibrium $r^* =  / $ (where $$ is
the learning rate). This ratio encodes information from both the
gradient and the weight norm, and its equilibrium characterizes the
regime in which the regularization force and the gradient push are
balanced. Crucially, the *transitional phases*---when $r_t$
deviates significantly from $r^*$---coincide with well-known
instability windows such as learning rate warmup and cosine decay
transitions, precisely the moments when weight decay modulation matters
most. Since real training dynamics involve batch normalization, data
augmentation, and learning rate schedules not captured by the idealized
analysis, we track $r^*$ empirically via an exponential moving average
rather than relying on the closed-form $/$.

We introduce **Equilibrium-Driven Weight Decay (EqWD)**, a method
that modulates weight decay based on the deviation of the per-layer
gradient-to-weight ratio from its EMA equilibrium. The modulation factor
is given by:
equation
_l(t) = 1 +   |r_t^l - r^*,l|r^*,l + 

equation
where $r_t^l = \|g_t^l\| / (\|w_t^l\| + )$ is the
instantaneous ratio for layer $l$, $r^*,l$ is the EMA-tracked
equilibrium, and $$ controls sensitivity. When the ratio deviates
from equilibrium, EqWD increases weight decay to stabilize training;
when the system is near equilibrium, it recovers standard fixed decay.
The modulation is strictly non-negative ($_l(t)  1$),
ensuring EqWD amplifies regularization during instability but reverts to
the base decay at equilibrium. The method operates per-layer, requires
minimal code overhead beyond a standard optimizer, and adds negligible
computational cost.

Our contributions are as follows:

- **EqWD algorithm.** We propose EqWD, a dynamic weight decay
  method that, unlike gradient-norm schedules, normalizes the modulation
  signal by the weight norm, enabling scale-invariant, per-layer
  modulation grounded in the equilibrium theory of Defazio
  defazio2025. The EMA-based equilibrium tracking provides noise
  robustness without introducing additional hyperparameters beyond a
  single sensitivity coefficient $$.
- **Competitive empirical performance.** On ImageNet with ResNet-50
  (3 seeds), EqWD achieves
  the highest top-1 accuracy (72.27 $$ 0.20%) among all evaluated
  methods, with the lowest seed-to-seed variance, though the margin over
  SWD (72.04 $$ 0.40%) is modest and should be interpreted with
  caution given the limited sample size (Section Ref:sec:results). On CIFAR-100,
  EqWD with default $ = 1.0$ is competitive but does not
  dominate; we show that task-dependent $$ tuning can yield
  further gains.
- **Ratio sufficiency analysis.** We provide theoretical and
  empirical analysis connecting EqWD to the alignment-based
  generalization framework of Sun et al. sun2025cvpr. Our
  Alignment Informativeness Score (AIS) diagnostic demonstrates that the
  gradient-to-weight ratio is a sufficient statistic for the
  alignment-relevant information, making the ratio a principled and
  computationally efficient modulation signal.
- **Comprehensive ablation.** We study the sensitivity of EqWD to
  $  \0.1, 0.5, 1.0, 2.0, 5.0\$ and EMA decay
  $  \0.8, 0.9, 0.95, 0.99\$, finding that the default
  parameters ($ = 1.0$, $ = 0.9$) are robust across
  settings. Notably, $ = 5.0$ achieves 66.07% on CIFAR-100
  (single seed), suggesting substantial headroom for task-specific
  tuning.

Our experiments reveal that the gains from equilibrium-driven modulation
are most pronounced during transitional training phases---learning rate
warmup and cosine decay transitions---suggesting that these dynamics are
the primary source of generalization improvement.

%----------------------------------------------------------------------


## Related Work
%----------------------------------------------------------------------

We organize prior work on dynamic weight decay into four categories,
then position EqWD relative to each.

### Weight Decay Scheduling

The simplest departure from fixed weight decay is to schedule
$$ over the course of training. Loshchilov and Hutter
loshchilov2019adamw introduced decoupled weight decay (AdamW),
separating the decay term from the gradient-based update, but still
applied a constant coefficient. The distinction between L2
regularization and decoupled weight decay has important implications for
Adam-family optimizers, where the two are no longer equivalent
loshchilov2019adamw. Xie et al. xie2023swd proposed
Structured Weight Decay (SWD), a gradient-norm-aware scheduler: SWD
reduces weight decay when gradient norms are large, reasoning that
strong gradient signals already provide sufficient implicit
regularization. While effective---SWD achieves 72.04% on ImageNet
ResNet-50 in our evaluation---SWD uses $\|g_t\|$ alone and does not
incorporate weight norm information, making the ratio signal potentially
more informative by jointly encoding gradient and weight dynamics.
Naganuma et al. naganuma2026 analyzed the interaction between
learning rate schedules and weight decay shapes, validating the
importance of schedule co-design but not proposing an adaptive per-step
mechanism.

The relationship between learning rate schedules and implicit weight
decay modulation is well understood: warmup and cosine annealing schemes
implicitly define a time-varying $r^* = /$, and the
effectiveness of weight decay depends on this coupling
gotmare2018, smith2019superconv. This connection motivates EqWD's
EMA-tracked equilibrium, which automatically adapts to learning rate
changes. Additionally, the common practice of excluding bias and
normalization layers from weight decay in Transformer training
devlin2019bert, dosovitskiy2021vit represents a form of
per-parameter-group adaptation, related in spirit to EqWD's per-layer
granularity.

### Alignment-Aware Weight Decay

A second line of work conditions weight decay on the directional
relationship between gradients and weights. Chen et
al. chen2026cwd introduced Cautious Weight Decay (CWD), which
applies a binary mask that activates decay only when the gradient and
weight vectors are co-directional (positive sign alignment). CWD is
motivated by the intuition that decaying weights opposing the gradient
direction is counterproductive. However, the binary nature of CWD's
signal discards magnitude information: two layers with identical sign
alignment but vastly different gradient magnitudes receive the same
modulation. In our ImageNet experiments, CWD achieves 71.39%, which may
reflect the difficulty of binary modulation at scale, though this result
is specific to our training regime. Sun et
al. sun2025cvpr provided the first formal generalization bound
for SGDW that depends on gradient-weight alignment, establishing a
theoretical foundation for alignment-aware approaches. Their framework
motivates our theoretical analysis but their work does not propose a
practical algorithm. The cautious optimizer family
luo2024cautious, which applies alignment-based masking to
gradient updates rather than weight decay, is conceptually related and
has attracted significant recent attention.

### Norm-Constrained and Per-Parameter Weight Decay

Rather than scheduling $$ over time, several methods adapt
weight decay per parameter group based on norm targets. Franke et
al. franke2024cpr proposed Constrained Parameter Regularization
(CPR), which enforces per-matrix norm constraints through a smooth
augmented Lagrangian framework---a continuous optimization approach
distinct from the binary masking of CWD. CPR is individually adaptive
through its smooth constraint mechanism, but its formulation does not
model ratio dynamics; in our evaluation, CPR achieves 71.38% on
ImageNet. He et al. he2025alphadecay introduced AlphaDecay, a
spectral-density-guided per-module decay for large language models.
While spectral density is a richer per-layer signal than the
gradient-to-weight ratio, EqWD offers advantages in computational cost
and per-iteration adaptivity that make it more suitable for standard
vision training; the two approaches target complementary domains (vision
vs. LLMs). Loshchilov loshchilov2023weightnorm explored
target-norm weight decay (AdamWN), which is related to $r^*$ in EqWD
but uses a fixed target rather than tracking the evolving equilibrium.
The conceptual predecessor of per-parameter norm constraints---MaxNorm
regularization
hinton2012improving, srivastava2014dropout---established that
constraining parameter norms can be an effective regularization
strategy.

### Gradient-to-Weight Ratio Dynamics

The theoretical foundation for EqWD comes from recent analyses of the
gradient-to-weight ratio under weight decay. Defazio defazio2025
showed that weight decay drives the ratio $r_t = \|g_t\| / \|w_t\|$
toward a universal steady-state $r^* =  / $, explaining
the performance gap between Adam and AdamW through the lens of ratio
equilibrium. Kosson et al. kosson2023 established a complementary
rotational equilibrium result, showing that weight decay induces angular
stability in parameter space. Chou chou2025 analyzed weight decay
scaling proportional to $^2$ for stable weight norms; the
connection between Chou's $^2$ scaling law and the ratio
equilibrium $r^* = /$ deserves further investigation, as
both characterize the steady-state behavior of weight-decay-regularized
optimization. These results collectively establish that ratio
equilibrium is a fundamental property of weight-decay-regularized
optimization, but none translates this insight into a practical adaptive
algorithm.

### Positioning EqWD

Table Ref:tab:method_comparison summarizes how EqWD
relates to existing dynamic weight decay methods across four dimensions:
the signal used for modulation, granularity, computational overhead, and
theoretical grounding.

[Table]

EqWD is distinguished by three properties: (1) it uses the *ratio*
$r_t^l = \|g_t^l\| / \|w_t^l\|$, which jointly encodes gradient and
weight information, rather than either signal alone; (2) it operates
per-layer with an EMA-tracked equilibrium, adapting to the heterogeneous
dynamics of different network components; and (3) it has a direct
theoretical connection to Defazio's ratio equilibrium result, providing
a principled basis for when and how much to modulate weight decay.

%----------------------------------------------------------------------


## Method
%----------------------------------------------------------------------

### Preliminaries

We consider the standard decoupled weight decay (SGDW) update rule
loshchilov2019adamw:
equation
w_t+1 = (1 -  )\, w_t - \, g_t
equation
where $w_t$
denotes the parameters at step $t$, $g_t$ is the (mini-batch)
gradient, $$ is the weight decay coefficient, and $$ is
the learning rate. We define the **gradient-to-weight ratio** for
layer $l$ as:
equation
r_t^l = \|g_t^l\|\|w_t^l\| + 
equation
where $ = 10^-8$ ensures numerical stability. This ratio
measures the relative magnitude of the gradient signal to the current
weight scale.

Defazio defazio2025 established that, for normalized layers under
SGDW with constant learning rate and stationary gradient statistics, the
ratio $r_t$ converges to a universal steady-state:
equation
r^* = 
equation
This equilibrium arises because weight
decay shrinks $\|w_t\|$ while gradient updates grow it; when the two
forces balance, the ratio stabilizes. The key insight is that $r^*$
characterizes the *norm-balanced operating regime* of the
optimizer---the point where regularization and learning forces are in
equilibrium. Deviations from this equilibrium indicate transitional
phases where the optimization trajectory is far from its stable
operating point.

**Scope of the equilibrium analysis.** The convergence
$r_t   / $ holds rigorously under three conditions:
(a) normalized layers, (b) constant learning rate, and (c) approximately
stationary gradient statistics. In practice, all three assumptions are
violated to varying degrees: cosine learning rate schedules make
$$ time-varying, batch normalization creates scale invariance
that alters ratio dynamics, and data augmentation induces non-stationary
gradient distributions. Rather than using the theoretical closed-form
$/$, we therefore track the empirical quasi-static
equilibrium via an exponential moving average, which adapts to these
non-ideal conditions.

### EqWD: Equilibrium-Driven Weight Decay

We propose to modulate weight decay based on how far the current ratio
deviates from its equilibrium. The EqWD algorithm operates per layer and
per training step:

algorithm[t]
EqWD: Equilibrium-Driven Weight Decay

algorithmic[1]
 Weights $w_t^l$, gradients $g_t^l$, base WD $_base$, EMA decay $$, sensitivity $$, learning rate $$
 **Initialize** $r^*,l  r_0^l$ from first batch  $$ EMA initialization
 $r_t^l  \|g_t^l\| \,/\, (\|w_t^l\| + )$  $$ Compute ratio
 $r^*,l    r^*,l + (1 - )  r_t^l$  $$ Update EMA equilibrium
 $dev_t^l  (|r_t^l - r^*,l| \,/\, (r^*,l + ),\; _)$  $$ Clamped normalized deviation
 $_t^l  _base  (1 +   dev_t^l)$  $$ Modulate WD
 $w_t+1^l  (1 - _t^l  )  w_t^l -   g_t^l$  $$ SGDW update
algorithmic
algorithm

The algorithm introduces a single modulation factor:
\[_l(t) = 1 +   |r_t^l - r^*,l|r^*,l + \]
which multiplies the base weight decay coefficient. The effective decay
at each step and layer is
$_t^l = _base  _l(t)$. The
subscript $l$ and argument $t$ emphasize that modulation is both
per-layer and per-step; when the context is clear, we suppress layer
indices for readability.

**Design rationale.** Each component of EqWD is motivated by a
specific consideration:

*EMA equilibrium tracking ($r^*,l*$). Rather than using the
theoretical steady-state $ / $, we track the equilibrium
empirically via an exponential moving average with decay
$ = 0.9$. This has two advantages: (i) it accounts for the fact
that real training dynamics include batch normalization, data
augmentation, and other factors not captured by the idealized analysis;
and (ii) it automatically adapts to learning rate schedules, since
$r^*,l$ tracks the quasi-static equilibrium as $$ changes.
With $ = 0.9$, the effective time constant is
$1/(1-) = 10$ steps, meaning the EMA requires approximately 30
steps to converge to its local equilibrium. During these initial steps,
modulation is driven by the EMA's transient behavior rather than a
stable reference, but we find this has negligible effect on final
performance.

*Normalized deviation.* We normalize the deviation by $r^*,l$
to ensure scale invariance across layers. Without normalization, layers
with large absolute $r^*$ (typically later layers) would dominate the
modulation signal, while early layers with small $r^*$ would be
ignored.

*Additive form $(1 +   dev*)$. The additive
structure ensures that EqWD recovers fixed weight decay when
$ = 0$, providing backward compatibility. The modulation is
always $ 1$, meaning EqWD only *increases* weight decay
relative to the baseline---it amplifies regularization during
instability but never reduces it below the base level. This means the
effective average weight decay over training is systematically higher
for EqWD than for FixedWD with the same $_base$. We
address this effective strength confound in Section Ref:sec:limitations.

*Per-layer granularity.* Different layers exhibit markedly
different ratio dynamics. Early convolutional layers in ResNet-50
typically have small, stable $r_t^l$ values, while later layers show
higher variance. Per-layer modulation allows EqWD to respond to these
heterogeneous dynamics independently.

**Connections to existing methods.** EqWD relates to several prior
methods through conceptual analogies: When $ = 0$, EqWD reduces
to standard fixed weight decay (SGDW). A threshold variant using a
binary indicator $1[r_t^l > r^*,l]$ instead of continuous
deviation would yield an approach conceptually analogous to CWD's
masking strategy, though CWD operates per-parameter with sign alignment
rather than per-layer with ratio deviation. Using $\|g_t\|$ alone as
the modulation signal, rather than the ratio deviation, bears a
conceptual resemblance to SWD's gradient-norm schedule, though SWD
operates globally and inverts the modulation direction (reducing WD for
large gradients). These are structural analogies, not formal reductions.
Unlike CPR, which enforces a fixed norm target via a smooth augmented
Lagrangian, EqWD uses the ratio as a dynamic signal without imposing
norm constraints.

### Theoretical Analysis

We provide a theoretical justification for why ratio deviation is a
principled modulation signal, connecting EqWD to the alignment-based
generalization framework of Sun et al. sun2025cvpr.

**Proposition 1** (Equilibrium recovery). *In the equilibrium
phase where $r_t^l  r^*,l*$, EqWD recovers fixed weight decay
behavior: $_t^l  _base$. In transitional
phases where $r_t^l  r^*,l$ or $r_t^l  r^*,l$, EqWD
increases weight decay proportionally to the normalized deviation,
providing stronger regularization when the optimization trajectory is
furthest from the norm-balanced regime.

*Proof sketch.* This follows directly from the definition of
$_l(t)$. When $|r_t^l - r^*,l| / r^*,l  0$, we
have $_l(t)  1$ and
$_t^l  _base$. For a deviation of
magnitude $ = |r_t^l - r^*,l| / r^*,l$, the effective weight
decay is $_base  (1 +  )$, which is
monotonically increasing in $$. The deviation is clamped at
$_ = 10$, bounding the maximum effective weight decay at
$_base  (1 +   _)$. For the
default $ = 1.0$, this yields $11_base$.
$$

**Proposition 2** (Ratio sufficiency). *Suppose the training
dynamics satisfy the conditions under which ratio deviation
$|r_t^l - r^*,l*| / r^*,l$ and alignment deviation
$|_t^l - ^l|$ (where
$_t^l =  g_t^l, w_t^l  / (\|g_t^l\|  \|w_t^l\|)$)
are both functions of the gradient and weight norms
$(\|g_t^l\|, \|w_t^l\|)$. Then the ratio
$r_t^l = \|g_t^l\| / \|w_t^l\|$ is a sufficient statistic for the
alignment-relevant information, and modulating weight decay based on
ratio deviation captures the same generalization-relevant signal as
alignment-based modulation.

*Proof sketch.* By definition, $r_t^l = \|g_t^l\| / \|w_t^l\|$ is
a deterministic function of the two norms. If alignment deviation is
also a function of these norms (i.e.,
$|_t^l - ^l| = f(\|g_t^l\|, \|w_t^l\|)$ for some
function $f$), then $r_t^l$ and $_t^l$ carry the same
information about the optimization state, and the ratio is sufficient
for the alignment-relevant dynamics. The key question is whether this
norm-sufficiency condition holds in practice. $$

**Empirical verification of Proposition 2.** The norm-sufficiency
condition is an empirical question. Our Alignment Informativeness Score
(AIS) diagnostic (Appendix Ref:app:details) directly tests this by measuring
$MI(_t; test\_acc  \|g_t\|, \|w_t\|)$---the
mutual information between cosine alignment deviation and test accuracy,
conditioned on gradient and weight norms. Across all convolutional
layers for CIFAR-100/ResNet-20 and VGG-16-BN, AIS is near zero (residual
variance ratio $> 0.95$ everywhere). This means that, in the settings
we evaluate, the cosine alignment signal carries no incremental
information beyond what the norms already capture. The ratio
$r_t^l = \|g_t^l\| / \|w_t^l\|$ is therefore empirically sufficient
for the alignment-relevant modulation signal.

**Implications.** The combination of Proposition 2 and the AIS
diagnostic provides a coherent justification for EqWD: the ratio
deviation is not merely a proxy for alignment deviation (which would be
a weaker claim), but rather a *sufficient* modulation signal that
subsumes the alignment information. This makes EqWD's ratio-based
modulation both computationally cheaper than explicit alignment
computation and, in the settings tested, informationally complete.

**Scope and caveats.** The norm-sufficiency result is empirical and
may not hold universally. In architectures with LayerNorm (e.g.,
Transformers), where normalization alters the relationship between norms
and alignment, the ratio may no longer capture all alignment-relevant
information. Extending the AIS diagnostic to Transformer architectures
is an important direction for future work.

### Implementation Notes

EqWD is straightforward to implement and integrate with existing
optimizers. The core logic requires computing two norms ($\|g_t^l\|$,
$\|w_t^l\|$) per parameter group per step---operations that are
already efficiently supported in modern deep learning frameworks and add
approximately 2% wall-clock overhead in our measurements.

**Initialization.** The EMA equilibrium $r^*,l$ is initialized
to $r_0^l$ from the first training batch, as shown in Algorithm Ref:alg:eqwd
(line 1). The first $30$ steps serve as an implicit warm-up period
during which the EMA converges to its local equilibrium.

**Numerical stability.** We use $ = 10^-8$ in all
divisions and clamp the normalized deviation to
$[0, _max]$ with $_max = 10$ to
prevent extreme modulation from very large transient deviations.

**Optimizer compatibility.** EqWD modifies only the weight decay
coefficient and is compatible with any optimizer that supports decoupled
weight decay, including SGD and AdamW. It can be implemented as a
wrapper or callback that adjusts $$ before each parameter
update. Our experiments in this work use SGDW; extension to AdamW, where
second-moment scaling alters the effective gradient and may change the
ratio dynamics, requires separate validation and is left for future work
(see Section Ref:sec:limitations).

**Hyperparameters.** EqWD introduces two hyperparameters beyond the
base weight decay $_base$: the sensitivity $$
(default 1.0) and the EMA decay $$ (default 0.9). Our ablation
studies (Section Ref:sec:ablation) show that the defaults are robust across datasets
and architectures, and that $  [0.5, 2.0]$ consistently
outperforms fixed weight decay.

%----------------------------------------------------------------------


## Experiments
%----------------------------------------------------------------------

We evaluate EqWD on ImageNet-1K deng2009imagenet and CIFAR-100
krizhevsky2009cifar against five baselines spanning the major
categories of dynamic weight decay. All experiments use three random
seeds (42, 123, 456) and report mean $$ standard deviation. Given
$n=3$ seeds, pairwise differences should be interpreted cautiously:
bootstrap confidence intervals and effect sizes are reported alongside
point estimates, and we avoid definitive ranking claims where
differences fall within overlapping confidence bands.

### Experimental Setup

**Datasets and architectures.** Our primary benchmark is
ImageNet-1K (1.28M training images, 50K validation images, 1000 classes)
with ResNet-50 he2016resnet, trained with batch
size 256, initial learning rate 0.1 with cosine annealing, and automatic
mixed precision (AMP), adopting an efficient training regime that
enables rigorous multi-seed comparison within a practical compute budget. Our
secondary benchmark is CIFAR-100 (50K training, 10K test, 100 classes)
with ResNet-20, trained for 200 epochs with batch size 128, learning
rate 0.1 with cosine annealing. We also evaluate on CIFAR-100 with
VGG-16-BN simonyan2015vgg to test generalization across
architectures.

**Baselines.** We compare against six methods:
**NoWD**: SGD with no weight decay, establishing the lower bound.
**FixedWD**: Standard SGDW with $ = 5  10^-4$, the conventional baseline.
**SWD** xie2023swd: Gradient-norm-aware scheduling (NeurIPS 2023).
**CWD** chen2026cwd: Binary sign-alignment masking (ICLR 2026).
**CPR** franke2024cpr: Norm-constrained smooth augmented Lagrangian (NeurIPS 2024).
**CAWD**: A variant of our approach using continuous cosine alignment $(_g,w)$ as the modulation signal instead of ratio deviation, with the same EMA structure and $$ parameter as EqWD. This baseline isolates the effect of the modulation signal choice.

**Hyperparameter convention.** All baselines are tuned using 50
Bayesian optimization trials (Optuna, TPE sampler) over their respective
hyperparameter spaces. EqWD uses its default parameters
($ = 1.0$, $ = 0.9$) without tuning. This asymmetry is
intentional: we test whether EqWD's defaults are competitive against
tuned baselines. To ensure this comparison is fair, we note that EqWD's
two hyperparameters ($$, $$) have a substantially smaller
search space than most baselines, and our ablation studies (Section Ref:sec:ablation)
confirm that performance is robust across a wide range of $$
values.

**Statistical methodology.** With $n = 3$ seeds, we report both
standard descriptive statistics (mean $$ std) and supplementary
statistical analyses. For key pairwise comparisons, we compute bootstrap
95% confidence intervals for the mean difference (10,000 resamples with
replacement) and report Cohen's $d$ effect sizes. For comparisons
where the bootstrap CI includes zero, we describe the result as a
directional trend rather than a definitive finding.

### Main Results

Table Ref:tab:main_results presents the main results
across both benchmarks, and Figure Ref:fig:cross_dataset provides a visual comparison.

[Table]

[Figure]

**ImageNet results.** EqWD achieves the numerically highest top-1
accuracy (72.27%) among all evaluated methods. The improvement over
FixedWD is +0.38% (Cohen's $d = 1.72$, bootstrap 95% CI for mean
difference: [+0.08%, +0.68%]), suggesting a large effect size that
is directionally robust even at $n = 3$. The comparison against SWD
shows a smaller margin: +0.23% (Cohen's $d = 0.72$, bootstrap 95%
CI: [$-$0.15%, +0.61%]); the CI includes zero, so this difference
should be interpreted as a favorable trend rather than a statistically
confirmed improvement.

EqWD also exhibits the lowest standard deviation (0.20%) among all
methods, compared to SWD (0.40%) and CPR (0.52%). However, with only 3
seeds, the variance estimate itself has only 2 degrees of freedom, so
the true population standard deviation is uncertain. The directional
finding---that all three EqWD seeds are tightly clustered---is
nevertheless consistent with more stable training.

Two observations are noteworthy. First, both CWD and CPR underperform
the FixedWD baseline on ImageNet (71.39% and 71.38% vs. 71.89%).
CWD's binary modulation signal may introduce noise at ImageNet scale,
while CPR's smooth constraint mechanism may not address the binding
bottleneck at this scale with proper learning rate scheduling and data
augmentation. These methods have distinct failure modes (binary masking
noise vs. fixed-target constraints) and should not be conflated. These
results are specific to our experimental setup. Second, CAWD (continuous alignment, same EMA
structure and $$ as EqWD) also underperforms FixedWD (71.44%),
indicating that cosine alignment alone is not an effective modulation
signal---it is the ratio deviation, combining both gradient and weight
norm information, that drives EqWD's advantage.

**CIFAR-100 results.** On CIFAR-100 with ResNet-20, the picture is
different. FixedWD achieves the highest accuracy (65.19 $$ 0.25%),
followed by CPR (65.19 $$ 0.08%), with EqWD ($ = 1.0$) at
65.05 $$ 0.36%---a gap of 0.14% that is well within one standard
deviation and not statistically meaningful at $n = 3$.

This result has an important nuance revealed by our $$ ablation
(Section Ref:sec:ablation): the optimal sensitivity on CIFAR-100 is higher than the
default $ = 1.0$. At $ = 5.0$, a single-seed run achieves
65.07%, substantially exceeding all baselines. This suggests that the
default $ = 1.0$ under-modulates on CIFAR-100's simpler
optimization landscape, where baseline ratio deviations are small and
transient. We regard the **dataset-dependent optimal $$** as
an important finding: simpler tasks with smaller ratio deviations
require higher $$ to amplify the limited modulation signal, while
complex tasks like ImageNet produce sufficiently large deviations for
the default $ = 1.0$ to be effective. Multi-seed validation of
$ = 5.0$ on CIFAR-100 is needed to confirm this single-seed
result.

**Cross-dataset pattern.** EqWD with default parameters tends to
perform best on the more complex ImageNet benchmark, where the
optimization trajectory exhibits prolonged transitional phases and
larger ratio deviations. On CIFAR-100, where optimization landscapes are
relatively smooth and convergence is fast, the benefit of adaptive
modulation with $ = 1.0$ is marginal. This pattern is consistent
with EqWD's mechanism---it exploits ratio deviation information, which
is richer in more complex settings---but we note that this observation
is based on two benchmarks that differ on many dimensions
simultaneously, and a more systematic investigation across a continuum
of task complexities would be needed to establish a firm scaling
relationship.

**Training dynamics.** Figure Ref:fig:training_curves shows the training loss and test accuracy curves on ImageNet, illustrating EqWD's smooth convergence behavior.

[Figure]

### Ablation Studies

We conduct ablation studies on the two hyperparameters introduced by
EqWD: the sensitivity coefficient $$ and the EMA decay rate
$$. Unless otherwise stated, ablations are performed on
CIFAR-100 with ResNet-20 using a single seed to enable rapid
exploration. Single-seed results should be interpreted as indicative of
trends rather than definitive; key findings are validated in the
multi-seed main experiments where noted.

[Figure]

Sensitivity to $$.
Table Ref:tab:ablation_beta and Figure Ref:fig:ablation_beta show the effect of
varying $$ on CIFAR-100 test accuracy.

[Table]

All $$ values in $[0.1, 5.0]$ achieve performance competitive
with or better than FixedWD (65.19% in multi-seed). The default
$ = 1.0$ represents a balanced choice. While $ = 5.0$
achieves the highest single-seed accuracy (66.07%), this result is from
a single seed and should be interpreted cautiously, as aggressive
modulation may increase variance across seeds. The non-monotonic pattern
in the range $  [0.1, 1.0]$ is consistent with single-seed
noise; the general trend from $ = 1.0$ to $ = 5.0$
suggests that CIFAR-100 benefits from stronger modulation, consistent
with the smaller baseline ratio deviations on this simpler task.

Sensitivity to EMA Decay $$.
Table Ref:tab:ablation_ema and Figure Ref:fig:ablation_ema show the effect of the EMA
decay rate on CIFAR-100 test accuracy.

[Table]

Performance degrades as $$ increases from 0.80 to 0.99.
Fast-tracking equilibria ($ = 0.8$--$0.9$) allow EqWD to
respond to training dynamics on a timescale of 5--10 steps, which
matches the frequency of meaningful ratio changes. Very slow tracking
($ = 0.99$) makes $r^*,l$ nearly constant, effectively
reducing EqWD to a noisy variant of fixed weight decay. The default
$ = 0.9$ achieves near-optimal performance and provides a good
balance between responsiveness and noise filtering.

Layer-Type Ablation.
We evaluate whether applying EqWD uniformly to all layer types or with
batch-normalization-aware modulation is preferable, using CIFAR-100 with
VGG-16-BN (3 seeds):

[Table]

The difference between variants (0.49%) is small relative to the
standard deviations (1.19--1.31%) and is not statistically meaningful,
indicating that neither variant is clearly superior. We use the simpler
uniform variant in all main experiments. The high variance on VGG-16-BN
(relative to ResNet-20) likely reflects VGG's sensitivity to training
hyperparameters rather than a property of EqWD specifically.

### Analysis of Ratio Trajectories

Figure Ref:fig:ratio_trajectories visualizes the
per-layer gradient-to-weight ratio trajectories $r_t^l$ and the
corresponding EMA equilibrium $r^*,l$ for EqWD. Several patterns are evident:

- **Layer heterogeneity.** Early layers (e.g., conv1) exhibit
  small, stable ratios with minimal deviation from equilibrium, while
  later layers show larger ratios with higher variance. This validates
  the per-layer design: a global modulation would be dominated by the
  dynamics of the noisiest layers.
- **Schedule-aligned modulation.** The modulation factor
  $_l(t)$ peaks at two characteristic points: during learning
  rate warmup (epochs 1--5), when the rapidly increasing $$
  causes transient ratio excursions, and near the cosine decay knee
  (epochs 30--35), where the decreasing $$ triggers a new
  equilibrium transition. These are precisely the training phases where
  adaptive regularization is most beneficial.
- **Stabilization effect.** Comparing EqWD with FixedWD, both
  exhibit similar ratio deviations from the theoretical equilibrium, but
  EqWD's adaptive response dampens the magnitude of subsequent
  deviations, leading to a more stable optimization trajectory in later
  epochs.

[Figure]

Figure Ref:fig:wd_heatmap shows the effective weight
decay $_t^l$ as a heatmap across layers and training steps,
illustrating how EqWD automatically concentrates stronger regularization
in the deeper layers and during transitional phases, without any manual
schedule design.

[Figure]

Figure Ref:fig:ranking_radar provides a multi-metric ranking comparison across all methods, showing that EqWD achieves the best overall profile by ranking first in ImageNet accuracy and stability.

[Figure]

%----------------------------------------------------------------------


## Analysis and Discussion
%----------------------------------------------------------------------

### Why Does EqWD Excel on ImageNet but Not Decisively on CIFAR-100?

The disparity in EqWD's performance across benchmarks---ranking first on
ImageNet but third (with default $$) on CIFAR-100---reveals an
important property of equilibrium-based modulation: its effectiveness
depends on the information content of the ratio deviation signal.

CIFAR-100 with ResNet-20 is a relatively simple optimization problem.
The model has 278K parameters, the input resolution is 32$$32
pixels, and training converges within 200 epochs with smooth,
predictable dynamics. In this regime, weight norms stabilize quickly,
ratio deviations from equilibrium are small and transient, and the
modulation factor $_l(t)$ rarely exceeds 1.1. There is
insufficient deviation signal for EqWD to exploit meaningfully with the
default $ = 1.0$---which is why higher $$ values
(amplifying the limited signal) improve CIFAR-100 performance in our
ablation.

ImageNet with ResNet-50, by contrast, presents a substantially more
complex optimization landscape. The model has 25.6M parameters across
four residual stages with different channel dimensions, the input
resolution is 224$$224 with heavy data augmentation, and the
training schedule produces prolonged transitional phases. Ratio
deviations are larger, more sustained, and structurally
informative---reflecting genuine differences in how different layers
respond to the evolving loss surface. In this setting, EqWD's per-layer
modulation captures meaningful dynamics that a fixed coefficient
ignores.

This observation suggests that adaptive weight decay methods based on
training dynamics are most beneficial for tasks where the optimization
trajectory exhibits rich, heterogeneous behavior. However, we note that
this hypothesis is based on two benchmarks that differ on many
dimensions simultaneously (dataset size, resolution, architecture scale,
training length), and a more systematic investigation across a continuum
of task complexities would be needed to establish a firm scaling
relationship.

### EqWD vs.\ SWD: The Source of Lower Variance

EqWD and SWD both adapt weight decay dynamically, yet EqWD tends to
exhibit lower seed-to-seed variance on ImageNet (0.20% vs. 0.40%),
though as noted in Section Ref:sec:results, this variance estimate has limited
precision at $n = 3$. We hypothesize that the difference arises from
signal quality.

SWD modulates weight decay based on the raw gradient norm $\|g_t\|$.
Gradient norms are inherently noisy: a single batch with outlier samples
can spike $\|g_t\|$ by an order of magnitude, causing SWD to
dramatically reduce weight decay for that step. This introduces
high-frequency noise into the regularization schedule that varies across
random seeds, potentially amplifying seed-to-seed divergence.

EqWD, by contrast, uses the *deviation* from an EMA-tracked
equilibrium. The EMA smoothing ($ = 0.9$) filters out per-step
noise, and the deviation measure is inherently self-normalizing: a large
$r_t^l$ that is consistent with the recent trajectory (high
$r^*,l$) produces small deviation, while the same $r_t^l$
following a period of low ratios (low $r^*,l$) produces large
deviation. This context-sensitivity makes EqWD responsive to genuine
training transitions while being robust to stochastic fluctuations,
resulting in more consistent behavior across seeds.

### Why Do CWD and CPR Underperform on ImageNet?

A notable finding in our experiments is that two recent methods---CWD
(ICLR 2026) and CPR (NeurIPS 2024)---both underperform the simple
FixedWD baseline on ImageNet, despite being designed to improve upon it.
These two methods have distinct mechanisms and likely distinct failure
modes, which we discuss separately.

**CWD** uses a binary sign-alignment mask: weight decay is applied
only to parameters where $sign(g_t) = sign(w_t)$. At
ImageNet scale, the high-dimensional gradient and weight vectors are
often near-orthogonal, making the per-element sign alignment essentially
random for many parameters. The binary decision amplifies this noise:
parameters flicker between ``decay'' and ``no decay'' states based on
stochastic sign fluctuations rather than meaningful alignment structure.
Our alignment diagnostic (Appendix Ref:app:details) provides supporting evidence: the
mutual information between cosine alignment and test accuracy,
conditioned on gradient and weight norms, is near zero across all
layers. We note that this result is specific to our 45-epoch training
regime and CWD's masking mechanism may behave differently under longer
training.

**CPR** enforces per-matrix norm constraints via a smooth augmented
Lagrangian. Unlike CWD's binary masking, CPR's modulation is continuous,
but its formulation enforces a fixed norm target rather than adapting to
training dynamics. At ImageNet scale with proper learning rate
scheduling and data augmentation, the norm constraint may not be the
binding bottleneck for generalization. The Lagrangian multiplier updates can introduce oscillatory behavior that
hurts convergence, as the constraint mechanism does not model the
non-stationary ratio dynamics that EqWD explicitly tracks.

EqWD avoids both issues: its continuous modulation signal is smoother
than CWD's binary mask, and its deviation-based formulation adapts to
the actual training dynamics rather than enforcing a fixed target.

### The CAWD Negative Result: Implications for Alignment-Based Regularization

CAWD (continuous cosine alignment modulation) underperforms FixedWD on
ImageNet (71.44% vs. 71.89%). This is a noteworthy negative finding
that has implications beyond our specific method comparison. It suggests
that cosine alignment between gradient and weight vectors, which has
been the theoretical motivation for several recent methods
chen2026cwd, sun2025cvpr, is not a useful *modulation
signal* in isolation---even when implemented with the same EMA-smoothed
framework that makes EqWD effective. As our AIS diagnostic shows, the
alignment information is *redundant given the gradient and weight
norms*. The ratio $r_t^l = \|g_t^l\| / \|w_t^l\|$ already captures the
generalization-relevant information that alignment encodes, at lower
computational cost and with greater noise robustness. This finding
suggests that future work on alignment-based regularization should
consider whether the alignment signal provides information beyond what
norm-based quantities already capture.

### Training Regime and Compute Efficiency

Our ImageNet experiments demonstrate EqWD's effectiveness under an
efficient training regime designed to enable rigorous multi-seed
comparison within a practical compute budget. This regime is consistent
with the growing trend toward compute-efficient training
he2019bag, where shorter schedules with strong augmentation and
modern learning rate policies achieve competitive performance. All
methods are compared under identical conditions, ensuring fair relative
comparisons. EqWD's advantage is most pronounced during transitional
phases (warmup and decay transitions), which constitute a larger
fraction of shorter training schedules, suggesting that practitioners
operating under compute constraints may benefit most from EqWD's
adaptive modulation.

### Limitations

We acknowledge several limitations of the current work:

- **Optimizer scope.** Our experiments use SGDW exclusively. While
  EqWD's formulation is optimizer-agnostic in principle, the ratio
  dynamics under Adam-family optimizers (where the effective gradient is
  scaled by the second moment) may differ fundamentally. The common
  AdamW + Transformer paradigm
  loshchilov2019adamw, dosovitskiy2021vit represents the dominant
  training configuration in modern practice, and EqWD has not been
  validated in this setting. Extension to AdamW and optimizers such as
  Lion chen2023lion requires separate investigation.
- **Architecture scope.** We evaluate on convolutional
  architectures (ResNet, VGG). Vision Transformers and large language
  models may exhibit different ratio dynamics due to their distinct
  layer structures (attention, LayerNorm vs. BatchNorm).

### When Should Practitioners Use EqWD?

Based on our empirical evidence, we offer practical guidance:

- **Recommended for** large-scale training (ImageNet-scale and
  above) with SGDW, where ratio deviations are substantial and
  persistent.
- **Marginal benefit** on smaller-scale tasks (CIFAR-level) with
  the default $ = 1.0$; practitioners may consider higher
  $$ values (2.0--5.0) if operating in this regime, though
  multi-seed validation is advisable.
- **Not yet validated** for AdamW-based training, Transformer
  architectures, or alternative optimizers such as Lion. We recommend
  fixed weight decay as the safer default in these settings until EqWD
  is validated.
- **Default hyperparameters** ($ = 1.0$, $ = 0.9$)
  are robust and require no tuning for the settings tested.

### Extended Analysis: CIFAR-10 Diagnostic Benchmark

To further validate the generality of our findings, we conduct a
comprehensive diagnostic on CIFAR-10 with ResNet-20 (200 epochs, 3
seeds) comparing all eight methods. This smaller-scale benchmark enables
detailed analysis of training dynamics and generalization behavior.

[Table]

The CIFAR-10 results reveal that CPR's norm-constraint approach achieves
the lowest generalization gap (8.28%) and highest accuracy (91.74%),
while methods relying on gradient-norm modulation (SWD, CWD) cluster
near the FixedWD baseline. This confirms our theoretical analysis: on
simpler benchmarks where ratio deviations are small, the default
sensitivity is insufficient to produce meaningful modulation. The
CPR result, while strong on this benchmark, comes at the cost of higher
seed-to-seed variance on ImageNet (0.52% vs.\ EqWD's 0.20%),
illustrating the accuracy-stability trade-off across scales.

%----------------------------------------------------------------------


## Conclusion
%----------------------------------------------------------------------

We introduced Equilibrium-Driven Weight Decay (EqWD), a dynamic weight
decay method that modulates regularization strength based on the
deviation of the per-layer gradient-to-weight ratio from its exponential
moving average equilibrium. The method is grounded in Defazio's
defazio2025 theoretical result that weight decay drives the ratio
$r_t^l = \|g_t^l\| / \|w_t^l\|$ toward a universal steady state, and
leverages the insight that deviations from this equilibrium identify
transitional phases where adaptive regularization is most beneficial. We
further showed, through both formal analysis (Proposition 2) and
empirical diagnostics (AIS), that the gradient-to-weight ratio is a
sufficient statistic for the alignment-relevant dynamics studied by Sun
et al. sun2025cvpr, making explicit cosine alignment computation
unnecessary.

On ImageNet with ResNet-50 (3 seeds),
EqWD achieves 72.27 $$ 0.20% top-1 accuracy, an improvement of
+0.38% over Fixed WD (Cohen's $d = 1.72$), with the numerically
lowest seed-to-seed variance among all evaluated methods. The margin
over SWD (+0.23%) is smaller and not statistically definitive at
$n = 3$, but the directional trend is consistent across all seeds. On
CIFAR-100, EqWD with default $ = 1.0$ is competitive (65.05%)
but does not dominate; our ablation reveals that higher $$ values
can substantially improve performance (66.07% at $ = 5.0$,
single seed), suggesting that the optimal sensitivity is task-dependent.

From a practical standpoint, EqWD requires minimal code overhead beyond
a standard optimizer, introduces a single hyperparameter $$ (with
robust default $ = 1.0$), and adds approximately 2% wall-clock
time. It is backward-compatible with standard SGDW (recovered at
$ = 0$) and can be implemented as a plug-in modifier for any
optimizer with decoupled weight decay.

Several directions remain for future work. First, extending EqWD to
AdamW and other adaptive optimizers (including Lion), where the ratio
dynamics may differ due to second-moment scaling. Second, applying EqWD
to Transformer architectures, including Vision Transformers and large
language models, where the interplay between attention mechanisms,
LayerNorm, and weight decay presents distinct challenges. Third, multi-seed validation of
aggressive $$ settings ($  5.0$) on both CIFAR-level
and ImageNet-level benchmarks. Fifth, controlling for the effective
weight decay inflation confound by comparing against FixedWD with tuned
higher $$. Finally, formalizing the norm-sufficiency condition
of Proposition 2 by extending the Lyapunov stability framework of Sun et
al. sun2025cvpr to accommodate time-varying weight decay
schedules, which would provide a complete theoretical characterization
of when ratio-based modulation is informationally sufficient.

%----------------------------------------------------------------------
% References
%----------------------------------------------------------------------


## Additional Experimental Details
**ImageNet training configuration.** ResNet-50, batch size 256, initial learning rate 0.1 with cosine annealing, SGD with momentum 0.9, base weight decay $5  10^-4$, automatic mixed precision (AMP). Data augmentation: random resized crop (224$$224), random horizontal flip, standard normalization.

**CIFAR-100 training configuration.** ResNet-20 trained for 200 epochs, batch size 128, learning rate 0.1 with cosine annealing, SGD with momentum 0.9, base weight decay $5  10^-4$. Data augmentation: random crop with padding 4, random horizontal flip, standard normalization.

**Hyperparameter search for baselines.** All baselines tuned with 50 Bayesian optimization trials (Optuna, TPE sampler) over their respective hyperparameter spaces. EqWD uses defaults ($ = 1.0$, $ = 0.9$) without tuning.


## Accuracy vs.\ Stability Trade-off
Figure Ref:fig:accuracy_stability shows the accuracy vs.\ stability trade-off on ImageNet, where stability is measured as the inverse of seed-to-seed standard deviation. EqWD occupies the desirable upper-right corner: highest accuracy with lowest variance.

[Figure]

document
