# Backup Ideas for Potential Pivot

## Alternative 1: BCM Sliding Threshold Weight Decay (BCM-WD)

### Source
Interdisciplinary perspective — computational neuroscience (BCM theory, Bienenstock-Cooper-Munro 1982).

### Core Idea
Replace the static equilibrium-based WD modulation of EqWD with a biologically-grounded sliding threshold mechanism from BCM theory. The BCM sliding threshold theta_M adapts as a super-linear function of historical alignment activity, providing homeostatic stability through a negative feedback loop.

### Algorithm
```
delta_hat_t = <g_t, w_t> / (||g_t|| ||w_t|| + eps)   [alignment signal]
theta_t = (1-alpha_theta) * theta_{t-1} + alpha_theta * delta_hat_t^2  [sliding threshold EMA]
alignment_deviation = 1.0 - delta_hat_t / (sqrt(theta_t) + eps)
lambda_t = lambda_base * (1 + beta * alignment_deviation)
lambda_t = clamp(lambda_t, 0.1 * lambda_base, 10 * lambda_base)
w_{t+1} = (1 - lambda_t * gamma_t) * w_t - gamma_t * g_t
```

### Why This Could Succeed Where EqWD Fails
- If the ratio deviation signal is too noisy at ImageNet scale, BCM-WD uses a different signal (alignment relative to historical baseline) that may be more stable
- BCM stability theory provides formal convergence guarantees under bounded-correlation assumptions
- The sliding threshold mechanism has a fundamentally different character: it modulates WD based on deviation from the neuron's own alignment history, not from a ratio equilibrium
- Addresses a complementary aspect: EqWD is about "when is the system far from equilibrium?"; BCM-WD is about "when is the alignment signal surprising relative to history?"

### When to Pivot
- If EqWD shows no improvement over SWD on CIFAR (H2 falsified)
- If the ratio deviation signal has insufficient variance at ImageNet scale (Risk 3 materialized)
- If the alignment informativeness diagnostic (H3) shows alignment carries information but ratio does not capture it

### Key Risk
BCM's WD direction (decay more when misaligned) CONFLICTS with CWD's direction (decay when aligned). The correct direction is an empirical question. Must test both BCM-WD and BCM-WD-inverted.

### Novelty
9/10 — BCM has never been applied to gradient-based optimizer WD scheduling. The mapping (delta_hat_t -> postsynaptic activity, sliding theta -> dynamic WD strength) is completely unexplored.

### Resource Estimate
Same as EqWD — negligible overhead. CIFAR pilot: 30 min. Full evaluation: same timeline.

---

## Alternative 2: Purely Empirical Budget Equivalence Framework (No New Algorithm)

### Source
Empiricist perspective — the "null hypothesis" approach.

### Core Idea
If EqWD and other dynamic WD methods fail to beat optimally-tuned fixed WD under matched compute budgets (H6 falsified), pivot to a purely empirical contribution: the first rigorous budget-equivalent comparison of all four WD streams, establishing standardized evaluation metrics and definitively answering whether dynamic WD provides genuine benefit.

### Paper Framing
"Does Dynamic Weight Decay Actually Help? A Rigorous Budget-Equivalent Evaluation" — analogous to GLUE/SuperGLUE establishing benchmarks for NLP rather than proposing new methods.

### Contributions
1. Budget Equivalence test across all methods under matched Bayesian optimization budgets
2. Three standardized metrics (BEM, CSI, AIS) validated across architectures
3. Per-layer diagnostic visualizations showing ratio trajectories under each WD strategy
4. Comprehensive cross-architecture comparison (ResNet, VGG, ViT) on CIFAR + ImageNet
5. Layer-type analysis: which layers benefit from alignment-aware WD?

### Why This Could Be a Strong Paper
- The empiricist perspective correctly notes this is "the most important experiment in the whole space and it has not been done"
- A rigorous negative finding (fixed WD matches dynamic WD) is highly publishable at NeurIPS/ICML as a corrective empirical contribution
- The theoretical framework (Theorem 1, Proposition 3) still provides value as analytical tools
- The evaluation protocol itself is a lasting contribution regardless of the specific results

### When to Pivot
- If H2 AND H7 are both falsified (neither EqWD nor CAWD beats baselines)
- If H6 shows null result (no dynamic WD beats tuned fixed WD)
- If the core theoretical contributions (Theorem 1) have proof gaps that cannot be resolved

### Key Risk
Survey/benchmark papers without novel algorithms face higher reviewer skepticism at top venues. Must ensure the metrics and evaluation protocol provide genuinely new insights, not just "we ran more baselines."

### Novelty
6/10 for the evaluation framework alone; 8/10 if paired with the theoretical framework (Cumulative Contraction bound + ratio equilibrium characterization remain novel even without a winning algorithm).

### Resource Estimate
Higher compute requirement: ~236 GPU-hours (per empiricist's estimate) for comprehensive Bayesian optimization search across all methods.

---

## Alternative 3: Continuous Alignment-Aware Weight Decay (CAWD) as Primary Algorithm

### Source
Pragmatist perspective.

### Core Idea
If the ratio deviation signal proves uninformative (EqWD fails), fall back to the simpler continuous cosine alignment modulation: lambda_t = lambda_base * (1 + cos(w, g)) / 2. This extends CWD's binary mask to the full continuous spectrum.

### Algorithm
```
cos_sim = cosine_similarity(w.flatten(), g.flatten())
wd_scale = (1.0 + cos_sim) / 2.0   [maps [-1,1] to [0,1]]
w = (1 - lr * wd * wd_scale) * w    [modulated decay]
```

### Why This Could Succeed Where EqWD Fails
- Simpler signal: directly uses the alignment quantity delta_t from Sun et al. (CVPR 2025) rather than a derived ratio
- Zero hyperparameters beyond lambda_base
- Directly extends CWD (binary -> continuous) — clearer novelty narrative
- Already validated conceptually by CWD's success: the sign works, the magnitude should add information

### When to Pivot
- If EqWD's ratio deviation has insufficient variance (ratio is too stable to modulate WD)
- If alignment diagnostic (H3) confirms alignment is informative but ratio is not

### Key Risk
Improvement over CWD may be within noise (+0.1-0.2%), making it hard to demonstrate statistical significance. Need 5+ seeds for ImageNet.

### Novelty
7/10 — directly extends CWD from binary to continuous. The extension is "obvious" given CWD, but no one has published it. Strongest when paired with the theoretical framework (Theorem 1 provides formal justification for continuous over binary).

### Resource Estimate
Same as EqWD.

---

## Pivot Decision Matrix

| Condition | Action |
|---|---|
| H2 confirmed, H6 confirmed | Stay with EqWD (front-runner) |
| H2 falsified, H3 confirmed | Pivot to Alternative 1 (BCM-WD) or Alternative 3 (CAWD) |
| H2 falsified, H3 falsified | Pivot to Alternative 2 (Budget Equivalence Framework) |
| H2 confirmed, H6 falsified | Reframe: EqWD provides modest improvement; emphasize theory + evaluation framework |
| H2 confirmed, H4 falsified | Drop layer-type-aware extension; use uniform EqWD |
| All hypotheses falsified | Pivot to Alternative 2 with strong negative findings; theoretical contributions remain |
