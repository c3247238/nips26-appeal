# Backup Ideas for Pivot

## Alternative A: FastProbe-Absorb — Automated Probe-Based Screening for Feature Absorption

### Motivation
The Chanin/SAEBench absorption metric is accurate but slow (~26 min per SAE) and requires manual first-letter task design. The community lacks a scalable baseline for absorption screening that can be run in seconds per SAE as a preprocessing step.

### Core Idea
Develop a lightweight, automated probe-projection screening method that detects absorption-like behavior across thousands of SAE latents without expensive causal ablations. For each SAE latent, train lightweight logistic regression probes on diverse token properties (not just first letters). Compute the fraction of probe projection variance captured by "non-main" latents with similar probe directions. Flag absorption when the ratio exceeds a relaxed threshold (τ_c = 0.4, following Chanin Appendix A.13).

### Validation Plan
1. Run FastProbe-Absorb on 8–12 pretrained SAELens SAEs.
2. Validate against SAEBench gold-standard absorption scores.
3. Target correlation r > 0.7 and runtime < 1 min per SAE.

### Why It Is a Strong Backup
- **Low implementation risk**: Builds directly on `sae_spelling.probing` and SAEBench.
- **Clear community utility**: A fast screen would be genuinely useful for SAE practitioners.
- **Natural pivot path**: If the construct-validity study (front-runner) shows that first-letter absorption is a poor proxy, a fast screen based on *diverse semantic probes* becomes even more valuable.

### Pivot Trigger
Pivot to this idea if the front-runner's pilot reveals that semantic-hierarchy probes are too noisy or that the correlation is uninterpretable. In that case, shift focus from *validating* the existing metric to *replacing* it with a more scalable alternative.

---

## Alternative B: The Rate-Distortion Origin of Feature Absorption — A Theoretical Bound

### Motivation
Chanin et al. (2024) proved that absorption decreases loss for a *pair* of hierarchical features, but there is no general theorem predicting absorption for arbitrary tree-structured hierarchies. The field lacks a unified theoretical framework.

### Core Idea
Prove a general combinatorial theorem: for an SAE with m ≥ N latents optimizing reconstruction + λ·sparsity on a tree-structured hierarchy with N nodes, **absorption is inevitable for any λ > 0**. Derive an explicit bound on absorption depth as a function of sparsity penalty, branching factor, and co-occurrence probability.

### Empirical Validation
1. **Synthetic validation**: Generate synthetic activations from a 3-level binary tree (7 features) with controlled co-occurrence probabilities. Train small SAEs with varying λ and measure absorption via the δ-absorption metric.
2. **Real SAE test**: Use SAEBench absorption evaluation on GPT-2 small SAEs with varying L0 targets to test whether absorption increases monotonically with effective sparsity penalty.

### Why It Is a Strong Backup
- **Theoretically crisp**: A clean theorem with an explicit threshold is rare in the SAE literature.
- **Directly testable**: Synthetic experiments are fast and fully controlled.
- **High novelty**: No published work generalizes Chanin et al.'s two-feature proof to arbitrary hierarchies.

### Pivot Trigger
Pivot to this idea if the front-runner's empirical study is confounded by probe quality or frequency-matching difficulties. A theory paper with synthetic validation is less vulnerable to these empirical noise sources.

---

## Alternative C: Rethinking Feature Absorption — A Validity-Aware Analysis of SAE Architectures

### Motivation
The field assumes that reducing absorption scores is a meaningful proxy for improving SAE feature quality. But recent work shows that randomized SAE baselines match trained SAEs on standard metrics (Korznikov et al., 2026), and Matryoshka SAEs trade absorption for hedging. This challenges the validity of absorption as an optimization target.

### Core Idea
Systematically test whether architectures with lower absorption scores (Matryoshka, OrtSAE) actually produce more valid or actionable features than standard SAEs when evaluated by alternative criteria:
1. **Random-baseline margin**: Compute the gap between trained and frozen-random-decoder baselines on AutoInterp, sparse probing, and TPP.
2. **Causal actionability proxy**: Use activation patching on factual recall prompts to measure whether intervening on SAE latents changes model behavior.
3. **Label-free geometric proxy**: Compute decoder-weight cosine-similarity hierarchies to quantify "geometric absorption" without ground-truth probes.

### Validation Plan
1. Select 6–8 pretrained SAEs spanning high-to-low absorption architectures.
2. Run SAEBench evaluations + random-baseline comparisons.
3. Run activation patching on 50–100 factual recall prompts.
4. Test whether low-absorption architectures outperform high-absorption ones on validity-aware metrics.

### Why It Is a Strong Backup
- **Provocative and high-impact**: Directly challenges a core assumption of the field.
- **Builds on existing infrastructure**: Uses SAEBench, SAELens, and TransformerLens.
- **Actionable regardless of outcome**: If low-absorption architectures *do* win, the result validates the community's focus. If they *don't*, it reframes the research agenda.

### Pivot Trigger
Pivot to this idea if the front-runner yields a strong positive correlation (first-letter absorption *does* generalize), because the next natural question is: **Does reducing absorption actually improve feature validity?** This backup directly attacks that follow-up question.

---

## Pivot Decision Framework

| Scenario | Recommended Pivot |
|----------|-------------------|
| Semantic-hierarchy probes are too noisy / correlation is uninterpretable | **Alternative A** (FastProbe-Absorb) |
| Empirical study is confounded by frequency / probe quality issues | **Alternative B** (Theoretical bound + synthetic validation) |
| Front-runner shows strong generalization; next question is causal impact | **Alternative C** (Validity-aware analysis) |
| Front-runner shows weak generalization; need constructive alternative | **Alternative A** or **Alternative C** |
