# Backup Ideas for Pivot

## Alternative 1: Towards a Task-Agnostic Absorption Metric

### Title
**Towards a Task-Agnostic Absorption Metric: Automated Hierarchy Discovery for Generalizable Feature Absorption Measurement in Sparse Autoencoders**

### Core Idea
The canonical absorption metric (Chanin et al., 2024) is tied to a hand-designed first-letter spelling task. This severely limits our ability to study absorption in general semantic domains and to benchmark new architectures fairly. We propose a task-agnostic absorption metric constructed by combining automated hierarchical concept discovery (via LLM-based feature labeling) with the causal ablation framework of Chanin et al.

### Hypothesis
A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with the causal ablation framework of Chanin et al., will correlate with the original first-letter absorption benchmark while enabling absorption measurement across arbitrary semantic domains. SAEs with higher scores on this metric will exhibit measurably worse downstream interpretability utility.

### Method
1. **Automated Hierarchy Discovery:** Use an LLM judge to label top-N SAE features and organize them into validated hierarchies (geography, biology, colors).
2. **Probe Training:** For each parent-child pair, train a logistic regression probe on residual-stream activations.
3. **Absorption Detection:** Perform k-sparse probing, detect false negatives, and run integrated-gradients ablation to identify absorbing latents.
4. **Validation:** Correlate the new metric with the original first-letter benchmark across 20-50 pretrained SAEs.

### Why Pivot Here
- If the multi-objective evaluation reveals that the field's main problem is not "tradeoffs" but "we don't even have a good way to measure absorption outside spelling," then a methods paper on metric generalization becomes the higher-impact contribution.
- Fully training-free and aligned with project constraints.

### Risks
- LLM-generated hierarchies may be noisy or hallucinated.
- New metric may not correlate with the first-letter benchmark, creating an ambiguous result.
- Higher engineering complexity than the front-runner.

---

## Alternative 2: Absorption Increases Effective Complexity — A Learning-Theoretic Analysis

### Title
**Absorption Increases Effective Complexity: A Generalization Bound for Sparse Autoencoders that Scales with Feature Absorption**

### Core Idea
While prior work has characterized absorption as an optimization artifact (Chanin et al., 2024) and a spurious minimum (Tang et al., 2025), no prior work has asked whether absorption increases the effective complexity of the SAE representation class. We derive a generalization bound showing that higher absorption increases the complexity term, and validate this prediction using SAEBench data.

### Formal Claim (Sketch)
Let $\mathcal{F}$ be a ground-truth feature set of size $P$ with a subset forming hierarchical parent-child relationships. Let an SAE achieve absorption rate $\alpha \in [0,1]$. Then with probability at least $1 - \delta$:

$$R(h) \leq \hat{R}_S(h) + \mathcal{O}\left( B \sqrt{ \frac{P \ln(em/P) + \alpha P \ln m + \ln(1/\delta)}{N} } \right).$$

### Empirical Validation
1. **SAEBench Correlation Analysis:** Extract absorption and sparse probing metrics for 200+ pretrained SAEs. Compute partial correlation between absorption and sparse probing accuracy, controlling for L0 and loss recovered.
2. **Architecture Comparison:** Evaluate sparse probing accuracy separately on parent-level features vs. non-hierarchical features for Standard SAE, TopK, OrtSAE, and Matryoshka SAE.

### Why Pivot Here
- If the multi-objective evaluation produces messy or noisy empirical results, a theory-driven paper with clean mathematical claims and targeted empirical validation may be more publishable.
- Bridges optimization theory, learning theory, and mechanistic interpretability in a novel way.

### Risks
- The combinatorial growth argument in the bound assumes absorbed pairs are independently selectable, which is an approximation.
- Real absorption rates may be too small for the $\alpha P \ln m$ term to dominate the base term, making the bound empirically indistinguishable from existing bounds.

---

## Alternative 3: Is Feature Absorption a Training Artifact? (Conditional)

### Title
**Is Feature Absorption a Training Artifact? A Controlled Baseline Study with Random-Decoder Sparse Autoencoders**

### Core Idea
If feature absorption were primarily a pathology of flawed training dynamics, then SAEs with randomly initialized, frozen decoder weights should exhibit substantially lower absorption rates than fully trained SAEs. Conversely, if absorption rates remain comparable, absorption is better understood as a geometric consequence of sparse dictionary learning on hierarchically structured data.

### Method
1. Load a trained SAE baseline (e.g., `gpt2-small-res-jb`).
2. Initialize decoder weights randomly, freeze them, and train only the encoder to matched L0 sparsity.
3. Run the `sae-spelling` absorption metric on both conditions.
4. Compare absorption rates. If random-decoder absorption is within ~20% of trained SAE absorption, conclude geometry dominates.

### Why Pivot Here
- This is the sharpest, most falsifiable experiment among all candidates. A clear result would immediately reframe how the field thinks about absorption.
- Korznikov et al. (2026) showed random-decoder baselines match trained SAEs on standard metrics, but **nobody has run the absorption metric on them**.

### Risks
- **Explicitly violates the project's training-free constraint** ("实验类型: training-free（分析现有预训练 SAE，不重新训练）"). This can only be pursued if the constraint is relaxed for a minimal pilot.
- Random-decoder SAE may fail to converge to reasonable MSE.
- Result may fall in an ambiguous range (50-80% of trained absorption), making interpretation difficult.

### Pivot Condition
Only pursue this alternative if:
1. The training-free constraint is explicitly relaxed, OR
2. The front-runner and Backup 1 both yield inconclusive results and a sharp, minimal training experiment is approved as an exception.

---

## Pivot Decision Rules

| Scenario | Pivot To |
|----------|----------|
| H1 (downstream causal cost) attenuates or flips on real SAEBench data | Backup 1 (task-agnostic metric) or Backup 2 (theory) |
| Metrics are too noisy for clean causal analysis | Backup 2 (theory + targeted correlation) |
| Task-agnostic metric shows strong correlation with first-letter benchmark | Analyze divergence as a negative result; may stay with front-runner or pivot to Backup 3 if training allowed |
| All empirical results are messy / inconclusive | Pivot to Backup 2 (theory-driven) |
