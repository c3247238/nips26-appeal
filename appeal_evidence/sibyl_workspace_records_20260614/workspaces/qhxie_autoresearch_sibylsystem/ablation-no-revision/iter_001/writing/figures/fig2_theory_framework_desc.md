# Figure 2: Theory Framework Diagram

## Type
Architecture / Analogy Diagram (generated via Python matplotlib)

## Description
A side-by-side comparison illustrating the mapping between chemical kinetics concepts and LLM multi-sample reasoning. The diagram consists of two columns:

**Left column — Chemical Kinetics:**
- Activation energy $E_a$: the energy barrier reactant molecules must overcome
- Temperature $T$: thermal energy driving molecular collisions
- Catalyst: a substance that lowers the activation energy
- Pre-exponential factor $A$: frequency of molecular collisions
- Reaction rate: speed of product formation per unit time
- Arrhenius equation: $k = A e^{-E_a/RT}$

**Right column — LLM Reasoning:**
- Problem difficulty: the cognitive barrier to producing a correct answer
- Sampling diversity / compute: multiple independent samples providing reasoning diversity
- Tools or external knowledge: external scaffolding that lowers problem difficulty
- Model base capability: the model's inherent ability to approach the problem
- $P$(\textit{correct}) per sample: probability of a correct answer per sample
- Exponential saturation equation: $P_k = P_\infty (1 - e^{-k/k_0})$

**Bottom — Unified Equation:**
$$P_k = P_\infty \cdot (1 - e^{-k/k_0})$$

where $P_\infty$ is the asymptotic accuracy ceiling and $k_0$ is the characteristic sampling count.

## Generation Script
`gen_fig2_theory_framework.py` — uses matplotlib to draw two rounded boxes with paired bullet items and connecting arrows.

## Output
`fig2_theory_framework.pdf`
