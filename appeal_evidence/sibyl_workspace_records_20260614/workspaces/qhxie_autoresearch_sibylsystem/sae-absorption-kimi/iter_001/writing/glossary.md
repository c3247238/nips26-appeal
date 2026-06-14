# Glossary

This document unifies key terminology definitions used throughout the paper. All section writers should reference this file for consistency.

---

## Core Concepts

**Sparse Autoencoder (SAE)**
: An unsupervised neural network that learns an overcomplete dictionary of features by encoding high-dimensional activations into a sparse latent representation and reconstructing the original input. Used extensively for mechanistic interpretability of language models.

**Feature Absorption**
: A pathology in SAEs where a general "parent" feature (e.g., "animal") is not represented by its own latent but is instead encoded through a more specific "child" feature (e.g., "dog"). This makes the parent feature effectively invisible in the SAE's latent space.

**Feature Hedging**
: An opposite pathology where a single SAE latent incorrectly mixes multiple correlated features, failing to achieve monosemanticity. First formalized by Chanin et al. (2025) as a failure mode of narrow SAEs.

**Monosemanticity**
: The property of a neural network representation where individual neurons or latents encode single, interpretable concepts. A core goal of SAE research.

**Polysemanticity**
: The property where a single neuron or latent encodes multiple unrelated concepts. The standard superposition hypothesis posits that polysemanticity arises from compressed representations.

## Metrics and Evaluation Terms

**First-Letter Absorption Benchmark**
: The canonical absorption metric introduced by Chanin et al. (2024), which measures whether SAEs correctly represent the first letter of a word (parent) independently of the full word spelling (child). Used as the primary comparability benchmark in absorption research.

**Task-Agnostic Absorption Metric**
: A proposed generalization of absorption measurement that uses automated hierarchy discovery (e.g., continent -> country) rather than a single supervised task. Piloted in Experiment 3 of this paper.

**Explained Variance (EV)**
: The fraction of variance in the original activations that is recovered by the SAE reconstruction. Higher is better.

**CE Loss Recovered**
: The ratio of original cross-entropy loss to cross-entropy loss when SAE-reconstructed activations are substituted back into the model. Values above 1.0 indicate that reconstruction preserves or improves next-token prediction; values below 1.0 indicate degradation.

**Dead Neuron**
: A latent in the SAE dictionary that never (or almost never) activates on a held-out corpus. High dead-neuron rates indicate inefficient dictionary utilization.

**Sparse Probing F1**
: The F1 score of a sparse linear classifier trained on SAE latents to predict downstream concepts. Measures whether SAE features are useful for downstream tasks.

**RAVEL (Causality Benchmark)**
: A benchmark for causal interpretability that evaluates whether model interventions on specific features produce targeted, isolated effects. Cause and Isolation are two sub-metrics.

## Architecture Terms

**Standard SAE**
: The baseline sparse autoencoder architecture trained with an $L_1$ sparsity penalty on latents.

**TopK SAE**
: An SAE variant that enforces sparsity by keeping only the top-$K$ latents with highest activations and zeroing the rest.

**JumpReLU**
: An SAE variant that uses a ReLU with an adaptive jump threshold, allowing latents to turn on only when activations exceed a learned cutoff.

**Gated SAE**
: An SAE architecture with separate gating and magnitude pathways, designed to improve feature separability.

**Matryoshka SAE**
: A multi-scale dictionary learning approach that organizes features at multiple granularities, reported to reduce absorption.

**Feature-Splitting SAE**
: An architecture that splits the residual stream into multiple subspaces before applying separate SAEs, designed to improve reconstruction and reduce dead neurons.

**PAnneal**
: A penalty-annealed training strategy that gradually adjusts the sparsity penalty during training.

## Methodological Terms

**Training-Free Evaluation**
: An evaluation paradigm that uses publicly released pretrained SAE checkpoints without training new SAEs. All experiments in this paper follow this paradigm.

**Pareto Front**
: In multi-objective optimization, the set of solutions where no objective can be improved without worsening another. Used in Experiment 1 to evaluate whether any SAE architecture dominates across metrics.

**Stochastic Dominance**
: A statistical concept meaning that one distribution is consistently better than another across all quantiles. Tested via Mann-Whitney U tests in Experiment 1.

**Partial Correlation**
: A correlation between two variables with the linear effect of one or more control variables removed. Used in Experiment 2 to isolate absorption's unique effect on downstream utility.

**Cluster-Robust Standard Errors**
: Standard errors adjusted for correlation within groups (here, architecture families). Used in Experiment 2 regression analyses.

## Preferred Phrasing

- Use "fine-tuning" (not "finetuning").
- Use "few-shot" (not "few shot").
- Use "downstream" as an adjective (e.g., "downstream performance").
- Use "checkpoints" to refer to pretrained SAE releases (not "models" or "weights").
- Use "architecture family" to group SAEs by design (Standard, TopK, JumpReLU, etc.).
- Use "metric" for a quantitative measurement; use "benchmark" for a standardized evaluation protocol.
- Use "latent" (not "neuron" or "feature") when referring to SAE dictionary elements, unless explicitly comparing to neurons.

## Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| SAE | Sparse Autoencoder |
| LLM | Large Language Model |
| CE | Cross-Entropy |
| MSE | Mean Squared Error |
| EV | Explained Variance |
| F1 | F1 Score (harmonic mean of precision and recall) |
| RAVEL | Relational Attribution for Visual Explanation and Learning (causality benchmark) |
| TPP | Token-level Predictive Performance |
| SCR | Sparse Coding Recall |
| HF | HuggingFace |
| OLS | Ordinary Least Squares |
| SE | Standard Error |
| CI | Confidence Interval |
