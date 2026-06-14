# Critique: Planning (experiment_plan.json, methodology.md)

## Summary

The experiment plan was well-structured with clear task dependencies and pass criteria. However, critical issues were not addressed: H1's hypothesis is internally inconsistent (falsified at layer 8 but confirmed at layer 4), H4's experiment is fundamentally uninformative (both subsets yield 0.0 because 90% subsetting destroys reconstruction), H2 was abandoned despite layer 4 having 260x more data, and no contingency plans existed for the outcomes observed. Additionally, the plan specified full-scale (1,024-sequence) experiments but only pilot-scale (100 sequences) was executed.

## Critical Issues

### 1. H1 Falsification Criterion Cannot Be Evaluated as Stated

**Planned criterion**: "<10% prevalence across layers 4-10 falsifies H1"

**Results**:
- Layer 8: 0.19% (below 10%, falsified)
- Layer 4: 49.3% (above 20%, NOT falsified)

**Problem**: The falsification criterion tests across "layers 4-10" but layer 4 (49.3%) contradicts the hypothesis (>20%), not supports it. The hypothesis cannot be falsified across a range when one layer in that range confirms it.

**What should have been in the plan**:
1. Layer-specific pass/fail criteria (e.g., "<10% at layer 8 specifically"), OR
2. Aggregate metric across layers with explicit formula (mean, min, etc.), OR
3. Specify one primary layer for H1 evaluation

**Current presentation misleads**: Readers are told H1 is "falsified" but layer 4 actually confirms it. The plan should have been explicit about which layers trigger falsification and how layer-specific results are aggregated.

### 2. H4 Plan Had No Contingency for Uninformative Result (0.0 for Both Subsets)

**Planned design**: Select bottom-10% and top-10% absorption latents, zero out 90%, test patching

**Problem**: The plan did not anticipate that subsetting 90% of the dictionary would destroy reconstruction quality entirely, yielding 0.0 for both subsets. The plan gated h4_full on h4_pilot completion but did not specify what to do when the pilot produces an uninformative result.

**What should have been in the plan**:
- Decision trigger: "If both absorption subsets yield faithfulness = 0.0, flag H4 redesign before proceeding to h4_full"
- Redesign option: "Compare FULL SAE representations at layers with different absorption profiles (layer 4 vs layer 8) rather than subsets of a single layer"

### 3. H2 Termination Decision Was Not Based on Actual Data

**Planned**: h2_full depends on h2_pilot completion

**What happened**: H2 was never run. The plan cited "early termination after H1/H3/H4 falsification."

**Critical data that was ignored**: Layer 4 has 49.3% absorption (~12,000 latents) vs layer 8's 0.19% (~46 latents) — 260x more data for H2 analysis. The plan never specified: "If H1 shows layer-specific results, run H2 at the layer with highest absorption."

**What should have been in the plan**: A decision tree: "If absorption rate at layer 8 is <1%, evaluate H2 at layer 4 where absorption is highest. Only terminate H2 if layer 4 also shows insufficient variance."

### 4. Full-Scale Experiments Were Never Executed

**Planned**: Full experiments use 1,024 sequences; pilots use 100 sequences

**Actual execution**: All results reported use only 100 sequences (pilot scale). The methodology states 1,024-sequence experiments but these were never run.

**What should have been in the plan**: Explicit commitment to full-scale experiments or a decision to scope as pilot-only. The 10x scale gap is material for rare phenomenon characterization.

## Major Issues

### 5. H3 Fallback Was Not Triggered When L0 Proxy Failed

**Planned**: "If no lambda sweep exists, train 3 mini-SAEs with L1=[4e-5,8e-5,1.6e-4] on 10M tokens"

**What happened**: L0 proxy failed (Spearman r=0.086, p=0.872) but mini-SAEs were not trained. The inverted-U pattern was reported without mechanistic understanding.

**What should have been in the plan**: Automatic trigger condition: "If H3 pilot shows no monotonic correlation (Spearman r not significant), auto-trigger h3_mini_sae training task before declaring H3 falsified."

### 6. No Metric Validation Task Before Experiments

**Planned**: H1-H5 experiments on pretrained SAEs

**Missing**: Positive control showing the metric can detect real absorption when present

**Problem**: The random dictionary control validates the null case (0.0 absorption by construction) but provides no evidence that the absorption metric can detect actual absorption at any threshold. If the metric is only sensitive to complete absorption (A_f = 1.0), partial absorption would be systematically missed.

**What should have been in the plan**: A metric validation task before h1_full:
- Generate synthetic SAEs with injected absorption patterns
- Verify metric recovers them across different absorption levels
- This gates all subsequent experiments

### 7. No Seed Ablation Specification

**Planned**: Single-seed pilots (seed=42)

**Problem**: The dramatic layer discrepancy in H1 results (layer 4 at 49.3% vs layer 8 at 0.19%) could be a seed artifact. Without seed ablation, no finding is robust.

**What should have been in the plan**: "Each key finding (H1 layer comparison, H3 inverted-U) requires confirmation across 3 seeds (42, 43, 44) before declaring pass/fail. Results must replicate across seeds for the finding to be considered stable."

### 8. H5 Dictionary Subsampling Is Not Equivalent to Trained Dictionaries

**Planned**: Simulate smaller dictionaries by subsampling 24K to 2K/8K

**Problem**: Subsampling a trained 24K dictionary does not produce the same absorption profile as a dictionary trained with 2K or 8K latents from initialization. Training dynamics differ — a 2K dictionary trained from scratch has different feature specialization than one extracted from a 24K dictionary.

**What should have been in the plan**: Either:
1. Use actual SAELens checkpoints at different dictionary sizes (if available in gpt2-small-res-jb), OR
2. Explicitly acknowledge in the methodology that subsampling is an approximation and may not reflect trained dictionary properties

### 9. H4 Uses Single Prompt Without Generalization Check

**Planned**: "The capital of France is ___" (clean) vs "The capital of Germany is ___" (corrupted)

**Problem**: Using one factual recall prompt does not generalize to other circuits. The faithfulness metric is task-specific and may not reflect circuit behavior more broadly.

**What should have been in the plan**: "H4 uses 5-10 factual recall prompts with different subject-object pairs (e.g., 'The capital of Japan is', 'The capital of Italy is'). Mean faithfulness across prompts is the primary metric."

## Recommendations

1. **Add metric validation task** before h1_full (synthetic positive controls) — gates all experiments
2. **Add H4 redesign trigger** (if both subsets yield 0.0, redesign before h4_full)
3. **Add H2 decision tree** (if layer 8 insufficient, try layer 4; only terminate if both layers fail)
4. **Add H3 auto-trigger** for mini-SAE training if L0 proxy fails (r not significant)
5. **Add seed ablation** (3 seeds) for all key findings
6. **Specify H5 dictionary source** (actual checkpoints vs subsampling approximation)
7. **Add H4 multi-prompt requirement** (5-10 prompts, not 1)
8. **Clarify H1 falsification criterion** (layer-specific vs aggregate; which layers are primary)

The plan's strength was clear task dependencies and pass criteria. The weakness was no contingency for uninformative results, no metric validation, no decision framework for when pilots fail, and no seed ablation. Future plans should include explicit decision trees and triggers for all major hypothesis outcomes.