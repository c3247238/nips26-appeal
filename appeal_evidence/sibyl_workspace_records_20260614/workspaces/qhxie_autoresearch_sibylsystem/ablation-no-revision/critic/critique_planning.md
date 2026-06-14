# Critique of Planning: Quantifying Feature Absorption in Sparse Autoencoders

## Summary

The experimental planning is generally sound with appropriate pre-registered hypotheses and falsification criteria. The pilot-before-full-experiment strategy is correct. However, three issues reduce the plan's robustness: (1) H2 has no fallback plan if H1 fails; (2) H4's design flaw was predictable and should have been addressed in the pre-registration; (3) the H5 subsampling limitation was not pre-registered as a known caveat.

---

## What Works

### Pre-registered falsification criteria
Each hypothesis has a clear falsification criterion stated before experiments. This is good scientific practice that enables fair evaluation.

### Pilot-before-full strategy
The plan correctly identifies pilots (100 sequences, ~10 min runtime) as the appropriate first step before committing to full experiments. The NO-GO decision framework (based on pilot results) is sound.

### Multiple hypothesis approach
Testing five hypotheses in parallel is efficient. The hypothesis summary table in the plan is clear and complete.

### Random control baseline
Pre-registering the random dictionary control as a baseline is excellent practice.

---

## Critical Issues

### 1. H2 Has No Fallback Plan

The plan states H2 will be tested but does not pre-register what happens if H1 shows insufficient absorption variance. H1's falsification criterion (<10% at layers 4-10) was expected to still leave enough variance for H2's token-frequency correlation. However, if absorption is too rare at layer 8 (46 of 24,576 = 0.19%), there is no variance to correlate with token frequency.

The plan should have pre-registered: "If H1 shows <1% prevalence at layer 8, H2 will use layer 4 as the primary test layer (which has higher absorption per H3 results), with explicit justification for why layer 4's absorption profile would enable the token-frequency correlation analysis."

### 2. H4 Design Flaw Was Predictable

The H4 hypothesis tests whether "high-absorption SAE patching reduces faithfulness by >=5pp vs. low-absorption." The operationalization selects low/high absorption latents corpus-wide, then patches them on a specific circuit (France/Paris). This is a two-step selection that is inherently flawed: corpus-wide absorption scores do not predict circuit-level causal importance.

A predictable concern is that the France/Paris circuit may be driven by latents that are neither high-absorption nor low-absorption by corpus standards, making the experiment uninformative. The plan should have pre-registered: "If corpus-wide absorption selection proves uninformative for circuit-level importance, H4 will be redesigned to compare full SAEs at different layers (layer 4 vs. layer 8), holding dictionary size constant while varying absorption level."

### 3. H5 Subsampling Limitation Not Pre-Registered

The plan uses cumulative subsampling to simulate dictionary sizes 2K and 8K from the full 24K SAE. This is a reasonable approach but has a known limitation: independently trained SAEs may have qualitatively different dictionaries at different sizes, not just a subset of a larger dictionary.

The plan should have pre-registered this as a known caveat: "H5 results are based on cumulative subsampling from the full 24K SAE, not independently trained SAEs. Results may differ for independently trained SAEs with different dictionary sizes."

---

## Minor Issues

### H3 L0 proxy concern

The plan uses L0 as a proxy for lambda (sparsity penalty strength) without pre-registering what happens if L0 and lambda are not monotonic across layers. This is exactly what was observed (layer 8 has the highest L0 but lowest absorption among mid layers), which is interesting but was not expected. The plan should have acknowledged this as a risk.

### Layer choice for H1

The plan states H1 will be tested at "mid-to-deep layers" with a falsification criterion of "<10% across layers 4-10." The layer-8-specific reporting (0.19%) is appropriate as the primary test layer, but the plan should have explicitly stated that layer 8 is the primary layer with layers 4/6 as exploratory.

---

## Assessment

The planning is generally sound with appropriate pre-registration and a correct pilot-before-full strategy. The main weaknesses are: (1) H2 has no fallback plan if H1 fails; (2) H4's design flaw was predictable and should have been addressed in the pre-registration; (3) the H5 subsampling limitation was not pre-registered as a known caveat. These are planning-level issues that, if addressed, would have improved the robustness of the experimental program.