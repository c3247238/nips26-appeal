# Planning Critique -- Iteration 6

## Overall Assessment

The methodology document is detailed, well-organized, and demonstrates several best practices: pre-registered decision gates, explicit SAE configuration tables, four-control suite design, statistical correction pre-specification (Benjamini-Hochberg), bootstrap CI protocol (10k resamples, seed=42), and language discipline guidelines. However, the plan had three structural failures that materially degraded the study outcome: controls were not blocking gates in the execution sequence, the confound decomposition definition was not stress-tested against tautological outcomes, and resource allocation was breadth-first when the three most informative experiments were not included in the plan at all.

---

## Critical Issues

### 1. Controls Were Not Blocking Gates

The methodology specifies controls (C1-C4) and a decision gate: "After Exp 1a: If first-letter controls not calibrated (shuffled > 20% after refinement), investigate before proceeding." In execution, all 23 experiments ran to completion before the universal control failure was discovered. The shuffled control produced 74.6% on first-letter (far exceeding the 20% gate), yet all cross-domain experiments proceeded.

**Consequence**: 4+ experiments (cross-domain city-country, city-continent, city-language, animal-class) produced absorption measurements that the paper itself declares "uninterpretable." These GPU-hours could have been redirected to activation patching and CMI replication at L0=22.

**Fix for next iteration**: Enforce decision gates as actual blocking conditions in the execution pipeline. After Exp 1a, verify shuffled < 20% before launching Exp 1b-1e. This requires integrating the control check into the experiment scheduler.

### 2. Confound Decomposition Definition Not Stress-Tested

The methodology defines three false-negative categories (hedging, reconstruction error, hierarchy-driven) but does not consider what happens when the L0 range spans an 8x factor (22 to 176). At L0=176, only 10/1,195 tokens are false negatives, meaning the permissive hedging definition (resolution at ANY higher L0) will classify 99.2% of L0=22 false negatives as hedging regardless of mechanism.

**What should have been specified**: Both a permissive and a strict classification, where strict requires the specific parent-associated latent to fire at higher L0. The methodology should have included a calibration analysis: "What fraction of false negatives resolve because the parent latent fires vs. because other mechanisms resolve them?"

### 3. Three Highest-Value Experiments Not Planned

| Missing Experiment | Information Gain | GPU Cost | Why It Was Not Planned |
|---|---|---|---|
| Activation patching on persistent core words | Highest -- only causal evidence | 0.5-1h | Not in the proposal despite being mentioned as "immune escape test" in the Interdisciplinary perspective |
| Control failure analytical diagnosis | High -- explains WHY controls fail | 0h (CPU only) | Not recognized as a separate experiment; the control results were treated as side evidence rather than the central finding |
| CMI at L0=22 with perfect probes | High -- eliminates F1 confound | 1h | The CMI analysis was planned only at L0=82; L0=22 was not considered for CMI despite being the configuration with perfect probes |

**Root cause**: The planning prioritized novelty (cross-domain characterization, rate-distortion theory, unsupervised detection) over validity (metric validation, causal verification, confound control). The plan would have been stronger if it had adopted a "validate first, measure second" philosophy.

---

## Major Issues

### 4. Vocabulary Standardization Not Specified

The methodology specifies "single-token alphabetic words" but does not enforce a single canonical vocabulary across all experiments. The result: three different vocabularies (1,092 in CMI, 1,196 in confound decomposition, 1,204 in first_letter_improved) for the same first-letter task, producing incompatible absorption rates.

The planning should have specified: "All first-letter experiments use the same tokenization run, producing a single canonical word list. All pipelines import this list rather than generating their own."

### 5. Statistical Methodology for CMI Was Underspecified

The methodology specifies k-NN MI estimation at d'={10,20,30,50} but does not:
- Pre-register which d' is primary
- Specify MI estimation diagnostics (bootstrap CIs, convergence, k-sensitivity)
- Plan for the probe quality confound (partial correlation, quality-gated subanalysis)
- Specify correction for multiple testing across dimensions (4 tests)

This underspecification enabled post-hoc selection of d'=10 as "primary" and left the probe quality confound uncontrolled.

### 6. Hierarchy Predictor Analysis Was Underpowered by Design

With n=5 domains, the hierarchy predictor analysis (H6) had near-zero statistical power. Bootstrap CIs span [-1.0, +1.0] for all predictors. The methodology should have either (a) acknowledged this as an exploratory analysis from the start or (b) designed the study to include 10+ domains (e.g., splitting first-letter into consonants/vowels, adding city-language by language family, etc.).

---

## Positive Findings

### 1. Multi-L0 Sweep Design (Innovative)
The four L0 operating points {22, 41, 82, 176} covering an 8x range was a well-chosen design that enabled the L0 phase transition discovery and the confound decomposition. This should be adopted as standard practice for absorption studies.

### 2. Cross-Layer Validation
Testing at layers 10, 12, and 20 for cross-layer stability was a good design choice that produced the most confident finding (CV < 10%). The triple-layer validation at L0=82 provides genuine robustness evidence.

### 3. Pre-Specified Language Discipline
Explicitly specifying "statistical association" not "causal effect" and "statistical mediation" not "causal mediation" demonstrates learning from iteration 5's overclaiming. However, this discipline was not applied to the CMI claims in the actual paper.

### 4. Resource Estimates Were Conservative
Planned: 6-7 GPU-hours. Actual: ~30 minutes. The 10x overestimate is consistent across iterations and indicates room for additional experiments within the time budget. The system correctly recognized this resource margin was available.

### 5. Control Suite Design (Exemplary)
Four controls covering different failure modes (random direction baseline, label permutation, dense probe ceiling, untrained SAE null), applied per domain, with explicit expected outcomes. This is the most rigorous control design in any published absorption study.
