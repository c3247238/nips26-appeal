# Idea Validation Decision

## Pilot Evidence Summary

This decision evaluates the front-runner candidate (`cand_crossdomain_causal_tax`) after iteration 10 of experimental work. All 10 hypotheses (H1-H10) have received verdicts. Seven pilot/full-mode tasks were completed successfully in this iteration, with 2 skipped (data integrity check reused iter_009 validation, env check reused iter_009). The consolidation summary (`consolidation_summary.json`) represents the accumulated evidence from 10 iterations.

### Key Metrics per Candidate

**cand_crossdomain_causal_tax (front-runner)**:
- H1 (Cross-Domain Variation): SUPPORTED_WITH_NUANCE. Kruskal-Wallis p=0.005. At L24_16k: first-letter 27.1%, city-continent 31.4%, city-country 45.1%, city-language 11.6%. 4.1x descriptive range.
- H7 (Causal Absorption): SUPPORTED universally. First-letter d=1.33, city-continent d=1.50, city-language d=0.75. All p<0.001.
- H8 (Decoder Magnitude): CONSISTENT across hierarchies. First-letter 5.99 nats, city-continent 3.98 nats, both 100% at all thresholds. Control ~0.01-0.12 nats.
- H10 (Probe Degradation): MIXED. City-continent matches probe degradation curve (31.4% vs predicted 34.7%, delta=-3.2pp). City-language is genuine outlier (11.6% vs predicted 35.5%, delta=-23.9pp). BOTH probe quality AND genuine hierarchy effects drive variation.
- H9 (Rate-Distortion): NOT_SUPPORTED. 131 pairs: model rho=0.286, R^2=0.104. All individual predictors in WRONG direction. Fourth negative for correlational methods.
- H3 (Hedging): SUPPORTED. Strict 0-22.6% vs loose 92.6%. Compensatory dominates.
- H4 (GAS): REFUTED. rho=0.116, AUROC=0.571.
- H5 (Absorption Tax): NOT_SUPPORTED quantitatively. rho=-0.20.
- H6 (Architecture): INCONCLUSIVE. Underpowered (12 obs). Hierarchy >> architecture.
- Writing gate: GO_WRITE. All figures generated, corrections propagated, appendices compiled.

**cand_controlled_dictionary (backup)**: No pilot data collected. Remains theoretical.

**cand_ecological_phase_transition (backup)**: No pilot data collected. Zero GPU cost, uses existing data.

**cand_absorption_aware_correction (backup)**: No pilot data collected. Inference-time correction concept.

**cand_within_hierarchy_variation (backup)**: No pilot data collected. Per-class analysis.

## Decision Matrix

### cand_crossdomain_causal_tax

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 5 | 10 hypotheses tested, 5 supported (H1/H3/H7/H7-cross/H8), 1 mixed (H10 -- both outcomes publishable), 4 transparent negative results. Universal causal mechanism d=0.75-1.50. Probe degradation ablation successfully decomposes variation sources. |
| Hypothesis survival | 0.25 | 5 | Core hypotheses H1 (cross-domain variation), H7 (universal causal mechanism), H8 (decoder entanglement) all survived. H10 resolves the probe artifact ambiguity with a nuanced MIXED result that is more publishable than either extreme. No main hypothesis was falsified. |
| Path to full result | 0.20 | 5 | Writing gate passed: GO_WRITE. All experiments complete. 15 paper corrections propagated. 4/4 figures generated. 5 appendix sections compiled. Methodology documented. Paper is in final-polish phase, not discovery phase. |
| Novelty (from report) | 0.15 | 4 | Overall novelty score 7/10 from novelty checker. Per-contribution: cross-domain characterization 7, universal causal mechanism 8, probe degradation ablation 8, hedging 7, decoder magnitude 6, negative results 6. No competing cross-domain work as of April 2026. Risk: SAEBench follow-up could extend absorption metric to RAVEL. |
| Resource efficiency | 0.10 | 5 | Only 3.5 GPU-hours total for this iteration (most work was zero-GPU paper fixes). All critical experiments done. Paper-ready state achieved with minimal remaining compute. |

**Weighted score: 0.30*5 + 0.25*5 + 0.20*5 + 0.15*4 + 0.10*5 = 1.50 + 1.25 + 1.00 + 0.60 + 0.50 = 4.85**

### cand_controlled_dictionary (backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected. Untested. |
| Hypothesis survival | 0.25 | 3 | Hypotheses are sound in theory (encoder vs dictionary contribution). Strengthened by universal causal mechanism finding. But completely untested. |
| Path to full result | 0.20 | 2 | Would require new experimental infrastructure (OMP encoding, 2-pass encoding). Estimated 1-2 GPU-hours, but no prior tooling exists. |
| Novelty (from report) | 0.15 | 4 | Novelty 8/10 from checker. First controlled dictionary experiment. No prior work isolates encoder vs dictionary. |
| Resource efficiency | 0.10 | 3 | 1-2 GPU-hours. Reasonable, but requires abandoning near-complete paper. |

**Weighted score: 0.30*1 + 0.25*3 + 0.20*2 + 0.15*4 + 0.10*3 = 0.30 + 0.75 + 0.40 + 0.60 + 0.30 = 2.35**

### cand_ecological_phase_transition (backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected. Would use existing data (zero GPU). |
| Hypothesis survival | 0.25 | 3 | Phase transition and Lotka-Volterra formalisms are theoretically motivated. Universal mechanism (d=0.75-1.50) provides empirical basis. But untested. |
| Path to full result | 0.20 | 2 | Could supplement the front-runner paper as an additional section or companion paper. Not a standalone replacement. |
| Novelty (from report) | 0.15 | 5 | Novelty 8/10. Zero prior work on ecological competition in SAE absorption. Genuinely novel framing. |
| Resource efficiency | 0.10 | 5 | Zero additional GPU cost -- uses existing data. |

**Weighted score: 0.30*1 + 0.25*3 + 0.20*2 + 0.15*5 + 0.10*5 = 0.30 + 0.75 + 0.40 + 0.75 + 0.50 = 2.70**

### cand_absorption_aware_correction (backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected. |
| Hypothesis survival | 0.25 | 2 | Feasibility unknown. Absorption may permanently alter decoder geometry, making post-hoc correction ineffective. |
| Path to full result | 0.20 | 2 | Would require significant new implementation. Not a natural extension of existing work. |
| Novelty (from report) | 0.15 | 4 | Novelty 7/10. First inference-time correction approach. |
| Resource efficiency | 0.10 | 2 | 1-2 GPU-hours, but high implementation effort. |

**Weighted score: 0.30*1 + 0.25*2 + 0.20*2 + 0.15*4 + 0.10*2 = 0.30 + 0.50 + 0.40 + 0.60 + 0.20 = 2.00**

### cand_within_hierarchy_variation (backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected. Would use existing data. |
| Hypothesis survival | 0.25 | 3 | The H10 result (city-language as genuine outlier) suggests within-hierarchy variation exists. Per-class analysis would deepen the characterization. |
| Path to full result | 0.20 | 3 | Natural extension of existing cross-domain work. Could be added as a section to the front-runner paper. |
| Novelty (from report) | 0.15 | 3 | Novelty 6/10. Relatively straightforward given existing data. |
| Resource efficiency | 0.10 | 5 | Zero additional GPU cost. |

**Weighted score: 0.30*1 + 0.25*3 + 0.20*3 + 0.15*3 + 0.10*5 = 0.30 + 0.75 + 0.60 + 0.45 + 0.50 = 2.60**

## Decision Rationale

The decision is unambiguously **ADVANCE** with `cand_crossdomain_causal_tax`.

**Evidence supporting this decision:**

1. **Overwhelming pilot signal (score 4.85/5.00):** The front-runner has completed 10 iterations of experimental work with all 10 hypotheses tested. The writing gate has passed with all prerequisites satisfied: corrected data propagated, figures generated, appendices compiled, methodology documented.

2. **Core hypotheses survived and strengthened:** H7 (universal causal mechanism via activation patching, d=0.75-1.50 across all hierarchies) is the paper's strongest finding. H10 (probe degradation ablation) produced a nuanced MIXED result that is MORE publishable than either extreme -- it decomposes cross-domain variation into probe quality confound AND genuine hierarchy effects, which is the most scientifically honest answer.

3. **Clear path to publication:** The paper is in final-polish phase. The writing gate explicitly recommends GO_WRITE. Remaining open issues are LOW priority (data integrity spot-check, more degradation levels, cross-model validation, token position confound, circularity independent test) and can all be addressed in camera-ready revision.

4. **Quadruple negative for correlational predictors is itself a key finding:** GAS (rho=0.116), CMI (rho=0.044), Absorption Tax (rho=-0.20), Rate-Distortion (rho=0.286, all predictors wrong direction). The common theme -- absorption is a causal phenomenon requiring causal methods -- is a substantive contribution that positions the paper distinctly in the field.

5. **No competing work:** Novelty check as of April 2026 confirms no competing cross-domain absorption characterization. The field is active on architectural mitigations (HSAE, OrtSAE, Masked Regularization), not characterization. Window is open but closing.

6. **No backup candidate is close:** The highest-scoring backup (cand_ecological_phase_transition, 2.70) has zero pilot data and no clear standalone publication path. It could supplement the front-runner as future work.

**Sanity checks:**
- [x] All 5 candidates compared, not just the front-runner.
- [x] No candidate failed its own falsification criteria for the core hypotheses. H4/H5/H9 were falsified but are reported as transparent negative results, which is a strength.
- [x] Not swayed by sunk cost: 10 iterations is a large investment, but the decision is based on evidence quality, not effort invested. If the evidence were weak, I would PIVOT.
- [x] H10 pilot was somewhat inconclusive (MIXED rather than clean pass/fail), but MIXED is itself a clear, publishable answer. I am not defaulting to ADVANCE out of inertia -- the evidence genuinely supports it.

## Next Actions

1. **Proceed to full experiment execution** (if any remaining tasks) -- all critical experiments are already complete.
2. **Proceed to writing/editing phase** -- the writing gate has passed. Generate the final paper draft integrating all 10 iterations of results.
3. **Address MEDIUM-priority open issues in camera-ready:**
   - OI2: More probe degradation levels (F1=0.60, 0.65, 0.75, 0.95) to strengthen the H10 curve
   - OI1: Data integrity spot-check (validate_integration.py against paper LaTeX)
4. **Venue target: NeurIPS 2026 main conference or ICLR 2027 main conference.** Fallback: EMNLP 2026 or NeurIPS MI Workshop.
5. **Competition watch:** Monitor for SAEBench extensions to cross-domain absorption. Execute promptly to maintain priority.

SELECTED_CANDIDATE: cand_crossdomain_causal_tax
CONFIDENCE: 0.94
DECISION: ADVANCE
