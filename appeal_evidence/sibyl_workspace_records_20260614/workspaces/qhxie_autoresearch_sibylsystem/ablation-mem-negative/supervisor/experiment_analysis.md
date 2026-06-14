# Experiment Result Analysis

## Key Results Summary

### UAD (Unsupervised Absorption Detection)
- **F1**: 0.725 (full, 500 features) vs 0.522 (pilot, 100 features) -- exceeds 0.6 threshold by +0.104
- **Precision**: 0.569 (Wilson 95% CI: [0.43, 0.70])
- **Recall**: 1.000 (29/29 supervised collisions found, zero false negatives)
- **Multi-seed consistency**: std = 0.000 across seeds 42, 123, 456
- **Cross-layer**: Mean F1 = 0.561 (layers 4, 8, 10); layer 4 drops to 0.432
- **Runtime**: 7.6s per run

### DFDA (Dynamic Feature De-Absorption)
- **Pilot (4 pairs)**: 11.14% mean improvement, 3/4 positive, 388 parameters (0.004% of SAE)
- **Scaling (8 pairs)**: 99.5% mean improvement, 8/8 positive, 1,544 parameters
- **Critical caveat**: Metric is artifactual -- MLP learns to predict near-zero parent values in child-dominant positions

### Original Primary Hypotheses (H1-H4: CAAB, causal, sparsity, layer)
- All |r| < 0.11, p > 0.86 -- definitively noise
- Collision rate != absorption; proxy metric conflation exposed

---

## Debate Perspectives Summary

- **Optimist**: UAD F1=0.725 with perfect recall is a genuinely strong signal. DFDA 8/8 positive suggests underlying mechanism is real. Scaling trend (0.522 -> 0.704) is positive. Super-absorber finding (feature 18486 absorbing 5 letters) is unexpected bonus.

- **Skeptic**: UAD's "ground truth" is itself an unvalidated heuristic (first-letter collisions, not Chanin-ablation-confirmed absorption). F1 measures agreement between two heuristics, not detection of true absorption. DFDA's 99.5% is a mathematical tautology (predict near-zero on near-zero targets), not evidence of recovery. Zero baselines executed. H1-H4 are dead ends.

- **Strategist**: 2 strong signals (UAD detection, DFDA compensation) vs 4 noise channels (H1-H4). Cross-model validation is highest-leverage next step. If F1 >= 0.55 on Gemma-2B and Pythia-2.8B, contribution is solidified. Resource estimate: ~2.5 GPU-hours for workshop-ready paper.

- **Comparativist**: UAD is genuinely novel -- no prior work eliminates supervision requirement for absorption detection. DFDA is qualitatively different (only inference-time, training-free compensation) but metrically compromised. Venue: NeurIPS/ICLR Workshop now, main conference after validation.

- **Methodologist**: Reproducibility score 2/5. No random baselines, no ablations, no bootstrap CIs. DFDA metric-claim alignment is "Poor." Train-test overlap (UAD pairs used for DFDA). Single model, single layer, single concept domain. Perfect seed consistency reflects determinism, not robustness.

- **Revisionist**: Original mental model conflated collision rate (polysemanticity) with absorption (hierarchical suppression). Contribution hierarchy must be inverted: UAD/DFDA (exploratory) outperform CAAB/causal (primary). DFDA metric was measuring the wrong thing entirely.

---

## Analysis

### 1. Method Feasibility

**UAD: PARTIALLY VALIDATED**
- The co-occurrence clustering mechanism works -- it finds feature pairs that match the collision heuristic with F1=0.725.
- However, whether it detects "absorption" specifically or merely "correlation" is unvalidated. Chanin-style ablation confirmation was not performed.
- The cluster-based approach naturally captures multi-parent absorption (super-absorbers), which supervised methods cannot.
- No ablations were executed -- cannot claim HAC, phi coefficient, or 500-feature window are necessary design choices.

**DFDA: CONCEPTUALLY SOUND, EMPIRICALLY INVALIDATED**
- The residual compensation architecture is reasonable in principle.
- However, the evaluation metric is tautological: the MLP learns the training distribution mean (near-zero), achieving massive percentage improvement on near-zero targets.
- This does NOT demonstrate recovery of absorbed parent activations on examples where the parent SHOULD fire.

### 2. Performance

**UAD**: F1=0.725 exceeds the 0.6 threshold. Perfect recall (1.0) is a critical property -- no missed absorbed pairs. However, precision=0.569 means 43% false positives. The signal is real but noisy.

**DFDA**: The 99.5% figure must be discarded entirely. The true test (recovery on parent-positive examples) was not performed.

**H1-H4**: All noise. |r| < 0.11 across cross-architecture variation, causal links, sparsity trends, and layer patterns.

### 3. Improvement Headroom

**UAD**: Clear improvement paths exist:
- Post-hoc filtering could improve precision from 0.54 to >0.7 without sacrificing recall
- Cross-model validation (Gemma-2B, Pythia-2.8B) could solidify generalization claims
- Chanin validation would anchor the absorption claim
- Semantic hierarchy validation (WordNet) would broaden applicability

**DFDA**: Requires fundamental metric rebuild before any claim can be made. The current evaluation protocol must be replaced with parent-positive recovery measurement.

### 4. Time-Cost Tradeoff

- **UAD validation**: ~1.5 GPU-hours (Chanin validation + cross-model + random baseline)
- **DFDA metric rebuild**: ~0.5 GPU-hours (new evaluation protocol + parent-positive test)
- **Total to workshop-ready**: ~2.5 GPU-hours + ~2 hours writing
- **Pivot cost**: Would discard a genuinely novel contribution (UAD) for alternatives that are either less novel (co-occurrence toolkit) or require new infrastructure (vision)

The sunk cost of H1-H4 should not influence the decision. Those hypotheses are noise and should be dropped regardless.

### 5. Critical Objections

**Skeptic's concerns**:
1. Ground truth is unvalidated -- VALID. UAD's F1 measures heuristic agreement, not ground-truth detection.
2. DFDA metric is tautological -- VALID. The 99.5% figure is mathematically meaningless.
3. Zero baselines -- VALID. Cannot claim F1=0.7 is meaningful without random baseline.

**However**: None of these are fatal to the project. They are fatal to specific claims (DFDA effectiveness, UAD accuracy magnitude) but not to the core contribution (UAD as first unsupervised detection method). The project can be reframed honestly.

---

## Decision Rationale

The project has one genuinely novel contribution: **UAD is the first method to detect feature absorption-like relationships in SAEs without ground-truth parent features or supervised probe directions.** This novelty holds regardless of the exact F1 magnitude or ground truth validation status.

The empirical support is weaker than initially assessed:
- UAD's F1 is unanchored (no random baseline, unvalidated ground truth)
- DFDA's metric is fatally flawed (tautological, not merely artifactual)
- H1-H4 are dead ends built on proxy metric conflation

However, the path to a publishable contribution is clear:
1. Chanin validation anchors the absorption claim
2. Cross-model validation solidifies generalization
3. DFDA metric rebuild enables honest compensation claim
4. Honest reframing drops H1-H4, leads with UAD

Pivoting would discard a genuinely novel idea for alternatives that are either less novel (co-occurrence toolkit) or contradicted by data (nonidentifiability -- seed variance is 0, not high). The minimum viable critique paper is a fallback, not a pivot.

The synthesis verdict is correct: **PROCEED with UAD as sole primary contribution, HALT DFDA until metric rebuilt, execute Chanin validation as highest-priority next step.**

---

## DECISION: PROCEED
