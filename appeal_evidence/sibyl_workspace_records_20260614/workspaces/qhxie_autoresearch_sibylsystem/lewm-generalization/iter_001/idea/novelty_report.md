# Novelty Report: LeWM Compositional Generalization Research

**Date:** 2026-04-14
**Checker:** sibyl-novelty-checker
**Search rounds:** 12 (arXiv, Google Scholar via WebSearch, WebFetch)

---

## Overall Verdict: HIGH NOVELTY

All four candidates are defensible with novelty scores 7–9. The front-runner (cand_a) achieves score **8** at the combination level: individual components have partial precedents, but the specific integrated proposal has no direct prior art.

---

## Search Coverage

| Query | Key Papers Found |
|-------|-----------------|
| JEPA world model compositional generalization physical parameter holdout | SVIB (NeurIPS 2023), What Drives Success in JEPA-WM (arXiv:2512.24497) |
| LeWM/SIGReg background | LeWorldModel (arXiv:2603.19312), Balestriero & LeCun 2025 (LeJEPA) |
| Uselis geometric conditions | arXiv:2602.24264, arXiv:2507.07102 |
| Interventional probing latent space | Causal-JEPA (arXiv:2602.11389), Learning Robust Intervention (ICLR 2026, arXiv:2508.04492), PIWM (arXiv:2412.12870) |
| Group-structured latent prior | Delliaux et al. (arXiv:2506.01529), Schwarcz 2026 (arXiv:2603.27134) |
| LoRA encoder-predictor world model | V-JEPA 2-AC two-phase training (no specific LoRA+JEPA paper found) |
| Physical coupling irreducibility | No direct hit; closest: Four Principles (arXiv:2503.02143), PNAS 2024 dynamical subprocesses |
| Data diversity and compositionality | Uselis/Redhardt ICML/NeurIPS 2025 |
| Probing dissociation | Liang et al. arXiv:2501.18797, Uselis 2026 |

---

## Per-Candidate Analysis

---

### Candidate A — Front-Runner
**Title:** Geometric Factorization Meets Physical Reality: A Systematic Compositional Generalization Benchmark for JEPA World Models
**Novelty Score: 8/10**
**Recommendation: PROCEED**

#### Core Contribution Claims
1. First CoGenT-style physics-parameter holdout benchmark for a JEPA world model (LeWM)
2. First empirical test of whether SIGReg's isotropic Gaussian prior promotes or competes with the Uselis et al. geometric conditions for compositional generalization
3. Interventional Independence Score (IIS) — novel metric for causal independence of physical concept representations in world model latent spaces
4. LoRA as encoder-predictor bottleneck diagnostic — first application to JEPA world models
5. Regime-aware evaluation: distinguish model-fixable from physically irreducible compositional failures

#### Prior Art Search Findings

**Most relevant precedent (partial overlap):**

**SVIB — Systematic Visual Imagination Benchmark** (Kim et al., NeurIPS 2023, arXiv:2311.09064):
- What it does: Introduces a CoGenT-style benchmark for systematic generalization in visual world models; holdout combinations of visual factors (shape, color, texture) in dSprites/CLEVR/CLEVRTex.
- Overlap: CoGenT-style factor holdout evaluation philosophy for world models.
- *Why it does NOT preempt cand_a:* SVIB tests visual image-to-image transformation rules, not physics simulation parameters. It does not evaluate physical dynamics (friction, gravity, mass) or JEPA architectures, and it provides no geometric analysis, IIS, LoRA diagnostics, or coupling regime classification.

**Uselis et al. 2026 (arXiv:2602.24264) — Compositional Generalization Requires Linear, Orthogonal Representations:**
- What it does: Proves geometric conditions for compositional generalization; provides principal angle analysis codebase; tests CLIP, DINO, SigLIP on visual concept holdouts.
- Overlap: Geometric analysis methodology we plan to apply.
- *Why it does NOT preempt:* Tests static vision encoders only. Does not address world models with temporal predictors, SIGReg vs. VICReg, IIS, LoRA adaptation, or physical coupling analysis. We explicitly build on their codebase and extend to the world model setting.

**Uselis/Redhardt 2025 (arXiv:2507.07102, arXiv:2507.07207) — Data Diversity Drives Compositional Generalization:**
- Overlap: Confirms that data combinatorial diversity (not scale) drives compositional generalization. Supports cand_a's training protocol design.
- *Why it does NOT preempt:* No world model evaluation, no JEPA, no physics parameters, no interventional probing.

**Causal-JEPA (Nam et al., arXiv:2602.11389, ICLR 2026):**
- What it does: Introduces object-level masking in a JEPA world model to induce causal inductive bias; demonstrates improved counterfactual reasoning (+20%) on CLEVRER and Push-T.
- Overlap: Involves latent interventions and JEPA world model — superficially similar to IIS contribution.
- *Why it does NOT preempt:* Causal-JEPA uses interventions as a *training objective* (masking), not as a *test-time diagnostic metric* for causal independence between physical concepts. No physics-parameter holdout benchmark, no SIGReg/VICReg comparison, no LoRA, no coupling regime analysis. The IIS metric (test-time interventional probing) is methodologically distinct from Causal-JEPA's training approach.

**Learning Robust Intervention Representations (ICLR 2026, arXiv:2508.04492):**
- What it does: Introduces Causal Delta Embedding (CDE) — delta vectors between pre/post-intervention latent states for OOD-robust action representation in object-centric settings.
- Overlap: Delta-vector representation of interventions in latent space; independence property.
- *Why it does NOT preempt:* CDE is for representing *actions* (not physical concept independence), targets OOD action classification in object-centric domains, does not evaluate JEPA world models or physical parameter holdouts, and does not propose a diagnostic metric like IIS.

**PIWM — Physically Interpretable World Models (arXiv:2412.12870):**
- What it does: Aligns latent dimensions with physical quantities (position, velocity) using weak supervision; tests on Cart Pole, Lunar Lander, Donkey Car.
- Overlap: Physical interpretability of world model latent spaces.
- *Why it does NOT preempt:* Uses explicit weak supervision to align latents (supervised setting); does not test compositional generalization across held-out physics combinations; does not evaluate JEPA-style models or SIGReg.

**DINO-WM (arXiv:2411.04983, ICML 2025):**
- Role: Key baseline. Generalizes to new visual configurations (maze walls, object shapes), not physics parameter combinations.
- Classification: related_work (baseline).

**What drives success in JEPA-WM (arXiv:2512.24497):**
- What it does: Systematic study of architecture, training objective, and planning algorithm choices in JEPA-WMs.
- Overlap: Studies JEPA-WM design; mentions DINO-WM and V-JEPA-2-AC.
- *Why it does NOT preempt:* Does not test compositional generalization over physics holdouts; no geometric/interventional analysis; no SIGReg/VICReg comparison.

#### Summary Gap Confirmation
The specific combination of LeWM + physics-parameter CoGenT holdout + SIGReg-orthogonality analysis + IIS metric + LoRA bottleneck diagnostic + regime-aware evaluation is absent from the literature. **Gap confirmed on all five contribution dimensions.**

---

### Candidate B — Backup Pivot
**Title:** Group-Structured vs. Isotropic Latent Priors for Compositional Physical Generalization
**Novelty Score: 8/10**
**Recommendation: PROCEED AS PIVOT**

#### Prior Art Search Findings

**Delliaux et al., 2025 (arXiv:2506.01529) — Learning Abstract World Models with a Group-Structured Latent Space:**
- What it does: Implements group-structured latent space (SO(2), translations) for MDP world models; improves RL performance and latent prediction quality in environments with rotational/translational symmetry.
- Overlap: This is precisely the method cand_b would compare against SIGReg.
- *Severity: partial_overlap.* Delliaux et al. do not compare to SIGReg, do not use CoGenT-style physics holdouts, and do not measure principal angle orthogonality or IIS. The comparison on ComPhys-LeWM is novel.

**Schwarcz 2026 (arXiv:2603.27134) — Semantic Interaction Information (RCC architectures):**
- Overlap: Alternative approach to compositional generalization in latent spaces via variational inference.
- Classification: related_work.

#### Gap Confirmation
No paper has compared group-structured vs. isotropic Gaussian latent priors on a physics compositional generalization benchmark. **Gap confirmed.**

---

### Candidate C — Backup Pivot
**Title:** Physics-Informed Fair Evaluation of World Model Compositional Generalization
**Novelty Score: 9/10**
**Recommendation: PROCEED AS PIVOT**

#### Prior Art Search Findings

**Four Principles for Physically Interpretable World Models (arXiv:2503.02143, 2025):**
- Overlap: Advocates for physics-aligned world model design; discusses OOD generalization improvements from physical principles.
- *Why it does NOT preempt:* Model design paper, not evaluation framework. Does not distinguish model-fixable from physically irreducible failures. No coupling analysis.

**Decomposing dynamical subprocesses for compositional generalization (PNAS 2024):**
- Overlap: Studies compositionality of dynamical subprocesses in human cognition.
- Classification: related_work — neuroscience/cognitive science context.

**Bifurcation/Phase transition literature:**
- No paper applies bifurcation theory to classify world model compositional generalization failures as physically irreducible.

#### Gap Confirmation
No prior work distinguishes "model-fixable" from "physically irreducible" compositional generalization failures using physical coupling analysis. This framework (coupling strength as regime classifier, tested across multiple model architectures) is entirely absent from the literature. **Strong gap confirmed.**

---

### Candidate D — Embedded Contribution
**Title:** The Probing Illusion: Geometric Diagnostics for Compositional Generalization in JEPA World Models
**Novelty Score: 7/10**
**Recommendation: PROCEED AS EMBEDDED (within cand_a, not standalone)**

#### Prior Art Search Findings

**Liang et al., 2025 (arXiv:2501.18797) — Compositional Generalization Requires More Than Disentangled Representations:**
- What it does: Shows that disentangled representations are insufficient for compositional generalization; argues against using probing accuracy as a proxy.
- Overlap: Conceptual argument that probing accuracy ≠ compositional generalization.
- *Severity: partial_overlap.* The specific quantification of probing-generalization dissociation in a JEPA world model (r < 0.3 between in-distribution and holdout probing) has not been done. But the conceptual claim is established.

**Uselis et al., 2026 (arXiv:2602.24264):**
- Overlap: Advocates for geometric metrics over probing; shows partial linear factorization in CLIP/DINO correlates better with compositional generalization than in-distribution accuracy.
- *Severity: partial_overlap.* Static encoders only; JEPA temporal dynamics not studied.

#### Assessment
The probing-generalization dissociation for JEPA world models is a novel empirical finding, but conceptually anticipated. Best positioned as a section within the multi-level evaluation of cand_a (demonstrating that Level 1 probing is insufficient while Levels 2-3 are predictive) rather than a standalone paper.

---

## Key Prior Art Summary

| Paper | ArXiv/DOI | Relevance | Severity |
|-------|-----------|-----------|----------|
| SVIB (Kim et al., NeurIPS 2023) | arXiv:2311.09064 | CoGenT-style world model benchmark | partial_overlap |
| Uselis et al. 2026 | arXiv:2602.24264 | Geometric conditions for CG | partial_overlap |
| Uselis et al. ICML 2025 | arXiv:2507.07102 | Data diversity drives CG | related_work |
| Redhardt et al. NeurIPS 2025 | arXiv:2507.07207 | Scaling and CG | related_work |
| Causal-JEPA (Nam et al. 2026) | arXiv:2602.11389 | Latent interventions in JEPA | partial_overlap |
| Learning Robust Intervention (ICLR 2026) | arXiv:2508.04492 | Delta-vector intervention representations | partial_overlap |
| Delliaux et al. 2025 | arXiv:2506.01529 | Group-structured latent space world model | partial_overlap |
| DINO-WM (Zhou et al. ICML 2025) | arXiv:2411.04983 | JEPA-WM baseline | related_work |
| PIWM (Mao et al. 2024) | arXiv:2412.12870 | Physical alignment in world model latents | related_work |
| Liang et al. 2025 | arXiv:2501.18797 | Probing insufficient for CG | partial_overlap |
| Schwarcz 2026 | arXiv:2603.27134 | RCC architectures | related_work |
| Decomposing dynamical subprocesses (PNAS 2024) | doi:10.1073/pnas.2408134121 | CG of dynamical processes (cognitive science) | related_work |
| Four Principles Physically Interpretable WM (2025) | arXiv:2503.02143 | Physical OOD generalization | related_work |

---

## Recommendations

**Cand_a (PRIMARY — PROCEED):**
All five core contributions are confirmed novel. The most important differentiation to emphasize in the paper:
- "First evaluation of compositional generalization in a JEPA world model over physics parameter holdouts (not visual attributes, not static encoders)"
- "First empirical test of the SIGReg-Uselis geometric condition connection"
- "IIS metric is distinct from Causal-JEPA's training-time masking: IIS is a post-hoc diagnostic that measures causal independence without modifying the training procedure"

**Cand_b (PROCEED AS PIVOT):** Activate if H2 is falsified or to extend cand_a with explicit group-structured prior comparison.

**Cand_c (PROCEED AS PIVOT):** Activate if H1 is falsified (LeWM generalizes perfectly) — the physics regime framework is the most novel single contribution and deserves standalone treatment in that case.

**Cand_d (EMBED IN CAND_A):** Do not pursue as standalone; incorporate probing-generalization dissociation as part of multi-level evaluation comparison in cand_a.

---

## Anti-Patterns Checked

- No rubber-stamping: Each claim was searched with at least 3 targeted queries.
- SVIB was carefully distinguished: it is the closest conceptual predecessor but covers a substantially different domain (visual attributes, not physics dynamics; toy environments, not DMControl; non-JEPA baselines).
- Causal-JEPA was carefully distinguished: training-time causal inductive bias vs. test-time causal independence diagnostic are different contributions.
- Liang et al. 2025 partial overlap on cand_d is accurately classified as partial_overlap, not dismissed as "unrelated."

---

*Generated by sibyl-novelty-checker | 2026-04-14*
