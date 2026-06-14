# Section Critique: Related Work (Revision Round 2)

**Section**: Related Work (Section 2)
**Reviewer**: Sibyl Section Critic
**Date**: 2026-03-18
**Basis**: related_work.md + final_review.md (score 5.5/10)
**Previous round score**: 6.5/10

---

## Overall Assessment: 6.5 / 10

The Related Work section is well-organized along the four Phi modulation axes and effectively positions the paper against key concurrent works. However, several issues from round 1 remain unresolved, and the final review exposes additional vulnerabilities---particularly around novelty overlap, the BN confound, and overclaiming about untested regime boundaries.

---

## Strengths

1. **Clean organizational structure.** The four-axis decomposition (temporal, directional, spatial, target-norm) mirrors the Phi framework and makes the taxonomy immediately useful to readers.

2. **Honest positioning statement.** The closing paragraph of Section 2.2 ("we do not propose a new dynamic weight decay method") is clear and sets appropriate expectations.

3. **Section 2.4 direct comparisons.** The explicit "vs." format against Wang & Aitchison, D'Angelo et al., and Chou is effective and transparent.

4. **Evaluation fragmentation argument (Section 2.3).** The observation that no two papers share identical experimental conditions directly motivates the paper's contribution.

---

## Critical Issues

### RW-C1: Defazio (2025) is cited in Sections 1.3 and 4 but completely absent from Related Work [PERSISTS FROM ROUND 1]

Defazio (2025) derives R_* = lambda/eta for normalized layers---the exact rho quantity that Theorem 1 calls "Dual Characterization." A reviewer familiar with Defazio will immediately flag this as an uncredited related result. The paper cannot claim rho as a novel construct when Defazio independently derives the same quantity.

**Fix**: Add 3-4 sentences in Section 2.1 (after the Xie & Li paragraph): (a) Defazio derives R_* = lambda/eta steady-state for normalized layers; (b) his focus is end-of-training instability correction via SGDC/AdamC; (c) this paper's novelty is the Trichotomy regime structure and cross-optimizer comparison, not the rho quantity itself. Update Theorem 1 attribution accordingly.

### RW-C2: BN confound not acknowledged, NoBN ablation reference is problematic [NEW, from final review M9]

The paper states in Section 2.4 (vs. D'Angelo): "The NoBN ablation experiment (Section 6) directly distinguishes these mechanisms." However, the final review (M9) identifies that this NoBN ablation does not exist in the current experimental results. Citing a nonexistent experiment as evidence is a serious scholarly issue.

D'Angelo et al.'s BN scale-invariance argument provides an alternative explanation for the paper's entire AdamW invariance finding. The Related Work must either:
- (a) Remove the NoBN reference until the experiment is completed, and explicitly acknowledge the BN confound as an open question
- (b) If the NoBN experiment will be added before submission, keep the reference but note it as "planned" rather than completed

### RW-C3: Understates novelty overlap with prior work [NEW, from final review Novelty weakness]

The section acknowledges related work but systematically understates how much the core findings are anticipated:
- **Xie & Li (2024)**: The l-infinity constraint result directly predicts AdamW insensitivity to small WD perturbations. A knowledgeable reviewer could predict the paper's main finding without running experiments.
- **Wang & Aitchison (2024)**: The rho ~ 1/EMA-timescale connection is buried as an aside ("suggesting the two perspectives may unify"). This is actually critical: the rho quantity is not new.
- **Defazio (2025)**: Derives the same steady-state ratio (see RW-C1).

**Fix**: Add a paragraph explicitly acknowledging that individual observations (AdamW insensitivity, the rho ratio) are anticipated by prior work. Clarify that novelty lies in: (a) the controlled multi-method comparison under identical conditions, (b) the Phi framework as a unifying abstraction, (c) the systematic budget-normalized evaluation (BEM).

---

## Major Issues

### RW-M1: Section 2.4 Chou comparison makes untested regime claims [NEW, from final review M8, M13]

The text states: "when rho remains in the low regime throughout training, the particular functional form of the schedule is immaterial, and their proposed schedule is one of many equivalent choices." This assumes the regime boundary is established, but the final review (M8) identifies that the regime boundary has zero empirical support beyond rho=0.5. The Trichotomy is currently tested at exactly one operating point.

**Fix**: Soften to: "our preliminary analysis at rho=0.5 suggests that..." or "if the Phi Invariance Conjecture holds in the low-rho regime, then..."

### RW-M2: Two high-priority papers from novelty analysis are absent [PERSISTS FROM ROUND 1]

arXiv:2510.19093 ("Weight Decay May Matter More Than muP for LR Transfer") and arXiv:2510.15262 ("Robust Layerwise Scaling Rules by Proper Weight Decay Tuning") are missing. Both are directly relevant: the former provides independent support for rho as a fundamental control quantity; the latter addresses the spatial modulation axis.

**Fix**: Add 1-2 sentences each in appropriate subsections.

### RW-M3: "Case for Null Results" promised in title but not delivered in body [PERSISTS FROM ROUND 1]

Section 2.3 discusses evaluation fragmentation but never develops the null-results argument. Given that the paper's central finding is a null result (seven strategies are statistically indistinguishable), this is a missed contribution-narrative opportunity.

**Fix**: Either (a) add 2-3 sentences on publication bias against null results and the scientific value of controlled negative findings, or (b) rename to "Evaluation Fragmentation" to match actual content.

### RW-M4: ADANA, AdamO, SPD appear in Section 2.2 but have no Phi formulations in Table 1 [PERSISTS FROM ROUND 1]

These methods are described as dynamic WD approaches but not expressed as Phi modulators. This creates an inconsistency: if the framework cannot express them, its claimed universality is overstated; if it can but they were excluded, the catalog is incomplete.

**Fix**: Add Phi forms to Table 1, or add explicit exclusion rationale in Section 2.2 (e.g., "SPD is designed for fine-tuning and requires adaptation of budget-equivalence normalization").

### RW-M5: Evaluation fragmentation critique exposes the paper's own scope limitation [PERSISTS FROM ROUND 1]

Section 2.3 criticizes prior work for narrow evaluation, but this paper's own experiments are CIFAR-only with ResNet-20. The final review (M6, M10) identifies this as a critical weakness. A reviewer who reads the fragmentation critique before seeing CIFAR-only results will notice the irony.

**Fix**: Add one honest sentence acknowledging that the present paper's validation is also currently limited in scale, and that large-scale replication is identified as future work.

---

## Minor Issues

### RW-m1: Incomplete reference details [from final review m6]

Multiple citations lack venues or arXiv IDs: Chou (2025), Defazio (2025), He et al. (2025), Kosson et al. (2023), Loshchilov (2023), Tian et al. (2024), Ferbach et al. (2026), Chen et al. (2026a, 2026b). All references must have complete bibliographic information for NeurIPS/ICML submission.

### RW-m2: Wang & Aitchison comparison buries a key insight

"Notably, their EMA timescale ~ 1/rho in our notation, suggesting the two perspectives may unify" is treated as an afterthought. If this unification is genuine, it significantly strengthens the framework and deserves more prominent discussion.

### RW-m3: Kosson et al. rotational equilibrium not connected to Regime I

The paper describes Kosson's rotational equilibrium mechanism but does not connect it to the Phi Invariance Trichotomy. Adding one sentence ("Kosson et al.'s rotational equilibrium may correspond to Regime I of our Trichotomy") would strengthen the theoretical narrative.

### RW-m4: Sun et al. (CVPR 2025) cited in Section 4 but absent from Related Work

Sun et al.'s alignment-optimal schedule analysis (used in Proposition 2) should appear in Section 2.2 under directional modulation.

### RW-m5: D'Angelo "never" claim is too strong

"weight decay is *never* useful as explicit regularization in modern settings" should be scoped: "in modern BN-equipped architectures under standard training regimes."

### RW-m6: Terminology inconsistency with Section 4

Section 2.4 describes the mechanism as "AdamW's sign normalization rendering time-varying modulation irrelevant," but Section 4 uses "l-infinity constraint" (from Xie & Li). These are compatible but use different vocabulary. Standardize on "l-infinity constraint" throughout.

### RW-m7: Ferbach 2026 / Chen 2026a/b dating

Papers dated 2026 in a 2026 submission should have preprint status annotated (e.g., "arXiv preprint, 2025") to clarify they are not concurrent submissions.

---

## Cross-Reference to Final Review Issues

| Final Review Issue | Related Work Impact | Status |
|-------------------|---------------------|--------|
| M8 (Regime boundary untested) | RW-M1: Chou comparison assumes established boundary | **Unresolved** |
| M9 (BN confound) | RW-C2: NoBN ablation reference is problematic | **Unresolved** |
| M12 (Overclaiming) | RW-M1: regime claims; RW-C3: novelty overlap | **Unresolved** |
| M13 (Regime presented as established) | RW-M1: Chou comparison | **Unresolved** |
| m6 (Incomplete refs) | RW-m1: multiple refs incomplete | **Unresolved** |
| Novelty concern | RW-C3: overlap with Xie & Li, Wang & Aitchison, Defazio | **Unresolved** |

---

## Actionable Revision Checklist (Priority Order)

| ID | Priority | Action |
|----|----------|--------|
| RW-C1 | **Critical** | Add Defazio (2025) to Section 2.1 with proper positioning |
| RW-C2 | **Critical** | Remove/qualify NoBN ablation reference; acknowledge BN confound |
| RW-C3 | **High** | Add paragraph acknowledging novelty overlap; clarify what is genuinely new |
| RW-M1 | **High** | Soften Chou comparison to reflect untested regime boundary |
| RW-M2 | **High** | Add arXiv:2510.19093 and arXiv:2510.15262 |
| RW-M3 | **Medium** | Develop null-results framing or rename Section 2.3 |
| RW-M4 | **Medium** | Resolve Table 1 completeness for ADANA/AdamO/SPD |
| RW-M5 | **Medium** | Acknowledge own scope limitation in Section 2.3 |
| RW-m1 | **Medium** | Complete all reference entries |
| RW-m2 | **Medium** | Elevate rho ~ 1/EMA-timescale connection |
| RW-m3 | **Low** | Connect Kosson to Regime I |
| RW-m4 | **Low** | Add Sun et al. to Section 2.2 |
| RW-m5 | **Low** | Scope D'Angelo "never" claim |
| RW-m6 | **Low** | Standardize terminology (l-infinity constraint) |
| RW-m7 | **Low** | Annotate 2026-dated papers with preprint status |

---

*Reviewed by: Sibyl Section Critic*
*Standard: NeurIPS/ICML main track*
*Revision round: 2*
