# Planning Critique — SAE Absorption Paper

## Plan Execution Assessment

### Successful Pivots

The iter_002 methodology was a well-executed pivot from the overly ambitious iter_001 proposal. Key successful decisions:

1. **ASI → EDA**: The ASI metric (AUROC=0.476) was correctly abandoned after pilot falsification. EDA (AUROC=0.681 pilot → 0.650 full) became the primary contribution. This pivot was appropriate and properly documented.

2. **GPT-2 Small primary, Gemma as aspirational**: The decision to use GPT-2 Small with SAELens pre-trained SAEs was correct given resource constraints. Gemma Scope is mentioned throughout the paper as "required future work" without pretending Gemma results exist.

3. **Pre-Writing Audit Gate**: The audit_report.json system that blocked writing until critical issues were resolved is good process design. However, three critical flags were still present at writing time (per audit: `writing_blocked: true`, 3 CRITICAL flags), and writing proceeded anyway. This bypassed the gate.

### Failed Gates

**Critical: Pre-writing audit was bypassed.** The audit_report.json shows `writing_blocked: true` with 3 CRITICAL flags at the time writing was initiated:
- B1_7_direction_check (H1 falsified)
- C1_4_city_country_shuffle (city_country excluded)
- D2_10_asi_fails (H3 falsified, ASI fails)

The writing agent correctly incorporated the H1 and H3 falsifications into the paper. However, the four numerical inconsistencies in the paper (AUROC=0.206, absorption range, Spearman rho, missing abstract) suggest that the writing agent had access to partial or stale experimental data rather than the canonical JSON files.

**Root cause analysis:** The paper's Introduction and Conclusion cite numbers (0.919–0.968, ρ=0.191) that appear in the B2 scaling curve pilot analysis but not in the full E1 phase transition analysis. The writing agent may have used the methodology.md expected values rather than the final experiment JSONs for some claims.

### Scope Creep Not Delivered

The original proposal promised contributions that were not delivered and some of these gaps are not clearly acknowledged in the paper:

| Original Promise | Status in Paper |
|---|---|
| Cross-domain absorption on RAVEL (entity-type) | Not attempted — GPT-2 Small fallback |
| Gemma Scope validation | Not attempted |
| Absorption Susceptibility Index (ASI) | Failed (AUROC=0.476), reported honestly |
| Phase transition with hysteresis | Not testable due to saturation, reported honestly |
| Absorption Impossibility Theorem | Downgraded to Proposition 2 Depth Bound, not proven |
| F2 Architectural Mitigation Empirical | Skipped, no alternative SAE available |

The paper handles most of these well by either reporting falsification honestly or noting them as future work. The scope is accurately presented as "GPT-2 Small, first-letter hierarchy, L1-penalized SAEs." The issue is in the Introduction and Related Work, which occasionally use language from the broader original scope.

### Time Budget

The methodology specifies ~8–10 GPU-hours total. From the DONE task files, the actual experiments completed:
- B1, B2, B3 (decoder geometry, scaling, cross-arch)
- C1, C2-REDESIGN, C3 (probe training, cross-domain redesign, hierarchy correlation)
- D1, D2, D3 (EDA validation, variants, cross-domain)
- E1, E2, E3 (phase transition, hysteresis, width analysis)
- F1 (theory revision), F2 (mitigation, skip)
- STAT_audit

Total tasks: ~16 major tasks, consistent with the ~8–10 GPU-hour budget. No GPU waste on failed directions — the pivot was efficient.

### What Should Be Planned Differently Next Iteration

1. **Ground truth labels first**: The FeatureAbsorptionCalculator should be run at full scale (all 26 letters) before any detection experiment. Having n_pos=18 as the evaluation set for a model with 24,576 features is the single largest limitation of the entire study.

2. **Matched architecture comparisons**: Any cross-architecture or cross-training-regime comparison (jb vs. AJT, L1 vs. TopK) must use the same hook point and matched L0. The B3 comparison was confounded from the start.

3. **Cross-domain at larger scale from the beginning**: If semantic hierarchies require n_words >= 500 for detection power, plan for this in the experimental design. The C2 redesign with n=20 words was doomed to be underpowered. Either test on Gemma Scope (richer semantic hierarchies) or increase n_words by 25x.

4. **Pre-writing consistency gate**: Add an automatic cross-check that compares Abstract, Introduction, and Conclusion claim values against the canonical JSON files before any writing agent runs. The numerical inconsistencies in this paper (AUROC phantom, absorption range, Spearman rho) are all detectable by a simple script that parses these sections.
