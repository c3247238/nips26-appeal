# Experiment Result Analysis

## Key Results Summary

All 12 experiments across pilot and full phases completed successfully with zero failures.

### Core Negative Result: UAD Failure
- **F1 = 0.00048** (1 true positive out of 4,155 detected pairs, 6 false negatives out of 7 ground truth pairs)
- **Precision = 0.024%**, **Recall = 14.3%**
- UAD detects exactly the same number of true positives as randomly sampling 4,155 pairs from within clusters
- All of UAD's complexity (phi filtering, dead feature filtering, hierarchical clustering, specificity checks) provides zero improvement over trivial random baseline
- Bootstrap 95% CI for F1: [0.00012, 0.00102]

### Ablation Results
| Variant | Detected Pairs | TP | Precision | Recall | F1 |
|---------|---------------|----|-----------|--------|-----|
| Full UAD | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No dead filter | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No phi filter | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No clustering | 106,864 | 3 | 0.003% | 42.9% | 0.00006 |
| Single linkage | 102,832 | 0 | 0.0% | 0.0% | 0.0 |
| K-means | 3,243 | 6 | 0.185% | 85.7% | 0.0037 |

### Collision Rate Internal Consistency
- **Spearman r = 0.869** across 56 pairs (numbers + punctuation hierarchies)
- 95% CI: [0.780, 0.938]
- Numbers only: r = 0.598
- Punctuation only: r = 0.693
- Pilot (first letters, n=10): r = 0.711

### Root Cause Evidence
Token-level activations show complete mutual exclusivity: feature 11513 fires only on "three" (29.4), feature 12413 fires only on "two" (15.3) and "one" (15.3), feature 22971 fires only on "two" (24.2), feature 24189 fires on "four" through "eight" (14.3-18.9). No two absorption features ever activate on the same token.

---

## Debate Perspectives Summary

- **Optimist**: All 12 reviewer issues have concrete fixes; most are writing-only. Expected score improvement from 6.0 to 8.3. Core finding (F1 = 0.00048) is robust and impressive. Token-level mutual exclusivity explanation is elegant.

- **Skeptic**: Even with all fixes, realistic expected score is 7.0-7.5 (not 8.3). Structural limits (n=7 GT, circular definition residue) cap the ceiling. Collision rate reframe carries reviewer acceptance risk. K-means 85.7% recall is a double-edged sword. Negative result acceptance at main track is uncertain.

- **Strategist**: Highest ROI path is Tier 1 writing fixes (1 hour, +1.3 points), then Tier 2 statistical improvements (1 hour, +0.8 points), then Tier 3 polish (1 hour, +0.4 points). Total budget: 3-4 hours. Stop at 4 hours regardless.

- **Comparativist**: Iteration 3 is at ICBINB average quality (6.5). Iteration 4 targets ICBINB excellent (7.5-8.0). NeurIPS main track (8.5+) is out of reach due to structural limitations. ICBINB is the correct target venue.

- **Methodologist**: All improvements must be verifiable. Tier 1 fixes (delete fabricated claims, fix data mismatch, soften universal claims) are fully verifiable via text search. Bootstrap CI and figure generation are verifiable via file checks. Collision rate reframe and K-means analysis require careful execution.

- **Revisionist**: Key cognitive updates: (1) "proxy validation" framework was epistemically wrong, reframed as "operationalization consistency"; (2) K-means difference is systematic, not noise; (3) negative result ceiling is ~8.0, not NeurIPS main track; (4) honesty matters more than perfection for negative-result papers.

---

## Analysis

### 1. Method Feasibility
The core method (UAD co-occurrence clustering) works as implemented but fails structurally for its intended purpose. All 12 experiments completed without technical errors. The UAD pipeline executes correctly---it simply produces the wrong outputs because the underlying assumption (absorbed features co-occur) is false for token-disjoint hierarchies. The method is not broken; the theory behind it is mismatched to the phenomenon.

### 2. Performance
Results do not outperform baselines---they are the baseline. UAD achieves F1 = 0.00048, identical to random sampling. This is the intended finding: the paper's contribution is demonstrating that a proposed method fails, not that a new method succeeds. The negative result is the result.

The collision rate operationalization shows strong internal consistency (r = 0.869, n = 56), which is a positive secondary finding. However, this is explicitly reframed as "operationalization consistency" not "proxy validation"---the metrics are not independent.

### 3. Improvement Headroom
Clear improvement path exists and is entirely writing-based:
- Tier 1 fixes (collision rate reframe, remove fabrications, fix data, soften claims, K-means analysis): +1.3 points
- Tier 2 improvements (bootstrap CI, tables, heatmap): +0.8 points
- Tier 3 polish (terminology, scatter plot, LaTeX): +0.4 points

No new experiments are needed. All improvements are achievable within 3 hours.

### 4. Time-Cost Tradeoff
Continuing to optimize the current direction requires ~3 hours of writing and figure generation. Pivoting to an alternative (e.g., decoder weight similarity) would require new experiments with uncertain outcomes. The current work is nearly complete---all experiments done, all data collected, only writing and figures remain.

### 5. Critical Objections
The skeptic's concerns are addressable:
- **Small n=7 GT**: Mitigated by bootstrap CI and explicit discussion of limited power
- **Collision rate reframe risk**: Made explicit and prominent in the paper; epistemically sound distinction
- **K-means double-edged sword**: Framed as undermining UAD's robustness, not as a positive finding
- **Negative result ceiling**: Accepted---target is ICBINB (7.5-8.0), not NeurIPS main track

None of the skeptic's concerns are fatal. All are acknowledged and mitigated in the proposed revision.

---

## Decision Rationale

1. **The core finding is honest and valuable**: UAD fails structurally for token-disjoint hierarchies due to token-level mutual exclusivity. This prevents the community from wasting effort on a dead-end direction.

2. **All critical issues have concrete fixes**: The 12 Iteration 3 reviewer issues are all addressable with writing changes. No new experiments needed.

3. **The projected score (7.5) is competitive for ICBINB**: ICBINB excellent papers score ~8.0. We are within reach after 3 hours of targeted fixes.

4. **Pivoting is worse than proceeding**: Starting over would waste all completed experimental work. The decoder weight similarity direction (Alternative A) is promising but unvalidated; pivoting to it would require new experiments with uncertain outcomes and would abandon a solid negative result.

5. **The synthesis verdict already recommends PROCEED**: The result debate synthesis (verdict.md) explicitly recommends "PROCEED with ICBINB-targeted paper revision" with a 3-hour budget.

6. **Structural limitations are accepted, not hidden**: The revisionist's cognitive updates correctly identify that honesty matters more than perfection for negative-result papers. All limitations are explicitly acknowledged.

---

## DECISION: PROCEED
