# Ideation Critique: Activation Energy Theory

## Overall Assessment

The ideation process shows evidence-driven iteration across five rounds, with honest reporting of negative results. The pivot from EDW-DPO (Round 1) to CCAR (Round 2) to Activation Energy Theory (Round 4) demonstrates adaptive reasoning. However, the current direction has a narrow novelty window, relies on a single negative result, and proposes follow-up hypotheses (H4, H5) that are not delivered in the paper.

**Novelty Score: 6/10** (accurately assessed in novelty_report.json)

---

## Critical Issues

### 1. Novelty Rests Entirely on a Single Negative Result

The novelty_report.json correctly identifies that H1/H2 collide with Yang et al. (2025) on the exponential saturation formula. The only genuinely novel claim is the H3 negative result: "Ea-based routing does NOT predict single-pass solveability."

**Risk**: A paper whose sole novelty is a negative result on n=50 problems with a single model is fragile. Reviewers may ask: (a) is this just a failed experiment rather than a publishable finding? (b) does the negative result generalize beyond Qwen2.5-Math-7B?

**Mitigation in current paper**: The paper frames the negative result as "first systematic falsification" with diagnostic analysis. This is defensible but borderline.

### 2. H4/H5 Are Promised But Not Delivered

The proposal.md (Round 5 Synthesis) and hypotheses.md introduce H4 (error classification) and H5 (entropy routing) as central next steps. The alternatives.md presents Alternative A (Entropy Routing) as the "front-runner." However:

- H4 error classification appears only as a qualitative Table 8 with no quantitative data
- H5 entropy routing is listed as "pending" in Table 9 and never tested
- The paper promises diagnostic analysis and alternative signals but delivers only the H3 falsification

**Risk**: The paper feels incomplete. Readers expecting the promised H4/H5 analysis will be disappointed.

**Fix**: Either (a) complete H4/H5 experiments and integrate results, or (b) reframe the paper as a focused falsification study and move H4/H5 to Future Work.

### 3. Decision Tree in Alternatives Is Overly Optimistic

The alternatives.md decision tree assumes that if H4 explains the routing failure, then H5 (entropy routing) will succeed. But the novelty_report.json flags H5 as "HIGH RISK" because ACAR (2026) found entropy routing ineffective. The decision tree does not account for this risk.

**Fix**: Update the decision tree to reflect the ACAR finding and the high probability that H5 also fails.

---

## Major Issues

### 4. Collision with Yang et al. Is Not Fully Acknowledged

The paper states the collision in the proposal and novelty report, but the abstract and introduction still present H1 (Arrhenius kinetics) as a finding rather than a replication. The abstract says "aggregate accuracy follows Arrhenius-like kinetics with R²=0.924" without immediately noting that this replicates Yang et al.

**Fix**: In the abstract and introduction, explicitly label H1 as a replication/cross-validation of Yang et al. rather than an original finding. This strengthens the paper by showing intellectual honesty.

### 5. The "Activation Energy" Framing May Be Misleading

The paper adopts "activation energy" as a framing device, but:
- The physical analogy is weak (chemical activation energy vs. answer consistency)
- The two "activation energy" measures (Ea and k0) are uncorrelated, undermining the unified framework
- The bimodal Ea distribution does not resemble any physical activation energy spectrum

**Risk**: Reviewers may view the "activation energy" framing as gratuitous physics-washing of a simple consistency metric.

**Fix**: Consider de-emphasizing the physics analogy and presenting the work as a study of "consistency-derived difficulty metrics" rather than "activation energy theory." The physics framing adds little and risks skepticism.

### 6. Alternatives Document Proposes Training-Based Approaches That Are Ignored

Alternative B (EDTT Training) and Alternative D (Step-DPO Redux) propose GPU-based training experiments. The GPU is available (PyTorch 2.11.0 + Blackwell), but the paper focuses entirely on inference-based analysis.

**Risk**: The training-based alternatives may have higher novelty than the inference-based negative result.

**Assessment**: Given the project's history (Round 1 EDW-DPO falsified, Round 2 Step-DPO loss 0.694 > 0.5), training-based approaches have poor track record. The inference-based direction is pragmatic but limited in scope.

---

## Minor Issues

### 7. Round 3 API Block Is Not Mentioned in Paper

The paper does not mention that Round 3 (Training-Free API Inference) was blocked due to lack of API key. This is fine---failed directions need not be in the final paper---but it explains why the project pivoted to the current direction.

### 8. Pilot vs. Full Data Inconsistencies

- Round 4 pilot (n=30): H2 Spearman=0.578, p=0.0008
- Full analysis (n=50): H2 Spearman=0.448, p=0.001

The correlation weakened with more data. The paper uses the full n=50 values but the pilot values appear in hypotheses.md and proposal.md. This is not a problem if the paper is consistent, but it shows the pilot was optimistic.

### 9. The "Irreducible Error Floor" Claim Is Speculative

The paper claims a ~25pp irreducible error floor for consistency-based routing, comparing it to ACAR's ~8pp for variance-based routing. But:
- The 25pp figure comes from post-hoc thresholding
- ACAR is not cited with full bibliographic details
- The comparison may not be fair (different models, different datasets, different routing paradigms)

**Fix**: Soften the comparison or provide more evidence.

---

## Summary

The ideation process produced a defensible but narrow contribution. The H3 negative result is genuine and honestly reported, but the paper promises more (H4 diagnostic, H5 alternatives) than it delivers. The novelty is fragile and depends on resolving the formula-data inconsistency. Consider either:

1. **Narrow and deep**: Focus solely on H3 falsification with robust diagnostic analysis (quantitative H4), or
2. **Broad and comparative**: Test multiple routing signals (consistency, entropy, confidence) on the same dataset to provide a comparative benchmark.

Option 1 is more achievable; Option 2 would be more impactful but requires additional experiments.
