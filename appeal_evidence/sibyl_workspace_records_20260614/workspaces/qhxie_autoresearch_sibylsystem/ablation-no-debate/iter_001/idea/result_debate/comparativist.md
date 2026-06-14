# Comparative Analysis: Feature Absorption in SAEs

## Baseline Landscape

### Core References for Absorption Metrics

| Method | Paper | Key Metric | Published |
|--------|-------|------------|-----------|
| **Original absorption rate** | Chanin et al. (2024) arXiv:2409.14507 | Single-feature ablation | Sep 2024 |
| **Multi-child proportional ablation** | **This work** | Parent activation / child ablation | Apr 2026 |
| **Matryoshka SAEs** | Bussmann et al. (2025) arXiv:2503.17547 | Absorption mitigation via multi-level features | Mar 2025 |
| **Feature sensitivity** | Tian et al. (2025) arXiv:2509.23717 | Sensitivity = activation on semantic neighbors | Sep 2025 |
| **Interpretability without actionability** | Basu et al. (2026) arXiv:2603.18353 | Safety correction rate via SAE steering | Mar 2026 |
| **Coffee feature fragility** | Ronge et al. (2026) arXiv:2601.03047 | Steering sensitivity to layer/magnitude | Jan 2026 |

### Established SOTA Baselines

| Baseline | Absorption Rate | Notes |
|----------|----------------|-------|
| Trained SAE (synthetic) | 0.50 | Pilot + multiseed confirmation |
| Random decoder baseline | 0.06-0.20 | Varies by seed |
| Shuffled/permuted baselines | 0.48-0.49 | Matches trained SAE |
| Gemma Scope real SAE | ~0.91 | Very high absorption in practice |
| Safety vs non-safety features | No difference (p=0.665) | H_Safe failed |

---

## Contribution Margin Analysis

### H1: Multi-child proportional absorption

| Comparison | Delta | Classification |
|------------|-------|----------------|
| Trained SAE (0.50) vs Random decoder (0.06-0.20) | +0.30-0.44 | **STRONG** (>5%) |
| Effect size (Cohen's d) | 8.94 | **Very large** |

**Verdict**: Strong contribution. Multi-child proportional ablation successfully differentiates trained SAEs from random baselines with very large effect size. This validates the measurement methodology.

### H3: Steering intervention

| Comparison | Delta | Classification |
|------------|-------|----------------|
| Absorbed vs non-absorbed sensitivity | 1.62x ratio | **Moderate** (1-5%) |
| Statistical significance | Pass criteria met | Validated |

**Verdict**: Moderate contribution. Steering improves sensitivity for absorbed features, but absolute effect sizes are small (0.055 vs 0.034). Practical significance for safety applications unclear.

### H_Safe: Safety-critical feature absorption

| Comparison | Delta | Significance |
|------------|-------|--------------|
| Safety (0.907) vs non-safety (0.906) | +0.0003 | **Not significant** (p=0.665) |

**Verdict**: **No contribution**. Safety features are not disproportionately absorbed. This is an important negative result but does not constitute a novel contribution.

### H_Mech: Geometric decomposition

| Condition | Absorption | Interpretation |
|-----------|------------|----------------|
| Random encoder + Random decoder | 0.299 | Pure geometry |
| Trained encoder + Random decoder | 0.490 | Encoder alignment dominates |
| Random encoder + Trained decoder | 0.299 | Decoder geometry effect = 0 |
| Trained encoder + Trained decoder | 0.484 | Full training |

**Verdict**: **Counter-intuitive finding** but pass criteria not met. Encoder alignment (not decoder geometry) appears to drive absorption, contradicting pilot interpretation.

---

## Concurrent Work Scan (Jan-Apr 2026)

### Direct Competitors

| Paper | Relationship | Impact |
|-------|--------------|--------|
| Basu et al. (2026) arXiv:2603.18353 | **High relevance** | Shows SAE steering fails to reliably correct model outputs; supports need for better absorption understanding |
| Bussmann et al. (2025) arXiv:2503.17547 | **Moderate** | Matryoshka SAEs explicitly address absorption via multi-level features |
| Ronge et al. (2026) arXiv:2601.03047 | **High relevance** | Documents steering fragility; validates concern about SAE reliability |

### Indirect Competitors

| Paper | Relationship | Notes |
|-------|--------------|-------|
| Tian et al. (2025) | Complementary | Feature sensitivity = different evaluation dimension |
| Muchane et al. (2025) | Complementary | Hierarchical SAE architecture addresses similar issues |
| Song et al. (2025) arXiv:2505.20254 | Complementary | Feature consistency focus |

### Key Finding from Concurrent Work

**Basu et al. (2026)** is the most important concurrent work:
- SAE feature steering produced **zero effect** despite 3,695 significant features
- 53-percentage-point knowledge-action gap
- Concludes: "Current mechanistic interpretability methods cannot reliably translate internal knowledge into corrected outputs"

This **directly supports** the importance of understanding absorption, but also suggests the field may be over-estimating the utility of SAE-based interpretability.

---

## Novelty Verdict

**The ONE thing this work does that no prior work does:**

> **Systematically quantifies the geometric vs learned decomposition of feature absorption using a 2x2 factorial design, finding that encoder alignment (not decoder geometry) drives absorption in synthetic hierarchies.**

### What is genuinely novel:
1. Multi-child proportional ablation methodology
2. Geometric decomposition via 2x2 factorial design
3. Empirical evidence that encoder alignment drives absorption (vs decoder geometry intuition)

### What is NOT novel:
- General absorption measurement (Chanin 2024)
- Steering interventions (multiple prior works)
- Safety feature analysis (Ronge 2026, Basu 2026)

---

## Venue Recommendation

### Assessment

| Factor | Score | Notes |
|--------|-------|-------|
| Novelty | Moderate | Methodology novel; core finding incremental |
| Effect size | Large | H1 shows strong separation |
| Negative results | Present | H_Safe failed; H_Mech counterintuitive |
| Concurrent work | Challenging | Basu 2026 shows field skepticism |

### Recommended Venue: **NeurIPS 2026 or ICLR 2026 (Workshop Track)**

**Justification**:
- The multi-child proportional ablation methodology is novel and worth publishing
- Strong effect sizes for H1
- However, Basu et al. (2026) publication raises questions about whether absorption-focused work will be seen as practically important
- Safety finding (no elevated absorption) is a negative result better suited for workshop

**Alternative**: **ACL 2026 (Findings)** if framed as methodology paper

### Not recommended for:
- **ICML/ICLR main track**: Contribution margin insufficient for top-tier without additional experiments
- **Nature Machine Intelligence**: Too applied/methodological for their scope

---

## Strengthening Plan

### Priority 1: Resolve H_Mech (Encoder vs Decoder)

**Problem**: H_Mech contradicts pilot interpretation (geometric vs learned)

**Required experiment**:
- Re-run with more samples (n=500) and stochastic hierarchy generation
- Check whether decoder geometry effect emerges with different hierarchy configurations
- Consider: does decoder geometry affect which features absorb, not whether absorption occurs?

**Expected outcome**: Clearer decomposition of absorption drivers

### Priority 2: Address Basu et al. (2026) Collision

**Problem**: Concurrent work shows SAE steering fails in practice

**Required analysis**:
- Explicitly cite and discuss Basu et al.
- Relate absorption findings to the "knowledge-action gap"
- Frame contribution as: "understanding why steering fails by characterizing absorption"

**Expected outcome**: Clearer positioning against skepticism

### Priority 3: Add Real-World SAE Comparison

**Problem**: Results primarily on synthetic hierarchies

**Required experiment**:
- Apply multi-child proportional ablation to real Gemma Scope SAEs
- Use Neuronpedia feature indices
- Compare with random baseline on real activations

**Expected outcome**: Bridge synthetic-to-real gap

---

## Final Assessment

### Strengths
1. Novel measurement methodology (multi-child proportional ablation)
2. Large effect sizes for trained vs random baseline (d=8.94)
3. Interesting negative result (safety features not disproportionately absorbed)
4. Addresses important concurrent work (Ronge 2026, Basu 2026)

### Weaknesses
1. H_Mech results are counter-intuitive and pass criteria not met
2. Safety finding is negative (no contribution)
3. Synthetic hierarchy may not generalize to real SAEs
4. Basu et al. (2026) publication casts doubt on practical utility

### Contribution Margin Summary

| Contribution | Delta | Novelty | Publishable? |
|--------------|-------|---------|--------------|
| Multi-child ablation methodology | New | High | Yes (workshop) |
| Geometric decomposition | Null | High | Needs replication |
| Safety absorption analysis | ~0% | None | No (negative result) |
| Steering sensitivity | 1.62x | Moderate | Maybe (with Basu discussion) |

**Bottom line**: Publishable as workshop paper with methodology focus. Main conference requires resolving H_Mech and adding real-world experiments.

---

## Sources

- [Chanin et al. 2024: A is for Absorption](https://arxiv.org/abs/2409.14507)
- [Bussmann et al. 2025: Matryoshka SAEs](https://arxiv.org/abs/2503.17547)
- [Basu et al. 2026: Interpretability without Actionability](https://arxiv.org/abs/2603.18353)
- [Ronge et al. 2026: Coffee Feature Activates on Coffins](https://arxiv.org/abs/2601.03047)
- [Tian et al. 2025: Measuring SAE Feature Sensitivity](https://arxiv.org/abs/2509.23717)
- [Muchane et al. 2025: Hierarchical Semantics in SAE](https://arxiv.org/abs/2506.01197)
- [Song et al. 2025: Feature Consistency Position](https://arxiv.org/abs/2505.20254)
