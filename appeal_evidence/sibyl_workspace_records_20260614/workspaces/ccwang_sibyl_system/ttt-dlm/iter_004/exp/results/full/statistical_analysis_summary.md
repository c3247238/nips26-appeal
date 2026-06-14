# Statistical Analysis — PILOT Summary (n=16)

**Task**: statistical_tests | **Mode**: PILOT | **Verdict**: GO

## Countdown-16 Accuracy Ranking

| Rank | Method | Accuracy | n_correct/16 |
|------|--------|----------|-------------|
| 1 | DMI (α=0.3) | 12.5% | 2 |
| 1 | A-CFG (w=1.5) | 12.5% | 2 |
| 1 | A-CFG (w=2.0) | 12.5% | 2 |
| 4 | BSD (k=0.75) | 6.2% | 1 |
| 4 | A-CFG (w=1.0) | 6.2% | 1 |
| 4 | BSD+ACFG combo | 6.2% | 1 |
| 4 | vanilla (256 steps) | 6.2% | 1 |
| 8 | vanilla (128 steps) | 0.0% | 0 |

## GSM8K-16 Accuracy Ranking

| Rank | Method | Accuracy | n_correct/16 |
|------|--------|----------|-------------|
| 1 | **A-CFG (w=1.5)** | **37.5%** | 6 |
| 2 | Vanilla | 25.0% | 4 |
| 2 | DMI | 25.0% | 4 |
| 4 | BSD | 18.8% | 3 |

## McNemar Tests

**No comparisons reached statistical significance** (all p ≥ 0.45).

This is expected: with n=16, McNemar's exact test requires extremely large effect sizes (~25+pp in a single direction with zero flips in the other). The maximum discordant pair count observed was (b=5, c=2) for A-CFG vs BSD on GSM8K (p=0.453).

- Bonferroni-corrected α = 0.05/36 = 0.00139 (Countdown)
- 0 / 36 comparisons significant

## Bootstrap 95% CI (vs vanilla)

### Countdown-16
| Method | Mean Δ | 95% CI | Includes 0? |
|--------|--------|--------|-------------|
| DMI | +0.125 | [0.0, +0.25] | Yes (boundary) |
| A-CFG (w=1.5) | +0.125 | [0.0, +0.25] | Yes (boundary) |
| BSD | +0.063 | [−0.063, +0.188] | Yes |
| BSD+ACFG | +0.063 | [−0.063, +0.188] | Yes |

### GSM8K-16
| Method | Mean Δ | 95% CI | Includes 0? |
|--------|--------|--------|-------------|
| A-CFG | +0.125 | [−0.125, +0.375] | Yes |
| DMI | 0.000 | [−0.250, +0.250] | Yes |
| BSD | −0.063 | [−0.313, +0.188] | Yes |

## Effect Sizes (Cohen's h)

Key comparisons vs vanilla (128 steps):
- DMI vs vanilla: h=+0.72 (**medium**)
- A-CFG vs vanilla: h=+0.72 (**medium**)
- BSD vs vanilla: h=+0.51 (**medium**)
- BSD+ACFG vs vanilla: h=+0.51 (**medium**)

## Difficulty-Stratified Analysis (Countdown-16)

| Difficulty | Count | Vanilla | DMI | BSD | A-CFG | Combo |
|-----------|-------|---------|-----|-----|-------|-------|
| Easy | 1 | 0% | 100% | 0% | 100% | 0% |
| Medium | 4 | 0% | 0% | 0% | 0% | 0% |
| Hard | 11 | 0% | 9% | 9% | 9% | 9% |

Most problems (11/16 = 69%) are "hard" (target ≥ 100 or ≥ 5 numbers). DMI uniquely solves the easy problem.

## Statistical Power Warning

With n=16 samples, the statistical tests have **very low power**:
- McNemar cannot detect effects < ~25pp
- Bootstrap CIs are wide (±12-31pp)
- Full-scale evaluation (n=500, 3 seeds) is **required** for publication-ready conclusions

## PILOT Pass Criteria

✅ All statistical tests computed without error
✅ Framework validated for full-scale analysis
