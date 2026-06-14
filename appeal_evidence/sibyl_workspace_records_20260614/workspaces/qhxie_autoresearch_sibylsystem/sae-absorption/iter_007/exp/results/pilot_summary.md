# Tightened Hedging Classification — Pilot + Full Results

## Task
For each of 656 false negatives at L0=22, check whether the SPECIFIC k=5 parent-associated latents (from probe at L0=22) fire at L0=176. Classify as "strict hedging" only if at least one parent latent fires.

## Key Results

| Classification | Count | Rate | 95% CI |
|---|---|---|---|
| Strict hedging (parent fires at L0=176) | 41 | 6.2% | [4.4%, 8.2%] |
| Non-hedging (NO parent fires at L0=176) | 615 | 93.8% | [91.9%, 95.6%] |
| Permissive hedging (iter_006 baseline) | 648 | 98.6% | — |

## Critical Finding

**The 98.6% permissive hedging rate is almost entirely an artifact of the definition.** When we check whether the SPECIFIC parent-associated latents fire at L0=176 (strict check), only 6.2% qualify. The remaining 93.8% of FN tokens resolve at L0=176 through mechanisms OTHER than parent-latent recovery — e.g., the SAE's overall feature decomposition changes so dramatically at higher L0 that different features capture the letter information.

## Control Validation

| Metric | Value |
|---|---|
| True strict rate | 6.25% |
| Shuffled control mean | 3.44% |
| Z-score | 3.51 |
| P-value | 0.00044 |

The true strict rate is significantly higher than shuffled controls (p=0.0004), confirming a small but real parent-specific hedging signal above chance.

## Parent Latent Firing Distribution

| N latents firing (of 5) | Count | Fraction |
|---|---|---|
| 0 | 615 | 93.8% |
| 1 | 41 | 6.2% |
| 2-5 | 0 | 0.0% |

When parent latents do fire, they fire weakly (only 1 of 5) with mean activation 6.29.

## Notable Per-Letter Results

- **G (anomaly)**: 19/21 FNs show strict hedging (90.5%) — one probe latent strongly activates at L0=176
- **B, D, F, H, M, N, O, P, R, T, W**: 0% strict hedging (all FNs are non-hedging)
- Most letters have 0-13% strict hedging rates

## Mechanism Insight

All 615 non-hedging FN tokens have `still_fn_at_l0_176 = False` — they are NOT false negatives at L0=176 despite their parent latents NOT firing. This means FN resolution at high L0 occurs through compensatory features, NOT parent-specific recovery. The permissive definition incorrectly attributes this compensatory resolution to "hedging."

## Decision Gate

**MAJOR NARRATIVE REVISION NEEDED.** The 98.6% headline must be reframed as a permissive upper bound. The strict rate (6.2%) should become the primary reported number. The paper should discuss the 92.3 percentage-point gap as evidence that the permissive hedging definition conflates compensatory feature resolution with genuine parent-latent hedging.

## GO/NO-GO: GO
Experiment completed successfully with clear, interpretable results that resolve a 3-iteration blocking issue.
