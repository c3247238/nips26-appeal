# Result Debate Verdict

**Iteration 9 FULL Mode | 2026-04-16 | 6 perspectives synthesized**

---

## Score: 7.5 / 10

A paper with genuinely novel and important findings (layer-dependent absorption, cross-domain causal mechanism, comprehensive negative results), strong statistical methodology, and exemplary honest reporting -- penalized by unresolved probe quality concerns and scope limitations on the pathological absorption claim.

---

## Key Conclusion

This project has produced the first cross-domain characterization of SAE feature absorption, revealing that absorption is layer-dependent (15x range, confound-free), causally driven by competitive exclusion across all tested hierarchy types (d=0.75-1.50 after methodology correction), and always pathological within tested scope (0% benign, ~1000x effect ratio). Nine definitive negative results establish that absorption resists all tested correlational predictors.

**Critical correction identified in this synthesis**: The consolidation summary erroneously reports cross-domain activation patching as "FAILED" (d=-0.91, reverse direction). This was based on a buggy pilot. The corrected FULL-run data shows strong positive results: city-continent recovery 61.9% (d=1.50, p<1e-20), city-language recovery 34.2% (d=0.75, p<1e-18). H7 is upgraded from "first-letter only" to "cross-domain supported." This substantially strengthens the paper.

---

## Action Plan

### Verdict: PROCEED

### Three blocking actions before submission:

1. **Probe degradation ablation** (1-2 GPU-hours). Inject label noise into first-letter probes to F1={0.70, 0.80, 0.85, 0.90}, re-measure absorption at L24. This resolves the paper's Achilles heel: the correlation between probe quality and measured absorption rate. All non-Optimist perspectives demand it.

2. **Update consolidation with corrected cross-domain patching** (30 min CPU). Replace the stale "FAILED" verdict for H7 with the FULL-run results (d=1.50, d=0.75). Without this, the writing agent will propagate incorrect claims.

3. **Implement validate_integration.py** (1.5 hr CPU). Automated cross-check of all numerical claims against source JSON files. The 12.3% hallucinated number incident proves this is necessary for submission quality.

### Two high-priority experiments:

4. **L24 activation patching for first-letter** (1-2 GPU-hours). Current causal evidence is at L12 (5.7% absorption). Headline results are at L24 (27.1%). This gap invites reviewer skepticism.

5. **Benign/pathological replication on first-letter** (0.5-1 GPU-hour). Extends the "100% pathological" claim beyond a single hierarchy.

### Paper reframing:

- Lead with layer-dependent absorption (confound-free) and universal causal mechanism (corrected cross-domain patching)
- Report cross-domain variation with probe quality caveats; remove city-country (F1=0.73) from primary comparison
- Change title away from "The Absorption Tax" (quantitative framework failed)
- Target venue: NeurIPS 2026 main conference

### Total budget: ~4-6 GPU-hours + ~4 CPU-hours. Wall-clock: ~1 day.
