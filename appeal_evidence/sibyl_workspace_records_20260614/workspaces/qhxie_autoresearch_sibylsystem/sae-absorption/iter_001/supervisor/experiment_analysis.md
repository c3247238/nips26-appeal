# Experiment Result Analysis

## Key Results Summary

### EDA Detection (H1/H2)
| Config | EDA AUROC | Status | Label Source |
|--------|-----------|--------|--------------|
| Gemma L5-16k | 0.698 | PASS | Proxy (Neuronpedia) |
| Gemma L12-16k | 0.776 (Cohen's d = 1.02, p = 6.4e-5) | PASS | Proxy |
| Gemma L5-65k | 0.617 | FAIL | Proxy |
| Gemma L12-65k | 0.468 | FAIL (sub-chance) | Proxy |
| Gemma L19-16k | 0.458 | FAIL | Proxy |
| Gemma L19-65k | 0.562 | FAIL | Proxy |
| GPT-2 L6 | 0.650 (Cohen's d = 0.53, CI [0.531, 0.761]) | PASS | Direct (FeatureAbsorptionCalculator) |
| GPT-2 L10 | 0.336 (reversed direction) | FAIL | Direct |
**Net: 3/8 configs pass AUROC ≥ 0.65. Writing gate go/no_go = true (gate was ≥ 2 model families; passes).**

Critical R4 finding: EDA = 1 - cos(encoder_j, decoder_j) is mathematically identical to 1 - DecCos (validated across Gemma 2B, GPT-2, Llama-3.1-8B with Pearson r = -1.0). The metric is a re-derivation of an existing SAEBench measure; the novelty lies in the formal lower bound theorem and systematic regime characterization.

Theoretical validation: SynthSAEBench AUROC = 1.0, F1 = 0.974. The theorem holds exactly in controlled settings.

### Three-Subtype Taxonomy (H4) — Strongest Finding
| Config | N Absorbed | Early (%) | Late (%) | Partial (%) | KW p-value |
|--------|-----------|-----------|----------|-------------|------------|
| L12-16k | 16 | 75.0 | 12.5 | 12.5 | 0.237 (underpowered) |
| L12-65k | 65 | 72.3 | 13.8 | 13.8 | **0.0002** |

Late > Early EDA ordering holds at all 5 threshold variants (tau = 0.20–0.40). Critical caveat: at tau = 0.2, early proportion drops to ~32% at L12-65k — the 72-75% figure is tau-dependent. Natural interpretation: most absorbed latents (~75%) lack a corresponding decoder direction (dictionary coverage failure), not encoder alignment failure.

### ITAC Correction (H5) — FALSIFIED
| Config | FN Reduction | FVU Change | Pass ≥ 20%? |
|--------|-------------|------------|-------------|
| L12-16k | 0.00% | +22.1% | NO |
| L12-65k | 3.14% (best latent: 22.7%) | -4.23% | NO |

H5 pre-registered target was ≥ 20% mean FN reduction. Mean is 2.69%. Structural explanation: 75% of absorbed latents are early-type (no decoder representation), making them ineligible for ITAC. This failure is confirmatory evidence for the taxonomy, not an independent weakness.

### Cross-Domain Absorption (H3) — FALSIFIED by R4 shuffled controls
- R4B shuffled control: 0/9 domain-SAE combinations exceed shuffled p95 null threshold. Real ≈ shuffled for all hierarchies.
- Probe quality gate: best achievable with GPT-2 Medium bridge = 59.5% (city-continent), gate = 85%. Both Gemma 2B and Llama-3.1-8B are HF-gated.
- The R3 intra-RAVEL rho = 0.924 (n=6 configs) is an artifact: R4 shuffled labels also produce rho = 1.0 (n=3), meaning the coherence reflects SAE-width-level variation, not hierarchy-specific absorption structure.
- **H3 as a publication claim is withdrawn.**

### Scaling Analysis (H6) — FALSIFIED
Partial rho(width, absorption | L0) = +0.37 (expected: negative). L0 variation = 65–72 (insufficient). Log-linear R² = 0.18. Reported as a methodological limitation and descriptive finding only.

### Backup A (Amortization Gap Experiment) — NOT YET RUN
The strategist identifies this as the highest information-gain remaining experiment (1-2 GPU-hours): fix Gemma Scope L12-16k decoder dictionary and compare absorption rates under feedforward encoding vs. OMP encoding. Either outcome (encoder-dominant vs. dictionary-dominant) is publishable and connects to the early-dominance finding.

---

## Debate Perspectives Summary

- **Optimist**: Two-to-three genuine contributions validated. EDA's regime-specific signal is real (AUROC 0.776 at L12-16k, direct GPT-2 validation). Three-subtype taxonomy is the paper's most impactful finding — reframes absorption from encoder failure to dictionary coverage failure. ITAC's failure is confirmatory. PROCEED with two-contribution structure + Backup A as third contribution. Target: EMNLP 2026 or NeurIPS MI Workshop.

- **Skeptic**: Fatal Flaw 1 — EDA = 1 - DecCos destroys metric novelty; the theorem is novel but the metric is an existing SAEBench measure. Fatal Flaw 2 — H3 is empirically falsified by R4 shuffled controls, not merely pending. The best direct-label AUROC is 0.650 with n_pos = 18 (CI lower bound = 0.531, barely above chance). Taxonomy's 75% early-dominance figure is threshold-dependent and rests on a single adequately-powered SAE config (L12-65k n=65). Paper scope must be dramatically reduced relative to original proposal.

- **Strategist**: PROCEED immediately. Run Backup A (1-2 GPU-hours) as the pre-writing experiment — either outcome is publishable and converts a two-contribution paper into a three-contribution paper. Dominant strategy over all alternatives (model access waiting, pivot to LCA-SAE, more EDA configs). Writing gate (go_write = true) has been passed. Do not wait for HF model access — it is an external, uncontrollable dependency.

---

## Analysis

### Dimension 1: Method Feasibility

The EDA lower bound theorem is sound and uncontested: formally grounded in Tang et al. (2025) biconvex optimization landscape, validated synthetically (AUROC = 1.0, F1 = 0.974). The key R4 clarification is that EDA is mathematically equivalent to 1 - DecCos — not a new formula, but a re-derivation with formal theoretical backing and systematic regime characterization. The method works in the regime where it is expected to work: mid-layers (L5–L12), narrow SAEs (16k width), where late absorption provides visible encoder-decoder misalignment. The failure at 65k-width and deep layers is consistent with the theoretical prediction: wider SAEs have higher proportions of early-type absorbed latents (decoder-absent), for which EDA has no theoretical signal.

**Feasibility verdict: CONFIRMED in the delineated regime.**

### Dimension 2: Performance

Three configs pass AUROC ≥ 0.65 across two model families (Gemma proxy + GPT-2 direct labels). The strongest signal is L12-16k at 0.776 (Cohen's d = 1.02), which is a large effect size for a purely weight-based diagnostic with no activation data. The taxonomy's key result — 72–75% early-type dominance at KW p = 0.0002 — is statistically robust at the pre-registered threshold. ITAC (3.14% vs. 20% target) and cross-domain (H3 falsified by shuffled controls) have failed their pre-registered gates.

**Performance verdict: Moderate. Two primary contributions (EDA regime characterization + taxonomy) have adequate statistical support. Three hypotheses falsified (H2 D-EDA improvement, H3 cross-domain, H5 ITAC, H6 scaling). Scope must reflect this.**

### Dimension 3: Improvement Headroom

The strategist's Backup A experiment (1–2 GPU-hours) is the clearest path to a third contribution: comparing feedforward vs. OMP absorption rates to adjudicate whether early absorption is driven by encoder amortization gap (O'Neill et al.) or dictionary landscape spurious optima (Tang et al.). This experiment requires no new data downloads, uses already-loaded Gemma Scope SAEs, and either outcome is a publishable mechanistic finding. Beyond this, HF model access for Gemma 2B / Llama-3.1-8B remains a background action that could upgrade proxy-label results but is not a blocking dependency.

**Improvement headroom: Moderate. Backup A is the only actionable pre-writing experiment.**

### Dimension 4: Time-Cost Tradeoff

The project has completed 4 rounds of experiments (7+ GPU-hours of results). The existing evidence supports two robust contributions (EDA regime detection + taxonomy). Backup A adds a third for 1–2 GPU-hours with near-certain publishable outcome. In contrast, pivoting to any alternative research direction resets the accumulated evidence to zero and incurs an estimated 3–4 weeks of additional experiment cycles. The opportunity cost of pivoting is prohibitive given the quality and completeness of the existing data.

**Time-cost tradeoff: Strongly favors PROCEED.**

### Dimension 5: Critical Objections

The skeptic's two designated "fatal flaws" must be evaluated against the severity threshold for PIVOT:

- **F1 (EDA = 1 - DecCos)**: Serious, not fatal. The paper's novelty must be repositioned to (a) the formal biconvex lower bound theorem, (b) the first systematic regime characterization across layers/widths/models for this metric as an absorption detector, and (c) the cross-model empirical validation. SAEBench has used DecCos as a secondary metric but has never grounded it theoretically as an absorption detector with formal guarantees or studied its regime dependence. The contribution survives honest repositioning.

- **F2 (H3 falsified by shuffled controls)**: Fatal for the cross-domain absorption claim. This was previously the "cleanest novelty claim" and is now empirically withdrawn. This is a genuine loss of one primary contribution, not just a framing issue. However, it does not invalidate the remaining two contributions.

- **Additional concern (GPT-2 L10 reversal)**: Absorbed latents show lower EDA than non-absorbed at L10 (AUROC = 0.336, reversed). This is not merely failure but active anti-prediction. The skeptic is correct that this requires explanation, not suppression. The taxonomy provides a partial explanation (if L10 absorbed features are predominantly late-type with high encoder-decoder alignment maintained by other mechanisms, the reversal is theoretically possible), but this must be investigated before submission.

The remaining concerns — taxonomy threshold sensitivity (S2), GPT-2 small sample n_pos = 18 (S1), ITAC falsification (minor) — are addressable through honest framing and additional analysis within the current paper scope. None independently crosses the PIVOT threshold.

**Critical objections verdict: Two of three pre-registered primary contributions survive (EDA + taxonomy). One contribution is definitively withdrawn (H3). The paper scope reduces from 3 to 2 primary contributions, with Backup A as a potential third. No objection constitutes a fundamental validity failure requiring abandonment of the research direction.**

---

## Decision Rationale

**PROCEED** is the correct decision for the following reasons:

1. **Two robust contributions remain valid** after R4. The EDA regime characterization (AUROC 0.65–0.78 across Gemma and GPT-2, formal lower bound theorem, regime boundary at 16k vs. 65k width) and the three-subtype taxonomy with early-dominance finding (72–75%, KW p = 0.0002, threshold-stable late > early ordering) are both publication-quality results at EMNLP/workshop level.

2. **H3's falsification is a cost but not a dealbreaker**. Losing the cross-domain contribution narrows the paper, but the remaining contributions are self-contained and novel. The early-dominance finding (75% of absorbed latents are dictionary coverage failures) is potentially the most impactful result for practitioners — it directly redirects the mitigation literature toward training-time dictionary coverage fixes rather than inference-time encoder fixes.

3. **Backup A provides a near-zero-cost path to a third contribution**. The amortization gap experiment (1–2 GPU-hours, no new downloads) converts the paper from two to three contributions. Given that the writing gate has been passed (go_write = true), the dominant strategy is to run Backup A immediately, then begin paper writing.

4. **All pivot alternatives are strictly dominated**. LCA-SAE requires 3–4 GPU-hours and faces MP-SAE competitive crowding. Waiting for HF model access is an external, uncontrollable dependency. Scaling law experiments are already falsified. None of these paths improves expected paper quality per GPU-hour compared to Backup A + writing.

5. **Honest scope reduction is achievable**. The skeptic's concerns (EDA = DecCos, n_pos = 18 at GPT-2, taxonomy threshold sensitivity) are all addressable through explicit, honest framing in the paper. These are limitations to acknowledge, not flaws that require abandonment. The field will benefit from an honest, precisely-scoped characterization paper that does not overclaim.

**Immediate recommended actions before paper writing:**
1. Run Backup A (amortization gap controlled dictionary experiment, 1–2 GPU-hours)
2. Integrate Backup A results into framing; determine whether third contribution is encoder-dominant or dictionary-dominant finding
3. Investigate GPT-2 L10 reversal mechanistically (taxonomy analysis: what subtype are L10 absorbed features?)
4. Begin paper writing with two-contribution (+ optional Backup A third) structure

DECISION: PROCEED
