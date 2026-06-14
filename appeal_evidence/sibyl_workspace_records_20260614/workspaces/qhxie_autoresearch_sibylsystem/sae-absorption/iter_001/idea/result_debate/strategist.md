# Strategist Analysis: SAE Absorption — Post-R4 Final Assessment

**Timestamp:** 2026-04-13
**Round:** 4 (post-R4 final writing gate)
**Verdict:** PROCEED to paper writing (two-contribution paper + Backup A as third contribution)

---

## 1. Signal Strength Assessment

| Result Area | Signal | Key Evidence | Justification |
|-------------|--------|--------------|---------------|
| EDA metric (H1) | **Moderate** | 3/8 configs pass AUROC >= 0.65: Gemma L5-16k (0.698), L12-16k (0.776), GPT-2 L6 (0.650 direct). Cross-model. Cohen's d = 0.53-1.02. | Reliable in a clearly delineated regime (16k-width, early-mid layers). Regime-specific framing is defensible. 5/8 failures are documented and explained. Cross-model consistency (Gemma + GPT-2) raises this above "single-model anecdote." |
| Three-subtype taxonomy (H4) | **Strong** | KW p = 0.0002 (L12-65k); late > early EDA ordering stable at 5/5 thresholds; 72-75% early-type at both L12-16k and L12-65k. | The most robust and highest-impact finding. Early-dominance reframes the entire absorption mitigation literature. Consistent across both tested SAE widths despite very different n_absorbed (16 vs 65). |
| Cross-domain absorption (H3) | **Collapsed** | R4B: all 3 RAVEL hierarchies fail probe quality gate (best: 59.5%, gate: 85%). Shuffled control: real rates indistinguishable from null for all 9 domain-SAE combinations. | H3 is effectively falsified by methodological limitation, not by evidence of non-existence. Same-model probes (Gemma 2B, Llama-3.1-8B) remain HF-gated. Intra-RAVEL rho = 0.924 from R3 is downgraded (n=6, Qwen proxy artifact). |
| D-EDA as separate metric (H2) | **Noise** | R4A: EDA and decoder-cosine AUROC are mathematically identical (delta = 0.0). EDA = 1 - DecCos validated across 3 architectures (r = -1.0). | H2 is definitively falsified. D-EDA adds zero information over EDA. The D-EDA formulation collapses to a monotone transform of dec_cos. Drop entirely. |
| ITAC correction (H5) | **Weak** | 2.69% mean FN reduction (target: 20%). 1/10 targets show notable improvement. R4D (real activations) was not run. | H5 falsified. ITAC is ineligible for 75% of absorbed latents (early-type). Retain only as a negative result / discussion item. |
| Scaling analysis (H6) | **Null** | Partial rho(width, absorption | L0) = +0.37. Log-linear R^2 = 0.18. L0 range = 65-72 (insufficient variation). | H6 falsified. Methodological limitation (canonical SAEs provide no L0 variation). Report as supplementary negative result. |
| GPT-2 + Llama cross-model | **Moderate** | GPT-2 L6 direct AUROC = 0.650 (PASS); GPT-2 L10 = 0.336 (FAIL, reversed). Llama-3.1-8B: weight-only EDA computed, model gated, no AUROC. | Cross-model evidence exists for 2 families (Gemma, GPT-2). Llama adds weight-level characterization (3 families) but no AUROC validation. Sufficient for "cross-model consistency" claim but not for "universal." |
| Backup A (amortization gap) | **Untested** | Not yet run. 1-2 GPU-hours. No new downloads. Immediate execution possible. | Highest expected information-gain-per-GPU-hour of any remaining experiment. Adjudicates the Tang vs. O'Neill mechanistic debate. Either outcome is publishable and connects to the 75% early-dominance finding. |

---

## 2. Opportunity Cost Analysis

| Direction | GPU Cost | Expected Info Gain per GPU-hr | Risk | Priority |
|-----------|----------|-------------------------------|------|----------|
| Backup A: Amortization gap controlled dictionary experiment | 1-2 hrs | **Very High** | Low (no new downloads, clean hypothesis, either outcome is publishable) | **1** |
| Paper writing (two-contribution structure) | 0 hrs (LLM time only) | **High** (reduces project duration) | Low | **2** |
| Model access acquisition (Gemma 2B / Llama-3.1-8B HF gates) | 0 GPU-hrs, unknown calendar time | **High but uncertain** (unlocks H3 validation + Llama AUROC + Gemma direct labels) | High (gating is external, uncontrollable) | **3 (background)** |
| ITAC on real activations (R4D) | 2-3 hrs | **Low** (even best case: 10-15% FN reduction on ~13% of absorbed latents = marginal contribution) | Medium (likely confirms R3 negative result) | **4 (optional, post-paper)** |
| Additional cross-model configs | 3-4 hrs | **Low** (diminishing returns; 2 model families already present) | Low | **5 (deferred to revision)** |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|----------------|----------|------|-----------------|
| **Backup A + paper writing** | Untested (Backup A) + Moderate-Strong (existing) | 1-2 hrs | **Low** | **Best path**: adds a clean third contribution with minimal cost. Three-contribution paper: EDA + taxonomy + amortization gap. |
| Paper writing only (two contributions) | Moderate + Strong | 0 hrs | **Medium** (thin paper risk at top venues) | Publishable at mid-tier (EMNLP, MI Workshop) but may lack sufficient depth for NeurIPS/ICLR main. |
| Wait for model access, then complete H3 | Collapsed | Unknown | **Very High** (external dependency, may never resolve) | If resolved: restores cross-domain as third contribution. If not: delays paper indefinitely. **Unacceptable.** |
| Pivot to Backup B (LCA-SAE) | Untested | 3-4 hrs | **High** (new direction, crowded by MP-SAE, untested on real SAEs) | Requires complete reframing. Discards 4 rounds of accumulated evidence. Only justified if all current contributions collapse. |

---

## 4. PROCEED vs PIVOT Verdict

**PROCEED** — to paper writing, with Backup A as the immediate pre-writing experiment.

### Rationale

Two of the original contributions have solid-to-strong evidence:

1. **EDA as regime-specific weight-only absorption detector.** 3/8 configs pass across 2 model families. Regime characterization (16k-width, early-mid layers) is itself a finding. The EDA = 1 - DecCos equivalence, while reducing formula novelty, strengthens the metric's connection to existing SAEBench infrastructure and makes it immediately actionable for practitioners.

2. **Three-subtype taxonomy with early-absorption dominance.** The most impactful finding. 72-75% of absorbed latents are early-type (decoder-absent = dictionary coverage failure). This directly challenges the implicit framing of all ITAC-style, OrtSAE, and MP-SAE mitigation approaches, which target the minority late-absorption category. KW p = 0.0002. Threshold sensitivity acknowledged but baseline ordering robust.

The third contribution (H3 cross-domain) has collapsed due to model access limitations. Rather than wait for an external dependency that may not resolve, the dominant strategy is to **run Backup A (amortization gap controlled dictionary experiment)** as the third contribution. Backup A:
- Costs only 1-2 GPU-hours
- Requires no new downloads (uses already-loaded Gemma Scope SAEs)
- Directly adjudicates the Tang et al. vs. O'Neill et al. mechanistic debate
- Connects to the 75% early-dominance finding (is early absorption caused by the encoder or the dictionary?)
- Has 9/10 novelty (never done before)
- Either outcome is publishable and high-impact

### Why not PIVOT?
All three backup pivot options (LCA-SAE, scaling laws, cross-domain with different approach) are strictly dominated by the Backup A supplementation strategy. LCA-SAE requires 3-4 GPU-hours and faces MP-SAE crowding. Scaling laws are already falsified. Waiting for model access is an uncontrolled external dependency. None of these options improve expected paper quality more than Backup A does at 1-2 GPU-hours.

---

## 5. Recommended Next Steps (Priority Order)

### Step 1: Run Backup A — Amortization Gap Controlled Dictionary Experiment (1-2 GPU-hours)

**What:** Load Gemma Scope L12-16k decoder weights (already downloaded). Run three encoding methods on 10,000 tokens:
1. Standard feedforward encoder (original SAE)
2. Orthogonal Matching Pursuit on D (no learned encoder, same K)
3. 2-pass encoder: standard encoder + one residual correction pass

Compute absorption rates for all three encoding methods using first-letter Chanin et al. metric. Compare with paired Wilcoxon signed-rank test.

**Decision gate:**
- If OMP absorption < 50% of feedforward: amortization gap is the dominant cause. **Major finding** — practitioners should use iterative encoding for deployed SAEs.
- If OMP absorption is similar to feedforward: loss landscape (Tang et al.) dominates. **Major finding** — encoder changes alone cannot fix absorption; the dictionary itself is the problem.
- Both outcomes are high-impact. The experiment cannot fail to produce a publishable result.

**Why now:** This is the highest information-gain-per-GPU-hour experiment available. It converts the paper from a two-contribution to a three-contribution paper at minimal cost.

### Step 2: Finalize Paper Framing as Three-Contribution Paper (0 GPU-hours)

After Backup A completes, the paper structure becomes:

**Contribution 1: EDA as regime-specific weight-only absorption screening tool.**
- Formal lower bound theorem (biconvex optimization, Tang et al. 2025).
- Empirical validation: AUROC 0.65-0.78 in 16k-width SAEs at layers 5-12, cross-model (Gemma 2B + GPT-2 Small).
- Regime characterization: failure at 65k-width and deep layers explained by early-absorption prevalence and label density.
- EDA = 1 - DecCos equivalence validated on 3 architectures.
- Limitation: not universal; regime-specific. Frame as: "a diagnostic tool with known operating conditions."

**Contribution 2: Three-subtype taxonomy with early-absorption dominance.**
- Early (72-75%), late (~13%), partial (~13%).
- Early dominance reframes absorption as a dictionary coverage problem, not an encoder alignment problem.
- KW p = 0.0002; late > early EDA ordering at all 5 thresholds.
- Prescriptive implication: wider dictionaries or hierarchically-aware training objectives, not inference-time encoder fixes.
- Threshold sensitivity reported as robustness check.

**Contribution 3: Amortization gap controlled dictionary experiment** (contingent on Step 1 results).
- First controlled dictionary experiment separating encoder from dictionary in absorption causation.
- Connects early-dominance finding to mechanistic theory.

**Supplementary / Negative Results:**
- ITAC proof-of-concept (2.69% FN reduction, H5 falsified) — frames as motivating evidence for early dominance.
- H6 scaling analysis (falsified — supplementary).
- Cross-domain existence evidence (intra-RAVEL rho = 0.924, R3) — acknowledged as suggestive but unvalidated without same-model probes. Flagged as future work.
- Llama-3.1-8B weight-level EDA characterization (3 model families, no AUROC).

### Step 3: Begin Paper Writing

Once Backup A is complete and results are integrated, proceed to writing phase. The writing gate (r4_writing_gate.json: go_write = true) has already been passed.

### Step 4 (Background, Low Priority): Continue Pursuing Model Access

Gemma 2B and Llama-3.1-8B HF access should be pursued in the background. If obtained before submission, re-run:
- EDA with direct Chanin labels on Gemma Scope SAEs (potentially upgrading from 3/8 to 5/8 passing configs).
- RAVEL probes trained on Gemma 2B directly (potentially validating H3 for a camera-ready revision).
- Llama AUROC validation (upgrading from weight-only to full validation).

These are revision-stage improvements, not blocking dependencies.

---

## 6. Resource Allocation Recommendation

| Time Window | Activity | GPU Cost |
|-------------|----------|----------|
| Next 1-2 hours | Run Backup A (amortization gap experiment) | 1-2 GPU-hrs |
| Following 1 hour | Analyze Backup A results, integrate into paper framing | 0 |
| Remainder | Begin paper writing (outline + drafting) | 0 |
| Background | Model access requests (Gemma 2B, Llama-3.1-8B) | 0 |

**Total remaining compute before paper writing: 1-2 GPU-hours.**

---

## 7. Risk Assessment for Recommended Path

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Backup A produces ambiguous result (OMP absorption = 60-80% of feedforward) | Medium | Medium | Report the partial reduction with confidence intervals. Frame as "both mechanisms contribute" — still a finding. |
| Reviewer concern: "EDA is just 1 - DecCos, not novel" | High | High | Acknowledge equivalence upfront. Argue novelty is in (a) formal biconvex lower bound theorem, (b) systematic regime characterization, (c) cross-model empirical validation, and (d) connection to absorption theory. DecCos existed in SAEBench but was never studied as an absorption detector with formal guarantees. |
| Reviewer concern: "Taxonomy rests on only 2 SAE configs" | Medium | High | Report threshold sensitivity prominently. Argue both configs show consistent pattern (75% early). N=65 at L12-65k provides reasonable statistical power. N=16 at L12-16k acknowledged as underpowered but consistent. |
| Reviewer concern: "No cross-domain validation" | Medium | High | Report as a limitation and future work. The intra-RAVEL rho = 0.924 is mentioned as suggestive. Backup A + taxonomy are sufficient for a two/three-contribution paper without cross-domain. |
| Backup A experiment fails to run (code bugs, GPU issues) | Low | Low | Fallback: proceed with two-contribution paper. The taxonomy finding alone justifies a mid-tier publication. |
| "Thin paper" rejection at top venues | Medium | Medium | Backup A as third contribution materially strengthens the paper. If Backup A produces a clean result, the paper has three genuine contributions: (1) detection, (2) characterization/taxonomy, (3) mechanistic explanation. This is sufficient for NeurIPS/ICLR MI Workshop or EMNLP main. |

---

## 8. Venue Targeting Update

| Scenario | Venue Target |
|----------|-------------|
| Two contributions only (EDA + taxonomy) | EMNLP 2026 main or NeurIPS 2026 MI Workshop |
| Three contributions (+ Backup A clean result) | EMNLP 2026 main (strong) or NeurIPS 2026 main (competitive) |
| Three contributions + model access obtained + H3 validated | NeurIPS 2026 main or ICLR 2027 (strongest possible framing) |

---

## Summary

The project has reached a clear decision point. After R4, the evidence landscape is:
- **Two robust contributions**: EDA regime-specific detector (moderate signal, cross-model) and three-subtype taxonomy with early-dominance (strong signal, highest impact).
- **One collapsed contribution**: H3 cross-domain absorption (model access dependency, unresolvable in short term).
- **Three falsified hypotheses**: D-EDA improvement (H2), ITAC efficacy (H5), scaling sign reversal (H6).
- **One high-value untested experiment**: Backup A (amortization gap, 1-2 GPU-hours, either outcome publishable).

The dominant strategy is crystal clear: **run Backup A immediately, then write the paper.** This maximizes expected paper quality per GPU-hour invested. Any alternative — waiting for model access, pivoting to a new idea, running more EDA configs — has strictly lower expected return. The project should spend its remaining compute budget on Backup A and nothing else before entering the writing phase.
