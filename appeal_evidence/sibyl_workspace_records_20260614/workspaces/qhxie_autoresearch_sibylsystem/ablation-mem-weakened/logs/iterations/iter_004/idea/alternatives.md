# Backup Ideas for Pivot

## Alternative A: Rigorous Null-Result Study with Methodological Insights (Fallback)

**Status:** Strong fallback, always available.

**Core claim:** "Feature Absorption Does Not Significantly Degrade Steering or Probing: A Methodological Framework for SAE Evaluation"

**Why it works:**
- The data is honest and the methodology is sound (random baseline, MCP, precision-recall, EC50).
- Null results are increasingly accepted in ML when methodological rigor is high.
- Prevents wasted effort on similar studies.
- Reusable evaluation framework (baseline correction, precision-recall decomposition, EC50).

**Content:**
- Report all null results honestly with exact statistics.
- Emphasize methodological contributions.
- Include H6 falsification as valuable negative result.
- Discuss implications: absorption may not be the right target for architectural innovation.

**Venue:** Workshop (ICML MiF, NeurIPS XAI) or arXiv preprint.

**Risk:** May be dismissed as "we found nothing." Needs strong framing.

**When to pivot:** If optimal compression framing is rejected by reviewers.

---

## Alternative B: Random SAE Baseline Comparison

**Status:** Moderate backup, addresses methodological crisis.

**Core claim:** Current feature absorption metrics may measure dictionary-size artifacts rather than genuine SAE behavior. Systematic comparison of trained vs. random/frozen SAE baselines.

**Why it works:**
- Directly responds to Korznikov et al. (2026) "Sanity Checks for SAEs."
- Either outcome advances the field: if metrics distinguish trained from random, validated; if not, field must develop better metrics.
- Clean, falsifiable experimental design.

**Experiments:**
1. Generate frozen-decoder and pure-random baselines for GPT-2 Small (~30 min)
2. Run identical absorption metric on trained, frozen, and random SAEs (~1 hr)
3. Statistical comparison: t-test, Cohen's d, bootstrap CIs (~15 min)

**Risk:** If random baselines match trained SAEs, paper undermines absorption subfield --- including own prior work. Intellectually honest but uncomfortable.

**When to pivot:** If metric validity concerns emerge during review.

---

## Alternative C: The Absorption-Pathology Trade-off as a Pareto Frontier (Theoretical)

**Status:** Moderate backup, theoretically ambitious.

**Core claim:** For any SAE architecture, there exists a Pareto frontier in (reconstruction error, sparsity, absorption rate) space. No SAE can simultaneously achieve zero reconstruction, minimal sparsity, and zero absorption.

**Why it works:**
- Generalizes sparsity-reconstruction trade-off to include absorption as third axis.
- Provides unifying framework for comparing SAE architectures.
- Project data (absorption coexists with functional steering) consistent with frontier permitting functional performance.

**Experiments:**
1. Measure (R, S, A) for 5+ SAE architectures on GPT-2 Small layer 8 (~1 hr)
2. Test if points lie on 2D manifold (PCA dimensionality test) (~15 min)
3. Show absorption-reducing interventions increase other pathologies (~30 min)

**Risk:** Pareto frontier may not be smooth; outlier architectures may break manifold claim. Theory elegant but may not be empirically tight.

**When to pivot:** If optimal compression claim needs stronger empirical support.

---

## Alternative D: Scaling Laws for Feature Absorption Rate

**Status:** Weak backup, requires more data.

**Core claim:** Feature absorption rate follows scaling law: A(N) = alpha_0 * N^{beta_N} * exp(-gamma * N / N^*(L)), where N is dictionary size and N^*(L) is critical dictionary size depending on layer depth.

**Why it works:**
- First scaling law for absorption rate.
- Predicts non-monotonic inverted-U curve.

**Experiments:**
1. Measure absorption on GemmaScope SAEs at 16K and 65K (layer 12) (~30 min)
2. Fit preliminary scaling curve (~15 min)
3. Validate on additional dictionary sizes if available (~30 min)

**Risk:** Power-law form is phenomenological, not derived from first principles. Measurement noise may bias estimates.

**When to pivot:** If scaling patterns are interesting but other directions fail.

---

## Alternative E: Encoder-Correlation-Based Absorption Prediction

**Status:** Exploratory, follows from H6 falsification.

**Core claim:** Since decoder correlations do NOT predict absorption (H6 falsified), perhaps encoder correlations (W_enc^T W_enc) or encoder-decoder cross-correlations do.

**Why it works:**
- H6 falsification implies absorption dynamics are in the encoder, not the decoder.
- Encoder correlations capture competitive dynamics during inference.
- Testable with existing data.

**Experiments:**
1. Compute encoder correlation matrix for GPT-2 Small SAE.
2. Test precision@k against absorption pairs (same protocol as H6).
3. Compare encoder vs. decoder prediction power.

**Risk:** May also fail; the mechanism may be more complex than simple correlations.

**When to pivot:** If reviewers ask "what does predict absorption?" after H6 falsification.
