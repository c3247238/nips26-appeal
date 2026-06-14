# Backup Ideas for Pivot

## Alternative A: Feature Absorption as Optimal Compression -- A Trade-off Analysis

**Status:** Strong backup, directly supported by existing data.

**Core claim:** Feature absorption is not a pathology to fix but a predictable consequence of the interpretability-compression trade-off. Absorption-fixing interventions (Matryoshka, OrtSAE, ATM) increase other pathologies (dead neurons, hedging, reconstruction error).

**Why it works:**
- The project's data already shows absorption coexists with functional steering and perfect precision --- consistent with absorption being a benign byproduct of optimal compression.
- No new experiments needed for the core claim; existing data supports it.
- The contrarian angle is intellectually fresh in a field that treats absorption as a uniform pathology.

**Experiments:**
1. Compare standard SAE vs OrtSAE on GPT-2 (layer 8): measure absorption + dead neuron rate + reconstruction error (~1 GPU-hr)
2. Fit Pareto frontier for (absorption, reconstruction, dead neurons) (~30 min)
3. Cross-architecture comparison using SAEBench data (~30 min)

**Risk:** May be seen as "we found nothing, so we reframed it." Needs strong theoretical hook.

**When to pivot:** If H1 (graph prediction) is not validated.

---

## Alternative B: Validating Feature Absorption Metrics Against Random Baselines

**Status:** Moderate backup, addresses methodological crisis.

**Core claim:** Current feature absorption metrics (Chanin et al., SAEBench) may measure dictionary-size artifacts rather than genuine SAE behavior. Systematic comparison of trained vs. random/frozen SAE baselines tests this.

**Why it works:**
- Directly responds to Korznikov et al. (2026) "Sanity Checks for SAEs" --- the most threatening concurrent work.
- Either outcome advances the field: if metrics distinguish trained from random, they are validated; if not, the field must develop better metrics.
- Clean, falsifiable experimental design.

**Experiments:**
1. Generate frozen-decoder and pure-random baselines for GPT-2 Small (~30 min)
2. Run identical absorption metric on trained, frozen, and random SAEs (~1 hr)
3. Statistical comparison: t-test, Cohen's d, bootstrap CIs (~15 min)

**Risk:** If random baselines match trained SAEs, the paper undermines the entire absorption subfield --- including the project's own prior work. This is intellectually honest but may be uncomfortable.

**When to pivot:** If metric validity concerns emerge during inhibition graph experiments.

---

## Alternative C: The Absorption-Pathology Trade-off as a Pareto Frontier (Theoretical)

**Status:** Moderate backup, theoretically ambitious.

**Core claim:** For any SAE architecture, there exists a Pareto frontier in (reconstruction error, sparsity, absorption rate) space. No SAE can simultaneously achieve zero reconstruction, minimal sparsity, and zero absorption. All existing architectures map to distinct points on this frontier.

**Why it works:**
- Generalizes the sparsity-reconstruction trade-off to include absorption as a third axis.
- Provides a unifying framework for comparing SAE architectures.
- The project's data (absorption coexists with functional steering) is consistent with the frontier permitting functional performance.

**Experiments:**
1. Measure (R, S, A) for 5+ SAE architectures on GPT-2 Small layer 8 (~1 hr)
2. Test if points lie on a 2D manifold (PCA dimensionality test) (~15 min)
3. Show that absorption-reducing interventions increase other pathologies (~30 min)

**Risk:** The Pareto frontier may not be smooth; outlier architectures may break the manifold claim. The theory is elegant but may not be empirically tight.

**When to pivot:** If the inhibition graph provides partial validation but the repair mechanism fails.

---

## Alternative D: Scaling Laws for Feature Absorption Rate

**Status:** Weak backup, requires more data.

**Core claim:** Feature absorption rate follows a scaling law: A(N) = alpha_0 * N^{beta_N} * exp(-gamma * N / N^*(L)), where N is dictionary size and N^*(L) is a critical dictionary size that depends on layer depth.

**Why it works:**
- First scaling law for absorption rate.
- Predicts non-monotonic inverted-U curve (absorption increases then decreases with dictionary size).

**Experiments:**
1. Measure absorption on GemmaScope SAEs at 16K and 65K (layer 12) (~30 min)
2. Fit preliminary scaling curve (~15 min)
3. Validate on additional dictionary sizes if available (~30 min)

**Risk:** Power-law form is phenomenological, not derived from first principles. Measurement noise may bias estimates.

**When to pivot:** If scaling patterns are interesting but inhibition graph fails.

---

## Alternative E: Write a Rigorous Null-Result Paper

**Status:** Fallback, always available.

**Core claim:** "Feature Absorption Does Not Significantly Degrade Steering or Probing: A Cautionary Tale About SAE Pathology Metrics"

**Why it works:**
- The data is honest and the methodology is sound (after fixing H3 bug and adding baselines).
- Null results are increasingly accepted in ML when methodological rigor is high and the question is important.
- Prevents wasted effort on similar studies.

**Content:**
- Report all null results honestly with exact statistics.
- Emphasize methodological contributions: baseline correction, precision-recall decomposition, power analysis.
- Discuss implications for the field: absorption may not be the right target for architectural innovation.

**Venue:** Workshop (ICML MiF, NeurIPS XAI) or arXiv preprint.

**Risk:** May be dismissed as "we found nothing." Needs strong framing.

**When to pivot:** If all other alternatives fail or produce null results.
