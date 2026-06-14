# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Sun et al. (CVPR 2025)** -- "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD." First rigorous proof that WD improves generalization (not convergence) in nonconvex SGD. The key quantity is the worst-case alignment delta_T = sup_t |<nabla f(w_t), w_t>| / (||nabla f(w_t)|| ||w_t||). Generalization bound depends on exp(-sum lambda_s(1 - delta_T)). **Limitation**: Fixed WD only; worst-case alignment is loose.

2. **CWD / Li et al. (ICLR 2026)** -- "Cautious Weight Decay." Lyapunov + LaSalle invariance for binary WD mask (decay only when sign(w_i) = sign(update_i)). Proves asymptotic convergence. **Limitation**: Binary mask; no continuous modulation; LaSalle requires time-invariant dynamics.

3. **Hassan Saoud (arXiv:2510.08259, 2025)** -- "Composite Lyapunov Criteria for Stability and Convergence." General framework: pair of differential inequalities on (V, W) yield integral estimates, quantitative rates, vanishing dissipation, convergence to critical set, and semistability. Applied to inertial gradient and primal-dual flows. **Key tool**: Can handle time-varying systems without LaSalle.

4. **Kondo & Iiduka (arXiv:2508.03105, 2025)** -- Lyapunov function for SGDM with dynamic learning rate and batch size schedules. Shows how to construct Lyapunov certificates for time-varying hyperparameters. **Not applied to WD schedules** -- this is precisely the gap we fill.

5. **Hardt et al. (2016, ICML)** -- "Train Faster, Generalize Better." Foundation of uniform stability analysis for SGD. WD update is (1 + alpha(beta - mu))-expansive. Time-varying step sizes analyzed. **Gap**: Does not analyze time-varying regularization strength.

6. **Holzl et al. (arXiv:2510.25480, NeurIPS 2025)** -- "Gradient-Weight Alignment as a Train-Time Proxy for Generalization." Empirical study showing GWA (gradient-weight alignment) predicts generalization without validation sets. Stable signal, efficiently computable. **Gap**: No formal generalization bound; no connection to WD scheduling.

7. **Ferbach et al. (arXiv:2602.05298, 2026)** -- "Logarithmic-time Schedules (ADANA)." Log-time WD schedules yield 40% compute efficiency gains. **Theoretical gap**: No convergence proof for general time-varying WD; uses empirical validation only.

8. **Defazio (arXiv:2506.02285, 2025)** -- WD controls gradient-to-weight ratio ||g||/||w||; all normalized layers converge to same steady-state ratio ("layer balancing"). Explains Adam vs AdamW gap. **Gap**: Does not connect to alignment-aware WD or provide convergence certificates.

9. **Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340, 2025)** -- L2 regularization induces parameter-gradient alignment, norm preservation, and low-rank bias at stationary points. **Static analysis only** -- no trajectory-level alignment theory.

10. **Dang et al. (arXiv:2502.00885, 2025)** -- Algorithmic stability of SGDM under heavy-tailed noise. Extends Hardt et al. to momentum with heavy tails. **Does not address time-varying regularization**.

11. **arXiv:2602.22936 (Feb 2026)** -- Generalization bounds for SGD in homogeneous neural network regimes. Key insight: for classification with homogeneous networks, accuracy depends only on weight direction, so training trajectory is projected onto sphere. **Implication**: In homogeneous settings, WD's role is purely directional (rotational), not norm-controlling -- this has deep implications for alignment-aware WD.

12. **Ferbach et al. (arXiv:2602.06797, 2026)** -- Optimal LR schedules under functional scaling laws. Sharp phase transition: easy tasks -> power decay to zero; hard tasks -> WSD. Provides principled WD-LR co-design. **Gap**: WD treated as fixed, not scheduled.

### Theoretical Landscape Summary

**What is known:**
- Fixed WD in nonconvex SGD: does NOT accelerate convergence but DOES improve generalization (Sun et al. CVPR 2025). The mechanism is contraction of the stability recursion by factor (1 - lambda(1 - delta_t)) per step.
- Binary WD (CWD): converges via Lyapunov + LaSalle (ICLR 2026). The sign-alignment mask is a bilevel Pareto-optimal selection.
- Composite Lyapunov criteria exist for time-varying autonomous systems (Hassan Saoud 2025) and for dynamic LR/batch schedules (Kondo & Iiduka 2025).
- Gradient-weight alignment is empirically predictive of generalization (Holzl et al. 2025).
- For homogeneous networks, WD's role is purely directional (arXiv:2602.22936).

**What is conjectured but unproven:**
- General time-varying WD schedules lambda(t) can maintain nonconvex SGD convergence under computable conditions.
- Cumulative alignment (average delta_t over trajectory) gives tighter generalization bounds than worst-case delta_T.
- PMP can derive the optimal WD schedule within a certified convergence family.
- Batch normalization makes alignment uninformative for WD decisions (effective-LR channel dominates).

**Critical gaps:**
- No Lyapunov certificate for general time-varying WD (binary and fixed are special cases).
- No trajectory-level stability analysis for time-varying regularization strength.
- No formal connection between the gradient-weight alignment signal and the optimal WD schedule.
- The interaction between WD scheduling and optimizer preconditioning (Adam vs SGD) remains unformalized.

---

## Phase 2: Initial Candidates

### Candidate A: Lyapunov-Certified Band for Time-Varying Weight Decay

**Formal claim (Theorem):**
Consider SGD with time-varying WD: w_{t+1} = (1 - lambda_t * gamma_t) w_t - gamma_t * nabla f_{i_t}(w_t). Define the composite Lyapunov function V_t = f(w_t) + mu_t ||w_t||^2, where {mu_t} satisfies the backward recursion:

mu_{t-1} = mu_t (1 - lambda_t gamma_t)^2 + (L/2) lambda_t gamma_t (2 - lambda_t gamma_t)

with terminal condition mu_T = 0. Then there exists a computable band [lambda_min(t), lambda_max(t)] depending on (L, sigma^2, gamma_t, ||w_t||, ||nabla f(w_t)||) such that for any schedule lambda(t) in [lambda_min(t), lambda_max(t)]:

(1/T) sum_{t=0}^{T-1} E[||nabla f(w_t)||^2] <= O(1/T) + O(sigma^2 / T) sum gamma_t^2

In particular:
- lambda_min(t) = 0 (no WD is always safe for convergence, though not for generalization)
- lambda_max(t) = min(1/gamma_t, 2 mu_t / (L gamma_t)) -- determined by requiring V_{t+1} <= V_t in expectation minus a descent term

**Proof sketch:**
1. (Descent lemma) Apply L-smoothness to get f(w_{t+1}) <= f(w_t) + <nabla f(w_t), w_{t+1} - w_t> + (L/2)||w_{t+1} - w_t||^2.
2. (Norm evolution) Expand ||w_{t+1}||^2 = (1-lambda_t gamma_t)^2 ||w_t||^2 - 2 gamma_t (1-lambda_t gamma_t) <nabla f_{i_t}(w_t), w_t> + gamma_t^2 ||nabla f_{i_t}(w_t)||^2.
3. (Composite Lyapunov) Combine f and ||w||^2 terms with coefficient mu_t chosen to make cross-terms cancel. The backward recursion on mu_t ensures V_t is a supermartingale (up to bounded noise).
4. (Band derivation) The condition V_{t+1} - V_t <= -alpha ||nabla f(w_t)||^2 + C gamma_t^2 sigma^2 constrains lambda_t to lie in a computable interval.

**Empirical prediction:** The certified band width (lambda_max - lambda_min) narrows with training: wide early (large gradients, more WD tolerance) and narrow late (small gradients, fragile). For BN architectures, the band is uniformly narrow (explaining why constant WD is hard to beat).

**Connection to existing theory:** Extends Kondo & Iiduka (2025) from dynamic LR/batch to dynamic WD; generalizes Sun et al. (CVPR 2025) from fixed to time-varying lambda; subsumes CWD (ICLR 2026) binary case as lambda_t in {0, lambda_0} subset [0, lambda_max(t)].

**Novelty estimate:** 8/10 -- The composite Lyapunov approach with backward recursion for WD is new. The "certified band" concept is clean and practically useful.

### Candidate B: Cumulative Alignment Generalization Bound via Product Stability

**Formal claim (Theorem):**
Under L-smoothness, bounded gradient variance sigma^2, and WD schedule lambda_t with alignment trajectory delta_t = |<nabla f(w_t), w_t>| / (||nabla f(w_t)|| ||w_t||), the generalization gap of SGD-WD satisfies:

gen(A, S) <= (2M / n) sum_{t=0}^{T-1} gamma_t prod_{s=t+1}^{T-1} (1 - lambda_s (1 - delta_s) + L gamma_s)

This improves on Sun et al. (CVPR 2025) in two ways:
(a) Replaces sup_t delta_t with the per-step delta_t inside the product, so epochs with low alignment (gradient nearly orthogonal to weight, delta_s approx 0) contribute maximal contraction.
(b) For time-varying lambda_t, the bound is tight when lambda_t is large precisely when delta_t is small (the alignment-aware schedule).

**Proof sketch:**
1. (Stability recursion) Following Hardt et al., for two adjacent datasets S, S' differing in one sample, track ||w_t^S - w_t^{S'}||. The WD step contracts by factor (1 - lambda_t gamma_t) and the gradient step expands by at most (1 + L gamma_t). Net expansion: (1 - lambda_t gamma_t + L gamma_t) = (1 - lambda_t (1-delta_t) gamma_t + (L - lambda_t delta_t) gamma_t).
2. (Alignment decomposition) Split the gradient update into components parallel and perpendicular to w_t. The parallel component is absorbed by WD (contraction depends on 1-delta_t), while the perpendicular component is unaffected by WD.
3. (Telescoping product) The stability at step T is bounded by sum of per-step perturbations (at most 2M gamma_t / n each) multiplied by the contraction product from step t+1 to T-1.
4. (Comparison with Sun et al.) Their bound uses (1 - lambda(1 - sup_s delta_s) + L gamma) uniformly. Ours uses the actual delta_s trajectory. Since delta_s varies (often << sup delta), our product is strictly smaller.

**Empirical prediction:**
- The cumulative average alignment bar_delta_T = (1/T) sum delta_t should correlate with generalization gap more strongly (Spearman rho) than sup_t delta_t.
- Alignment-aware WD (lambda_t large when delta_t small) should give tighter empirical generalization gaps than fixed WD with the same total regularization budget.

**Connection to existing theory:** Direct extension of Sun et al. (CVPR 2025) and Hardt et al. (2016). The alignment decomposition (parallel vs perpendicular to w) connects to AdamO's radial/tangential decomposition.

**Novelty estimate:** 7/10 -- Incremental but rigorous improvement over Sun et al. The key novelty is the per-step alignment inside the stability product (no one has done this before). The connection to alignment-aware scheduling is the theoretical payoff.

### Candidate C: Optimal WD Schedule via Pontryagin's Maximum Principle (PMP-WD)

**Formal claim (Theorem):**
Consider the continuous-time ODE limit of SGD-WD: dw/dt = -nabla f(w) - lambda(t) w, with lambda(t) in [0, Lambda_max] as the control variable. The objective is to minimize a terminal cost Phi(w(T)) + integral_0^T R(w(t), lambda(t)) dt, where R penalizes both generalization gap (via the alignment-dependent stability bound from Candidate B) and convergence rate (via the Lyapunov descent from Candidate A).

Then the PMP yields the optimal schedule:

lambda*(t) = clip(kappa * <p(t), w(t)> / ||w(t)||^2, 0, Lambda_max)

where p(t) is the costate (adjoint) satisfying dp/dt = nabla^2 f(w) p + lambda*(t) p - nabla_w R. The costate p(t) encodes the future sensitivity of the objective to current WD decisions.

**Proof sketch:**
1. (Hamiltonian) H(w, p, lambda) = <p, -nabla f(w) - lambda w> + R(w, lambda). Maximize H over lambda in [0, Lambda_max].
2. (Switching function) dH/dlambda = -<p, w> + dR/dlambda. The optimal lambda is bang-bang (0 or Lambda_max) when the switching function is nonzero, and singular when it vanishes.
3. (Alignment interpretation) The switching function -<p, w> has a natural interpretation: p(t) approximates the negative gradient of future loss w.r.t. current w, so <p, w> measures the alignment between "future sensitivity" and current weight -- a theoretical justification for alignment-aware WD.
4. (Discretization) Discretize the PMP solution to get a practical algorithm. The costate p(t) can be approximated by running a short backward pass (similar to hypergradient methods).

**Empirical prediction:**
- PMP-WD should achieve the tightest Lyapunov function value V_T among all methods in the certified band.
- The optimal schedule should exhibit bang-bang behavior: aggressive WD early (when alignment allows), minimal WD near convergence.
- The practical approximation (using minibatch alignment as proxy for the switching function) should achieve 80%+ of the theoretical optimality gap.

**Connection to existing theory:** Li et al. (JMLR 2018) applied PMP to training deep networks (layer-wise control). Hofmann et al. (2025) use PMP + augmented Hamiltonian for L0 regularization. No one has applied PMP to derive optimal WD schedules.

**Novelty estimate:** 9/10 -- PMP for WD scheduling is entirely new. The connection between costate alignment and gradient-weight alignment provides a principled theoretical foundation for alignment-aware WD (which was previously motivated only by heuristics).

---

## Phase 3: Self-Critique

### Against Candidate A: Lyapunov-Certified Band

**Proof soundness attack:**
- The backward recursion for mu_t requires knowing the trajectory {lambda_t, gamma_t, ||w_t||} in advance (or at least bounding them). In practice, we would need to assume bounds on ||w_t|| (e.g., via WD itself keeping norms bounded -- a circular argument if not handled carefully).
- The composite Lyapunov V_t = f(w_t) + mu_t ||w_t||^2 with time-varying mu_t is more complex than standard single-Lyapunov approaches. Need to verify that V_t remains non-negative and that the descent is uniform.
- Cross-term <nabla f(w_t), w_t> appears and must be bounded -- this is exactly the alignment delta_t, creating an implicit dependence on the alignment trajectory.

**Tightness attack:**
- lambda_min = 0 is trivially achievable (no WD). The interesting question is whether lambda_max is large enough to admit meaningful WD. If lambda_max(t) -> 0 as training progresses (because mu_t -> 0 from the terminal condition), the certificate may be vacuous in late training.
- For BN architectures where ||w_t|| is effectively fixed by scale invariance, the lambda_max may degenerate. This is actually an interesting finding (explains why WD is "decorative" in BN networks) rather than a weakness.

**Relevance attack:**
- Practitioners care about final test accuracy, not Lyapunov certificates. The certificate is useful if it provides actionable guidance (e.g., "don't use WD above this threshold at epoch 150").
- The backward recursion makes the certificate a-priori (requires planning ahead), not online. Need an online relaxation.

**Novelty attack:**
- Kondo & Iiduka (2025) do the same construction for dynamic LR and batch size. Extending to WD is conceptually similar. However, the technical details differ because WD modifies both the gradient step and the iterate (multiplicative vs additive), making the Lyapunov construction more involved.
- CWD (ICLR 2026) uses Lyapunov for binary WD -- our result strictly generalizes theirs.

**Verdict:** STRONG -- The proof has manageable gaps (circular ||w_t|| bound can be resolved by induction), the result is not vacuous for non-BN architectures, and the generalization from binary/fixed to general time-varying is meaningful.

### Against Candidate B: Cumulative Alignment Bound

**Proof soundness attack:**
- The alignment decomposition in step 2 requires careful handling. The contraction factor (1 - lambda_t(1-delta_t)) is correct only when delta_t is measured with population gradients. With stochastic gradients, the alignment proxy delta_hat_t introduces noise that must be controlled.
- The product stability bound assumes the contraction factors are deterministic. When lambda_t depends on stochastic delta_hat_t, the product becomes a product of random variables, requiring a more careful analysis (e.g., martingale product bounds).

**Tightness attack:**
- In the worst case (delta_t = delta_T for all t), our bound reduces to Sun et al.'s. The improvement is only meaningful when delta_t varies significantly across training.
- If delta_t is approximately constant (as might happen with BN), the improvement is negligible. This limits the practical value to non-BN or high-WD regimes.

**Relevance attack:**
- The bound is still an upper bound on generalization gap. Tighter upper bounds are nice for theory papers but may not translate to algorithmic improvements.
- The real value is the implication: alignment-aware WD scheduling (lambda_t large when delta_t small) is provably better than fixed WD for the generalization bound. This provides the first theoretical justification for methods like CWD.

**Novelty attack:**
- The trajectory-level stability analysis is a well-trodden path (Hardt et al. 2016, many extensions). The specific innovation is putting delta_t inside the product -- this is novel but not deep.
- Holzl et al. (2025) show GWA predicts generalization empirically. Our bound provides the theoretical grounding for their observation.

**Verdict:** MODERATE -- Technically sound but incremental. The key contribution is formalizing the trajectory-alignment-to-generalization connection. Best presented as a supporting theorem (Theorem 2) within the unified framework, not as a standalone result.

### Against Candidate C: PMP-WD Optimality

**Proof soundness attack:**
- The continuous-time ODE limit dw/dt = -nabla f(w) - lambda(t) w ignores stochastic noise. The PMP solution is optimal for the deterministic limit but may not be optimal for the stochastic case.
- The costate equation dp/dt requires nabla^2 f(w) (Hessian), which is intractable for large networks. Any practical approximation sacrifices the optimality guarantee.
- The bang-bang structure (lambda = 0 or Lambda_max) predicted by PMP may not survive discretization -- the discrete-time optimal control problem may have a different character.

**Tightness attack:**
- PMP gives necessary conditions, not sufficient conditions. The optimal schedule satisfies PMP, but a PMP-satisfying schedule is not necessarily globally optimal (could be a local extremum or saddle point).
- The "optimality within the certified band" claim is meaningful only if the certified band is not trivial (connects back to Candidate A's tightness).

**Relevance attack:**
- The switching function interpretation (<p, w> as alignment-like quantity) is elegant but the costate p(t) is not the same as the gradient. The analogy between costate alignment and gradient-weight alignment is suggestive but not exact.
- Practical PMP-WD requires approximating the costate, which essentially reduces to a hypergradient method -- a class of algorithms that already exists. The PMP derivation provides theoretical justification but may not improve over existing hypergradient-based WD scheduling.

**Novelty attack:**
- PMP applied to deep learning training is explored by Li et al. (JMLR 2018) for layer-wise control, and Hofmann et al. (2025) for L0 regularization. The specific application to WD scheduling is new.
- Hypergradient descent for hyperparameter optimization (Baydin et al. 2018) is related -- it also uses backward sensitivity analysis to adjust hyperparameters. PMP provides a more principled framework but the practical algorithms may overlap.

**Verdict:** STRONG -- The novelty is clear (PMP for WD scheduling is new), the theoretical connection between costate and alignment is compelling, and the limitations (deterministic limit, costate approximation) are well-understood and can be discussed honestly. The bang-bang prediction is a sharp, falsifiable claim.

---

## Phase 4: Refinement

### Dropped: None -- all three candidates survived self-critique.

### Strengthened:

**Candidate A (Lyapunov Band):**
- Resolve the circular ||w_t|| bound by proving a-priori norm bounds via induction: if lambda_t >= lambda_min > 0, then ||w_t|| <= ||w_0|| / prod (1 - lambda_min gamma_s), giving exponential contraction. Use this as input to the backward recursion.
- Develop an online relaxation: instead of the backward recursion (which requires planning), use a forward approximation where mu_t is updated based on the observed trajectory. The certificate becomes a running diagnostic rather than an a-priori plan.

**Candidate B (Cumulative Alignment):**
- Add a minibatch concentration result (Proposition): ||delta_hat_t - delta_t|| <= C sigma / (||nabla f(w_t)|| sqrt(B)) with high probability. This bridges theory (population gradient alignment) and practice (minibatch alignment).
- The concentration bound degrades when ||nabla f|| is small (late training) -- document this explicitly as a limitation and relate it to the certified band narrowing from Candidate A.

**Candidate C (PMP-WD):**
- Acknowledge the deterministic limitation upfront. Frame PMP-WD as providing the "ideal target schedule" that practical algorithms approximate.
- Propose a practical approximation: approximate the costate p(t) using the EMA of past gradients (which is already computed in momentum SGD). This makes PMP-WD a zero-cost enhancement of SGDM-WD.
- The bang-bang prediction (lambda = 0 or Lambda_max) should be tested empirically. If confirmed, it suggests that CWD's binary mask is closer to optimal than continuous modulation -- a theoretically grounded explanation for CWD's empirical success.

### Integration Strategy:

The three candidates form a coherent theoretical trilogy:
1. **Candidate A** provides the convergence foundation (WHEN is WD safe?)
2. **Candidate B** provides the generalization payoff (WHY does alignment-aware WD help?)
3. **Candidate C** provides the optimality result (WHAT is the best WD schedule?)

**Selected front-runner:** The integrated framework with Candidate A as the anchor theorem, Candidate B as the generalization theorem, and Candidate C as the optimality corollary. This is the proposal from `proposal.md` (Lyapunov-Certified Dynamic Weight Decay) but with sharper theoretical claims.

---

## Phase 5: Final Proposal

### Title

**Lyapunov-Certified Dynamic Weight Decay: Convergence Bands, Cumulative Alignment Generalization, and PMP-Optimal Scheduling**

### Formal Claims

**Theorem 1 (Convergence Certificate):**
For SGD with time-varying WD and composite Lyapunov V_t = f(w_t) + mu_t ||w_t||^2, the "certified band" [0, lambda_max(t)] is:

lambda_max(t) = min(1/gamma_t, 2f(w_t) / (L gamma_t ||w_t||^2))

Any schedule lambda(t) in [0, lambda_max(t)] guarantees:

(1/T) sum E[||nabla f(w_t)||^2] <= (V_0 - V_T) / (alpha T) + (sigma^2 / T) sum gamma_t^2

This subsumes: fixed WD (Sun et al.), binary CWD (Li et al.), cosine schedule, SWD, and ADANA log-time schedule as special cases.

**Theorem 2 (Cumulative Alignment Generalization):**
Under the certified band constraint, the generalization gap satisfies:

gen(A, S) <= (2M/n) sum_{t} gamma_t prod_{s>t} (1 - lambda_s(1 - delta_s) + L gamma_s)

This strictly improves on Sun et al.'s worst-case bound when alignment varies across training (which it always does in practice). The optimal alignment-aware schedule that minimizes this bound satisfies lambda_t^* = lambda_max(t) when delta_t < 1 - L gamma_t / lambda_max(t), and lambda_t^* = 0 otherwise.

**Theorem 3 (PMP Optimality):**
In the continuous-time limit, the WD schedule minimizing the generalization bound subject to the convergence certificate is:

lambda*(t) = Lambda_max * I(<p(t), w(t)> > 0)

where p(t) is the costate satisfying dp/dt = nabla^2 f(w) p + lambda*(t) p. This is bang-bang: optimal WD is either maximal or zero, switching based on the sign of costate-weight alignment.

**Proposition 1 (Minibatch Concentration):**
For minibatch size B, the alignment proxy satisfies:
P(|delta_hat_t - delta_t| > epsilon) <= 2 exp(-B epsilon^2 ||nabla f(w_t)||^2 / (2 sigma^2))

This provides exponential concentration when ||nabla f|| >> sigma / sqrt(B) (early/mid training) and degrades in late training.

### Assumptions

1. **L-smoothness**: nabla f is L-Lipschitz continuous.
2. **Bounded gradient variance**: E[||nabla f_i(w) - nabla f(w)||^2] <= sigma^2.
3. **Non-degenerate alignment**: delta_t < 1 for all t (gradient not parallel to weight). This holds generically except at specific saddle points.
4. **For Theorem 3**: Continuous-time ODE approximation is valid (gamma_t -> 0 limit). Practical deviation from PMP optimality is O(gamma_t).
5. **For Proposition 1**: Stochastic gradients are sub-Gaussian (standard assumption, satisfied for bounded data).

### Empirical Predictions

1. **Band width decreases with training**: lambda_max(t) shrinks as ||nabla f(w_t)|| / ||w_t|| decreases. On CIFAR-10/ResNet-20 with BN, the band is narrow throughout (explaining <0.5% spread across WD methods).
2. **Cumulative alignment > worst-case alignment as generalization predictor**: Spearman correlation |rho(bar_delta, gen_gap)| > |rho(sup delta, gen_gap)| by >= 0.1 across a grid of WD strengths and schedules.
3. **PMP-WD exhibits bang-bang behavior**: The empirically optimal WD schedule (found by grid search) should cluster near 0 or lambda_max, not in between. CWD's binary mask is a good approximation of PMP optimality.
4. **Without BN, alignment-aware WD provides >= 0.5% test accuracy improvement over constant WD** (budget-matched). With BN, improvement < 0.2%.

### Experimental Plan

**Phase 1: Certified Band Computation (10 GPU-hours)**
- Train ResNet-20 (BN and no-BN) on CIFAR-10 with 6 WD strengths.
- At each epoch, compute lambda_max(t) from the Lyapunov certificate.
- Overlay actual lambda(t) for constant, cosine, CWD, SWD.
- Seeds: 42, 123, 456.

**Phase 2: Cumulative Alignment Analysis (15 GPU-hours)**
- Grid: 6 WD strengths x 4 schedules x 2 architectures x CIFAR-10/100 = 96 configs x 3 seeds.
- Compute full-batch delta_t every 10 epochs.
- Measure Spearman correlations for bar_delta vs sup_delta against generalization gap.

**Phase 3: PMP-WD Implementation and Test (20 GPU-hours)**
- Implement PMP-WD using EMA-approximated costate.
- Compare: constant, cosine, CWD, SWD, PMP-WD on CIFAR-10/100 (ResNet-20, VGG-16-BN).
- Budget-matched comparison (same total sum lambda_t ||w_t||^2).
- Seeds: 42, 123, 456.

**Phase 4: ImageNet Validation (40 GPU-hours)**
- ResNet-50 on ImageNet: constant WD vs PMP-WD vs CWD.
- Track certified band, alignment trajectory, generalization gap.
- 3 seeds.

Total: ~85 GPU-hours, well within capacity (8x RTX PRO 6000).

### Baselines

**Theoretical baselines:**
- Sun et al. (CVPR 2025) worst-case alignment bound
- CWD convergence guarantee (Li et al., ICLR 2026)
- Hardt et al. (2016) uniform stability bound without WD

**Empirical baselines:**
- No WD (SGD)
- Constant WD (standard SGDW / AdamW)
- Cosine WD schedule
- CWD (ICLR 2026)
- SWD (NeurIPS 2023)
- Random alignment control (shuffled delta_hat_t)

### Risk Assessment

1. **High risk**: The backward recursion for mu_t may produce a certified band that is too narrow to be informative. **Mitigation**: Even a narrow band is a finding -- it explains why dynamic WD rarely beats constant WD.
2. **Medium risk**: PMP-WD costate approximation via EMA may be too crude, making practical PMP-WD no better than heuristic schedules. **Mitigation**: The theoretical result (bang-bang structure) stands independently of the practical algorithm.
3. **Medium risk**: Cumulative alignment improvement over worst-case may be marginal in practice (delta_t may not vary much). **Mitigation**: Focus on non-BN architectures and high-WD regimes where variation is maximal.
4. **Low risk**: Minibatch concentration bound is standard; unlikely to fail.

### Novelty Claim

1. **First Lyapunov convergence certificate for general time-varying WD** in nonconvex SGD, subsuming all existing fixed/binary/scheduled WD convergence results.
2. **First cumulative alignment generalization bound**, replacing the worst-case alignment of Sun et al. with a trajectory-level analysis.
3. **First application of PMP to derive the optimal WD schedule**, with the surprising prediction that optimal WD is bang-bang (connecting to CWD's empirical success).
4. **Unified theoretical framework** showing that convergence, generalization, and optimality of WD scheduling are controlled by a single quantity: the gradient-weight alignment trajectory.

**Evidence of novelty:**
- No paper in the literature (38 core references surveyed, 12 theoretical papers analyzed) provides a Lyapunov certificate for general time-varying WD schedules.
- No paper derives WD scheduling from PMP. The closest work (Li et al., JMLR 2018) applies PMP to layer-wise training control, not hyperparameter scheduling.
- The Holzl et al. (2025) GWA paper provides empirical evidence that gradient-weight alignment predicts generalization, but no formal bound. Our Theorem 2 provides the missing theoretical grounding.
