# Lessons from This Iteration

## Must Improve

1. **Validate metrics before using them as primary evidence**: MCC at chance level (~0.22) across all variants including Random makes the central causal claim uninterpretable. Before any experiment, run a metric validation check: does the metric discriminate trained from random? Does it show variance across conditions? If not, do not use it as primary evidence.

2. **Verify table values against raw data before writing**: Table 1 reported 0% dead latents for all variants, but raw JSON shows TopK 81.9% and Matryoshka 56.2% dead. The writer must cross-check every table cell against raw data. Do not trust summary statistics alone---inspect individual replicate files.

3. **Add computational sanity checks to experiment pipeline**: The 1.8-unit explained_variance gap and 2.8s training time per seed should have triggered automatic warnings. Add post-experiment sanity checks: (a) explained_variance should be in [-1, 1], (b) training_time should be > 10s for 2M tokens, (c) dead_latents should be < 50% for healthy dictionaries, (d) MCC should vary by > 0.01 across conditions.

4. **Remove duplicate variants before analysis**: Matryoshka and MultiScale are byte-for-byte identical in code and data. Code review must catch duplicate task definitions before experiments run. Check: do any two tasks use identical configs? If yes, consolidate.

5. **Add statistical tests or remove the promise from methodology**: For three consecutive iterations, the methodology has promised Welch's t-test, Cohen's d, and Bonferroni correction, but none have been delivered. Either implement the tests (use scipy.stats, estimated 30 minutes) or remove the promise before writing the Method section.

6. **Cross-check critic claims against raw data**: The critic incorrectly claimed lambda=0.02 achieves L0=50, but the actual pilot file shows L0=963. Always verify secondary-source claims (critic, supervisor, etc.) against primary data (raw JSON files) before acting on them.

## Watch Out

- **OrtSAE custom class may break eval functions**: The OrthogonalitySAE uses a custom torch.nn.Module (not SAELens SAE), which may cause eval_sae_on_synthetic_data() to compute metrics differently. This likely explains the explained_variance anomaly (+0.994 for OrtSAE vs -0.88 for Baseline). When using custom SAE classes, verify all eval metrics independently.

- **SAELens may cache trained models**: The 2.8s training time suggests SyntheticSAERunner may load from cache rather than train from scratch. Check for checkpoint loading behavior and add explicit "train_from_scratch" flags if needed.

- **"Falsifies" is too strong for null results**: The abstract uses "falsifying the hypothesized causal link" but MCC insensitivity makes this claim unsupported. Use "does not support" or "finds no evidence for" instead. Reserve "falsifies" for when the metric is known to be sensitive and the null is genuine.

- **Dead latents may artificially suppress absorption**: With 82% dead latents, TopK operates with only ~350 active features. Absorption rate (fraction of parent firings where child latents activate) could be lower simply because fewer latents fire. Always report dead latent rates and consider recomputing absorption on active latents only.

- **Scope reduction needs explicit justification**: Full experiments used 1024 features (same as pilots) instead of planned 16k. This is a significant deviation that must be acknowledged with justification and generalizability discussion.

## Keep Doing (success patterns)

- **Honest negative results reporting**: H2 falsified (absorption does not predict MCC), dead latent rates honestly captured in raw data, scope reduction acknowledged. This builds reviewer trust and is the paper's strongest aspect.

- **Ground-truth synthetic data**: SynthSAEBench eliminates probe calibration bias, a genuine methodological improvement over prior work. Continue using ground-truth metrics where available.

- **5-seed replication**: Provides appropriate statistical foundation for cross-variant comparisons. Continue this practice.

- **Well-crafted abstracts**: The abstract leads with problem, states method, reports key finding with numbers, states implication. Continue this structure.

- **L0-matching protocol (in conception)**: The honest reporting that Baseline L1 "cannot match" low L0 avoids presenting unachieved targets as achieved. The protocol is methodologically sound even if the claim is underpowered.

- **Supervisor cross-validation**: The supervisor's independent verification of pilot data (confirming L0=963 at lambda=0.02, not L0=50) demonstrates the value of fact-checking. Continue this rigorous cross-validation practice.

- **Limitations section honesty**: Section 5.4 flags synthetic data, small scale, metric insensitivity, and convergence concerns---all genuine issues. This proactive disclosure is excellent practice.
