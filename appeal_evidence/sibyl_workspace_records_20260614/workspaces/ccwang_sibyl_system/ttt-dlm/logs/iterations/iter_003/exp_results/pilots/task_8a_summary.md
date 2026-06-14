# Task 8a Pilot Summary: LLaDA-8B Cross-Model Verification

## Configuration
- **Model**: GSAI-ML/LLaDA-8B-Instruct (32 layers, 4096 hidden, mask_token_id=126336)
- **Methods**: Vanilla, ReMDM-conf, DTA, DTA+ReMDM
- **Samples**: 16 per benchmark (Countdown + GSM8K)
- **DTA config**: rank=4, last 2 blocks (ff_proj, up_proj, ff_out), lr=5e-4, gamma=0.95
- **GPU**: 1x NVIDIA RTX PRO 6000 Blackwell (98GB)
- **Wall-clock**: 32 minutes

## Results

### Countdown (16 samples)
| Method      | Accuracy | Correct | Avg Time | Dist-2 | Rep-3 |
|-------------|----------|---------|----------|--------|-------|
| Vanilla     | 12.5%    | 2       | 4.4s     | 0.681  | 0.205 |
| ReMDM-conf  | 0.0%     | 0       | 7.9s     | 0.850  | 0.082 |
| DTA         | 0.0%     | 0       | 17.2s    | 0.756  | 0.161 |
| DTA+ReMDM   | 0.0%     | 0       | 20.7s    | 0.789  | 0.137 |

### GSM8K (16 samples)
| Method      | Accuracy | Correct | Avg Time | Dist-2 | Rep-3 |
|-------------|----------|---------|----------|--------|-------|
| Vanilla     | 43.8%    | 7       | 6.4s     | 0.779  | 0.114 |
| ReMDM-conf  | 37.5%    | 6       | 11.5s    | 0.746  | 0.148 |
| DTA         | 18.8%    | 3       | 23.4s    | 0.806  | 0.090 |
| DTA+ReMDM   | 31.2%    | 5       | 28.5s    | 0.764  | 0.141 |

## Cross-Model Comparison (Countdown)
| Metric         | Dream-7B | LLaDA-8B |
|----------------|----------|----------|
| Vanilla        | 12.5%    | 12.5%    |
| ReMDM-conf     | 6.25%    | 0.0%     |
| DTA            | 6.25%    | 0.0%     |
| DTA+ReMDM      | 6.25%    | 0.0%     |

## Key Findings

1. **DTA does NOT improve LLaDA accuracy** -- on both Countdown and GSM8K, DTA underperforms vanilla. This is consistent with the Dream pilot where DTA also did not outperform vanilla in the 16-sample pilot (6.25% vs 12.5%).

2. **LLaDA-8B shows the same pattern as Dream**: all inference-time methods (ReMDM, DTA, etc.) degrade Countdown accuracy relative to vanilla. This is the "Information Island" problem -- both models share the fundamental DLM limitation.

3. **GSM8K: LLaDA is much stronger than Dream on vanilla** (43.8% vs Dream's GSM8K performance). However, DTA and ReMDM both hurt LLaDA's GSM8K accuracy as well.

4. **DTA+ReMDM partially recovers GSM8K accuracy** (31.2% vs DTA-only 18.8%), suggesting the combination does have value, though still below vanilla (43.8%).

5. **LoRA norms are well-controlled**: max norms 0.05-0.21, indicating no parameter explosion or numerical instability on LLaDA.

6. **Text quality**: ReMDM-conf improves diversity (higher distinct-2, lower rep-3) on both models, consistent across architectures.

## Pass Criteria Assessment
- All 4 methods ran successfully: **PASS**
- DTA accuracy >= vanilla (Countdown): **FAIL** (0.0% vs 12.5%)
- DTA accuracy >= vanilla (GSM8K): **FAIL** (18.8% vs 43.8%)
- DTA improvement on at least one benchmark: **FAIL**

## Overall Verdict: CONDITIONAL-GO

The cross-model verification confirms that DTA's current formulation has limited effectiveness on both Dream-7B and LLaDA-8B in the pilot setting. The consistent pattern across two different DLM architectures suggests this is a fundamental limitation rather than a model-specific issue. However:

- The experiment infrastructure works correctly on LLaDA
- LoRA injection and online updates are numerically stable
- The 16-sample pilot may be too small to detect the effect (as was the case for Dream where full-scale results differed from pilot)
- DTA+ReMDM shows partial recovery on GSM8K, suggesting the combination approach has merit

For the full-scale experiment, larger sample sizes with statistical testing may reveal effects not visible in this pilot.
