# Experiment Cycle - Interim Analysis (task_5a_full complete, task_5b_full_s42 running)

## task_5a_full: Countdown 500 x 3 seeds x 7 methods (COMPLETED)

### Accuracy Results (mean ± std across seeds 42, 123, 456)
| Method | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|--------|---------|----------|----------|------|-----|
| DMI | 7.8% | 9.6% | 10.6% | **9.3%** | ±1.2% |
| SCP | 8.4% | 9.4% | 9.4% | **9.1%** | ±0.5% |
| RCR | 5.4% | 5.4% | 6.4% | 5.7% | ±0.5% |
| DTA | 4.4% | 4.6% | 5.4% | 4.8% | ±0.4% |
| Vanilla | 4.0% | 5.0% | 5.2% | 4.7% | ±0.5% |
| ReMDM-conf | 4.8% | 5.2% | 3.2% | 4.4% | ±0.9% |
| DTA+ReMDM | 3.6% | 2.4% | 4.8% | 3.6% | ±1.0% |

### Key Findings
1. **DMI and SCP are clear winners** - nearly 2x vanilla accuracy
2. **DTA marginally improves** over vanilla (4.8% vs 4.7%), not statistically significant
3. **DTA+ReMDM DEGRADES** performance (3.6% vs 4.7% vanilla)
4. **ReMDM-conf does not help** (4.4% vs 4.7% vanilla)
5. **RCR shows modest improvement** (5.7% vs 4.7%)

### Implications for Paper
- Original hypothesis (DTA as main contribution) NOT strongly supported
- Paper focus should shift to DMI/SCP as primary methods
- DTA may need rethinking or framing as negative result
- Combination DTA+ReMDM is counterproductive

## task_5b_full_s42: GSM8K 1319 x 4 methods (RUNNING)
- vanilla: DONE
- remdm_conf: DONE
- dta: ~45% complete (as of 16:50)
- dta_remdm: NOT STARTED
- Estimated completion: ~10+ hours from now
