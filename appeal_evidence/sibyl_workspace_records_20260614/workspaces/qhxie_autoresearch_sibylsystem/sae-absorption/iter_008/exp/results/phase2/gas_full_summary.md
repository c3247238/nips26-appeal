# Phase 2: GAS on 5,000 Sequences — PILOT Results

## Task
Fair evaluation of GAS as absorption detector. Scaled from pilot (200 seq, rho=0.12) to 5,000 sequences.

## Configuration
- SAE: gemma-scope-2b-pt-res-canonical / layer_12/width_16k/canonical (16,384 features)
- Sequences: 5,000 x 128 = 640,000 tokens (25x pilot)
- Text: wikitext-2-raw-v1

## Key Results

### Primary Validation
| Metric | Value |
|--------|-------|
| Spearman rho | **0.1159** (p=0.5813) |
| Bootstrap 95% CI | [-0.3330, 0.5356] |
| Pearson r | 0.2729 (p=0.1869) |
| AUROC (GAS) | 0.5714 |
| AUROC (cos baseline) | 0.46825396825396826 |

### Pilot vs Full Comparison
| | Pilot | Full | Change |
|--|-------|------|--------|
| Sequences | 200 | 5000 | 25x |
| Tokens | 25,600 | 640,000 | 25x |
| Spearman rho | 0.1235 | 0.1159 | -0.0076 |

### Alternative Strategies
| Strategy | rho | p |
|----------|-----|---|
| Main GAS | 0.1159 | 0.5813 |
| weighted_gas_x_cos | 0.1159 | 0.5813 |
| log_gas | 0.1159 | 0.5813 |
| inverse_frequency | -0.1430 | 0.4953 |
| gas_div_frequency | 0.1159 | 0.5813 |

Best: **inverse_frequency** (rho=-0.1430)

## Verdict: **DEFINITIVE_NEGATIVE**
GAS fails as absorption detector: rho=0.1159 (CI: [-0.3330, 0.5356]). 25x more data vs pilot did not resolve. AUROC=0.5714 (near chance). Report as negative result in appendix.

## Failure Mode Analysis
- Scale-up: pilot (200 seq, 25,600 tok) rho=0.1235 -> full (5000 seq, 640,000 tok) rho=0.1159. Change: -0.0076. 25x more data did NOT improve GAS correlation — the signal is fundamentally absent.
- Signal overlap: 10 letters with GAS>0, 7 with absorption>0, 3 with both. GAS captures decoder geometry but NOT functional suppression dynamics.
- Root cause: Absorption is driven by competitive exclusion (child suppresses parent via encoder dynamics), but GAS only measures DECODER geometry. The decoder cosine similarity between features predicts potential for absorption but not which features actually get suppressed during encoding.
- The frequency asymmetry term freq(i)/freq(j) amplifies noise for rare features. Letter features are relatively frequent, making GAS scores low for them as victims. Child features that do the absorbing are captured by high GAS(child->parent), but this requires knowing which child feature to look at — defeating the unsupervised goal.

## Timing
- Total: **75.4s (1.3 min)**
- Activation collection: 54.8s (0.9 min)
- Cosine similarity: 0.7s
- GAS computation: 0.1s
