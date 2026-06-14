# Pilot: tau=0.0 Paradox Resolution

**Date**: 2026-04-14 16:43  
**Task**: pilot_tau0_comparison  
**Config**: N=50 GSM8K samples, seed=42  
**Elapsed**: 6.4 min  

## Results

| Condition | Acc | TPS | Speedup | AccRet | QAS |
|-----------|-----|-----|---------|--------|-----|
| CD-SSD(tau=0.9) | 0.480 | 144.9 | 4.672x | 0.674 | 3.149 |
| CD-SSD(tau=0.0) | 0.440 | 220.6 | 7.114x | 0.618 | 4.395 |
| naive-T16 | 0.440 | 234.3 | 7.554x | 0.618 | 4.667 |
| M1+CD-SSD(tau=0.9) | 0.480 | 144.7 | 4.666x | 0.674 | 3.145 |
| M1+naive-T16 | 0.380 | 229.1 | 7.387x | 0.534 | 3.942 |

**Baseline**: GSM8K exact_match=0.7122, TPS=31.013

## Decision Analysis

| Decision Rule | Delta QAS | Outcome |
|---------------|-----------|---------|
| tau=0.0 vs naive-T16 | -0.272 | naive wins or approx equal |
| tau=0.9 vs naive-T16 | -1.518 | accept gate marginal |
| M1+CDSSD vs M1+naive | -0.797 | step-reduction drives synergy |
| M1+naive-T16 Ortho | 1.010 | SUPER-MULTIPLICATIVE (>1.0) |

## Interpretation

**naive-T16 outperforms CD-SSD(tau=0.9) — step-reduction is the driver**

NO-GO: reframe CD-SSD as step-reduction method

## Qualitative Samples (CD-SSD tau=0.9, first 5)

- Q: The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra, who raises ...
  Pred: To find the total amount of money raised by the girls, we need to calculate first how much each girl... | Correct: False

- Q: Kalinda is working on a 360 piece puzzle with her mom. Kalinda can normally add 4 pieces per minute....
  Pred: To determine how many hours it will take Kalinda and her mom to complete the 360-piece puzzle, we ne... | Correct: False

- Q: Tom's ship can travel at 10 miles per hour.  He is sailing from 1 to 4 PM.  He then travels back at ...
  Pred: To determine how long it takes Tom to get back, we first need to calculate the distance he traveled ... | Correct: True
