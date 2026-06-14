# H1 Unification: Control Law Parameter Mapping (Pilot v2)

| Method | Family | K_p | K_i | K_d | Rel. Error (%) | Status |
|--------|--------|-----|-----|-----|---------------:|--------|
| FixedWD | baseline | 0.000 | 0.000 | 0.000 | 0.00 | TRIVIAL |
| CWD | alignment-based | 0.000 | -0.000 | -0.000 | 5.15 | PASS |
| SWD | scheduling-based | -0.000 | -0.002 | -0.000 | 29.24 | FAIL |
| CPR | constraint-based | -0.153 | 4.026 | -5.612 | 14.91 | PASS |
| DefazioCorrective | scheduling-based | -0.000 | 0.000 | -0.000 | 0.21 | PASS |
| NoWD | degenerate | 5.476 | 0.708 | -7.771 | 0.00 | PASS |

**Verdict**: GO
- Converged: 5/5
- Methods with < 15% error: 4/5 (CWD, CPR, DefazioCorrective, NoWD)
- Methods with > 20% error: 1 (SWD)

H1 NOT falsified (1/5 > 20%, threshold: >2 for falsification)

## Method Family Interpretation

| Family | Methods | Dominant Gain | Avg Error (%) |
|--------|---------|---------------|---------------|
| alignment-based | CWD | K_i | 5.2 |
| scheduling-based | SWD, DefazioCorrective | K_i | 14.7 |
| constraint-based | CPR | K_d | 14.9 |
| degenerate | NoWD | K_d | 0.0 |
