# 3. Research Questions and Hypotheses

## 3.1 Research Questions

We formalize three research questions:

- **RQ1**: Does feature absorption cause measurable degradation in downstream interpretability tasks (steering effectiveness and sparse probing accuracy)?
- **RQ2**: Is the absorption--degradation relationship consistent across model layers?
- **RQ3**: Can we derive actionable guidance for SAE practitioners regarding absorbed features?

## 3.2 Hypotheses

We test four directional hypotheses derived from RQ1 and RQ2:

- **H1 (Raw steering)**: Higher absorption rate leads to lower raw steering success rate. Directional prediction: negative Pearson correlation between $A(f)$ and $S(f, 50)$, with $p < 0.05$.
- **H1b (Delta steering)**: Higher absorption rate leads to lower delta steering effectiveness, where $\Delta S(f, 50) = S(f, 50) - S_{\text{rand}}(50)$. Directional prediction: negative Pearson correlation between $A(f)$ and $\Delta S(f, 50)$, with $p < 0.05$.
- **H2 (Probing)**: Higher absorption rate leads to lower sparse probing F1. Directional prediction: negative Pearson correlation between $A(f)$ and $\text{F1}(f, 5)$, with $p < 0.05$.
- **H3 (Consistency)**: The degradation relationship is consistent across layers. Prediction: regression slopes $\beta$ have the same sign and similar magnitude across layers, with $\text{CV} = \sigma / |\mu| < 0.5$.

The H1b hypothesis is critical because random feature steering achieves non-negligible success (34--38% in our data), indicating that raw steering metrics conflate feature-specific contribution with generic directional bias. H1b isolates the true feature-specific effect by subtracting the random baseline.

## 3.3 Falsification Criteria

H1, H1b, and H2 are **not supported** if the Pearson correlation is non-negative ($r \geq 0$) or fails to reach significance ($p \geq 0.05$). A negative but non-significant trend does not support the hypothesis. H3 is **not supported** if slopes have opposite signs or differ substantially in magnitude ($\text{CV} \geq 0.5$).

<!-- FIGURES
- None
-->
