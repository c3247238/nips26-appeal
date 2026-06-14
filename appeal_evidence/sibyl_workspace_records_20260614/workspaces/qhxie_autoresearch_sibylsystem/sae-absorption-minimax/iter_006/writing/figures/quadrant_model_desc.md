# Figure: Quadrant Model Architecture

## Description

This diagram illustrates the four-quadrant model for classifying SAE features by absorption and sensitivity status.

```
                    HIGH Sensitivity (AUC > 0.6)
                           |
                           |
     Q2 (absorbed +      |
     sensitive)           |
     (0 features)         |
                           |
LOW Absorption <-----------+----------> HIGH Absorption
(UAS >= 0.4)               |           (UAS < 0.4)
                           |
     Q4 (best-case)        |
     (0 features)          |
                           |
                    LOW Sensitivity (AUC < 0.6)
                           |
                           |
```

## Key Elements

1. **X-axis**: Absorption score (UAS) - higher values indicate more absorbed features
2. **Y-axis**: Sensitivity (paraphrase AUC) - higher values indicate more reliable features
3. **Q1 (High Absorption + Low Sensitivity)**: 15 features - doubly-compromised
4. **Q2 (High Absorption + High Sensitivity)**: 0 features - absorbed but reliable
5. **Q3 (Low Absorption + Low Sensitivity)**: 28 features - not absorbed but unreliable
6. **Q4 (Low Absorption + High Sensitivity)**: 0 features - best-case scenario

## Interpretation

The empty Q4 and Q2 quadrants indicate that features tend to fall in Q1 or Q3. The positive correlation between absorption and sensitivity (r = 0.59) means features cluster along the diagonal: high absorption pairs with low sensitivity, and low absorption pairs with low sensitivity.

## LaTeX/TikZ Implementation

```latex
\begin{tikzpicture}
\draw[thick,->] (0,0) -- (6,0) node[right] {Absorption (UAS)};
\draw[thick,->] (0,0) -- (0,6) node[above] {Sensitivity (AUC)};

% Quadrant labels
\node at (1.5, 4.5) {Q1};
\node at (4.5, 4.5) {Q2};
\node at (1.5, 1.5) {Q3};
\node at (4.5, 1.5) {Q4};

% Threshold lines
\draw[dashed] (3, 0) -- (3, 6) node[above] {UAS = 0.4};
\draw[dashed] (0, 3) -- (6, 3) node[right] {AUC = 0.6};

% Feature points (schematic)
\filldraw (2, 2) circle (3pt); % Q1 features
\filldraw (2, 2) circle (3pt);
\filldraw (4, 2) circle (3pt); % Q3 features
\filldraw (4, 2) circle (3pt);
\end{tikzpicture}
```

## Caption

The four-quadrant model classifies SAE features by absorption level (x-axis) and sensitivity (y-axis). Pilot results show 15 features in Q1 (doubly-compromised) and 28 in Q3 (low absorption + low sensitivity). Q2 and Q4 are empty, suggesting a trade-off between absorption and sensitivity.