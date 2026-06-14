# Figure 1: Experimental Pipeline Overview

## Type
TikZ flow diagram

## Description
A horizontal four-phase pipeline showing the flow from SAE loading to hypothesis testing.

## TikZ Specification

```latex
\begin{tikzpicture}[
    node distance=1.8cm,
    phase/.style={rectangle, rounded corners=3pt, draw, minimum width=2.8cm, minimum height=1.4cm, align=center, font=\small},
    data/.style={rectangle, draw=gray, dashed, minimum width=2.4cm, minimum height=0.7cm, align=center, font=\footnotesize},
    arrow/.style={->, >=stealth, thick},
    label/.style={font=\footnotesize\bfseries}
]

% Phase nodes
\node[phase, fill=blue!10] (p1) {Phase 1\\Absorption\\Detection};
\node[phase, fill=green!10, right=of p1] (p2) {Phase 2\\Feature\\Steering};
\node[phase, fill=orange!10, right=of p2] (p3) {Phase 3\\Sparse\\Probing};
\node[phase, fill=red!10, right=of p3] (p4) {Phase 4\\Correlation\\Analysis};

% Data nodes below phases
\node[data, below=0.4cm of p1] (d1) {$A(f)$ per feature};
\node[data, below=0.4cm of p2] (d2) {$S(f, s)$ per strength};
\node[data, below=0.4cm of p3] (d3) {$\text{F1}(f, k)$ per $k$};
\node[data, below=0.4cm of p4] (d4) {$r$, $p$, $R^2$, $\beta$};

% Arrows between phases
\draw[arrow] (p1) -- (p2) node[midway, above, font=\scriptsize] {feature IDs};
\draw[arrow] (p2) -- (p3) node[midway, above, font=\scriptsize] {same features};
\draw[arrow] (p3) -- (p4) node[midway, above, font=\scriptsize] {task metrics};

% Input arrow to Phase 1
\node[left=0.8cm of p1, align=center, font=\footnotesize] (input) {Pre-trained SAE\\(SAELens)};
\draw[arrow] (input) -- (p1);

% Output arrow from Phase 4
\node[right=0.8cm of p4, align=center, font=\footnotesize] (output) {H1--H3\\tested};
\draw[arrow] (p4) -- (output);

% Vertical dashed lines connecting phases to data
\draw[gray, dashed] (p1.south) -- (d1.north);
\draw[gray, dashed] (p2.south) -- (d2.north);
\draw[gray, dashed] (p3.south) -- (d3.north);
\draw[gray, dashed] (p4.south) -- (d4.north);

% Title
\node[above=0.6cm of p2, xshift=1.5cm, font=\large\bfseries] {Four-Phase Experimental Pipeline};

% Key takeaway box
\node[below=1.2cm of p3, rectangle, draw=gray, fill=gray!5, rounded corners=2pt, minimum width=10cm, minimum height=0.6cm, align=center, font=\footnotesize] {
    Training-free methodology: pre-trained SAEs $\rightarrow$ absorption detection $\rightarrow$ task evaluation $\rightarrow$ statistical hypothesis testing
};

\end{tikzpicture}
```

## Key Elements

1. **Phase 1 (Absorption Detection)**: Load pre-trained SAE via SAELens, run Chanin et al. differential correlation metric on first-letter features (A-Z), output absorption rate $A(f)$ per feature.

2. **Phase 2 (Feature Steering)**: Extract decoder directions $d_f = W_{\text{dec}}[\phi(f)]$, generate test prompts (100 per feature), apply steering at strengths $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$, measure success rate $S(f, s)$.

3. **Phase 3 (Sparse Probing)**: Train k-sparse linear probes ($k \in \{1, 5, 10, 20\}$) on SAE latents for first-letter classification, measure F1 score $\text{F1}(f, k)$.

4. **Phase 4 (Correlation Analysis)**: Compute Pearson/Spearman correlation between $A(f)$ and $S(f, 50)$ (H1), between $A(f)$ and $\text{F1}(f, 5)$ (H2), test cross-layer consistency via CV of regression slopes (H3).

## Color Scheme
- Phase 1: Blue (#2E86AB tint)
- Phase 2: Green (#4CAF50 tint)
- Phase 3: Orange (#F18F01 tint)
- Phase 4: Red (#C73E1D tint)
- Data boxes: Gray dashed outline
- Arrows: Black, thick, stealth tips
