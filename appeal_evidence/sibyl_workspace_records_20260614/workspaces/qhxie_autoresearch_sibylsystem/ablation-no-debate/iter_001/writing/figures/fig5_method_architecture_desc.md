# Figure 5: Multi-Child Proportional Ablation Architecture

## Description

This TikZ diagram illustrates the synthetic hierarchy generation and multi-child proportional ablation procedure.

## TikZ Code

```latex
\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning, calc, shadows.blur}

\begin{document}
\begin{tikzpicture}[
    node distance=2cm,
    level distance=3cm,
    sibling distance=2cm,
    every node/.style={draw, rounded rectangle, blur shadow={shadow blur steps=5, shadow blur extra=1}},
    edge from parent/.style={draw, thick, ->},
    feature_node/.style={circle, draw, minimum size=8mm, fill=blue!20},
    child_node/.style={circle, draw, minimum size=8mm, fill=green!20},
    grandchild_node/.style={circle, draw, minimum size=6mm, fill=orange!20},
    baseline/.style={dashed, draw=gray, fill=gray!10}
]

% Hierarchy levels
\node[feature_node, fill=red!30] (parent) at (0,0) {$p$}
    child { node[child_node] (c1) {$c_1$}
        child { node[grandchild_node] (g1) {$g_1$} }
        child { node[grandchild_node] (g2) {$g_2$} }
    }
    child { node[child_node] (c2) {$c_2$}
        child { node[grandchild_node] (g3) {$g_3$} }
        child { node[grandchild_node] (g4) {$g_4$} }
    };

% Cosine annotations
\node[above=1cm of parent, font=\small] {cos(p, $c_i$) = 0.67};
\node[right=1cm of g1, font=\small] {cos($g_i, g_j$) $\approx$ 0};
\node[left=1cm of c1, font=\small, anchor=east] {Children span parent};

% Ablation illustration
\node[baseline, right=5cm of parent, minimum width=4cm, minimum height=6cm, label={Ablation: k=5}] (ablation_box) {};

% Before ablation
\node[feature_node, fill=red!30] at (7, 1) {$p'$};
\node[child_node] at (6, -0.5) {$c_1$};
\node[child_node] at (8, -0.5) {$c_2$};

% After ablation - crossed out children
\node[child_node, draw=red, line width=2pt, opacity=0.5] at (6, -2) {$c_1$};
\node[draw=red, line width=2pt] (x1) at (6, -2) {};
\draw[red, thick] (x1) +(-0.3,-0.3) -- +(0.3,0.3);
\draw[red, thick] (x1) +(-0.3,0.3) -- +(0.3,-0.3);

\node[child_node, draw=red, line width=2pt, opacity=0.5] at (8, -2) {$c_2$};
\node[draw=red, line width=2pt] (x2) at (8, -2) {};
\draw[red, thick] (x2) +(-0.3,-0.3) -- +(0.3,0.3);
\draw[red, thick] (x2) +(-0.3,0.3) -- +(0.3,-0.3);

\node[feature_node, dashed] at (7, -3.5) {$abs_5 = 0.50$};

% Arrow
\draw[thick, ->] (ablation_box.west) -- (parent) node[midway, above] {Synthetic activations};

% Measurement formula
\node[draw, align=left, right=1cm of ablation_box, font=\small] (formula) {
    $abs_k = \frac{act(p | c_{1..k} \text{ ablated})}{act(p)}$ \\
    $k=1$: saturates ($abs \approx 1.0$) \\
    $k=5$: differentiates trained vs. random
};

% Legend
\node[below=2cm of parent, anchor=north, font=\small] (legend) {
    \begin{tabular}{cl}
        \tikz\node[feature_node, minimum size=4mm, fill=red!30] {}; & Parent feature \\
        \tikz\node[child_node, minimum size=4mm] {}; & Child feature \\
        \tikz\node[grandchild_node, minimum size=3mm] {}; & Grandchild feature \\
        \tikz\draw[red, line width=2pt] (0,0) -- (0.3,0); & Ablated feature
    \end{tabular}
};

\end{tikzpicture}
\end{document}
```

## Key Visual Elements

1. **3-Level Hierarchy Structure**: Parent (red) at top, two children (green) in middle, four grandchildren (orange) at bottom
2. **Geometric Constraints**: Annotations showing cosine similarity values
3. **Ablation Process**: Right panel showing before/after ablation with crossed-out children
4. **Measurement Formula**: Inline formula box showing the absorption rate calculation
5. **Legend**: Color-coded feature type identification

## Intended Key Takeaway

Single-child ablation ($k=1$) saturates because remaining children can reconstruct the parent. Multi-child ablation ($k=5$) tests whether children collectively substitute for the parent, revealing differentiation between trained SAEs (0.50) and random decoder (0.059).

## Style Notes

- Use sans-serif font for readability
- Maintain consistent node sizes across hierarchy levels
- Use dashed borders for baseline/measurement elements
- Color scheme: Red for parent, green for children, orange for grandchildren
