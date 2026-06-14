# Figure 2: Theorem 1 Regime Illustration (TikZ)

## Description
Two-curve diagram showing the stability-cost vs alignment-benefit tradeoff from Theorem 1.

## Layout
- X-axis: AIS (Alignment Informativeness Score), range [0, 1]
- Y-axis: Net benefit (alignment benefit minus stability cost), range [-0.5, 0.5]
- "Alignment benefit" curve: monotonically increasing from origin, proportional to AIS * lambda_bar
- "Stability cost" horizontal line: constant at C*sigma^2/n * Delta_CSI / lambda_bar (positive value, approximately 0.15)
- Crossing point labeled AIS* (threshold)
- Left of AIS*: shaded red, labeled "Constant WD optimal"
- Right of AIS*: shaded green, labeled "Adaptive WD optimal"
- Annotation arrows pointing to where CIFAR-10 BN experiments fall (AIS ~ 0.2-0.4, left of threshold)

## Key Elements
- Alignment benefit curve: blue, solid, increasing
- Stability cost line: red, dashed, horizontal
- AIS* threshold: vertical dotted line at intersection
- Data point annotation: "CIFAR-10 BN experiments" cluster, left of threshold
- Legend: "Alignment benefit (proportional to AIS)", "Stability cost (proportional to Delta_CSI)"

## Caption
"Theorem 1 predicts constant WD is optimal when AIS falls below the threshold AIS* = (C*sigma^2/n) * Delta_CSI / lambda_bar. All CIFAR-10 BN experiments (AIS in [0.18, 0.40]) fall in the constant-WD-optimal region."
