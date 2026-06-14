# Figure 5: Mechanism Diagram - Robust vs Fragile Absorption

## Diagram Description

This architecture diagram illustrates the proposed causal mediation mechanism explaining why coefficient of variation (CV) predicts steering effectiveness for absorbed features.

## Layout

```
                    Parent Feature (Absorbed)
                           |
              +------------+------------+
              |                         |
              v                         v
     [Context-Sensitive          [Stable Child Channel]
         Child Channel]                  |
              |                         |
         +----+----+               +----+----+
         |         |               |         |
         v         v               v         v
     High-CV   Low-CV         Bypass    Fixed
     Robust    Fragile       Routing   Compensation
         \         /               |         |
          \       /                v         v
           \     /            Zero Net Effect
            \   /
             \ /
      Steering Effect
      (Effective)           (Ineffective)
```

## Node Descriptions

### Parent Feature (Absorbed)
- The SAE latent being absorbed by child features
- When steered, activation flows through child before affecting outputs

### Context-Sensitive Child Channel (High-CV / Robust)
- Child feature activation varies significantly across contexts
- Routing coefficient depends on input context
- Allows steering modulation to propagate to output
- **Result**: Steering EFFECTIVE

### Stable Child Channel (Low-CV / Fragile)
- Child feature activation is approximately constant across contexts
- Fixed routing coefficient regardless of input
- Compensates for parent steering through fixed contribution
- **Result**: Steering INEFFECTIVE (bypass routing)

## Key Insight

The coefficient of variation captures the routing regime of the child channel:
- **High CV** → Context-sensitive routing → Robust absorbed → Steerable
- **Low CV** → Stable bypass routing → Fragile absorbed → Non-steerable

## Styling Notes

- Use color coding: green for high-CV/robust path, red for low-CV/fragile path
- Arrows indicate activation flow direction
- Include labels for steering effectiveness at each terminal node
- Keep diagram clean and minimal, suitable for grayscale printing
