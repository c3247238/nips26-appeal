# Pilot Summary: Semantic-Hierarchy Probe

**Overall:** GO

## Results

| SAE | Mean Hierarchy Absorption | Mean Control Absorption |
|-----|---------------------------|-------------------------|
| MatryoshkaBatchTopK_trainer_0 | 0.2833 | 0.2125 |
| TopK_trainer_0 | 0.3389 | 0.1875 |

## Pass Criteria

- All finite: True
- All probe AUROCs > 0.6: True
- Matryoshka < TopK ordering: True
- Hierarchy > control for at least 1 SAE: True
