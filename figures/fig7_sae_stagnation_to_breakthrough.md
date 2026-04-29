# Supplementary Figure: SAE Absorption Stagnation to Experiment-first Recovery

## 目的

用一个具体深案例解释“高效 harness 才能看到 insight emergence”。`sae-absorption` 中，多轮 writing-only iteration 没有解决 scientific bottleneck；experiment-first strategy、validation pressure 和 negative results 才改变了系统的 mental model。

## 正文位置

Section 6.3 或附录。正文主图优先使用 `fig7_dynamic_wd_refine_repair_advance.md`；本图作为 SAE 长周期案例的备用/附录图。

## 核心论证点

- Writing-only loop 会形成 stagnation：文本变多，但科学状态不前进。
- Missing validation 会让 data integrity issue 反复出现。
- Experiment-first loop 通过新 trial 产生 patching、probe degradation 和 negative results，迫使 claim 重排。
- Insight emergence 来自 trial + memory + boundary，而不是单纯写作优化。

## ASCII 草图

```text
Phase A: Stagnation

writing revision
   -> stale numbers reused
      -> validation TODO repeated
         -> data integrity still unresolved
            -> score stagnation / regression
               -> writing revision again

                    missing hard boundary
                    missing validation gate

Phase B: Experiment-first Breakthrough

reflection detects stagnation
   -> supervisor pushes experiment-first strategy
      -> activation patching trial
      -> probe degradation trial
      -> negative / null result analysis
         -> claim hierarchy revised
            -> stronger limits, weaker but credible main story
               -> new insight for SAE absorption research
```

## 图形版草图

```text
+-----------------------------+        +--------------------------------+
| Writing-only Loop           |        | Experiment-first Loop           |
|                             |        |                                |
| revise text                 |        | detect stagnation               |
| reuse stale data            |        | run targeted experiments         |
| repeat validation TODO      |  --->  | patching / probe / null results |
| no hard gate                |        | revise claim hierarchy          |
| stagnation                  |        | insight emergence               |
+-----------------------------+        +--------------------------------+
        |                                           |
        v                                           v
   failure mode                            harness lesson
   polishing is not progress               trial + boundary + memory
```

## 正式插图注意事项

- 使用左右两段流程，左侧灰色/红色表示 stagnation，右侧蓝/绿表示 experiment-first progress。
- 不要写具体未审计数字，除非后续完成 path audit。
- `validate_integration.py` 可以作为一个小标签出现，但不要让图变成实现细节堆叠。
- 图注应强调：该图是 process evidence，不是 SAE domain claim 的最终证明。

## 图像生成提示词

```text
Create a clean case-flow diagram titled "SAE Absorption: From Writing-only Stagnation to Experiment-first Insight". Left side: a loop labeled Writing-only Loop with nodes revise text, stale data reused, validation TODO repeated, data integrity unresolved, stagnation. Add a warning label "polishing is not progress". Right side: a forward flow labeled Experiment-first Loop with nodes reflection detects stagnation, targeted experiments, activation patching, probe degradation, negative/null results, claim hierarchy revised, insight emergence. Add a bridge arrow from left to right labeled supervisor pushes experiment-first strategy. Style: NeurIPS systems paper figure, vector-like, white background, compact labels, no decorative gradients, no cartoon characters.
```
