# Figure 3: Evidential Maturity Ladder

## 目的

定义本文的核心 boundary object：`evidence maturity` 与 `task completion` 是两件事。该图支撑 Boundary Harness 和 Evaluation Agenda。

## 正文位置

Section 4 Theory: Scientific Trial-and-Error Harness，或 Section 7 Evaluation Agenda。也可以作为 `augmentation-order` 深案例旁边的小图。

## 核心论证点

- Agent 可以完成 task，但 claim 未必成熟。
- Pilot result 只能提供方向信号，不能自动进入 paper-ready 状态。
- Harness 的职责是控制 claim 在 maturity ladder 上的升级。

## ASCII 草图

```text
M0 Runnable
   code runs, no scientific claim
      |
      v
M1 Pilot Signal
   small seed/sample, directional only
      |
      v
M2 Analysis Ready
   enough data for descriptive comparison
      |
      v
M3 Claim Ready
   ablations, null controls, standardized baselines
      |
      v
M4 Paper Ready
   claim traced to artifacts and validated tables/figures
```

## 示例映射

- `augmentation-order`：在 full-scale experiments 完成前，许多 findings 应停留在 M1 附近。
- `dlm-acceleration`：部分 claims 曾被写得过强，随后被 reflection 和 review 降级。
- `sae-absorption-kimi`：即使有 artifacts，data integrity failure 也会阻止 claim 进入 M4。

## 正式插图注意事项

- 可画成 ladder 或 vertical funnel。
- 侧边加入注释：`completed task does not imply mature claim`。
- 每一层右侧加一个 gate 标签：run check、pilot gate、analysis gate、claim gate、paper gate。
- 图要足够简洁，作为全文反复使用的 visual anchor。

## 图像生成提示词

```text
Create a minimal academic ladder diagram titled "Evidential Maturity Ladder". Five vertical levels: M0 Runnable, M1 Pilot Signal, M2 Analysis Ready, M3 Claim Ready, M4 Paper Ready. Add one short description under each level. Add a side note: "Task completed does not imply claim mature." Use white background, crisp vector style, no decorative graphics, readable in a NeurIPS paper.
```
