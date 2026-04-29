# Historical Draft: Figure 2 Cross-Workspace Process Evidence Table

> Historical version. The authoritative figure specifications and generated
> assets now live under `figures/final/`, especially
> `figures/final/prompts.md` and
> `figures/final/fig4_workspace_process_evidence.png`.
> Keep this file only as provenance for earlier figure planning.

## 目的

用一张中性、积极的 results/process evidence table 展示：多个 Sibyl workspaces 如何把 trial signal 或 evidence challenge 转化为后续科研行为更新。这个图不是 failure catalog，也不是 domain benchmark table；它服务于本文核心指标 `trial-to-behavior conversion`。

## 正文位置

Section 6：Process Evidence from Sibyl Workspaces。

## 核心论证点

- workspace 不是用来证明领域科学发现已成立，而是作为 process evidence。
- 每行都展示 project context、trial signal、Sibyl mechanism、behavior update 和 methodological evidence。
- 负面 signal 用来说明 harness 如何触发 refinement、claim calibration、validation、resource reallocation 或 narrative restructuring。
- 详细路径放 `../evidence_registry.md`。

## 草稿表格

| Workspace | Research objective | Trial signal / evidence challenge | Sibyl mechanism | Research behavior update | Methodological evidence |
|---|---|---|---|---|---|
| `dynamic-wd` | dynamic weight decay / UDWDC | stability + budget + raw/paper mismatch | Boundary + Efficiency + Trace | REFINE -> UDWDC-v2 repair -> tests -> ADVANCE | trial signal becomes plan/algorithm/claim update |
| `dlm-acceleration` | DLM acceleration composition | unsupported stats, QAS/baseline mismatch, overhead | Boundary + Perspective + Efficiency | remove p-values, recalibrate metrics, sanity-check lesson | negative results calibrate belief |
| `sae-absorption` | SAE absorption mechanisms | writing stagnation, validation gaps | Memory + Boundary + Orchestration | experiment-first strategy, claim hierarchy revised | insight needs trial + memory + boundary |
| `sae-absorption-kimi` | component-isolated absorption | measurement crisis, duplicate data, component settings | Boundary + Trace | validation before narrative, controlled measurement framing | bad artifacts must not drive prose |
| `ablation-no-debate` | no-debate workflow diagnostic | pilot runs, measurement narrow | Perspective diagnostic | perspective ablation evidence | debate needs ablation, not assumption |
| `ablation-mem-positive` | absorption proxy diagnostic | broken proxy, REFINE | Boundary + Measurement | redesign metric / feature-pair search | completed runs can still be invalid evidence |
| `augmentation-order` | augmentation ordering | pilot-only evidence | Boundary + Trace | require full Tier 1 | pilot/full gate prevents overclaim |
| `lewm-generalization` | compositional world models | long GPU tasks early | Efficiency | scheduler/recovery case | compute is research infrastructure |

## ASCII 版

```text
Workspace
   -> Research objective
      -> Trial signal / evidence challenge
         -> Sibyl harness mechanism
            -> Research behavior update
               -> Methodological evidence

dynamic-wd
   -> UDWDC / dynamic WD
      -> stability + budget confound + raw/paper mismatch
         -> boundary + efficiency + trace
            -> REFINE -> repair -> tests -> ADVANCE
               -> trial signal can become algorithm and plan update

dlm-acceleration
   -> DLM acceleration composition
      -> unsupported stats + metric mismatch + overhead
         -> boundary + perspective + efficiency
            -> remove stats, recalibrate metrics, sanity-check lesson
               -> negative results calibrate belief

sae-absorption
   -> SAE absorption mechanisms
      -> stagnation + validation gaps
         -> memory + boundary + orchestration
            -> experiment-first strategy, claim hierarchy revised
               -> behavior update is operationalized research intuition
```

## 正式插图注意事项

- 建议做成横向矩阵或 compact table，标题使用 process evidence / trial-to-behavior conversion 语气。
- 每格只写 3-8 个词，避免正文段落塞进图里。
- 颜色建议按 harness function 标注：Boundary、Memory、Efficiency、Perspective、Orchestration、Trace。
- Caption 应说明：该图总结 trial-to-behavior conversion 的 process evidence。
- `needs audit` 数字不要出现在正式图中。

## 图像生成提示词

```text
Create a clean NeurIPS-style matrix figure titled "Sibyl Workspace Evidence for Trial-to-Behavior Conversion". Rows are eight workspaces: dynamic-wd, dlm-acceleration, sae-absorption, sae-absorption-kimi, ablation-no-debate, ablation-mem-positive, augmentation-order, lewm-generalization. Columns are Research Objective, Trial Signal / Evidence Challenge, Sibyl Mechanism, Research Behavior Update, Methodological Evidence. Use compact labels, not paragraphs. Use subtle color tags for Boundary, Memory, Efficiency, Perspective, Orchestration, Trace. White background, vector-like, readable at paper column width, no decorative gradients, no cartoon characters.
```
