# 本轮迭代教训

## 必须改进

- [benefit buckets 不能再停留在 future work]: 下一轮必须真正产出 fixed / harmed / no-effect bucket 证据，否则 diagnostic thesis 仍然只是 aggregate story。
- [seed spot-check 仍是硬缺口]: 至少对 headline GSM8K pairwise comparison 做最小不确定性检查，不能继续只靠单次结果撑质量门。
- [signal audit 定义要写透]: `d(s)` / `g(s)` 的来源、含义、可比性必须写成 appendix-grade protocol note。
- [runtime fairness 需要 reviewer-friendly artifact]: honest compute 不能只散落在 JSON 和表里，要有 asset-lineage / runtime fairness appendix。

## 需要注意

- [cross-task evidence 仍是 boundary slices]: MATH500 和 HumanEval 目前足以支持边界判断，不足以支持更强的 regime 宣言。
- [paper.md 与 main.tex 不能继续不同步]: markdown review trail 和最终 LaTeX 产物要保持一致。
- [版本目录卫生需要在迭代边界处理]: 不要在活跃 stage 中途搬目录，但进入新 iteration 时要把 legacy 内容安全归到 `iter_000`，后续自然进入 `iter_002`。

## 做得好的（继续保持）

- [证据驱动 pivot]: 当 method-forward 叙事站不住时，及时切换到 diagnostic framing，这个判断是对的。
- [负面结果诚实保留]: code boundary 和 observer-controller mismatch 都被保留并转化成了真正有价值的论点。
- [LaTeX 收口完成]: 图、引用、模板、PDF 编译链都已经跑通，说明 pipeline 的 paper package 能真正落地。
- [三路 review 有效]: critic / supervisor / codex reviewer 的独立意见高度收敛，说明问题定位已经足够稳定，可以进入 focused revision。
