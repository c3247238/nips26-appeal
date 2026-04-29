# Figure 5: Trial Signal to Institutional Memory

## 目的

解释 Research Memory：trial signals、负结果、审查意见和流程问题不被丢弃，而是转化为 routed lessons，并改变未来 research behavior。该图也帮助区分 `long context` 与 `institutional memory`。

## 正文位置

Section 4 Theory: Scientific Trial-and-Error Harness，或 Section 5 Sibyl as a Research Harness。

## 核心论证点

- Trial signal 只有进入可复用 memory，并改变后续行为，才算形成系统层面的研究直觉。
- Memory 必须被路由到具体 agent role，而不是写进一个大文本 blob。
- Memory 需要 effectiveness tracking，否则 lessons 只是日志。

## ASCII 草图

```text
trial signal / evidence challenge
   -> reflection issue
   -> category: EXPERIMENT / ANALYSIS / PIPELINE / WRITING / EFFICIENCY
   -> normalized lesson
   -> routed to role
        planner       -> changes task plan
        experimenter  -> adds sanity check / validation
        critic        -> checks claim weakness
        supervisor    -> blocks or downgrades claim
   -> injected into future prompts
   -> checked for effectiveness
   -> retained / decayed / revised
```

## 与案例的对应

- `dlm-acceleration`：负结果和 metric confusion 应进入 critic/planner memory。
- `sae-absorption`：反复出现的 validation gap 应进入 experimenter/supervisor memory。
- `augmentation-order`：pilot insufficiency 应进入 boundary and statistics memory。

## 正式插图注意事项

- 将 issue categories 画成小型 taxonomy。
- 展示 lessons 路由到不同 roles，而不是进入单个 memory blob。
- 用 feedback arrow 表达 time decay / effectiveness tracking。
- Caption 应明确对比 institutional memory 与 long context。

## 图像生成提示词

```text
Create a minimal vector-style academic diagram titled "Trial Signal to Institutional Memory". Flow from Trial Signal / Evidence Challenge to Reflection Issue to Issue Category to Normalized Lesson. Then split into routed roles: Planner, Experimenter, Critic, Supervisor. Show arrows from roles to Future Trial Behavior Update. Add a feedback arrow labeled effectiveness tracking / decay / revision. White background, clean typography, no decorative icons.
```
