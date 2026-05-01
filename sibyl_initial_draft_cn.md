# Sibyl 论文初始草稿（中文笔记）


## abs
自主科研系统现在越来越多了。AI Scientist、Agent Laboratory、各种 co-scientist，基本上都是给 agent 配工具、配角色、配 pipeline，让它从想法跑到论文。这条路能跑通，但缺一个东西。

缺的是什么呢？缺的是经验积累。

人做研究不是这样的。一个在某个 benchmark 上做了半年的人，他知道哪个指标容易出问题，哪个 baseline 结果不太对劲，哪个 pilot 结果还不够格写进论文，什么时候该收手不做了。这些东西不是模型权重里存的，也不是 prompt 里写死的，是一次次试出来的——做错了、被 reviewer 怼了、发现指标有问题、发现之前的假设不成立。

我们管这个叫 **research judgment**（研究判断力）。说白了就是：之前踩过的坑，能不能让后面的行为不一样。

现在的问题是，很多自主科研系统虽然能把 pipeline 跑完，但是这次的坑下次还会再踩。Reflection 写了，但 planner 不看。Critic 提了反对意见，但 writer 不理。某个指标上次就发现问题了，下次计划里还是用它。

所以我们提了一个概念：**trial-to-behavior conversion**（试错到行为转换）。就是说，一个试验信号出来之后，后面的行为要真的变。不是嘴上说说"我们学到了什么"，而是 planner 改了计划、experiment agent 先跑验证、supervisor 把 claim 降级了、scheduler 换了个便宜的 sanity check 先跑。这些东西要可追溯、可审计。

## Scientific Trial-and-Error Harness

基于这个想法，我们提了一个设计目标：**Scientific Trial-and-Error Harness**（科学试错框架）。它不是一个 pipeline，而是围绕 agent 的一整套环境。

Harness 要做几件事：

1. **Trial orchestration** — 每个试验要有明确的问题、预期产出、依赖关系和停止条件。上一轮的结果应该影响下一轮的计划。

2. **Evidence maturity** — 不是跑完就是证据。pilot 信号、分析级证据、论文级证据、审计后的声明，是不同的状态。不能把一个 cheap pilot 当 full result 写进 abstract。

3. **Traceability** — 行为变化要能追溯。从 claim 到 table 到 script 到 log 到 review comment，这个链条要能走通。

4. **Routed memory** — reflection 不能只是存在那里。要给 planner、experimenter、critic、supervisor、writer 分别注入对应的教训。不是"记住"就够了，是要"送到能行动的人手里"。

5. **Perspective separation** — 乐观的人（innovator）和怀疑的人（skeptic）和关注方法的人（methodologist）要有不同的权限。critic 的反对意见不能只是"写进 discussion"，要变成 validation task 或者 claim downgrade。

6. **Resource-aware trial policy** — 研究和资源是绑定的。跑了一个很贵的实验发现没用，那下一次的 sanity check 顺序应该变。

7. **Harness self-evolution** — 有些失败不是项目本身的问题，是 harness 的问题。比如每次都缺 figure、每次都 telemetry 不全、每次都 paper 和 evidence 不同步。这些要变成 harness 级的修正：新的 gate、新的 prompt overlay、新的 artifact contract。

后面两个合起来，就是我们说的 **agent-harness co-evolution**：agent 在研究项目里积累经验，harness 在基础设施层面进化。两类东西都应该被观察到。

## Sibyl 是什么

Sibyl 是我们搭的一个具体实现，用来验证上面的想法能不能跑。

几个关键设计：

- **文件驱动的状态机**：所有东西都落盘——planning、experiment、review、reflection、decision，每个阶段都有对应的文件。一个 workspace 就是一个目录，里面有 iter_001、iter_002 这样的迭代记录。

- **决策门控**：不是写了就发。有 REFINE（需要修改）、GO/NO_GO（方向判断）、PIVOT（转向）、ADVANCE（推进到下一阶段）等决策类型。弱证据会被 gate 挡回来。

- **角色分离**：planner、experimenter、critic、supervisor、skeptic、methodologist、writer、editor 各干各的，权限不同。Writer 不能偷偷把一个 pilot 结果升级成 full claim。

- **Evolution memory**：每次 reflection 会被 normalize 成 issue category、severity、affected roles、suggestions。然后 prompt loader 会把相关教训注入到对应角色的 prompt 里。

- **GPU 调度和恢复**：实验是异步跑的，有依赖图、lease、监控、恢复机制。挂了会重试，重试还有问题就变成 harness 级的 repair task。

## 我们用哪些 workspace 验证

我们跑了几个实际的研究项目，不是 toy example：

- **dynamic-wd**：动态 weight decay 控制。从 controller 不稳定、budget 有问题、control 缺失，到最后修好 controller、跑通 9/9 stability test、把 claim 收窄。REFINE -> ADVANCE 的路径很清楚。

- **dlm-acceleration**：扩散语言模型加速。一开始算出来 unsupported statistics、p-value 是编的、alpha 和 accept rate 对不上、M1 是 no-op。后面整个 story 从"加速"转成"interference taxonomy"，加了 null ablation 和 full-scale replication prerequisite。

- **sae-absorption**：SAE 特征吸收。跑了 11 个迭代。中间有一段时间一直在 polish 文章但证据没跟上，然后强制转了 experiment-first。有一轮科学上有进展（probe degradation R²=0.777），但同时数据完整性问题爆了（CI inversion、stale headline、三个 incompatible rate）。下一轮直接变成 data integrity iteration。

- **sae-absorption-kimi**：这个最有意思。writing score 到了 8/10，但 supervisor 和 critic 给了 4.5 和 5。为什么？Matryoshka 和 MultiScale 的 replicate 有 4/5 是 byte-identical 的，feature count 1024 vs 16384 对不上，所有 trained variant 的 explained variance 都是负的，TopK 有 81.6% dead latents。写作质量高但证据塌了——这就是我们说的"paper completion hides evidence immaturity"。

- **augmentation-order**：CIFAR-10 + ResNet18 pilot。pilot 给了 GO，但 reflection 打了 4.0/10，因为 Tier-1 experiment 都没跑。说明 cheap pilot 不能当 full claim 用。

- **lewm-generalization**：很快暴露了一个事实——真实的自研很快会变成 scheduling 和 recovery 问题。resource policy 不是 engineering detail，是 research loop 的一部分。

还有几个诊断性质的 ablation：

- **no-debate**：去掉 skeptic 的辩论，结果 pilot GO 信号过了但 coarse measurement 没被发现。
- **mem-negative**：质量 gate 直接 hard-block 了一个 review 缺失的 iteration。
- **no-revision**：长期停滞，issue 积累但没人强制 action。

## 整体数据

跨 7 个 primary workspace + 6 个 ablation，总共：
- 66 个 iteration 目录
- 411 个 tracked experiment tasks
- 382 completed / 9 failed / 15 running
- 225 个 REFINE marker
- 69 GO / 58 NO_GO
- 155 PIVOT / 112 ADVANCE

Evolution memory 的 central digest 里面有 416 个 recurring issue patterns：212 experiment、89 writing、84 analysis、20 system 等等。每个 issue 都被路由到了对应的 role。

## 我们想说的是什么

不是说 Sibyl 多厉害，也不是说它比别的系统强。核心观点就一句话：**自主科研系统应该被评价的是"试错历史有没有改变未来行为"，而不是"能不能产出一篇完整的论文"。**

论文只是证据状态的一个 snapshot。真正的能力是：能不能在多次尝试中积累判断力，让后来的计划更谨慎、验证更早发生、claims 更窄更准、资源策略更聪明、harness 本身越来越不容易被糊弄。

这些是我们在搭 Sibyl 的过程中反复撞到的墙。每次我们发现"信号保留了但行为没变"，框架就会被迫变得更具体、更可审计、更角色化。所以这篇 paper 的框架和实现是互相咬出来的，不是先想好理论再实现。

## 再说几个反方观点

有人会说"论文质量是终极指标"。我们的反例是 sae-absorption-kimi：写作 8/10 但证据烂了。好的 paper 不能只看 prose 好不好。

有人说"metric-driven loop 就够了"。我们的反例是 dlm-acceleration：metric 本身就是坏的，继续 optimize 同一个 metric 只会越跑越偏。

有人说"人在 loop 里就行，不需要长期自主记忆"。人的责任不能替代系统的可审计性。411 个 tasks 的 trace，人不可能一个一个看。需要系统自己把问题路由到该去的角色。

有人说"long context 就是 memory"。long context 能记住文字，但不能保证 planner 检查了那个教训、critic 用了那个反对意见、scheduler 改了那个策略。

## 最后

Autonomous research 不应该被当成"能不能写出一篇 paper"的问题。Paper 是证据的表达式。更难的问题是：系统能不能在一次次试错中变得更会做研究。

我们把这个能力叫做 trial-to-behavior conversion，提出了 Scientific Trial-and-Error Harness 作为设计目标，用 Sibyl 做了一个实例。不是 benchmark，不是 comparison，是一个 existence proof：这些东西是可以被观察到、被审计的。

下一代自主科研系统，应该比的不是谁的 paper 更流畅，而是谁的 trial history 更诚实地改变了后来的行为。


## 别人都做了什么

现在有几条线，分开看一下。

端到端 AI scientist：Lu et al. AI Scientist、Yamada v2 — 从 idea 跑到 paper。能跑通，但 paper completion 本身变成了奖励信号。paper 写完了就算成功，evidence 可能还是烂的。

metric-driven：Karpathy autoresearch loop、Analemma FARS — 给一个可见的 score 反复 optimize。score 可信的时候很好，但科研里很多 score 本身就是坏的 proxy，越 optimize 越偏。

研究助手：Agent Laboratory、co-scientist — hypothesis generation、literature synthesis、candidate ranking。有价值，但设计出发点是辅助人做判断，不是让 agent 自己积累 trial-to-trial 的经验。

AlphaEvolve：在有强 verifier 的 domain 可行，比如算法搜索。PaperBench：测 agent 能不能复现已有研究。这些跟我们要做的是互补的。

工程 harness：Anthropic 和 OpenAI 都在推。但我们想说 harness 不应该只为了 uptime——它应该成为 research method 的一部分，把失败转成行为变化。

## 现在系统缺什么：六类反复出现的失效模式

1. paper 写完了但证据跟不上。pipeline 把 draft 写出来了，看着流畅，底下数据可能是脏的、缺的、没验证的。"这个 evidence 还不够硬"这个信号没有传回 planning 或 validation 阶段。

2. pilot 信号被当成 full claim。跑了个 cheap pilot，一个 seed、小 sample、短训练，有个方向感。然后直接写进 abstract 当 claim。没有人标记"这是 pilot signal 不是 paper-ready evidence"。

3. 坏 metric 被反复 optimize。某个 metric 本身有问题，但系统不会质疑 metric——它只会继续 optimize 同一个 metric。需要 metric validity check 作为前置 gate。

4. reflection 写了没人看。反思文件写得很对，但 planner 还是原来的 plan，critic 的反对意见 writer 不理，supervisor 的降级建议没人执行。不是没记住，是没路由到能行动的人。

5. 每次跑都一样贵。上次有个实验跑了 8 小时结果没用，这次还是同样的顺序。不会先跑个 cheap sanity check。需要 resource-aware reordering。

6. 基础设施的坑一直重复。每次都缺 figure、telemetry 不全、paper 和 result 不同步。这是 harness 的问题，不是单个 project 的问题，但 harness 没有被修。需要 harness-level 的修正机制。

## 怎么评估

评估不应该是看 paper 好不好看。应该看几个东西：能不能从 trial 里找到行为变化的证据、能不能把 claim 追溯到原始 artifact、能不能测出来系统会在哪里过拟合 score。

还有 governance 的问题——harness 本身也会被 hack。用 paper quality 做 metric → 系统 optimize paper quality 而不是 evidence quality。用 reflection score 做 metric → reflection 会变成套路。需要 hidden injected failures、independent audit、不能只给一个总分。

## 实现细节

Sibyl 有六个 plane：
- control：决策门控（REFINE / GO / NO_GO / PIVOT / ADVANCE），跨 iteration 的信号路由
- evidence：workspace artifacts + trace chain（claim → table → script → log）
- memory：reflection → normalize 成 issue → 按 role 路由注入 prompt
- role：planner / experimenter / critic / supervisor / skeptic / methodologist / writer / editor
- compute：GPU 调度 / lease / 监控 / 恢复 / repair task
- evolution：全局记忆 + harness 级修正

实验是异步跑的。有依赖图、lease、自动重试。重试失败 → repair task。
所有 artifact 都落盘。workspace 目录结构是 iter_NNN/role/artifact convention。
Self-heal 模块：错误收集 → 分类 → 路由 → fix（prompt overlay 或 harness checkpoint）。


## 相关文献（继续补）

端到端 AI scientist: Lu et al., Yamada v2。idea → paper 全链跑通。问题在于 paper completion 自身变成 reward signal，evidence 质量不 check。pipeline 能产出流畅 draft 但底下数据可能没验证。

metric-driven loop: Karpathy autoresearch, Analemma FARS。score 可信时 ok。坏 proxy → 越 optimize 越偏。科研里很多 score 本身就是坏的 proxy，需要 metric validity check 前置。

研究助手: Agent Lab, co-scientist。hypothesis generation, literature synthesis, candidate ranking。设计初衷是辅助人做判断，不是让 agent 自己积累 trial-to-trial 的经验。

AlphaEvolve: 强 verifier domain 可行。PaperBench: 测复现能力。两条跟我们的互补。

工程 harness: Anthropic, OpenAI 都在推。核心观点：harness 不应该只为 uptime，应该成为 research method 的一部分——把失败转成行为变化。

## 失效模式再梳理

把最近碰到的六类问题整理得更清楚：

1. paper 写完 → evidence 脏/缺/未验证。没有 evidence → planning 的回传路径。写完不等于做完了，draft completeness ≠ evidence maturity。

2. pilot signal → 当 full claim。单 seed、小 sample → abstract 里当结论写。缺标记：pilot 还是 paper-ready，需要 evidence maturity tag。

3. 坏 metric → 反复 optimize。系统不质疑 metric 本身。需要 metric validity check 作为前置 gate，不能 blind optimize。

4. reflection → 写了但未路由到执行角色。planner 旧 plan 不动，critic 反对 writer 忽略，supervisor 降级无执行。不是没记住，是没送到能行动的人手里。

5. 实验 → 同样顺序、同样贵。上次 8h 白跑 → 不改 sanity check 优先级。需要 resource-aware reordering，让 cheap sanity check 先跑。

6. 基础设施坑 → 重复出现。缺 figure、telemetry 不全、paper/result 不同步。harness 自身未被修。需要 harness-level 的修正机制，不是 project-level 的规避。


## Harness 设计再细化

Trial orchestration: 每个 trial 要有 question、预期产出、依赖、停止条件。上轮结果 → 下轮计划修改。

Evidence maturity ladder: pilot signal → analysis-grade → paper-grade → audited claim。不许 cheap pilot 直接进 abstract。

Traceability: claim → table → script → log → review comment，链条不能断。

Routed memory: reflection → normalize 成 issue → 按 affected role 路由注入对应 prompt。不是"记住"，是"送到执行角色手里"。

Perspective separation: innovator（乐观）、skeptic（怀疑）、methodologist（方法）不同权限。critic 的反对 → validation task 或 claim downgrade，不是只写进 discussion。

Resource-aware trial policy: 贵实验失败 → 下次 sanity check 策略调序。

Harness self-evolution: 基础设施坑 → harness 级 gate、prompt overlay、artifact contract 修正。

后两项合成 agent-harness co-evolution。两类积累都应该被观察到。

## Sibyl 实现进一步记录

文件状态机：全落盘，iter_NNN/role/artifact convention。
门控类型：REFINE / GO / NO_GO / PIVOT / ADVANCE。弱 evidence → gate 挡回。
角色分离：planner, experimenter, critic, supervisor, skeptic, methodologist, writer, editor。writer 不能把 pilot 升级成 full claim。
Evolution memory: reflection → issue category / severity / affected roles / suggestions → prompt loader 注入到对应 prompt。
GPU 调度: 异步，依赖图，lease，监控，恢复。挂了自动重试 → 还不行 → repair task。


## Workspace 证据整理

dynamic-wd: controller 不稳 → budget 问题 → control 缺失 → 修好 controller → 9/9 stability test → claim 收窄。REFINE → ADVANCE 链路清晰。

dlm-acceleration: unsupported statistics → p-value 编造 → alpha 和 accept rate 对不上 → M1 是 no-op。整条线从"加速"转成"interference taxonomy"，加了 null ablation + full-scale replication prerequisite。

sae-absorption: 11 个迭代。中间 polish 阶段 evidence 没跟上 → 强制转 experiment-first。probe degradation R²=0.777 有进展，但同时 CI inversion、stale headline、三个 incompatible rate 一起爆 → 下一轮变成 data integrity iteration。

sae-absorption-kimi: 最有意思的一个。writing score 8/10，supervisor 给 4.5，critic 给 5。原因：4/5 replicate 是 byte-identical，feature count 1024 vs 16384 对不上，explained variance 全负，TopK 81.6% dead latents。写作质量高但证据塌了——paper completion hides evidence immaturity。

augmentation-order: CIFAR-10 + ResNet18 pilot。pilot 给 GO，reflection 打 4.0/10（Tier-1 experiment 都没跑）。cheap pilot ≠ full claim。

lewm-generalization: 真实自研 → 快速暴露 scheduling 和 recovery 瓶颈。resource policy 是 research loop 的一部分，不是 engineering detail。

消融诊断：
- no-debate: 去掉 skeptic → pilot GO 过了但 coarse measurement 漏检。
- mem-negative: 质量 gate 直接 hard-block 了一个 review 缺失的 iteration。
- no-revision: 长期停滞，issue 积压，没人强制 action。


## 主线再捋

核心判断：自主科研系统应该被评价的是"试错历史有没有改变未来行为"，而不是"能不能产出一篇完整的论文"。

Paper 只是证据状态的一个 snapshot。真正的能力是：多次尝试 → 积累判断力 → 后来的计划更谨慎、验证更早发生、claims 更窄更准、资源策略更聪明、harness 越来越难被糊弄。

框架和实现是互相咬出来的——每次发现"信号保留了但行为没变"，框架就被迫变得更具体、更可审计、更角色化。所以这篇 paper 的框架和实现不是先想好理论再实现，是互相咬合演进。

## 反方观点回应

"论文质量是终极指标" → sae-absorption-kimi 反例：writing 8/10，evidence 塌了。好的 paper 不能只看 prose。

"metric-driven loop 就够了" → dlm-acceleration 反例：metric 本身是坏的，继续 optimize 只会越跑越偏。

"人在 loop 里就行，不需要长期自主记忆" → 411 个 tasks 的 trace，人不可能逐个 audit。需要系统自己把问题路由到该去的角色。

"long context 就是 memory" → 记住文字 ≠ planner 检查了教训、critic 用了反对意见、scheduler 调整了策略。


## 评估体系

评估不应该是看 paper 好不好看。应该测：
- trial → 行为变化的证据
- claim → 原始 artifact 的追溯链
- 系统会在何处 overfit score

Governance 问题：harness 本身也会被 hack。
- paper quality 做 metric → optimize paper quality 而非 evidence quality
- reflection score 做 metric → reflection 套路化
需要 hidden injected failures + independent audit + 多维评分，不能只给一个总分。

## 实现架构（六个 plane 的完整关系）

control plane: 决策门控，跨 iteration 的信号路由。
evidence plane: workspace artifact + trace chain（claim → table → script → log）。
memory plane: reflection → normalize 成 issue → 按 role 路由注入。
role plane: planner / experimenter / critic / supervisor / skeptic / methodologist / writer / editor。
compute plane: GPU 调度 / lease / 监控 / 恢复 / repair task。
evolution plane: 全局记忆 + harness 级 checkpoint。

实验异步，依赖图 + lease + 自动重试。重试失败 → repair task。
全落盘。self-heal 模块: 错误收集 → 分类 → 路由 → fix（prompt overlay 或 harness checkpoint）。


## 论文框架展开：每段论述的中文对应

马上要把上面的中文笔记整理成正式的 LaTeX 稿件。目前的结构还是零散的——各个段落散落在不同笔记里，现在要按正式论文的结构重新组织。以下是从中文论述到英文 section 的段落级对应。

### Introduction 段落展开

第一段（现状）：AI Scientist、Agent Laboratory、co-scientist 这些系统的共性——它们证明了 agent 能参与科研 loop，但 pipeline completion 不等于研究能力的积累。对应中文笔记里"pipeline 能跑通但坑还会再踩"那一段。英文里要把这个 gap 说清楚：不是这些系统不好，是它们暴露了一个更深的设计问题——做完研究阶段 ≠ 积累研究判断。

第二段（六类失败模式）：从六个角度展开 trial completion 和 behavior change 之间的脱节。pilot 发现 metric 有问题 → 下次计划还用它。reviewer 提了反对意见 → writer 只润色不改论据。GPU 跑挂了 → scheduler 还按原来的顺序。每个失败模式要有具体的 observed signal + missing update path。这些对应中文笔记里的六类失效模式，但要写成正式的 failure-mode 描述。

第三段（trial-to-behavior conversion）：这是论文的核心概念。定义要精确：一个 signal 在 iteration t 出现，在 iteration t+k 必须能观察到行为改变——plan 改了、validation 提前了、claim 收窄了、branch 停了、scheduler 换顺序了。要有可观测的 artifact 链。

第四段（为什么这跟 prior work 不同）：对比现有系统——现有系统展示的是 agent 能不能完成研究阶段。我们问的是 trial history 能不能改变后续行为。两个不同的问题。

### Related Work 的结构

要拆成几条线。端到端 AI scientist（Lu et al., Yamada v2）：做到了 idea→paper，贡献在于证明了可行性。metric-driven（Karpathy, Analemma）：score 优化 loop，在 score 可信时有效。研究助手（Agent Lab, co-scientist）：辅助人做判断，人不参与时缺乏自主积累。AlphaEvolve / PaperBench：互补。工程 harness（Anthropic, OpenAI）：现有重点在 uptime 和 safety，我们想论证 harness 应该是 research method 的一部分。

每条线最后都要回到同一个问题：这些系统有没有从 trial 里积累可观测的行为变化？如果没有，那就不是记忆的问题，是更新路径缺失。

### Scientific Trial-and-Error Harness 七个函数

要把中文笔记里的七条从"功能列表"升级成"可审计的设计承诺"。每条原则都要有三个要素：这个函数做什么、需要什么基础设施支持、可观测的行为承诺是什么。

H1 Trial orchestration: 每个 trial 要有 question + expected evidence + dependencies → 上一轮的结果改变下一轮的 plan 或 task dependency。
H2 Evidence maturity: pilot / analysis-grade / paper-ready / audited claim → 不同状态不能混淆。claim 升级需要 validation + scope control + artifact links。
H3 Traceability: behavior update → 对应到具体 artifact（plan, config, log, table, review, writing change）。
H4 Routed memory: reflection → normalize → route to specific role → 对应 prompt injection。
H5 Perspective separation: innovator/skeptic/methodologist → 不同权限。critic 的反对 → validation task 或 claim downgrade。
H6 Resource-aware trial policy: 贵实验失败 → sanity check 顺序调整。
H7 Harness self-evolution: process failure → harness-level gate/prompt/contract change。

### 评估不要性能指标，要审计指标

我们的论证不需要"比 baseline 强 X%"。需要的是：能不能在 workspace trace 里找到从 trial signal 到 behavior change 的证据链。所以 Table 里的数字不是"accuracy 多少"，而是 iteration 数、task 数、decision marker 数、evolution issue 数。这些数字的意义是"这些信号存在，可以被审计"，不是"这些数字表示系统更好"。

### Governance 需要内建对抗

harness 本身也会被 hack。如果用 paper quality 做 metric，系统会 optimize paper quality 而不是 evidence quality。解决方案：hidden injected failures、negative evidence requirement、multi-perspective audit（不能一个分数定生死）。

### 整体段落布局

Section 1 Introduction（三段式 + failure modes）
Section 2 Related Work（四条线 → 同一问）
Section 3 Scientific Trial-and-Error Harness（七个函数，每个含行为承诺）
Section 4 Operationalizing in Sibyl（六个 plane 对应七个函数）
Section 5 Evidence from Workspaces（七个 workspace + ablation 的 trial-level 分析）
Section 6 Auditing（用 conversion lens 审计）
Section 7 Discussion & Limitations

这个结构是下面英文稿件重构（~515 行新增）的中文对应。每一段都能在笔记里找到源头。


## 主线论证和证据链细化

框架已经铺开了。现在要解决第二个问题：怎么让主线论证更密实，让每个 claim 都挂在具体的 workspace 证据上。下面的笔记是把主线和 evidence chain 再走一遍，为接下来的英文稿件大幅展开做铺垫。

### 核心论证链条（从 abstract 到 conclusion 贯通）

链 1：自主科研系统的评价标准错了。不该看"能不能产出一篇完整的 paper"，应该看"trial history 有没有改变后续行为"。paper 只是证据状态的 snapshot，不是能力的度量。

链 2：现有系统缺六类 update path。pilot→plan 回传断了、evidence→claim 升级缺门控、reflection→role action 没有路由、bad metric 被 blind optimize、resource waste 不改 scheduler、harness bugs 不被修正。

链 3：Solution 不是换一个更强的 model，是设计一个 harness 让 update path 成为基础设施。七个函数（H1-H7）就是这些 update path 的具象化。

链 4：Sibyl 实现了这七个函数。六个 plane（control/evidence/memory/role/compute/evolution）对应到七个 H。不是 engineering showcase，是 existence proof。

链 5：Workspace traces 展示了 trial-to-behavior 和 trial-to-harness-behavior 两类 conversion 的存在证据。不是 claim Sibyl 完胜，是 claim 这些信号可以被观测、被审计。

链 6：Discussion 退一步讨论可用性、局限、governance 和 broader impact。申明 scope：一个 harness builder 的自我审计，不是横跨科学的验证。

这个链条贯穿整篇论文，每个 claim 都要挂到具体 workspace 证据上。

### 关键 claims 和对应证据

Claim: "Paper completion hides evidence immaturity"
证据: sae-absorption-kimi workspace。writing score 8/10，supervisor 4.5/10，critic 5/10。4/5 replicate byte-identical，feature count 1024≠16384，所有 trained variant explained variance 为负，TopK 81.6% dead latents。Table in Section 5 直接放这些数字。

Claim: "Bad metric repeated optimization without validity check"
证据: dlm-acceleration workspace。一开始 unsupported statistics，p-value 编造，alpha vs accept rate 对不上，M1 no-op。系统继续 optimize 同一个 metric。后来的 PIVOT 转向 interference taxonomy + null ablation + full-scale replication prerequisite。Decision trace 从 REFINE → PIVOT → ADVANCE。

Claim: "Routed memory changes planner/critic/supervisor behavior"
证据: Evolution memory central digest。416 个 recurring issue patterns，按 category + severity + affected roles 分类后注入对应 prompt。Table 里的数字不是性能分数，是可审计的 issue 数量。

Claim: "Resource-aware reordering after failed expensive experiments"
证据: lewm-generalization workspace。真实自研 → scheduling 和 recovery 变成瓶颈 → resource policy 调整。不是 engineering detail，是 research loop 的一部分。

Claim: "Harness self-evolution corrects recurring process failures"
证据: 诊断 ablation（no-debate / mem-negative / no-revision）。去掉某个功能 → 可观测的失败。然后 harness-level 的修正（gate、prompt overlay、artifact contract）出现。

### 对反方观点的回应（论证层面）

"论文质量是终极指标" → writing 8/10 but evidence fails。如果只 optimize prose，系统会在写作上进步但证据上退化。需要 multi-dimensional audit。

"metric-driven loop 就够了" → dlm-acceleration 反例。system 需要先质疑 metric validity，再 optimize。gate 必须在 loop 之前。

"人在 loop 里就行" → 411 tasks 的 trace 人类不可能逐个审计。需要系统的 routed memory 把信号送到能行动的 role。

"long context 就是 memory" → 记住文字 ≠ 检查教训、用到决策、调整策略。需要 explicit update path 和 observable behavior change。

### 数据完整性论述

Sibyl 的数据不是 benchmark results。是 process counts——iteration 数、task 数、decision marker 数、issue pattern 数。这些数字的意义是"这些信号被记录了下来，并且可以在 trace 里被检索到"。不是"我们的数字比别人好"。

后面要做的英文稿件展开（~726 行新增）就是把上面的每一条论证链和对应证据写进正式论文里。英文里的每个 claim 都能在中文章节里找到前置论述。
