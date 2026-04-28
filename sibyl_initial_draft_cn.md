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
