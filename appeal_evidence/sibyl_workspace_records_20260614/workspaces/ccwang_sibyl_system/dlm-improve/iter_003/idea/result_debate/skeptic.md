# Skeptic View

## 怀疑结论

我同意当前应该沿 negative-case lane 继续推进，但前提是必须始终把这轮结果当作 **audited slice**，而不是扩写成“方法几乎成立”或“协议已经被验证”的故事。任何超出 `claim_scope_map.json` 的措辞，都会显著提高 reviewer 反击空间。

## 1. 统计与样本规模风险

当前最核心的数字都来自 100-sample slice：

- GSM8K 50 题
- MBPP 50 题
- 总样本 100

这足以支持方向性判断，但不足以支撑强泛化语言。尤其是：

- `CARD-84` 相对 `RAND-84` 在 GSM8K 上只有 `net_repaired = +1`
- 这意味着只要再有极少数样本翻转，当前局部差异的叙事就会变化

因此，最安全的写法只能是：

1. localized signal
2. audited negative case
3. current evidence limited to the audited slice

不能写：

- reliable improvement
- validated controller
- benchmark-level generalization

## 2. sham control implications 是本轮最不能被弱化的点

`decision.json` 明确写出：

- `gsm8k_card_beats_dnb84_by_2_net_repaired = true`
- `gsm8k_card_beats_rand84_by_2_net_repaired = false`

`claim_scope_map.json` 也把对应解释锁死了：`CARD-84` 对 `DNB-84` 的局部信号没有和 `RAND-84` 干净分离。这说明任何试图把 `CARD-84 > DNB-84` 直接升级成“entropy-guided revision 有效”的写法，都会立即被更强负对照推翻。

更直白地说：

- `DNB-84` 只能排除“多算力”解释
- `RAND-84` 才真正开始触及“是否只是 generic targeting / perturbation”这个问题

因此本轮负结果不是可选装饰，而是论文的中心。

## 3. proxy metric / wording inflation 风险

我最担心的不是数据本身，而是写作阶段把它写大。

### 3.1 最大风险：把 `CARD-84` 偷偷写回 winning method

`claim_scope_map.json` 已经把这条列为 forbidden claim：

- `CARD-84 is the winning inference method`

这是绝对不能碰的。原因不是措辞保守，而是事实不支持：

- `CARD-84` overall accuracy 只比 `RAND-84` 高 `0.01`
- GSM8K 上相对 `RAND-84` 的 `net_repaired` 只有 `+1`
- MBPP 上 `CARD-84` 与 `RAND-84` 都是 `0.04`

### 3.2 第二风险：把 entropy 从 `risk marker` 偷换成 targeting rule

allowed claim 只允许：

- entropy may be described as a `risk marker`

不允许：

- validated targeting rule

一旦把它写成“entropy-guided targeting works”，那 reviewer 只要指向 `CARD-84 ≈ RAND-84` 就足以击穿主文。

### 3.3 第三风险：把 minimal audit template 膨胀成 protocol paradigm

这也是 forbidden claim 之一。当前资产确实支持一套最小审计模板，但还不足以支持“建立了新 protocol paradigm”这种更大语言。

## 4. 仍缺失的证据

`validation_report.json` 说明产物是 joinable、current-only 且可复核，这很好；但它并不自动补足以下缺口：

1. 没有额外 seeds
2. 没有更大规模复现
3. 没有更强 sham control beyond current `RAND-84`
4. trajectory addon 被显式跳过

这些缺口不一定要在本轮补，但必须被主动披露。否则 negative case 会被 reviewer 质疑为“只在最方便的 slice 上成立”。

## 5. 我给当前主线的保留意见

我支持继续推进，但仅限于以下版本：

- 论文对象是 audited negative case
- 价值在于 stronger sham control 改写了一个原本容易被过度解释的 small gain
- supporting contribution 是最小审计资产包，而不是新方法胜利

只要保持这个边界，我支持 `PROCEED`；一旦写作中重新出现 “almost works” / near-win / validated controller 语言，我会认为这条线又滑回了危险区。
