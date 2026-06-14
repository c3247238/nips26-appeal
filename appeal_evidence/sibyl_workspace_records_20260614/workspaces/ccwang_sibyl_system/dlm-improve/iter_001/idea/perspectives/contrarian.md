# 反对者视角：也许这篇论文最重要的结果就是“不要再轻信 revision 胜利”

## 我反对的主流假设

### 假设 1：只要 revision 信号更聪明，方法就会继续变强

当前证据不支持。`TIGER` 比 `entropy revision` 更复杂，但主 benchmark 上只是打平。这说明问题可能不是“还缺一个更好的 ranking signal”，而是 revision 这条线本身已经接近收益上限。

### 假设 2：code 退化只是实现细节，修一修就好

我不接受这个说法。`gating` 只能部分修复 syntax failure，却没有恢复 Standard baseline，还让 reasoning 略掉点。这更像结构不兼容，而不是工程 bug。

### 假设 3：名义步数相近就足够公平

这也是错的。如果领域里很多方法都是靠不诚实的 compute accounting 得到“看起来更优”的曲线，那么 benchmark 本身就值得被重做。

## 我支持的主候选

### `cand_diag`

真正大胆的说法不是“我们又发明了一个 revision 变体”，而是：

**在强基线和真实 compute 对齐下，很多 revision 胜利会显著缩水，甚至消失。**

## proposal 必须保留的尖锐性

1. 必须明确写出：`TIGER` 没有赢过 `entropy revision`
2. 必须明确写出：`CORE-proxy` 仍然更强
3. 必须明确写出：`gating` 只是部分修补，不是解决

## 结论

最值得做的，不是继续证明 revision 还能赢，而是证明：

**当它不能赢时，我们应该怎样准确地承认、解释并重新定义贡献。**
