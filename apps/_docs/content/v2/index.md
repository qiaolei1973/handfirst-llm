# v2：进入机器学习的世界

v1 里你完成了一次完整的训练——从瞎猜到拟合。但你有没有发现，我们全程没有用任何"专业术语"？其实你刚才做的每一步，在机器学习里都有一个名字。

这一章分两半：上半场**给 v1 的操作起名**，下半场用这些名字来**改进 v1**。

---

## 上半场：你刚才做的事，机器学习里叫什么

### "猜一条直线" → 模型（Model）

`y = W·x + bias` 这一行公式，在 ML 里叫**模型**。W 和 bias 叫**参数**（Parameters）。

模型 = 公式，参数 = 公式里可以调的数。就这么简单。

```typescript
// 这就是你的第一个模型——线性模型
const linear = (x: number, W: number, bias: number) => x * W + bias;
```

### "看猜得有多差" → 损失函数（Loss Function）

把每个点的差距加起来取平均——这个**总差距**就叫**损失（Loss）**。算损失的那套公式就叫**损失函数**。

v1 用的是绝对值：`L = (1/n) · Σ |yPred - y|`。它在 ML 里的正式名字叫 **MAE**（Mean Absolute Error，平均绝对误差）。

```
预测值 yPred  = W·x + bias
差距    diff  = yPred - 真实值y
损失    L     = (1/n) · Σ |diff|        ← MAE：把所有 |diff| 加起来除以 n
```

![MAE 损失函数和符号梯度](/v1/mae-gradient.png)
<!-- gen_v1_images.py: #8 mae-gradient — 左=红色 V 形 L=|diff|，辅助线差±5→L=5；右=红蓝分段 grad=sign(diff)，标注差100/差0.01推力都是±1 -->

### "往正确的方向改" → 梯度（Gradient）

计算机不是靠试错来猜方向的。它用微积分算：把损失对 W 求导，就知道"W 变大一丢丢，损失是变大还是变小"。

这个导数就叫**梯度**。

对 MAE 求导的结果很有意思：`diff` 的绝对值被消掉了，只剩下正负号：

```
∂L/∂W    = (1/n) · Σ sign(diff) · x
∂L/∂bias = (1/n) · Σ sign(diff)
```

所以 v1 的梯度就是 `sign(diff) × x` 和 `sign(diff)`。**梯度只告诉你方向，不告诉你距离。** 这个特性等一下会变成问题。

### "走一小步" → 参数更新 & 学习率

算出梯度之后，沿着梯度的**反方向**走一步——损失就会变小。这叫**参数更新**。

步长由**学习率**（Learning Rate）控制。v1 的学习率是 0.1——只取梯度的 10%。

```typescript
W    -= 0.1 * gradW;    // ← 0.1 就是学习率
bias -= 0.1 * gradBias;
```

### 整个循环 → 梯度下降（Gradient Descent）

猜 → 算损失 → 算梯度 → 更新参数 → 再猜 → … 这个持续循环在 ML 里有一个名字：**梯度下降**。每一轮叫一个 **epoch**。

![Dashboard 全景](/v1/training-done.png)
<!-- dashboard 截图：v1 训练完成全貌，用作"你刚才做的就是梯度下降"的对照 -->

> 回头看一眼——你已经完成了**一次完整的梯度下降训练**。这些概念不是凭空造出来的，只是给你刚才做的事起了一个正式的名字。

---

## 下半场：改进 v1

v1 能工作，但有三个硬伤。现在用术语来描述这些问题（你会看到术语让讨论变清晰了），然后逐一解决。

### v1 的三个硬伤

| # | 问题 | ML 术语 | 后果 |
|---|------|---------|------|
| 1 | 梯度只看方向，不看差多远 | MAE 的梯度是常量 `±1` | 差 0.01 和差 100 的推力一样，该刹车时刹不住 |
| 2 | W 和 bias 的梯度量级差很远 | 梯度不对称 | W 杠杆放大（×x），bias 永远 ±1，收敛速度天差地别 |
| 3 | x 全是正数，W 的梯度永远同号 | 特征偏移 | 所有点都推同一边，只能靠 learning rate 往回兜 |

> 问题 4（全量数据）留到改进二处理。

---

## 改进一：让梯度带上距离——MSE

### 原因

MAE 的梯度是 `sign(diff)`，永远是 ±1。差 100 和差 0.01 的推动力完全一样。这意味着：**误差小的时候不知道减速，误差大的时候不会加速。**

![MAE vs MSE loss curves and gradients](/v2/loss-comparison.png)
<!-- 已有图片：左=MAE V 形 vs MSE U 形抛物线对比，右=MAE 梯度水平线 vs MSE 梯度斜线 -->

### 怎么做

把绝对值换成**平方**。这就是 **MSE**（Mean Squared Error，均方误差）：

```
MSE:  L = (1/n) · Σ (yPred - y)²
MAE:  L = (1/n) · Σ |yPred - y|         ← v1 用的
```

对 MSE 求导：

```
∂L/∂W    = (2/n) · Σ (yPred - y) · x    = (2/n) · Σ diff · x
∂L/∂bias = (2/n) · Σ (yPred - y)        = (2/n) · Σ diff
```

**`diff` 直接出现在梯度里了！** 不再是 `sign(diff)`。这解决了两件事：

1. **误差大 → 梯度大 → 大步走**（自动加速）
2. **误差小 → 梯度小 → 自然减速**（自动刹车）

```typescript
// v1（MAE）：只看方向
const sign = diff > 0 ? 1 : -1;
gradW += sign * x;
gradBias += sign;

// v2（MSE）：携带距离信息
gradW += 2 * diff * x;    // diff 本身带着大小
gradBias += 2 * diff;      // bias 也能大步走了
```

![bias 在 MSE 下收敛更快](/v1/loss-comparison.png)
<!-- 已有图片：并排 loss landscape，L(bias) 轨迹比 v1 更快到谷底 -->

> ⚠️ MSE 的梯度量级比 MAE 大很多（因为 `2×diff` 而不是 `±1`），所以**学习率要从 0.1 降到 0.01**，不然步长太大，参数会震荡甚至发散。

### 解决了什么

问题 1 ✅ — 梯度和误差成比例，大错大步走，小错小步走。
问题 2 ✅ — bias 的梯度不再是 ±1，也能随误差大小变化了。

### 没解决什么

问题 3（x 全正数，推力同号）还在。

---

## 改进二：每次只看几个点——SGD

### 原因

v1 每次 `step()` 都遍历**全部 12 个点**算梯度。这在 ML 里叫**批量梯度下降（BGD）**。

两个问题：
1. 数据量大会很慢（12 个点无所谓，但 1200 万个点呢？）
2. 每次都走完全一样的路，没有随机性帮你绕开"局部陷阱"

### 怎么做

每次只随机抽一小撮数据算梯度。这就是 **SGD**（Stochastic Gradient Descent，随机梯度下降）。这里每次抽 8 个：

```typescript
const batch = sampleBatch({ features: X, labels: Y }, 8);
// batch = [{ feature: 3.2, label: 15.8 }, { feature: -6.7, label: -4.1 }, ...]
//          ↑ 随机无放回采样，每次 step() 看到的不一样
```

然后只在这 8 个点上算梯度、更新参数。下一轮再随机抽 8 个——每次看到的子集都不一样。

![SGD loss 曲线有抖动但整体下降 vs BGD 平滑下降](/v2/sgd-fluctuation.png)
<!-- 已有图片：两条 loss 曲线，BGD 平滑单调下降 vs SGD 抖动但整体向下 -->

梯度虽然不如全量精准，但**总体方向对**。付出的代价是 loss 曲线会抖动，换来的好处是随机性带来的探索能力。

### 解决了什么

问题 4 ✅ — 不再是全量数据，有了随机性。

---

## 改进三：把数据摆对称——均值中心化

### 原因

回头看问题 3：x 全部 ≥ 0，所以 W 的梯度 `2·diff·x` 里，`x` 这部分永远同号。所有 12 个点都朝同一个方向推 W——推过头了也没有相反的力量拉回来。

### 怎么做

把所有 x 减去它们的平均值（mean）。变换后 x 以 0 为中心对称分布，正负各占一半：

```
x_centered = x - mean(x)       ← 均值 0 为中心
```

对 W 的梯度来说——正 x 的点推这边，负 x 的点推那边——推力自己就平衡了。

![中心化前后数据分布对比](/v2/centering.png)
<!-- 已有图片：上=原始 x 全在 [5,20]，下=中心化后以 0 对称分布 [-8,8] -->

```typescript
export function normalize(dataset) {
  const mean = dataset.features.reduce((s, v) => s + v, 0) / dataset.features.length;

  return {
    features: dataset.features.map((x) => x - mean),  // 全部减均值
    labels:   dataset.labels,

    // 训练后把参数恢复到原始空间
    recover(W, bias) {
      return { W, bias: bias - W * mean };
    },
  };
}
```

训练出来的是**中心化空间**的参数。用 `recover()` 就能拿回原始空间的值：

```
中心化空间:   W=2.0020,  bias=30.1501
恢复原始空间:  W=2.0020,  bias=10.1302    (真值: 2, 10)
```

> 数学上只是把 `W·(x－mean) + bias` 展开成 `W·x + (bias－W·mean)`。W 不受影响，bias 减掉一个常数就行。

### 解决了什么

问题 3 ✅ — 数据对称了，W 的梯度不再永远同号。

---

## 合在一起：v2 的完整 step()

```typescript
step(): EpochEvent {
  // SGD：随机抽一小撮
  const batch = sampleBatch({ features: X, labels: Y }, this._batchSize);

  let gradW = 0, gradBias = 0, batchLoss = 0;
  for (const { feature: x, label: y } of batch) {
    const yPred = x * this.params.W + this.params.bias;
    const diff = yPred - y;
    gradW += 2 * diff * x;         // MSE 梯度（自带距离）
    gradBias += 2 * diff;          // MSE 梯度
    batchLoss += diff * diff;      // MSE 损失
  }
  gradW /= this._batchSize;
  gradBias /= this._batchSize;

  this.params.W    -= this._lr * gradW;      // 学习率 0.01
  this.params.bias -= this._lr * gradBias;
  // ...
}
```

对比 v1 的 `step()`：

| | v1 | v2 | 为什么改 |
|---|---|---|---|
| 损失 | MAE（绝对值） | **MSE（平方）** | 梯度携带距离，该快则快该慢则慢 |
| 采样 | 全量 12/12 | **SGD 8/12** | 随机性 + 计算更快 |
| 数据 | 原始 x（全正） | **均值中心化** | 梯度推力平衡，不再同号偏斜 |
| 学习率 | 0.1 | **0.01** | MSE 梯度量级大，需要更小步长 |
| Loss 输出 | 写死为 0 | **真实计算** | 可以看到 loss 在下降 |

---

## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v2/train.ts

# 浏览器看实时图表
pnpm dev:v2
# 打开 http://localhost:3002
```

> 📸 **待截图**：打开 `http://localhost:3002`，启动训练，跑完 600 个 epoch，截整个 dashboard 全貌。确认六张图中 loss 曲线和梯度曲线都在动（不像 v1 那样是平的）。

---

## 总结

这一章做了两件事：

1. **命名**——v1 的每一步都有了 ML 术语对应（模型、损失函数、梯度、学习率、梯度下降）
2. **优化**——用 MSE 替代 MAE（梯度带距离了）、用 SGD 替代全量（有随机性了）、用均值中心化处理数据（推力平衡了）

但这才刚刚开始。v2 仍然是一次只用一个点去调整个模型。下一个问题：**如果模型不只是一条直线怎么办？** 那就是 v3 要聊的了。
