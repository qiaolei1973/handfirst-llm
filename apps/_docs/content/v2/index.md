# v2：进入机器学习的世界

v1 里你完成了一次完整的训练——从瞎猜到拟合。那些朴素的操作，在机器学习里都有自己的名字。这一章分两步：先**给 v1 的操作起名**，然后用这些名字来**解决 v1 遗留的问题**。

---

## v1 的操作在机器学习里叫什么

### 模型（Model）

`y = W·x + bias` 这一行公式，在 ML 里叫**模型**。W 和 bias 叫**参数**（Parameters）。

模型 = 公式，参数 = 公式里可以调的数。

```typescript
// 这就是你的第一个模型——线性模型
const linear = (x: number, W: number, bias: number) => x * W + bias;
```

### 损失函数（Loss Function）

把每个点的差距加起来取平均——这个**总差距**就叫**损失（Loss）**。算损失的公式叫**损失函数**。

v1 用的是绝对值取平均，官方名字叫 **MAE**（Mean Absolute Error，平均绝对误差）：

```
预测值 yPred  = W·x + bias
差距    diff  = yPred - y
MAE:    L     = (1/n) · Σ |diff|
```

![MAE 损失函数和符号梯度](/v1/mae-gradient.png)
<!-- gen_v1_images.py: #8 mae-gradient — 左=红色 V 形 L=|diff|，辅助线差±5→L=5；右=红蓝分段 grad=sign(diff)，标注差100/差0.01推力都是±1 -->

### 梯度（Gradient）

计算机不是靠试错来猜方向的。它用微积分算：把损失函数对参数求导，就知道"参数变大一丢丢，损失是变大还是变小"。这个导数叫**梯度**。

v1 对 MAE 求导：

```
∂L/∂W    = (1/n) · Σ sign(diff) · x
∂L/∂bias = (1/n) · Σ sign(diff)
```

求导后绝对值被消掉了，只剩下正负号。所以 MAE 的梯度**只看方向，不看距离**——差 100 和差 0.01，推得一样狠。

### 参数更新 & 学习率（Learning Rate）

算出梯度后，沿着梯度的**反方向**走一步——损失就会变小。这叫**参数更新**。

走多大步由**学习率**控制。v1 的学习率是 0.1——只取梯度的 10%：

```typescript
W    -= 0.1 * gradW;
bias -= 0.1 * gradBias;
```

### 训练 & 梯度下降（Gradient Descent）

猜 → 算损失 → 算梯度 → 更新参数 → 再猜 → … 这个循环叫**训练（Training）**。每一轮叫一个 **epoch**。

其中"更新参数"这一步叫**梯度下降（Gradient Descent）**：

```
拿到梯度 → w = w - 学习率 × 梯度
```

梯度下降只描述"怎么更新参数"，不关心梯度是怎么来的——全量算的还是随机抽的，它不关心。

> 训练 = 外层循环。梯度下降 = 内层那一行 `w -= lr * grad`。

![Dashboard 全景](/v1/training-done.png)
<!-- dashboard 截图：v1 训练完成全貌 -->

---

## 回顾 v1 的问题

v1 跑通了，但在最后我们留下了四个问题：

| # | 问题 | 根因 | 后果 |
|---|------|------|------|
| 1 | 梯度只看方向，不看差多远 | MAE 的梯度永远是 `±1` | 差 0.01 和差 100 推力一样，该刹车时刹不住 |
| 2 | W 和 bias 收敛速度差很远 | W 的梯度乘了 x（杠杆放大），bias 只有 ±1 | 一个跑得快，一个跟不上 |
| 3 | x 全是正数，W 的梯度永远同号 | `sign(diff) * x` 里 x ≥ 0 | 所有点推同一个方向，过冲了没有反方向的力量拉回来 |
| 4 | 每次扫全部数据 | 全量，没有随机性 | 浪费计算；路径确定，没法探索 |

下面逐一解决。

---

## 优化 v1

### MSE（Mean Squared Error）—— 让梯度带上距离

**解决：问题 1 + 问题 2**

v1 的 MAE 梯度永远是 `sign(diff)`——只看方向不看距离。现在把绝对值换成**平方**，这就是 **MSE**（Mean Squared Error，均方误差）：

```
MSE:  L = (1/n) · Σ (yPred - y)²
```

对 MSE 求导：

```
∂L/∂W    = (2/n) · Σ (yPred - y) · x    = (2/n) · Σ diff · x
∂L/∂bias = (2/n) · Σ (yPred - y)        = (2/n) · Σ diff
```

`diff` 直接出现在梯度里了（不再是 `sign(diff)`）。效果：

1. 误差大 → 梯度大 → 大步走（自动加速）
2. 误差小 → 梯度小 → 自然减速（自动刹车）
3. bias 的梯度也带上了 diff，不再永远是 ±1

```typescript
// v1（MAE）：只看方向
gradW += sign(diff) * x;
gradBias += sign(diff);

// v2（MSE）：携带距离
gradW += 2 * diff * x;
gradBias += 2 * diff;
```

![MAE vs MSE loss curves and gradients](/v2/loss-comparison.png)
<!-- gen_v2_images.py: #1 loss-comparison — 2x2: 左上=loss 曲线, 右上=梯度对比, 左下=放大, 右下=公式总结 -->

![bias 在 MSE 下收敛更快](/v1/loss-comparison.png)
<!-- gen_v2_images.py: #4 — L(bias) 曲面 MAE vs MSE 收敛轨迹 -->

> ⚠️ MSE 梯度量级比 MAE 大很多（`2×diff` vs `±1`），**学习率从 0.1 降到 0.01**，不然参数会震荡。

**问题 1 ✅ 问题 2 ✅**

---

### SGD（Stochastic Gradient Descent）—— 用随机性打破确定性路径

**解决：问题 4**

梯度下降不关心梯度怎么来的。v1 每次扫全部数据——每次走完全一样的路。

**采样方式和更新方式是两回事：**

```
采样方式              更新方式
────────              ────────
全量                 梯度下降（GD）                      ← v1
随机 Mini-batch      动量（Momentum）                    （后续讲）
单样本               Adam                               （后续讲）
```

现在把左边换成"随机 Mini-batch"——每次随机抽一小撮（batch=8）：

```typescript
const batch = sampleBatch({ features: X, labels: Y }, 8);
```

只在这 8 个点上算梯度、更新参数。下一轮再抽 8 个——每次看到的子集都不一样。**更新方式没变**，还是 `w -= lr * grad`。

`Stochastic = 随机采样`，`Gradient Descent = 梯度下降`。合起来就是 SGD。

![SGD loss 曲线有抖动但整体下降](/v2/sgd-fluctuation.png)
<!-- gen_v2_images.py: #2 sgd-fluctuation — BGD 平滑 + SGD 抖动但趋势相同 -->

梯度不如全量精准，但总体方向对。代价是 loss 抖动，换来的是随机性带来的探索能力。

**问题 4 ✅**

---

### 中心化（Centering）—— 让梯度推力自己平衡

**解决：问题 3**

x 全部 ≥ 0，W 的梯度里 `x` 永远同号——所有点推同一个方向，过冲了也没有反方向的力量拉回来。

把所有 x 减去它们的平均值：

```
x_centered = x - mean(x)      均值变成 0，正负各一半
```

正 x 的点推这边，负 x 的点推那边——推力自己就平衡了。

```typescript
const mean = features.reduce((s, v) => s + v, 0) / n;
features = features.map(x => x - mean);
```

![中心化前后数据分布对比](/v2/centering.png)
<!-- gen_v2_images.py: #3 centering — 上图=原始 x 全在右侧, 下图=中心化后以 0 对称 -->

> 叫"中心化"就够了——只减均值，不除标准差。这是三种预处理中最轻量的：中心化（减均值）、标准化（再除以标准差）、归一化（缩放到 [0,1]）。v2 只需要中心化。

**问题 3 ✅**

---

## 合在一起：v2 的完整 step()

```typescript
step(): EpochEvent {
  const batch = sampleBatch({ features: X, labels: Y }, this._batchSize);

  let gradW = 0, gradBias = 0, batchLoss = 0;
  for (const { feature: x, label: y } of batch) {
    const yPred = x * this.params.W + this.params.bias;
    const diff = yPred - y;
    gradW += 2 * diff * x;         // MSE 梯度
    gradBias += 2 * diff;          // MSE 梯度
    batchLoss += diff * diff;
  }
  gradW /= this._batchSize;
  gradBias /= this._batchSize;

  this.params.W    -= this._lr * gradW;
  this.params.bias -= this._lr * gradBias;
}
```

对比 v1：

| | v1 | v2 | 为什么改 |
|---|---|---|---|
| 损失 | MAE | **MSE** | 梯度带距离，自动加/减速 |
| 采样 | 全量 | **SGD** | 随机性 + 计算更快 |
| 数据 | 原始 x | **中心化** | 梯度推力平衡 |
| 学习率 | 0.1 | **0.01** | MSE 梯度量级更大 |

---

## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v2/train.ts

# 浏览器实时可视化
pnpm dev:v2     # → http://localhost:3002
```

> 📸 **待截图**：打开 `http://localhost:3002`，跑完 600 epoch，截 dashboard 全貌。

---

## 总结

给了名字之后，用名字去理解并解决问题。v2 的模型仍然是一条直线 `y = W·x + bias`——能处理的只有直线数据。下一个问题：**如果数据不是直线，模型该怎么画曲线？** 那就是 v3 要聊的了。
