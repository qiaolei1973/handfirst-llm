# v2：MSE + SGD + 归一化

## 从 v1 到 v2

v1 用 MAE 学会了最基本的梯度下降，但留下了三个问题：

1. **MAE 只看方向。** `sign(diff)` — 差 100 和差 0.1，推动力都是 1。
2. **参数收敛不一致。** W 梯度乘了 `x`，bias 梯度 ≤ 1，一个拿杠杆一个用手推。
3. **数据不平衡。** x 全部 ≥ 0，W 的梯度总是同号，一个方向推到黑。
4. **每次全量数据。** 12 个点全算一遍才走一步，没有随机性。

v2 逐一应对。

## 第一步：让梯度带上距离——MSE

MAE 的问题是梯度只有 `±1`，不感知"错多远"。改成平方：

```
MSE:  L = (1/n) · Σ (yPred - y)²
MAE:  L = (1/n) · Σ |yPred - y|
```

对平方求导：

```
∂L/∂W    = (2/n) · Σ (yPred - y) · x
∂L/∂bias = (2/n) · Σ (yPred - y)
```

关键变化：`diff` 直接出现在梯度里，不再是 `sign(diff)`。差得远 → 推得猛；差得近 → 自然减速。

```typescript
const diff = yPred - y;
gradW += 2 * diff * x;     // diff 本身带着距离信息
gradBias += 2 * diff;       // bias 也能大步走了
```

![MAE vs MSE loss curves and gradients](/v2/loss-comparison.png)

> 左图：同一个 diff=6，MAE 损失是 6，MSE 是 36——MSE 对大小误差的"惩罚"不在一个量级。
> 右图：MAE 梯度始终 ±1（红线持平），MSE 梯度随 diff 线性增长（蓝线）。所以误差大时 MSE 大步走，误差小时自然减速。

> **效果**：bias 的梯度不再 ≤ 1。误差大时 bias 也能大步前进，解决了 v1 里 bias 爬不动的问题。

学习率需要调低：MSE 梯度自带量级，lr 从 0.1 降到 0.01。

## 第二步：随机抽着学——SGD

v1 里每步都遍历所有 12 个点，每次都走同一条确定的路。没有意外，没有惊喜。

SGD（随机梯度下降）的做法：每步只随机抽一小撮数据算梯度。这里每步抽 8 个点：

```typescript
const batch = sampleBatch({ features: X, labels: Y }, 8);
// batch = [{ feature: 3.2, label: 15.8 }, { feature: -6.7, label: -4.1 }, ...]
```

`sampleBatch` 做的是**无放回随机采样**——每次 `step()` 看到的子集都不一样。

> 梯度虽然不如全量精准，但总体方向对。更重要的是：随机性让参数有机会"碰运气"，不会死卡在一条轨迹上。

![SGD noisy loss vs full-batch smooth loss](/v2/sgd-fluctuation.png)

> SGD 的 loss 曲线在抖动，但大方向向下。这不是 bug——随机抽样的梯度不可能次次精准，波动恰恰说明它在探索。

## 第三步：数据对称——均值中心化

v1 里 x 全是 ≥ 0 的正数。看 W 的梯度：`2·diff·x`。如果所有 x 同号，那所有样本对 W 的推力也同号——只能往一个方向推，推过了头也没有反向力拉回来。

**均值中心化**：把所有 x 减去均值（`mean`）。变换后 x 以 0 对标，正负各占一半：

![Data distribution before and after centering](/v2/centering.png)

> 中心化前所有 x ≥ 0，W 梯度的推力全同号。中心化后对称分布，正负推力互相抵消，不会推过头。

```typescript
export function normalize(dataset) {
  const mean = dataset.features.reduce((s, v) => s + v, 0) / dataset.features.length;

  return {
    features: dataset.features.map((x) => x - mean),
    labels:   dataset.labels,

    /** 用训练好的参数对原始 x 做预测 */
    predict(W, bias, x) {
      return W * (x - mean) + bias;
    },

    /** 把参数恢复到原始空间 */
    recover(W, bias) {
      return { W, bias: bias - W * mean };
    },
  };
}
```

> 数学上只是把 `W·(x-mean)+bias` 展开成 `W·x + (bias - W·mean)`。用 `recover()` 就能拿回原始空间的值。

## 合在一起

```typescript
const ds = normalize(linearData(12, 20));
const t = new Trainer({ features: ds.features, labels: ds.labels });

for (let epoch = 0; epoch < 600; epoch++) {
  t.step();
}

const raw = t.params;                    // 中心化空间
const orig = ds.recover(raw.W, raw.bias);// 恢复原始空间
```

`step()` 内部就是 MSE 梯度 + SGD 采样：

```typescript
step(): EpochEvent {
  const batch = sampleBatch({ features: X, labels: Y }, this._batchSize);

  let gradW = 0, gradBias = 0, batchLoss = 0;
  for (const { feature: x, label: y } of batch) {
    const yPred = x * this.params.W + this.params.bias;
    const diff = yPred - y;
    gradW += 2 * diff * x;
    gradBias += 2 * diff;
    batchLoss += diff * diff;
  }
  gradW /= this._batchSize;
  gradBias /= this._batchSize;

  this.params.W    -= this._lr * gradW;
  this.params.bias -= this._lr * gradBias;
  // ...
}
```

## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v2/train.ts

# 浏览器看实时图表
pnpm dev:v2
# 打开 http://localhost:3003
```

![训练全景截图](/v2/training.png)

> 📸 **待截图**：打开 `http://localhost:3003`，点 Play 跑完 600 步，截整个浏览器窗口。六张图都跑完了最好。

### 对比 v1 vs v2

| | v1 | v2 |
|----|----|----|
| 损失函数 | MAE（绝对值） | MSE（平方） |
| 梯度 | `sign(diff)` — 只看方向 | `2·diff` — 携带距离 |
| 采样 | 全量（12/12） | SGD（每次抽 8 个） |
| 数据预处理 | 无（x 全部 ≥0） | 均值中心化（x 对称） |
| 学习率 | 0.1 | 0.01 |
| bias 收敛 | 慢 | 快 |

### 参数恢复

训练出来的 W 在中心化空间里不变（还是 2），bias 需要通过 `recover()` 还原：

```
中心化空间: W=2.0020, bias=30.1501
恢复原始空间: W=2.0020, bias=10.1302  (真值: 2, 10)
```

跑一次终端 `pnpm exec tsx apps/v2/train.ts` 就能看到。
