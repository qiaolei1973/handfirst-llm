# v1：认识机器学习

## 什么是机器学习？

一句话：**让计算机从数据里自己找出规律。**

传统编程是你告诉计算机规则——"如果温度 > 30，就开空调"。机器学习反过来：你给计算机一堆（输入, 输出）的例子，让它自己琢磨出规则。

```
传统编程:  规则 + 输入 → 输出     （人写规则）
机器学习:  输入 + 输出 → 规则     （机器学规则）
```

比如这里的例子：给你 12 个 (x, y) 散点，你知道背后是 `y = 2x + 10` 这条直线。但计算机不知道。它要从零开始**学**出 W 和 bias，让 `W·x + bias` 算出来的 y 尽可能接近真实的 y。

这就是机器学习里最**简单**也最**核心**的问题——线性回归。

![数据空间截图](/v1/data-space.png)

> 计算机看到的只有这些点，它要从零学出一条直线。


## 怎么学？四个步骤

### 第一步：猜一条直线（模型）

什么都没有之前，先随便猜一个：`y = W·x + bias`。初始值随便给——比如 W=1, bias=0。

```typescript
const linear = (x: number, W: number, bias: number) => x * W + bias;
```

这就是**模型**。W 和 bias 是参数，等着数据来调教。


### 第二步：看这条直线有多差（损失函数）

预测值是 `yPred`，真实值是 `y`。每一步算所有数据点的平均误差——用**绝对值**，因为高了低了都算"差"：

```
L = (1/n) · Σ |yPred - y|
```

代码里就是两行：

```typescript
const diff = yPred - target;
totalLoss += Math.abs(diff);   // 绝对值误差，高了低了都算"差"
```

![MAE 损失函数和符号梯度](/v1/mae-gradient.png)

> 左图：MAE 是 V 形，|−5| 和 |5| 的 loss 都是 5——损失只看差多远，不关心方向。右图：梯度始终 ±1，不随差值大小而变。差 0.01 和差 100，推动力都一样。

### 第三步：算往哪调（梯度）

有了 L，想知道"W 变大一点，L 会变大还是变小"。对绝对值求导——结果就是 `sign`：只看正负，不关心大小。

```
∂L/∂W    = (1/n) · Σ sign(yPred - y) · x
∂L/∂bias = (1/n) · Σ sign(yPred - y)
```

对应的代码：

```typescript
const sign = diff > 0 ? 1 : -1;    // 只看方向
gradW += sign * x;                  // W 的梯度：方向 × 坐标
gradBias += sign;                   // bias 的梯度：只看方向

// 取平均，到这一步就得到了真实梯度
gradW /= n;
gradBias /= n;
```

> 预测大了 → sign = 1 → W 要**减**。预测小了 → sign = -1 → W 要**加**。直觉就是"错了往回拉"。


### 第四步：走一小步（参数更新）

算出梯度，沿**反方向**走一步——loss 就会变小：

```typescript
W    -= learnRate * gradW;
bias -= learnRate * gradBias;
```

`learnRate = 0.1` 意思是只取梯度的 10%。太大震荡，太小爬不动。


### 合在一起：训练循环

四步包进一个循环，反复跑 600 次：

```typescript
let W = 1, bias = 0;
const learnRate = 0.1;

for (let epoch = 0; epoch < 600; epoch++) {
  let gradW = 0, gradBias = 0;

  for (let i = 0; i < dataset.features.length; i++) {
    const x = dataset.features[i];
    const yPred = linear(x, W, bias);       // 第一步：模型
    const diff = yPred - dataset.labels[i];  // 第二步：误差
    const sign = diff > 0 ? 1 : -1;         // 第三步：梯度
    gradW += sign * x;                       //
    gradBias += sign;                        //
  }

  gradW /= dataset.features.length;          // 取平均
  gradBias /= dataset.features.length;

  W -= learnRate * gradW;                   // 第四步：更新
  bias -= learnRate * gradBias;
}
```

> 完整可运行文件：`apps/v1/train_simple.ts`。


## 浏览器里的可视化

上面代码里已经包含了全部算法。浏览器里看到的六张图，靠的是把同样逻辑包进 class，让 WebSocket 每次调用 `step()` 执行一轮：

```typescript
// train_class.ts —— step() 内部和上面一模一样，只是每轮只走一个 epoch
export class Trainer extends BaseTrainer {
  step(): EpochEvent {
    // ... 和上面完全相同的逻辑 ...
    this.history.push(ev);  // 记录下来，前端图表从这里取数据
    return ev;
  }
}
```


## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v1/train_simple.ts

# 浏览器看实时图表
pnpm dev:v1
# 打开 http://localhost:3001
```

你会看到六张图：

| 图 | 它告诉你什么 |
|----|-------------|
| 数据空间 | 左上角——当前直线拟合散点的样子 |
| L(W) 抛物线 | 左下——固定 bias 不动，W 在 loss 曲面上怎么滑 |
| L(bias) 抛物线 | 右下——固定 W 不动，bias 在 loss 曲面上怎么滑 |
| Loss 曲线 | 底部左——每步 loss 下降的轨迹 |
| 参数收敛 | 底部中——W、bias 一步步趋近真值 |
| 梯度衰减 | 底部右——梯度一步步减小 |

![训练全景](/v1/training.png)

> 注意 L(W) 和 L(bias) 两张图：W 大步跳跃，bias 几乎原地不动。


## 不足

回顾训练过程：W 一步冲到真值旁边，但 bias 蜗牛一样慢慢爬。

![L(W) vs L(bias) 收敛对比](/v1/loss-comparison.png)

> 同样的 MAE、同样的学习率、同样的步数——W 几步就逼近真值 2，bias 爬了 20 步才走到 10。

为什么？

![W 和 bias 的梯度量级对比](/v1/gradient-asymmetry.png)

> W 的梯度 = sign(diff) × x，被 x（最大 20）放大。bias 的梯度 = sign(diff)，永远是 ±1。一个拿杠杆，一个用手推。

四个明显的问题：

1. **损失函数只看方向。** MAE 用 `sign`——差 100 和差 0.1，推动力都是 1。该减速的时候不知道怎么减速。
2. **参数收敛不一致。** W 的梯度乘了 `x`（这里 x 最大 20），bias 的梯度 ≤ 1。一个拿杠杆，一个用手推。
3. **数据不平衡。** x 全部 ≥0，每个点对 W 的推动力都是同号——推过头了也没有反向力把它拉回来。
4. **每次算全量数据。** 每次都用全部数据算梯度，每次走的都是同一条确定的路，没有随机性帮它"碰运气"。
