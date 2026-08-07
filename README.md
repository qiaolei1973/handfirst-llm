# HandFirst LLM

从零开始，逐步深入学习机器学习与 LLM。

## 与众不同的地方

**HandFirst = 先动手，再看书。** 每个版本从一个具体问题出发——"这堆点怎么拟合？"——用最自然的直觉写代码，观察效果，然后问"能不能更好？"。v1 的 MAE 变成 v2 的 MSE，不是因为 MSE 更高级，而是因为你亲眼看见 MAE 的梯度该刹车时刹不住。

**完整的 Demo → 概念 → 算法 → 数学。** 概念不是词典条目——它先在 demo 里出现，你亲眼看见了，然后才被命名。命名之后跟上代码实现，最后用数学推导（辅以 matplotlib 图）解释为什么有效。每一步都踩在前一步的感受上。

**TypeScript 贯穿全程。** 训练算法、WebSocket 通信、Canvas 图表、React dashboard——全是一种语言。不用在 Python 和 JS 之间切来切去。另外：作者比较熟 TypeScript（这是主要原因，哈哈）。

**可交互的可视化。** 不是静态图，不是终端里的一串数字。播放、暂停、单步、调速、重置——每一步参数怎么变、loss 怎么降、神经元怎么"折"，全部实时可见。

## 快速开始

```bash
pnpm dev          # 教程 → localhost:3000
pnpm dev:v1       # v1 · 猜一条直线 → localhost:3001
pnpm dev:v2       # v2 · 进入机器学习的世界 → localhost:3002
pnpm dev:v3       # v3 · 画曲线 → localhost:3003

# 终端裸跑
pnpm exec tsx apps/v1/train_simple.ts
pnpm exec tsx apps/v2/train.ts
pnpm exec tsx apps/v3/train.ts
```

## 结构

```
apps/v1/          猜一条直线——零术语，纯直觉
apps/v2/          进入机器学习的世界——术语命名 + MSE/SGD/中心化
apps/v3/          画曲线——单隐藏层 ReLU 神经网络
apps/_docs/       教程站点（Markdown，含 matplotlib 生成的概念图）
packages/         utils（Trainer, Layer, WS）| datasets | charts | viz
```

MIT
