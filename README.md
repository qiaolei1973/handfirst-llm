# HandFirst LLM

从零开始，逐步深入学习机器学习与 LLM。编码优先，从代码中学习。

## 快速开始

```bash
# 阅读教程
pnpm dev          # 打开 http://localhost:3000

# 训练可视化
pnpm dev:v1       # v1 · MAE 梯度下降 → http://localhost:3001
pnpm dev:v2       # v2 · MSE + SGD + 归一化 → http://localhost:3003

# 终端裸跑
pnpm exec tsx apps/v1/train_simple.ts
pnpm exec tsx apps/v2/train.ts
```

## 结构

```
apps/
  _docs/           教程站点
  v1/              MAE 全量梯度下降
  v2/              MSE + SGD + 均值归一化
packages/
  utils/           Trainer 基类、WS server、React hook
  datasets/        数据生成
  charts/          Canvas2D 图表
  viz/             React 手术台 dashboard
```
