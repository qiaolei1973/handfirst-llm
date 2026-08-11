// ============================================================
// 线性回归 — 梯度下降训练（Class 版）
//
// 和 train_simple.ts 算法完全一致，只差了一层 class 包装。
// 终端可独立运行，也可作为 WS 服务的 Trainer。
// ============================================================

import { fileURLToPath } from 'node:url';
import { linearData } from './data';
import { Trainer as BaseTrainer } from '@handfirst/utils';
import type { EpochEvent } from '@handfirst/utils';

const linear = (x: number, W: number, bias: number) => {
  return x * W + bias;
};

// ===== Trainer =====

export class Trainer extends BaseTrainer {
  params = { W: 1, bias: 0 };
  private dataset: { features: number[]; labels: number[] };
  private learnRate = 0.1;

  constructor(dataset: { features: number[]; labels: number[] }) {
    super();
    this.dataset = dataset;
  }

  reset(): void {
    this.params = { W: 1, bias: 0 };
    this.history.length = 0;
  }

  step(): EpochEvent {
    const size = this.dataset.features.length;

    let gradW = 0;
    let gradBias = 0;

    for (let i = 0; i < size; i++) {
      const x = this.dataset.features[i];
      // 目标值
      const target = this.dataset.labels[i];
      // 计算预测值
      const yPred = linear(x, this.params.W, this.params.bias);
      // 计算预测值与真实值的差
      const diff = yPred - target;
      // MAE 损失:    L = (1/n) Σ |yPred - y|
      const sign = diff > 0 ? 1 : diff < 0 ? -1 : 0;
      // W 单次梯度 sign(yPred - y)·x
      gradW += sign * x;
      // bias 单次梯度 sign(yPred - y)
      gradBias += sign;
    }

    // 取平均得到真实梯度
    // W 的梯度:    ∂L/∂W    = (1/n) Σ sign(yPred - y)·x
    gradW /= size;
    // bias 的梯度: ∂L/∂bias = (1/n) Σ sign(yPred - y)
    gradBias /= size;

    // 参数更新（梯度下降）
    this.params.W -= this.learnRate * gradW;
    this.params.bias -= this.learnRate * gradBias;

    const ev: EpochEvent = {
      params: { W: this.params.W, bias: this.params.bias },
      grads: { W: gradW, bias: gradBias },
      loss: 0, // train_simple 不计算 loss，这里也不计
    };

    this.history.push(ev);
    return ev;
  }
}

// ===== 终端运行 =====
//   pnpm exec tsx train_class.ts

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = linearData(12, 20);
  const t = new Trainer({ features, labels });

  const MAX_EPOCHS = 600;
  for (let epoch = 0; epoch <= MAX_EPOCHS; epoch++) {
    t.step();
  }

  console.log(
    `训练完成，W: ${t.params.W}, bias: ${t.params.bias}`,
  );
}
