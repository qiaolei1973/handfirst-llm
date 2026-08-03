// ============================================================
// 线性回归 — 梯度下降训练
//
// 你学习时只看这个文件就够了。
// page.tsx 只是 UI，不包含任何算法。
//
// 终端验证: pnpm exec tsx lib/train.ts
// 浏览器 viz: pnpm dev
// ============================================================

import { fileURLToPath } from 'node:url';
import { sampleBatch } from '@handfirst/datasets';
import { Trainer as BaseTrainer } from '@handfirst/utils';
import type { EpochEvent } from '@handfirst/utils';

// ===== Trainer =====

export class Trainer extends BaseTrainer {
  params = { W: 1, bias: 0 };
  private readonly _lr = 0.01;
  private readonly _batchSize = 8;
  private readonly dataset: { features: number[]; labels: number[] };

  constructor(dataset: { features: number[]; labels: number[] }) {
    super();
    this.dataset = dataset;
  }

  reset(): void {
    this.params = { W: 1, bias: 0 };
    this.history.length = 0;
  }

  // ---- 一次 SGD 步骤 ----
  //
  // 模型预测:    yPred = x·W + bias
  // MSE 损失:    L = (1/n) Σ (yPred - y)²
  // W 的梯度:    ∂L/∂W    = (2/n) Σ (yPred - y)·x
  // bias 的梯度: ∂L/∂bias = (2/n) Σ (yPred - y)

  step(): EpochEvent {
    const { features: X, labels: Y } = this.dataset;

    // 随机采样（无放回 mini-batch）
    const batch = sampleBatch({ features: X, labels: Y }, this._batchSize);

    // 累积梯度 + batch MSE
    let gradW = 0;
    let gradBias = 0;
    let batchLoss = 0;
    for (const { feature: x, label: y } of batch) {
      const yPred = x * this.params.W + this.params.bias;
      const diff = yPred - y;
      gradW += 2 * diff * x;
      gradBias += 2 * diff;
      batchLoss += diff * diff;
    }
    gradW /= this._batchSize;
    gradBias /= this._batchSize;

    // 参数更新（梯度下降）
    this.params.W -= this._lr * gradW;
    this.params.bias -= this._lr * gradBias;

    const ev: EpochEvent = {
      params: { W: this.params.W, bias: this.params.bias },
      grads: { W: gradW, bias: gradBias },
      loss: batchLoss / this._batchSize,
    };

    this.history.push(ev);
    return ev;
  }
}

// ===== 终端运行 =====
//   pnpm exec tsx lib/train.ts

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const t = new Trainer({
    features: [0, 1, 2, 3, 4, 5],
    labels: [10, 12, 14, 16, 18, 20],
  });

  let last;
  while (t.history.length < 600) {
    last = t.step();
    const ep = t.history.length;
    if (ep <= 3 || ep % 200 === 0) {
      console.log(
        `epoch ${String(ep).padStart(4)}  W=${(last.params as {W:number,bias:number}).W.toFixed(5)}  bias=${(last.params as {W:number,bias:number}).bias.toFixed(5)}  loss=${last.loss.toFixed(4)}`,
      );
    }
  }
  console.log(`\n最终  W=${(last!.params as {W:number,bias:number}).W.toFixed(5)} (真值:2)  bias=${(last!.params as {W:number,bias:number}).bias.toFixed(5)} (真值:10)`);
}
