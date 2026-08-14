// ============================================================
// 线性回归 — MSE + SGD + 均值中心化
//
// 终端验证: pnpm exec tsx train.ts
// 浏览器 viz: pnpm dev:v2
// ============================================================

import { fileURLToPath } from 'node:url';
import { linearData, sampleBatch } from './data';
import { Trainer as BaseTrainer } from '@handfirst/utils';
import type { EpochEvent } from '@handfirst/utils';

// ---- 均值中心化 ----
//
// 训练前把所有 x 减去均值，让 x 分布在 0 两侧。
// 这样 W 的梯度 (2*diff*x) 不会总是同号，正负推力平衡。
//
// 训练后用 recover() 把参数还原到原始空间：
//   W · (x - mean) + bias  =  W·x + (bias - W·mean)
//                            ^^^^^   ^^^^^^^^^^^^^^^^^
//                           W 不变    原始空间的 bias

export function normalize(dataset: {
  features: number[];
  labels: number[];
  trueFn: (x: number) => number;
}) {
  const mean = dataset.features.reduce((s, v) => s + v, 0) / dataset.features.length;

  return {
    features: dataset.features.map((x) => x - mean),
    labels: dataset.labels,

    /** 用训练好的参数对原始 x 做预测（自动处理中心化） */
    predict(W: number, bias: number, x: number) {
      return W * (x - mean) + bias;
    },

    /** 中心化空间的真实函数（给 viz 画 true line 用） */
    trueFn(xCentered: number) {
      return dataset.trueFn(xCentered + mean);
    },

    /** 把中心化空间训练出来的参数，恢复到原始空间 */
    recover(W: number, bias: number) {
      return { W, bias: bias - W * mean };
    },
  };
}

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

  // 模型预测:    yPred = x·W + bias
  // MSE 损失:    L = (1/n) Σ (yPred - y)²
  // W 的梯度:    ∂L/∂W    = (2/n) Σ (yPred - y)·x
  // bias 的梯度: ∂L/∂bias = (2/n) Σ (yPred - y)

  step(): EpochEvent {
    const { features: X, labels: Y } = this.dataset;

    // 随机采样（无放回 mini-batch）—— SGD 的核心
    const batch = sampleBatch(X, Y, this._batchSize);

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

    // 参数更新
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
//   pnpm exec tsx train.ts

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const ds = normalize(linearData(12, 20));
  const t = new Trainer({ features: ds.features, labels: ds.labels });

  const MAX_EPOCHS = 600;
  for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    t.step();
  }

  const raw = t.params as { W: number; bias: number };
  const orig = ds.recover(raw.W, raw.bias);
  console.log(
    `训练完成  中心化空间: W=${raw.W.toFixed(4)}, bias=${raw.bias.toFixed(4)}`,
  );
  console.log(
    `恢复原始空间:       W=${orig.W.toFixed(4)}, bias=${orig.bias.toFixed(4)}  (真值: 2, 10)`,
  );
}
