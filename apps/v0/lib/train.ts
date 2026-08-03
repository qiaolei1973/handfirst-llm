// ============================================================
// 线性回归 — 梯度下降训练
//
// 你学习时只看这个文件就够了。
// page.tsx 只是 UI，不包含任何算法。
//
// 终端验证: pnpm check
// 浏览器 viz: pnpm dev
// ============================================================

import { linearData } from '@handfirst/datasets';

// ===== 1. 单步事件 =====

export interface EpochEvent {
  W: number;
  bias: number;
  loss: number;
  gradW: number;
  gradB: number;
}

// ===== 2. Trainer =====

export class Trainer {
  params = { W: 1, bias: 0 };
  epoch = 0;
  readonly history: EpochEvent[] = [];

  private readonly _lr = 0.01;
  private readonly _batchSize = 8;
  private readonly _X: number[];
  private readonly _Y: number[];

  constructor(dataset: { features: number[]; labels: number[] }) {
    this._X = dataset.features;
    this._Y = dataset.labels;
  }

  // ---- 一次 SGD 步骤 ----
  //
  // 模型预测:    yPred = x·W + bias
  // MSE 损失:    L = (1/n) Σ (yPred - y)²
  // W 的梯度:    ∂L/∂W    = (2/n) Σ (yPred - y)·x
  // bias 的梯度: ∂L/∂bias = (2/n) Σ (yPred - y)

  step(): EpochEvent {
    const n = this._X.length;

    // 随机采样（有放回）
    const batch: { x: number; y: number }[] = [];
    for (let k = 0; k < this._batchSize; k++) {
      const idx = Math.floor(Math.random() * n);
      batch.push({ x: this._X[idx], y: this._Y[idx] });
    }

    // 累积梯度 — 对 MSE 求导
    let gW = 0;
    let gB = 0;
    for (const { x, y } of batch) {
      const yPred = x * this.params.W + this.params.bias;
      const diff = yPred - y;
      gW += 2 * diff * x;
      gB += 2 * diff;
    }
    gW /= this._batchSize;
    gB /= this._batchSize;

    // 参数更新（梯度下降）
    this.params.W -= this._lr * gW;
    this.params.bias -= this._lr * gB;

    this.epoch++;

    // 全数据集 MSE（展示用，不影响训练）
    let totalLoss = 0;
    let fullGradW = 0;
    let fullGradB = 0;
    for (let i = 0; i < n; i++) {
      const yPred = this._X[i] * this.params.W + this.params.bias;
      const diff = yPred - this._Y[i];
      totalLoss += diff * diff;
      fullGradW += 2 * diff * this._X[i];
      fullGradB += 2 * diff;
    }

    const ev: EpochEvent = {
      W: this.params.W,
      bias: this.params.bias,
      loss: totalLoss / n,
      gradW: fullGradW / n,
      gradB: fullGradB / n,
    };

    this.history.push(ev);
    return ev;
  }

  get isDone(): boolean {
    return this.epoch >= 600;
  }

  reset(): void {
    this.params = { W: 1, bias: 0 };
    this.epoch = 0;
    this.history.length = 0;
  }
}

// ===== 3. 直接运行：tsx lib/train.ts =====
// 被 import 时（例如 page.tsx），下面的代码不会执行。

const _isMain =
  typeof process !== 'undefined' &&
  process.argv[1]?.replace(/\\/g, '/').endsWith('/lib/train.ts');

if (_isMain) {
  const { features, labels } = linearData(12);
  const t = new Trainer({ features, labels });

  console.log('W 初始值:', t.params.W.toFixed(3));
  console.log('bias 初始值:', t.params.bias.toFixed(3));
  console.log('');

  let last: EpochEvent | undefined;
  while (!t.isDone) {
    last = t.step();
    const ep = t.epoch;
    if (ep <= 3 || ep % 200 === 0) {
      console.log(
        `epoch ${String(ep).padStart(4)}  W=${last.W.toFixed(5)}  bias=${last.bias.toFixed(5)}  loss=${last.loss.toFixed(4)}  gradW=${last.gradW.toFixed(3)}  gradB=${last.gradB.toFixed(3)}`,
      );
    }
  }

  console.log('');
  console.log('=== 最终 ===');
  console.log(`W    = ${last!.W.toFixed(5)}  (真值: 2)`);
  console.log(`bias = ${last!.bias.toFixed(5)}  (真值: 10)`);
  console.log(`loss = ${last!.loss.toFixed(6)}`);
}
