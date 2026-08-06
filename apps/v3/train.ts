// ============================================================
// 神经网络 — 单隐藏层 ReLU + 反向传播（纯标量实现）
//
// 模型: y = Σ w_out_i · ReLU(w_i · x̂ + b_i) + b_out
//       x̂ ∈ [0, 1] 是归一化后的输入
//
// 终端验证: pnpm exec tsx train.ts
// 浏览器 viz: pnpm dev:v3
// ============================================================

import { fileURLToPath } from 'node:url';
import { sinData, sampleBatch } from '@handfirst/datasets';
import { Trainer as BaseTrainer, Layer } from '@handfirst/utils';
import type { EpochEvent } from '@handfirst/utils';

// ===== Parameter shape =====

export interface V3Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

// ===== Trainer =====

export class Trainer extends BaseTrainer<V3Params, EpochEvent> {
  params!: V3Params;
  private _hidden!: Layer; // 1→N, ReLU
  private _output!: Layer; // N→1, Linear
  private _numNeurons: number;
  private _lr = 0.02;
  private _batchSize = 40;
  private _dataset: { features: number[]; labels: number[] };

  constructor(
    dataset: { features: number[]; labels: number[] },
    numNeurons = 4,
  ) {
    super();
    this._dataset = dataset;
    this._numNeurons = numNeurons;
    this._init();
  }

  reset(): void {
    this.history.length = 0;
    this._init();
  }

  // ===== 核心: 一步训练（Mini-batch SGD） =====

  step(): EpochEvent {
    const batch = sampleBatch(this._dataset, this._batchSize);

    // ---- 清零梯度 ----
    this._hidden.zeroGrad();
    this._output.zeroGrad();

    let totalLoss = 0;

    // ---- 对 batch 中每个样本做 forward + backward ----
    for (let k = 0; k < batch.length; k++) {
      const x = batch[k].feature;
      const y = batch[k].label;

      // 前向传播
      const h = this._hidden.forward([x]);              // [1] → [N]
      const yPred = this._output.forward(Array.from(h)); // [N] → [1]

      // MSE loss: 累加 (yPred - y)²
      const diff = yPred[0] - y;
      totalLoss += diff * diff;

      // 反向传播（黑盒 — v4 展开讲数学）
      // ∂L/∂yPred = 2*diff / batchSize
      const gradOut = new Float64Array([(2 * diff) / this._batchSize]);
      const gradH = this._output.backward(gradOut);
      this._hidden.backward(gradH);
    }

    const loss = totalLoss / this._batchSize;

    // ---- 梯度下降更新 ----
    for (let i = 0; i < this._hidden.w.length; i++) {
      this._hidden.w[i] -= this._lr * this._hidden.gradW[i];
    }
    for (let i = 0; i < this._hidden.b.length; i++) {
      this._hidden.b[i] -= this._lr * this._hidden.gradB[i];
    }
    for (let i = 0; i < this._output.w.length; i++) {
      this._output.w[i] -= this._lr * this._output.gradW[i];
    }
    for (let i = 0; i < this._output.b.length; i++) {
      this._output.b[i] -= this._lr * this._output.gradB[i];
    }

    // ---- 同步 params ----
    this._syncParams();

    const ev: EpochEvent = {
      params: { ...this.params },
      grads: {}, // v3 不逐个暴露梯度（参数太多），v4 展开
      loss: Number(loss.toFixed(6)),
    };

    this.history.push(ev);
    return ev;
  }

  // ===== 预测 & 神经元信息（供图表用） =====

  predict(x: number): number {
    const h = this._hidden.forward([x]);
    return this._output.forward(Array.from(h))[0];
  }

  getNeurons(): { w: number; b: number }[] {
    const out: { w: number; b: number }[] = [];
    for (let i = 0; i < this._numNeurons; i++) {
      out.push({ w: this._hidden.w[i], b: this._hidden.b[i] });
    }
    return out;
  }

  // ===== 内部 =====

  private _init(): void {
    this._hidden = new Layer(1, this._numNeurons, 'relu');
    this._output = new Layer(this._numNeurons, 1, 'linear');
    this._syncParams();
  }

  private _syncParams(): void {
    this.params = {
      numNeurons: this._numNeurons,
      hiddenW: Array.from(this._hidden.w),
      hiddenB: Array.from(this._hidden.b),
      outputW: Array.from(this._output.w),
      outputB: this._output.b[0],
    };
  }
}

// ===== 终端运行 =====
//   pnpm exec tsx train.ts

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);

  const MAX_EPOCHS = 3000;
  for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || epoch === MAX_EPOCHS - 1) {
      console.log(
        `epoch ${epoch + 1}: loss=${ev.loss.toFixed(6)}  W1=${t.params.hiddenW.map((v) => v.toFixed(3)).join(',')}`,
      );
    }
  }

  // 验证: 预测几个点
  console.log('\n最终模型预测 vs 真实值:');
  for (const x of [0, 0.25, 0.5, 0.75]) {
    console.log(
      `  x̂=${x.toFixed(2)}  predict=${t.predict(x).toFixed(4)}  true=${trueFn(x).toFixed(4)}`,
    );
  }
}
