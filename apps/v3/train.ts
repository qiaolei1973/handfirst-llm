// ============================================================
// 神经网络 — 单隐藏层 ReLU + 反向传播
//
// 模型: y = Σ w_out_i · ReLU(w_i · x̂ + b_i) + b_out
//
// 终端验证: pnpm exec tsx apps/v3/train.ts
// 浏览器 viz: pnpm dev:v3
// ============================================================

import { fileURLToPath } from "node:url";
import { sinData, sampleBatch } from "@handfirst/datasets";
import { Trainer as BaseTrainer } from "@handfirst/utils";
import type { EpochEvent } from "@handfirst/utils";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { SGD } from "./nn/sgd";

// ===== Parameter shape =====

export interface V3Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

// ===== Trainer =====

const LR = 0.02;
const BATCH = 40;

export class Trainer extends BaseTrainer<V3Params, EpochEvent> {
  declare params: V3Params;
  private _model!: Sequential;
  private _opt!: SGD;
  private _numNeurons: number;
  private _dataset: { features: number[]; labels: number[] };

  constructor(
    dataset: { features: number[]; labels: number[] },
    numNeurons = 16,
  ) {
    super();
    this._dataset = dataset;
    this._numNeurons = numNeurons;
    this._init();
  }

  // ===== 一步训练 =====

  step(): EpochEvent {
    const batch = sampleBatch(this._dataset, BATCH);

    this._model.zeroGrad();

    let totalLoss = 0;
    for (let k = 0; k < batch.length; k++) {
      const x = batch[k].feature;
      const y = batch[k].label;

      // 前向
      const yPred = this._model.forward([x])[0];

      // MSE loss
      const diff = yPred - y;
      totalLoss += diff * diff;

      // 反向
      const gradOut = new Float64Array([(2 * diff) / BATCH]);
      this._model.backward(gradOut);
    }

    this._opt.step();
    this._syncParams();

    const ev: EpochEvent = {
      params: { ...this.params },
      grads: {},
      loss: Number((totalLoss / BATCH).toFixed(6)),
    };
    this.history.push(ev);
    return ev;
  }

  // ===== 接口 =====

  reset(): void { this.history.length = 0; this._init(); }

  predict(x: number): number {
    return this._model.forward([x])[0];
  }

  getNeurons(): { w: number; b: number }[] {
    return Array.from({ length: this._numNeurons }, (_, i) => ({
      w: (this._model.layers[0] as Linear).w[i],
      b: (this._model.layers[0] as Linear).b[i],
    }));
  }

  // ===== 内部 =====

  private _init() {
    // 模型: Linear(1→N)→ReLU→Linear(N→1)
    const hidden = new Linear(1, this._numNeurons);
    this._model = new Sequential([
      hidden,
      new ReLU(),
      new Linear(this._numNeurons, 1),
    ]);
    this._opt = new SGD(this._model.parameters(), LR);
    this._syncParams();
  }

  private _syncParams() {
    const hw = this._model.layers[0] as Linear;
    const ow = this._model.layers[2] as Linear;
    this.params = {
      numNeurons: this._numNeurons,
      hiddenW: Array.from(hw.w),
      hiddenB: Array.from(hw.b),
      outputW: Array.from(ow.w),
      outputB: ow.b[0],
    };
  }
}

// ===== 终端运行 =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);

  for (let epoch = 0; epoch < 3000; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || epoch === 2999) {
      console.log(
        `epoch ${epoch + 1}: loss=${ev.loss.toFixed(6)}`,
      );
    }
  }

  console.log("\n预测 vs 真实值:");
  for (const x of [0, 0.25, 0.5, 0.75]) {
    console.log(
      `  x̂=${x.toFixed(2)}  predict=${t.predict(x).toFixed(4)}  true=${trueFn(x).toFixed(4)}`,
    );
  }
}
