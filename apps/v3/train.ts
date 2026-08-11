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
import { Trainer as BaseTrainer, arr, setArr } from "@handfirst/utils";
import type { EpochEvent } from "@handfirst/utils";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { SGD } from "./nn/sgd";

// ===== Types =====

export interface V3Params {
  numNeurons: number; hiddenW: number[]; hiddenB: number[];
  outputW: number[]; outputB: number;
}

// ===== Trainer =====

const LR = 0.02, BATCH = 40;

export class Trainer extends BaseTrainer<V3Params, EpochEvent> {
  declare params: V3Params;
  private _model!: Sequential;
  private _opt!: SGD;
  private _N: number;
  private _data: { features: number[]; labels: number[] };

  constructor(data: { features: number[]; labels: number[] }, numNeurons = 16) {
    super();
    this._data = data;
    this._N = numNeurons;
    this._model = new Sequential([
      new Linear(1, numNeurons), new ReLU(), new Linear(numNeurons, 1),
    ]);
    this._opt = new SGD(this._model.parameters(), LR);
    this._sync();
  }

  reset(): void { this.history.length = 0; this._init(); }

  step(): EpochEvent {
    const batch = sampleBatch(this._data, BATCH);
    this._model.zeroGrad();
    let totalLoss = 0;

    for (let k = 0; k < batch.length; k++) {
      const yPred = this._model.forward([batch[k].feature])[0];
      const diff = yPred - batch[k].label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step(); this._sync();
    const ev: EpochEvent = { params: { ...this.params }, grads: {}, loss: Number((totalLoss / BATCH).toFixed(6)) };
    this.history.push(ev);
    return ev;
  }

  predict(x: number): number { return this._model.forward([x])[0]; }

  getNeurons(): { w: number; b: number }[] {
    const hw = this._model.layers[0] as Linear;
    return Array.from({ length: this._N }, (_, i) => ({ w: hw.w[i], b: hw.b[i] }));
  }

  // ── 内部 ──
  private _init() {
    const N = this._N;
    this._model = new Sequential([new Linear(1, N), new ReLU(), new Linear(N, 1)]);
    this._opt = new SGD(this._model.parameters(), LR);
    this._sync();
  }

  private _sync() {
    const hw = this._model.layers[0] as Linear;
    const ow = this._model.layers[2] as Linear;
    this.params = { numNeurons: this._N, hiddenW: arr(hw.w), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }
}

// ===== CLI =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);
  for (let e = 0; e < 3000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || e === 2999) console.log(`epoch ${e+1}: loss=${ev.loss.toFixed(6)}`);
  }
  console.log("\n预测 vs 真实值:");
  for (const x of [0, 0.25, 0.5, 0.75])
    console.log(`  x̂=${x.toFixed(2)}  predict=${t.predict(x).toFixed(4)}  true=${trueFn(x).toFixed(4)}`);
}
