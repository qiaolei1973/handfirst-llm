// ============================================================
// 神经网络 — 单隐藏层 ReLU + 反向传播
//
// 模型: y = Σ w_out_i · ReLU(w_i · x̂ + b_i) + b_out
// ============================================================

import { Trainer as BaseTrainer, arr } from "@handfirst/utils";
import type { EpochEvent } from "@handfirst/utils";
import { sampleBatch } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { SGD } from "./nn/sgd";

// ===== 参数形状（仪表盘协议） =====

export interface V3Params {
  numNeurons: number; hiddenW: number[]; hiddenB: number[];
  outputW: number[]; outputB: number;
}

// ===== Trainer =====

const LR = 0.02, BATCH = 40;

export class Trainer extends BaseTrainer<V3Params, EpochEvent> {
  get params(): V3Params {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    return { numNeurons: this._N, hiddenW: arr(hw.w), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }

  private _model!: Sequential;
  private _opt!: SGD;
  private _N: number;
  private _data: { features: number[]; labels: number[] };

  constructor(data: { features: number[]; labels: number[] }, numNeurons = 16) {
    super();
    this._data = data;
    this._N = numNeurons;
    this._init();
  }

  // ===== 一步训练 =====

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

    this._opt.step();
    const ev: EpochEvent = {
      params: this.params, grads: {},
      loss: Number((totalLoss / BATCH).toFixed(6)),
    };
    this.history.push(ev);
    return ev;
  }

  reset(): void { this.history.length = 0; this._init(); }

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
  }
}
