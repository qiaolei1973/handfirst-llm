// ============================================================
// 神经网络 — 单隐藏层 ReLU + 反向传播
//
// 模型: y = Σ w_out_i · ReLU(w_i · x̂ + b_i) + b_out
// ============================================================

import { Trainer as BaseTrainer, arr } from "@handfirst/utils";
import { DataLoader } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { SGD } from "./nn/sgd";

const LR = 0.02, BATCH = 40;

export class Trainer extends BaseTrainer<unknown, unknown> {
  get params() {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    return { numNeurons: hw.outDim, hiddenW: arr(hw.w), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }

  private _model: Sequential;
  private _opt: SGD;
  private _loader: DataLoader;

  constructor(features: number[], labels: number[], numNeurons = 16) {
    super();
    this._loader = new DataLoader(features, labels, BATCH);
    this._model = new Sequential([new Linear(1, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new SGD(this._model.parameters(), LR);
    this._setupReset(this._model, this._opt);
  }

  step() {
    const batch = this._loader.nextBatch();
    this._model.zeroGrad();
    let totalLoss = 0;

    for (const { feature, label } of batch) {
      const diff = this._model.forward([feature])[0] - label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();
    const ev = { params: this.params, grads: {}, loss: Number((totalLoss / BATCH).toFixed(6)) };
    this.history.push(ev);
    return ev;
  }

}
