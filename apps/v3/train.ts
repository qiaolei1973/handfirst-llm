// ============================================================
// 神经网络 — 单隐藏层 ReLU + 反向传播
//
// 模型: y = Σ w_out_i · ReLU(w_i · x̂ + b_i) + b_out
// ============================================================

import { Trainer as BaseTrainer } from "@handfirst/utils";
import { DataLoader } from "./data";
import type { Dataset } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { SGD } from "./nn/sgd";

const LR = 0.02, BATCH = 40;

export class Trainer extends BaseTrainer {
  readonly model: Sequential;
  private _opt: SGD;
  private _loader: DataLoader;

  constructor(dataset: Dataset, numNeurons = 16) {
    super();
    this._loader = new DataLoader(dataset, BATCH);
    this.model = new Sequential([new Linear(1, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new SGD(this.model.parameters(), LR);
  }

  step() {
    const batch = this._loader.nextBatch();
    this.model.zeroGrad();
    let totalLoss = 0;

    for (const { feature, label } of batch) {
      const diff = this.model.forward([feature])[0] - label;
      totalLoss += diff * diff;
      this.model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();
    const ev = { loss: Number((totalLoss / BATCH).toFixed(6)) };
    this.history.push(ev);
    return ev;
  }

}
