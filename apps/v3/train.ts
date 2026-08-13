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
    // 一个 epoch：遍历所有 batch，每个 batch 累加梯度后更新一次
    let totalLoss = 0, count = 0;
    let batch = this._loader.next();
    while (batch) {
      this.model.zeroGrad();
      for (const { feature, label } of batch) {
        const diff = this.model.forward([feature])[0] - label;
        totalLoss += diff * diff;
        count++;
        this.model.backward(new Float64Array([(2 * diff) / BATCH]));
      }
      this._opt.step();
      batch = this._loader.next();
    }

    const ev = { loss: Number((totalLoss / count).toFixed(6)) };
    this.history.push(ev);
    return ev;
  }

}
