// ============================================================
// 神经网络 + 标准化 + Adam + 训练/验证分离 + Early Stopping
//
// 模型: y = Σ w_out_i · ReLU(w_i · x_std + b_i) + b_out
// ============================================================

import { Trainer as BaseTrainer } from "@handfirst/utils";
import { DataLoader } from "./data";
import type { Dataset } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";

const BATCH = 40;

export class Trainer extends BaseTrainer {
  readonly model: Sequential;
  private _opt: Adam;
  private _train: DataLoader;
  private _val: DataLoader;

  constructor(train: Dataset, val: Dataset, numNeurons = 16) {
    super();
    this._train = new DataLoader(train, BATCH);
    this._val = new DataLoader(val, val.features.length);
    this.model = new Sequential([new Linear(1, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new Adam(this.model.parameters(), 0.001);
  }

  step() {
    // 训练：一个 epoch，遍历所有 batch
    let totalLoss = 0, count = 0;
    for (let batch = this._train.next(); batch; batch = this._train.next()) {
      this.model.zeroGrad();
      for (const { feature, label } of batch) {
        const diff = this.model.forward([feature])[0] - label;
        totalLoss += diff * diff;
        count++;
        this.model.backward(new Float64Array([(2 * diff) / BATCH]));
      }
      this._opt.step();
    }

    // 验证：batchSize = 全量，for 循环只跑一次
    let valLoss = 0;
    for (let batch = this._val.next(); batch; batch = this._val.next()) {
      for (const { feature, label } of batch) {
        const diff = this.model.forward([feature])[0] - label;
        valLoss += diff * diff;
      }
    }
    valLoss /= this._val.dataset.features.length;

    return {
      trainLoss: Number((totalLoss / count).toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
  }

}
