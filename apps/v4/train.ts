// ============================================================
// 神经网络 + 标准化 + Adam + 训练/验证分离 + Early Stopping
//
// 模型: y = Σ w_out_i · ReLU(w_i · x_std + b_i) + b_out
// ============================================================

import { Trainer as BaseTrainer } from "@handfirst/utils";
import { DataLoader } from "./data";
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

  constructor(trainF: number[], trainL: number[], valF: number[], valL: number[], numNeurons = 16) {
    super();
    this._train = new DataLoader(trainF, trainL, BATCH);
    this._val = new DataLoader(valF, valL, valF.length);
    this.model = new Sequential([new Linear(1, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new Adam(this.model.parameters(), 0.001);
  }

  step() {
    const batch = this._train.nextBatch();
    this.model.zeroGrad();
    let totalLoss = 0;

    for (const { feature, label } of batch) {
      const diff = this.model.forward([feature])[0] - label;
      totalLoss += diff * diff;
      this.model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();

    let valLoss = 0;
    for (let k = 0; k < this._val.features.length; k++) {
      const diff = this.model.forward([this._val.features[k]])[0] - this._val.labels[k];
      valLoss += diff * diff;
    }
    valLoss /= this._val.features.length;

    return {
      trainLoss: Number((totalLoss / BATCH).toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
  }

}
