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
    const batch = this._train.generate();   // 一批随机训练数据
    this.model.zeroGrad();
    let totalLoss = 0;

    for (let i = 0; i < batch.length; i++) {
      const { feature, label } = batch[i];
      const diff = this.model.forward([feature])[0] - label;
      totalLoss += diff * diff;
      this.model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();

    // 验证集：batchSize = 全量，一次全量评估
    const valBatch = this._val.generate();
    let valLoss = 0;
    for (let i = 0; i < valBatch.length; i++) {
      const diff = this.model.forward([valBatch[i].feature])[0] - valBatch[i].label;
      valLoss += diff * diff;
    }
    valLoss /= valBatch.length;

    return {
      trainLoss: Number((totalLoss / BATCH).toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
  }

}
