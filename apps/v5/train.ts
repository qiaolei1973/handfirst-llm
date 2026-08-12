// ============================================================
// 多维输入 — d 个特征 → N 个神经元 → ŷ
//
// 模型: h_j = ReLU(Σ_i W_{j,i}·x_i + b_j)
//        ŷ  = Σ_j w_out_j·h_j + b_out
// ============================================================

import { Trainer as BaseTrainer, arr, mat } from "@handfirst/utils";
import { DataLoader } from "./data";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";

export interface V5Params {
  inputDim: number; numNeurons: number;
  hiddenW: number[][]; hiddenB: number[];
  outputW: number[]; outputB: number;
}

export interface V5EpochEvent {
  params: V5Params; trainLoss: number; valLoss: number;
}

const BATCH = 40, LR = 0.005;

export class Trainer extends BaseTrainer<V5Params, V5EpochEvent> {
  get params(): V5Params {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    return { inputDim: hw.inDim, numNeurons: hw.outDim, hiddenW: mat(hw.w, hw.outDim, hw.inDim), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }

  private _model: Sequential;
  private _opt: Adam;
  private _train: DataLoader;
  private _val: DataLoader;
  constructor(
    trainF: number[][], trainL: number[], valF: number[][], valL: number[],
    numNeurons = 16,
  ) {
    super();
    this._train = new DataLoader(trainF, trainL, BATCH);
    this._val = new DataLoader(valF, valL, valF.length);
    this._model = new Sequential([new Linear(trainF[0].length, numNeurons), new ReLU(), new Linear(numNeurons, 1)]);
    this._opt = new Adam(this._model.parameters(), LR);
    this._setupReset(this._model, this._opt);
  }

  step(): V5EpochEvent {
    const batch = this._train.nextBatch();
    this._model.zeroGrad();
    let totalLoss = 0;

    for (const { feature, label } of batch) {
      const diff = this._model.forward(feature)[0] - label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();

    let valLoss = 0;
    for (let k = 0; k < this._val.features.length; k++) {
      const diff = this._model.forward(this._val.features[k])[0] - this._val.labels[k];
      valLoss += diff * diff;
    }
    valLoss /= this._val.features.length;

    const ev: V5EpochEvent = {
      params: this.params, trainLoss: Number((totalLoss / BATCH).toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
    };
    this.history.push(ev);
    return ev;
  }

}
