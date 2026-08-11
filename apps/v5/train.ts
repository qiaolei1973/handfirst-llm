// ============================================================
// 多维输入 — d 个特征 → N 个神经元 → ŷ
//
// 模型: h_j = ReLU(Σ_i W_{j,i}·x_i + b_j)
//        ŷ  = Σ_j w_out_j·h_j + b_out
// ============================================================

import { Trainer as BaseTrainer, arr, mat } from "@handfirst/utils";
import { sampleBatchMulti } from "@handfirst/datasets";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";
import { trainValSplit, standardizeMulti } from "./data";

// ===== 参数形状（仪表盘协议） =====

export interface V5Params {
  inputDim: number; numNeurons: number;
  hiddenW: number[][]; hiddenB: number[];
  outputW: number[]; outputB: number;
}

export interface V5EpochEvent {
  params: V5Params; trainLoss: number; valLoss: number;
  stopped?: boolean;
}

// ===== Trainer =====

const BATCH = 40, LR = 0.005, PATIENCE = 200;

export class Trainer extends BaseTrainer<V5Params, V5EpochEvent> {
  get params(): V5Params {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    return { inputDim: this._dim, numNeurons: this._N, hiddenW: mat(hw.w, this._N, this._dim), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }

  private _model!: Sequential;
  private _opt!: Adam;
  private _dim: number;
  private _N: number;

  private _trainF!: number[][]; private _trainL!: number[];
  private _valF!: number[][]; private _valL!: number[];
  private _means: number[] = []; private _stds: number[] = [];

  private _bestVal = Infinity;
  private _patience = 0; private _stopped = false;


  constructor(features: number[][], labels: number[], numNeurons = 16) {
    super();
    this._dim = features[0].length;
    this._N = numNeurons;

    const fi = trainValSplit(features.map((f, i) => ({ f, l: labels[i] })));
    this._trainF = fi.train.map(d => d.f); this._trainL = fi.train.map(d => d.l);
    this._valF = fi.val.map(d => d.f); this._valL = fi.val.map(d => d.l);

    const { means, stds } = standardizeMulti(this._trainF, this._dim);
    this._means = means; this._stds = stds;

    this._init();
  }

  // ===== 一步训练 =====

  step(): V5EpochEvent {
    if (this._stopped) return { ...this.history[this.history.length - 1], stopped: true };

    const batch = sampleBatchMulti(this._trainF, this._trainL, BATCH);
    this._model.zeroGrad();
    let totalLoss = 0;

    for (let k = 0; k < batch.length; k++) {
      const x = batch[k].feature.map((v, di) => (v - this._means[di]) / this._stds[di]);
      const yPred = this._model.forward(x)[0];
      const diff = yPred - batch[k].label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();
    const trainLoss = totalLoss / BATCH;
    const valLoss = this._evaluate();

    if (valLoss < this._bestVal) { this._bestVal = valLoss; this._patience = 0; }
    else { this._patience++; }

    const stopped = this._patience >= PATIENCE;
    if (stopped) this._stopped = true;

    const ev: V5EpochEvent = {
      params: this.params, trainLoss: Number(trainLoss.toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)), stopped: stopped || undefined,
    };
    this.history.push(ev);
    return ev;
  }

  reset(): void { this.history.length = 0; this._stopped = false; this._patience = 0; this._bestVal = Infinity; this._init(); }
  isDone(): boolean { return this._stopped; }

  predict(xRaw: number[]): number {
    return this._model.forward(xRaw.map((v, i) => (v - this._means[i]) / this._stds[i]))[0];
  }

  // ── 内部 ──

  private _init() {
    this._model = new Sequential([new Linear(this._dim, this._N), new ReLU(), new Linear(this._N, 1)]);
    this._opt = new Adam(this._model.parameters(), LR);
  }

  private _standardize(x: number[]): number[] {
    return x.map((v, i) => (v - this._means[i]) / this._stds[i]);
  }

  private _evaluate(): number {
    let loss = 0;
    for (let k = 0; k < this._valF.length; k++) {
      const diff = this._model.forward(this._standardize(this._valF[k]))[0] - this._valL[k];
      loss += diff * diff;
    }
    return loss / this._valF.length;
  }
}
