// ============================================================
// 神经网络 + 标准化 + Adam + 训练/验证分离 + Early Stopping
//
// 模型: y = Σ w_out_i · ReLU(w_i · x_std + b_i) + b_out
//       x_std = (x - μ) / σ
// ============================================================

import { fileURLToPath } from "node:url";
import { Trainer as BaseTrainer, arr } from "@handfirst/utils";
import { sinData, sampleBatch } from "@handfirst/datasets";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";
import { trainValSplit, standardize1D } from "./data";

// ===== 参数形状（仪表盘协议） =====

export interface V4Params {
  numNeurons: number; hiddenW: number[]; hiddenB: number[];
  outputW: number[]; outputB: number;
}

export interface V4EpochEvent {
  params: V4Params; trainLoss: number; valLoss: number;
  isBest: boolean; stopped?: boolean;
}

// ===== Trainer =====

const BATCH = 40, PATIENCE = 300;

export class Trainer extends BaseTrainer<V4Params, V4EpochEvent> {
  get params(): V4Params {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    return { numNeurons: this._N, hiddenW: arr(hw.w), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }

  private _model!: Sequential;
  private _opt!: Adam;
  private _N: number;

  private _trainF!: number[]; private _trainL!: number[];
  private _valF!: number[]; private _valL!: number[];
  private _xMean = 0; private _xStd = 1;

  private _bestVal = Infinity;
  private _bestState: Float64Array[] | null = null;
  private _patience = 0; private _stopped = false;

  constructor(features: number[], labels: number[], numNeurons = 16) {
    super();
    this._N = numNeurons;

    const fi = trainValSplit(features.map((f, i) => ({ f, l: labels[i] })));
    this._trainF = fi.train.map(d => d.f); this._trainL = fi.train.map(d => d.l);
    this._valF = fi.val.map(d => d.f); this._valL = fi.val.map(d => d.l);

    const { mean, std } = standardize1D(this._trainF);
    this._xMean = mean; this._xStd = std;

    this._init();
  }

  // ===== 一步训练 =====

  step(): V4EpochEvent {
    if (this._stopped) return { ...this.history[this.history.length - 1], stopped: true };

    const batch = sampleBatch({ features: this._trainF, labels: this._trainL }, BATCH);
    this._model.zeroGrad();
    let totalLoss = 0;

    for (let k = 0; k < batch.length; k++) {
      const x = (batch[k].feature - this._xMean) / this._xStd;
      const yPred = this._model.forward([x])[0];
      const diff = yPred - batch[k].label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();
    const trainLoss = totalLoss / BATCH;
    const valLoss = this._evaluate();

    let isBest = false;
    if (valLoss < this._bestVal) {
      this._bestVal = valLoss; isBest = true; this._patience = 0;
      this._bestState = this._model.stateDict();
    } else { this._patience++; }

    const stopped = this._patience >= PATIENCE;
    if (stopped) { this._stopped = true; this._model.loadStateDict(this._bestState!); }

    const ev: V4EpochEvent = {
      params: this.params, trainLoss: Number(trainLoss.toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)), isBest, stopped: stopped || undefined,
    };
    this.history.push(ev);
    return ev;
  }

  reset(): void { this.history.length = 0; this._stopped = false; this._patience = 0; this._bestVal = Infinity; this._bestState = null; this._init(); }
  isDone(): boolean { return this._stopped; }

  predict(xRaw: number): number {
    return this._model.forward([(xRaw - this._xMean) / this._xStd])[0];
  }

  getNeurons(): { w: number; b: number }[] {
    const hw = this._model.layers[0] as Linear;
    return Array.from({ length: this._N }, (_, i) => ({ w: hw.w[i], b: hw.b[i] }));
  }

  // ── 内部 ──

  private _init() {
    this._model = new Sequential([new Linear(1, this._N), new ReLU(), new Linear(this._N, 1)]);
    this._opt = new Adam(this._model.parameters(), 0.001);
  }

  private _evaluate(): number {
    let loss = 0;
    for (let k = 0; k < this._valF.length; k++) {
      const x = (this._valF[k] - this._xMean) / this._xStd;
      const diff = this._model.forward([x])[0] - this._valL[k];
      loss += diff * diff;
    }
    return loss / this._valF.length;
  }
}

// ===== 终端验证 =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = sinData(60);
  const t = new Trainer(features, labels, 16);
  for (let e = 0; e < 3000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || e === 2999 || ev.stopped) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${ev.isBest ? " ★" : ""}`);
    if (ev.stopped) break;
  }
}
