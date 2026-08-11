// ============================================================
// 神经网络 + 标准化 + Adam + 训练/验证分离 + Early Stopping
//
// 模型: y = Σ w_out_i · ReLU(w_i · x_std + b_i) + b_out
//       x_std = (x - μ) / σ    ← 标准化（Z-score）
//
// 终端验证: pnpm exec tsx apps/v4/train.ts
// 浏览器 viz: pnpm dev:v4
// ============================================================

import { fileURLToPath } from "node:url";
import { sinData, sampleBatch } from "@handfirst/datasets";
import { Trainer as BaseTrainer, arr, setArr } from "@handfirst/utils";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";
import { trainValSplit, standardize1D } from "./data";

// ===== Types =====

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
  declare params: V4Params;
  private _model!: Sequential;
  private _opt!: Adam;
  private _N: number;

  private _trainF!: number[]; private _trainL!: number[];
  private _valF!: number[]; private _valL!: number[];
  private _xMean = 0; private _xStd = 1;

  private _bestValLoss = Infinity; private _bestP: V4Params | null = null;
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

  step(): V4EpochEvent {
    if (this._stopped) return { ...this.history[this.history.length - 1], stopped: true };

    const batch = sampleBatch({ features: this._trainF, labels: this._trainL }, BATCH);
    this._model.zeroGrad();
    let totalLoss = 0;

    for (let k = 0; k < batch.length; k++) {
      const x = (batch[k].feature - this._xMean) / this._xStd;
      const yPred = this._model.forward([x])[0]; const diff = yPred - batch[k].label;
      totalLoss += diff * diff;
      this._model.backward(new Float64Array([(2 * diff) / BATCH]));
    }

    this._opt.step();
    const trainLoss = totalLoss / BATCH; const valLoss = this._evaluate();

    let isBest = false;
    if (valLoss < this._bestValLoss) {
      this._bestValLoss = valLoss; isBest = true; this._patience = 0; this._bestP = this._snap();
    } else { this._patience++; }

    const stopped = this._patience >= PATIENCE;
    if (stopped) { this._stopped = true; if (this._bestP) this._restore(this._bestP); }

    this._sync();
    const ev: V4EpochEvent = { params: { ...this.params }, trainLoss: Number(trainLoss.toFixed(6)), valLoss: Number(valLoss.toFixed(6)), isBest, stopped: stopped || undefined };
    this.history.push(ev);
    return ev;
  }

  reset(): void { this.history.length = 0; this._stopped = false; this._patience = 0; this._bestValLoss = Infinity; this._init(); }
  isDone(): boolean { return this._stopped; }
  predict(xRaw: number): number { return this._model.forward([(xRaw - this._xMean) / this._xStd])[0]; }

  getNeurons(): { w: number; b: number }[] {
    const hw = this._model.layers[0] as Linear;
    return Array.from({ length: this._N }, (_, i) => ({ w: hw.w[i], b: hw.b[i] }));
  }

  // ── 内部 ──
  private _init() {
    const N = this._N;
    this._model = new Sequential([new Linear(1, N), new ReLU(), new Linear(N, 1)]);
    this._opt = new Adam(this._model.parameters(), 0.001);
    this._sync();
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

  private _snap(): V4Params { this._sync(); return { ...this.params }; }
  private _restore(p: V4Params) {
    setArr((this._model.layers[0] as Linear).w, p.hiddenW);
    setArr((this._model.layers[0] as Linear).b, p.hiddenB);
    setArr((this._model.layers[2] as Linear).w, p.outputW);
    (this._model.layers[2] as Linear).b[0] = p.outputB;
    this._sync();
  }

  private _sync() {
    const hw = this._model.layers[0] as Linear, ow = this._model.layers[2] as Linear;
    this.params = { numNeurons: this._N, hiddenW: arr(hw.w), hiddenB: arr(hw.b), outputW: arr(ow.w), outputB: ow.b[0] };
  }
}

// ===== CLI =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = sinData(60);
  const t = new Trainer(features, labels, 16);
  for (let e = 0; e < 3000; e++) {
    const ev = t.step();
    if (e % 100 === 0 || e === 2999 || ev.stopped) console.log(`epoch ${e+1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${ev.isBest ? " ★" : ""}`);
    if (ev.stopped) break;
  }
}
