// ============================================================
// 多维输入神经网络 — d 个特征 → N 个 ReLU 神经元 → ŷ
//
// 模型: h_j = ReLU(Σ_i W_{j,i}·x_i + b_j)
//        ŷ  = Σ_j w_out_j·h_j + b_out
//
// 终端验证: pnpm exec tsx apps/v5/train.ts
// 浏览器 viz: pnpm dev:v5
// ============================================================

import { fileURLToPath } from "node:url";
import { surfaceData } from "@handfirst/datasets";
import { Trainer as BaseTrainer } from "@handfirst/utils";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";

// ===== Parameter shape =====

export interface V5Params {
  inputDim: number;
  numNeurons: number;
  hiddenW: number[][];  // [N × d]
  hiddenB: number[];     // [N]
  outputW: number[];     // [N]
  outputB: number;
}

// ===== Epoch event =====

export interface V5EpochEvent {
  params: V5Params;
  trainLoss: number;
  valLoss: number;
  isBest: boolean;
  stopped?: boolean;
}

// ===== Trainer =====

const BATCH = 40;
const PATIENCE = 200;

export class Trainer extends BaseTrainer<V5Params, V5EpochEvent> {
  declare params: V5Params;
  private _model!: Sequential;
  private _opt!: Adam;
  private _inDim: number;
  private _numNeurons: number;

  private _trainSet!: { features: number[][]; labels: number[] };
  private _valSet!: { features: number[][]; labels: number[] };
  private _xMeans: number[] = [];
  private _xStds: number[] = [];

  // Early stopping
  private _bestValLoss = Infinity;
  private _bestParams: V5Params | null = null;
  private _patienceCounter = 0;
  private _stopped = false;

  constructor(
    features: number[][],
    labels: number[],
    numNeurons = 16,
  ) {
    super();
    this._inDim = features[0].length;
    this._numNeurons = numNeurons;

    // ---- 80/20 train/val split ----
    const n = features.length;
    const nTrain = Math.floor(n * 0.8);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
    const trainIdx = new Set(indices.slice(0, nTrain));

    const trainF: number[][] = [], trainL: number[] = [];
    const valF: number[][] = [], valL: number[] = [];
    for (let i = 0; i < n; i++) {
      if (trainIdx.has(i)) {
        trainF.push(features[i]); trainL.push(labels[i]);
      } else {
        valF.push(features[i]); valL.push(labels[i]);
      }
    }
    this._trainSet = { features: trainF, labels: trainL };
    this._valSet = { features: valF, labels: valL };

    // ---- 标准化（每个特征维度独立） ----
    this._xMeans = Array(this._inDim).fill(0);
    this._xStds = Array(this._inDim).fill(1);
    for (let d = 0; d < this._inDim; d++) {
      const s = trainF.reduce((sum, f) => sum + f[d], 0);
      this._xMeans[d] = s / nTrain;
      const v = trainF.reduce(
        (sum, f) => sum + (f[d] - this._xMeans[d]) ** 2, 0,
      ) / nTrain;
      this._xStds[d] = Math.sqrt(v) || 1;
    }

    this._init();
  }

  // ===== 一步训练 =====

  step(): V5EpochEvent {
    if (this._stopped) {
      return { ...this.history[this.history.length - 1], stopped: true };
    }

    const batch = sampleBatchMulti(this._trainSet, BATCH);
    this._model.zeroGrad();

    let totalLoss = 0;
    for (let k = 0; k < batch.length; k++) {
      const x = this._standardize(batch[k].feature);
      const y = batch[k].label;

      const yPred = this._model.forward(x)[0];
      const diff = yPred - y;
      totalLoss += diff * diff;

      const gradOut = new Float64Array([(2 * diff) / BATCH]);
      this._model.backward(gradOut);
    }

    this._opt.step();
    const trainLoss = totalLoss / BATCH;
    const valLoss = this._evaluate();

    let isBest = false;
    if (valLoss < this._bestValLoss) {
      this._bestValLoss = valLoss; isBest = true; this._patienceCounter = 0;
      this._bestParams = this._snapshotParams();
    } else {
      this._patienceCounter++;
    }

    const stopped = this._patienceCounter >= PATIENCE;
    if (stopped) {
      this._stopped = true;
      if (this._bestParams) this._restoreParams(this._bestParams);
    }

    this._syncParams();
    const ev: V5EpochEvent = {
      params: { ...this.params },
      trainLoss: Number(trainLoss.toFixed(6)),
      valLoss: Number(valLoss.toFixed(6)),
      isBest,
      stopped: stopped || undefined,
    };
    this.history.push(ev);
    return ev;
  }

  // ===== 接口 =====

  reset(): void { this.history.length = 0; this._stopped = false; this._patienceCounter = 0; this._bestValLoss = Infinity; this._init(); }
  isDone(): boolean { return this._stopped; }

  predict(xRaw: number[]): number {
    return this._model.forward(this._standardize(xRaw))[0];
  }

  predictGrid(grid: number[][]): number[] {
    return grid.map((p) => this.predict(p));
  }

  // ===== 内部 =====

  private _init() {
    const hidden = new Linear(this._inDim, this._numNeurons);
    this._model = new Sequential([hidden, new ReLU(), new Linear(this._numNeurons, 1)]);
    this._opt = new Adam(this._model.parameters(), 0.005);
    this._syncParams();
  }

  private _standardize(x: number[]): number[] {
    return x.map((v, i) => (v - this._xMeans[i]) / this._xStds[i]);
  }

  private _evaluate(): number {
    let totalLoss = 0;
    for (let k = 0; k < this._valSet.features.length; k++) {
      const x = this._standardize(this._valSet.features[k]);
      const y = this._valSet.labels[k];
      const diff = this._model.forward(x)[0] - y;
      totalLoss += diff * diff;
    }
    return totalLoss / this._valSet.features.length;
  }

  private _matrixW(): number[][] {
    const hw = this._model.layers[0] as Linear;
    const rows: number[][] = [];
    for (let j = 0; j < this._numNeurons; j++) {
      rows.push(Array.from(
        hw.w.subarray(j * this._inDim, (j + 1) * this._inDim),
      ));
    }
    return rows;
  }

  private _copyMatrixTo(arr: number[][], dst: Float64Array) {
    for (let j = 0; j < arr.length; j++)
      for (let i = 0; i < arr[j].length; i++)
        dst[j * this._inDim + i] = arr[j][i];
  }

  private _snapshotParams(): V5Params {
    this._syncParams();
    return { ...this.params };
  }

  private _restoreParams(p: V5Params) {
    (this._model.layers[0] as Linear).b.set(new Float64Array(p.hiddenB));
    (this._model.layers[2] as Linear).w.set(new Float64Array(p.outputW));
    (this._model.layers[2] as Linear).b[0] = p.outputB;
    this._copyMatrixTo(p.hiddenW, (this._model.layers[0] as Linear).w);
    this._syncParams();
  }

  private _syncParams() {
    const hw = this._model.layers[0] as Linear;
    const ow = this._model.layers[2] as Linear;
    this.params = {
      inputDim: this._inDim,
      numNeurons: this._numNeurons,
      hiddenW: this._matrixW(),
      hiddenB: Array.from(hw.b),
      outputW: Array.from(ow.w),
      outputB: ow.b[0],
    };
  }
}

// ===== 多维 batch 采样 =====

function sampleBatchMulti(
  dataset: { features: number[][]; labels: number[] },
  size: number,
): { feature: number[]; label: number }[] {
  const n = dataset.features.length;
  const indices = [...Array(n).keys()]
    .sort(() => Math.random() - 0.5)
    .slice(0, Math.min(size, n));
  return indices.map((i) => ({
    feature: dataset.features[i],
    label: dataset.labels[i],
  }));
}

// ===== 终端运行 =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels } = surfaceData(200);
  const t = new Trainer(features, labels, 16);

  for (let epoch = 0; epoch < 2000; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || ev.stopped) {
      const bestMark = ev.isBest ? " ★" : "";
      console.log(
        `epoch ${epoch + 1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${bestMark}`,
      );
    }
    if (ev.stopped) break;
  }
}
