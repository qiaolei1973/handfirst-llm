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
import { Trainer as BaseTrainer } from "@handfirst/utils";
import { Linear } from "./nn/linear";
import { ReLU } from "./nn/relu";
import { Sequential } from "./nn/sequential";
import { Adam } from "./nn/adam";

// ===== Parameter shape =====

export interface V4Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

// ===== Epoch event =====

export interface V4EpochEvent {
  params: V4Params;
  trainLoss: number;
  valLoss: number;
  isBest: boolean;
  stopped?: boolean;
}

// ===== Trainer =====

const BATCH = 40;
const PATIENCE = 300;

export class Trainer extends BaseTrainer<V4Params, V4EpochEvent> {
  declare params: V4Params;
  private _model!: Sequential;
  private _opt!: Adam;
  private _numNeurons: number;

  // 数据集
  private _trainSet!: { features: number[]; labels: number[] };
  private _valSet!: { features: number[]; labels: number[] };

  // 标准化统计量（从训练集计算）
  private _xMean = 0;
  private _xStd = 1;

  // Early stopping
  private _bestValLoss = Infinity;
  private _bestParams: V4Params | null = null;
  private _patienceCounter = 0;
  private _stopped = false;

  constructor(
    dataset: { features: number[]; labels: number[] },
    numNeurons = 16,
  ) {
    super();
    this._numNeurons = numNeurons;

    // ---- 80/20 train/val split ----
    const n = dataset.features.length;
    const nTrain = Math.floor(n * 0.8);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
    const trainIdx = new Set(indices.slice(0, nTrain));

    const trainFeatures: number[] = [], trainLabels: number[] = [];
    const valFeatures: number[] = [], valLabels: number[] = [];
    for (let i = 0; i < n; i++) {
      if (trainIdx.has(i)) {
        trainFeatures.push(dataset.features[i]);
        trainLabels.push(dataset.labels[i]);
      } else {
        valFeatures.push(dataset.features[i]);
        valLabels.push(dataset.labels[i]);
      }
    }
    this._trainSet = { features: trainFeatures, labels: trainLabels };
    this._valSet = { features: valFeatures, labels: valLabels };

    // ---- 标准化 ----
    this._xMean = trainFeatures.reduce((s, v) => s + v, 0) / nTrain;
    const variance = trainFeatures.reduce(
      (s, v) => s + (v - this._xMean) ** 2, 0,
    ) / nTrain;
    this._xStd = Math.sqrt(variance) || 1;

    this._init();
  }

  // ===== 一步训练 =====

  step(): V4EpochEvent {
    if (this._stopped) {
      return { ...this.history[this.history.length - 1], stopped: true };
    }

    const batch = sampleBatch(this._trainSet, BATCH);
    this._model.zeroGrad();

    let totalLoss = 0;
    for (let k = 0; k < batch.length; k++) {
      const x = this._standardize(batch[k].feature);
      const y = batch[k].label;

      const yPred = this._model.forward([x])[0];
      const diff = yPred - y;
      totalLoss += diff * diff;

      const gradOut = new Float64Array([(2 * diff) / BATCH]);
      this._model.backward(gradOut);
    }

    this._opt.step();
    const trainLoss = totalLoss / BATCH;
    const valLoss = this._evaluate();

    // Early stopping
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

    const ev: V4EpochEvent = {
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

  predict(xRaw: number): number {
    return this._model.forward([this._standardize(xRaw)])[0];
  }

  getNeurons(): { w: number; b: number }[] {
    const hw = this._model.layers[0] as Linear;
    return Array.from({ length: this._numNeurons }, (_, i) => ({
      w: hw.w[i], b: hw.b[i],
    }));
  }

  // ===== 内部 =====

  private _init() {
    const hidden = new Linear(1, this._numNeurons);
    this._model = new Sequential([hidden, new ReLU(), new Linear(this._numNeurons, 1)]);
    this._opt = new Adam(this._model.parameters(), 0.001);
    this._syncParams();
  }

  private _standardize(x: number) { return (x - this._xMean) / this._xStd; }

  private _evaluate(): number {
    let totalLoss = 0;
    for (let k = 0; k < this._valSet.features.length; k++) {
      const x = this._standardize(this._valSet.features[k]);
      const y = this._valSet.labels[k];
      const diff = this._model.forward([x])[0] - y;
      totalLoss += diff * diff;
    }
    return totalLoss / this._valSet.features.length;
  }

  private _snapshotParams(): V4Params {
    this._syncParams();
    return { ...this.params };
  }

  private _restoreParams(p: V4Params) {
    const hw = this._model.layers[0] as Linear;
    const ow = this._model.layers[2] as Linear;
    for (let i = 0; i < p.hiddenW.length; i++) hw.w[i] = p.hiddenW[i];
    for (let i = 0; i < p.hiddenB.length; i++) hw.b[i] = p.hiddenB[i];
    for (let i = 0; i < p.outputW.length; i++) ow.w[i] = p.outputW[i];
    ow.b[0] = p.outputB;
    this._syncParams();
  }

  private _syncParams() {
    const hw = this._model.layers[0] as Linear;
    const ow = this._model.layers[2] as Linear;
    this.params = {
      numNeurons: this._numNeurons,
      hiddenW: Array.from(hw.w),
      hiddenB: Array.from(hw.b),
      outputW: Array.from(ow.w),
      outputB: ow.b[0],
    };
  }
}

// ===== 终端运行 =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);

  for (let epoch = 0; epoch < 3000; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || epoch === 2999 || ev.stopped) {
      const bestMark = ev.isBest ? " ★" : "";
      console.log(
        `epoch ${epoch + 1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${bestMark}`,
      );
    }
    if (ev.stopped) break;
  }
}
