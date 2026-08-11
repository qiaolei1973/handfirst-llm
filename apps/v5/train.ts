// ============================================================
// 多维输入神经网络 — d 个特征 → N 个 ReLU 神经元 → 预测 ŷ
//
// 模型: h_j = ReLU(Σ_i W_{j,i}·x_i + b_j),  ŷ = Σ_j w_out_j·h_j + b_out
//
// 终端验证: pnpm exec tsx train.ts
// 浏览器 viz: pnpm dev:v5
// ============================================================

import { fileURLToPath } from "node:url";
import { surfaceData } from "@handfirst/datasets";
import { Trainer as BaseTrainer, Layer } from "@handfirst/utils";
import type { EpochEvent } from "@handfirst/utils";

// ===== Parameter shape =====

export interface V5Params {
  inputDim: number;
  numNeurons: number;
  hiddenW: number[][];  // [N × d] 权重矩阵
  hiddenB: number[];    // [N]
  outputW: number[];    // [N]
  outputB: number;
}

// ===== Epoch event =====

export interface V5EpochEvent extends EpochEvent<V5Params> {
  trainLoss: number;
  valLoss: number;
  isBest: boolean;
  stopped?: boolean;
}

// ===== Adam state =====

interface AdamMatrix {
  m: number[][];
  v: number[][];
}

// ===== Trainer =====

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const PATIENCE = 200;

export class Trainer extends BaseTrainer<V5Params, V5EpochEvent> {
  private _hidden!: Layer;
  private _output!: Layer;
  private _inputDim: number;
  private _numNeurons: number;
  private _lr = 0.005;
  private _batchSize = 40;

  // 训练/验证集
  private _trainSet!: { features: number[][]; labels: number[] };
  private _valSet!: { features: number[][]; labels: number[] };

  // 标准化统计量
  private _xMeans!: number[];
  private _xStds!: number[];

  // Adam state
  private _t = 0;
  private _mHiddenW!: Float64Array;
  private _vHiddenW!: Float64Array;
  private _mHiddenB!: Float64Array;
  private _vHiddenB!: Float64Array;
  private _mOutputW!: Float64Array;
  private _vOutputW!: Float64Array;
  private _mOutputB = 0;
  private _vOutputB = 0;

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
    this._inputDim = features[0].length;
    this._numNeurons = numNeurons;

    // ---- 80/20 train/val split ----
    const n = features.length;
    const nTrain = Math.floor(n * 0.8);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
    const trainIdx = new Set(indices.slice(0, nTrain));
    const valIdx = new Set(indices.slice(nTrain));

    const trainFeatures: number[][] = [];
    const trainLabels: number[] = [];
    const valFeatures: number[][] = [];
    const valLabels: number[] = [];

    for (let i = 0; i < n; i++) {
      if (trainIdx.has(i)) {
        trainFeatures.push(features[i]);
        trainLabels.push(labels[i]);
      } else {
        valFeatures.push(features[i]);
        valLabels.push(labels[i]);
      }
    }

    this._trainSet = { features: trainFeatures, labels: trainLabels };
    this._valSet = { features: valFeatures, labels: valLabels };

    // ---- 标准化统计量 ----
    this._xMeans = Array(this._inputDim).fill(0);
    this._xStds = Array(this._inputDim).fill(1);
    for (let d = 0; d < this._inputDim; d++) {
      let sum = 0;
      for (const f of trainFeatures) sum += f[d];
      this._xMeans[d] = sum / nTrain;
      let v = 0;
      for (const f of trainFeatures) v += (f[d] - this._xMeans[d]) ** 2;
      this._xStds[d] = Math.sqrt(v / nTrain) || 1;
    }

    this._init();
  }

  reset(): void {
    this.history.length = 0;
    this._t = 0;
    this._bestValLoss = Infinity;
    this._bestParams = null;
    this._patienceCounter = 0;
    this._stopped = false;
    this._init();
  }

  isDone(): boolean { return this._stopped; }

  // ===== 一步训练 =====

  step(): V5EpochEvent {
    if (this._stopped) {
      const last = this.history[this.history.length - 1];
      return { ...last, stopped: true };
    }

    const batch = sampleBatchMulti(this._trainSet, this._batchSize);

    this._hidden.zeroGrad();
    this._output.zeroGrad();

    let totalLoss = 0;

    for (let k = 0; k < batch.length; k++) {
      const xRaw = batch[k].feature;
      const y = batch[k].label;
      const x = this._standardize(xRaw);

      // 前向
      const h = this._hidden.forward(x);
      const yPred = this._output.forward(Array.from(h));

      const diff = yPred[0] - y;
      totalLoss += diff * diff;

      // 反向
      const gradOut = new Float64Array([(2 * diff) / this._batchSize]);
      const gradH = this._output.backward(gradOut);
      this._hidden.backward(gradH);
    }

    const trainLoss = totalLoss / this._batchSize;

    // Adam 更新
    this._t++;
    this._adamStep(this._hidden.w, this._hidden.gradW, this._mHiddenW, this._vHiddenW);
    this._adamStep(this._hidden.b, this._hidden.gradB, this._mHiddenB, this._vHiddenB);
    this._adamStep(this._output.w, this._output.gradW, this._mOutputW, this._vOutputW);
    this._output.b[0] -= this._adamScalarStep(this._output.gradB[0], this._mOutputB, this._vOutputB);

    // 验证
    const valLoss = this._evaluate();

    let isBest = false;
    if (valLoss < this._bestValLoss) {
      this._bestValLoss = valLoss;
      this._patienceCounter = 0;
      isBest = true;
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

  // ===== 预测 =====

  predict(xRaw: number[]): number {
    const x = xRaw.map((v, i) => (v - this._xMeans[i]) / this._xStds[i]);
    const h = this._hidden.forward(x);
    return this._output.forward(Array.from(h))[0];
  }

  // 在整个网格上预测（给 dashboard 用）
  predictGrid(grid: number[][]): number[] {
    return grid.map((p) => this.predict(p));
  }

  // ===== 内部 =====

  private _init() {
    this._hidden = new Layer(this._inputDim, this._numNeurons, "relu");
    this._output = new Layer(this._numNeurons, 1, "linear");

    const hwLen = this._numNeurons * this._inputDim;
    this._mHiddenW = new Float64Array(hwLen);
    this._vHiddenW = new Float64Array(hwLen);
    this._mHiddenB = new Float64Array(this._numNeurons);
    this._vHiddenB = new Float64Array(this._numNeurons);
    this._mOutputW = new Float64Array(this._numNeurons);
    this._vOutputW = new Float64Array(this._numNeurons);
    this._mOutputB = 0;
    this._vOutputB = 0;
    this._t = 0;

    this._bestValLoss = Infinity;
    this._bestParams = null;
    this._patienceCounter = 0;
    this._stopped = false;
    this._syncParams();
  }

  private _standardize(x: number[]): number[] {
    return x.map((v, i) => (v - this._xMeans[i]) / this._xStds[i]);
  }

  private _adamStep(w: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) {
    for (let i = 0; i < w.length; i++) {
      m[i] = BETA1 * m[i] + (1 - BETA1) * g[i];
      v[i] = BETA2 * v[i] + (1 - BETA2) * g[i] * g[i];
      const mHat = m[i] / (1 - Math.pow(BETA1, this._t));
      const vHat = v[i] / (1 - Math.pow(BETA2, this._t));
      w[i] -= this._lr * mHat / (Math.sqrt(vHat) + EPS);
    }
  }

  private _adamScalarStep(g: number, mRef: number, vRef: number): number {
    const m = BETA1 * mRef + (1 - BETA1) * g;
    const v = BETA2 * vRef + (1 - BETA2) * g * g;
    const mHat = m / (1 - Math.pow(BETA1, this._t));
    const vHat = v / (1 - Math.pow(BETA2, this._t));
    return this._lr * mHat / (Math.sqrt(vHat) + EPS);
  }

  private _evaluate(): number {
    let totalLoss = 0;
    const n = this._valSet.features.length;
    for (let k = 0; k < n; k++) {
      const x = this._standardize(this._valSet.features[k]);
      const y = this._valSet.labels[k];
      const h = this._hidden.forward(x);
      const yPred = this._output.forward(Array.from(h))[0];
      const diff = yPred - y;
      totalLoss += diff * diff;
    }
    return totalLoss / n;
  }

  private _snapshotParams(): V5Params {
    return {
      inputDim: this._inputDim,
      numNeurons: this._numNeurons,
      hiddenW: this._matrixToArray(this._hidden.w, this._numNeurons, this._inputDim),
      hiddenB: Array.from(this._hidden.b),
      outputW: Array.from(this._output.w),
      outputB: this._output.b[0],
    };
  }

  private _restoreParams(p: V5Params) {
    this._arrayToMatrix(p.hiddenW, this._hidden.w);
    for (let i = 0; i < p.hiddenB.length; i++) this._hidden.b[i] = p.hiddenB[i];
    for (let i = 0; i < p.outputW.length; i++) this._output.w[i] = p.outputW[i];
    this._output.b[0] = p.outputB;
    this._syncParams();
  }

  private _syncParams() {
    this.params = {
      inputDim: this._inputDim,
      numNeurons: this._numNeurons,
      hiddenW: this._matrixToArray(this._hidden.w, this._numNeurons, this._inputDim),
      hiddenB: Array.from(this._hidden.b),
      outputW: Array.from(this._output.w),
      outputB: this._output.b[0],
    };
  }

  private _matrixToArray(data: Float64Array, rows: number, cols: number): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < rows; i++) {
      out.push(Array.from(data.subarray(i * cols, i * cols + cols)));
    }
    return out;
  }

  private _arrayToMatrix(arr: number[][], data: Float64Array) {
    for (let i = 0; i < arr.length; i++)
      for (let j = 0; j < arr[i].length; j++)
        data[i * arr[0].length + j] = arr[i][j];
  }
}

// Batch sampling for multi-dimensional features
function sampleBatchMulti(
  dataset: { features: number[][]; labels: number[] },
  size: number,
): { feature: number[]; label: number }[] {
  const indices = [...Array(dataset.features.length).keys()]
    .sort(() => Math.random() - 0.5)
    .slice(0, size);
  return indices.map((i) => ({
    feature: dataset.features[i],
    label: dataset.labels[i],
  }));
}

// ===== 终端运行 =====

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = surfaceData(200);
  const t = new Trainer(features, labels, 16);

  const MAX_EPOCHS = 2000;
  for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || epoch === MAX_EPOCHS - 1 || ev.stopped) {
      const bestMark = ev.isBest ? " ★" : "";
      console.log(
        `epoch ${epoch + 1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${bestMark}`,
      );
    }
    if (ev.stopped) {
      console.log(`\nEarly stopping at epoch ${epoch + 1}!`);
      console.log(`Best val loss: ${t["_bestValLoss"].toFixed(6)}`);
      break;
    }
  }
}
