// ============================================================
// 神经网络 + 标准化 + Adam + 训练/验证分离 + Early Stopping
//
// 模型: y = Σ w_out_i · ReLU(w_i · x_std + b_i) + b_out
//       x_std = (x - μ) / σ    ← 标准化（Z-score）
//
// 终端验证: pnpm exec tsx train.ts
// 浏览器 viz: pnpm dev:v4
// ============================================================

import { fileURLToPath } from 'node:url';
import { sinData, sampleBatch } from '@handfirst/datasets';
import { Trainer as BaseTrainer, Layer } from '@handfirst/utils';
import type { EpochEvent } from '@handfirst/utils';

// ===== Parameter shape =====

export interface V4Params {
  numNeurons: number;
  hiddenW: number[];
  hiddenB: number[];
  outputW: number[];
  outputB: number;
}

// ===== Epoch event — v4 adds trainLoss/valLoss/stopped =====

export interface V4EpochEvent {
  params: V4Params;
  trainLoss: number;
  valLoss: number;
  isBest: boolean;       // 当前是否为最佳验证 loss
  stopped?: boolean;      // early stopping 触发
}

// ===== Adam state for a scalar parameter =====

interface AdamState {
  m: number;
  v: number;
}

// ===== Trainer =====

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const PATIENCE = 300;          // 连续 300 epoch val loss 不降就停

export class Trainer extends BaseTrainer<V4Params, V4EpochEvent> {
  params!: V4Params;
  private _hidden!: Layer;
  private _output!: Layer;
  private _numNeurons: number;
  private _lr = 0.001;          // Adam 默认学习率
  private _batchSize = 40;

  // 训练/验证集
  private _trainSet!: { features: number[]; labels: number[] };
  private _valSet!: { features: number[]; labels: number[] };

  // 标准化统计量（从训练集计算）
  private _xMean = 0;
  private _xStd = 1;

  // Adam 状态
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

    // 随机打乱再切分
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
    const trainIdx = new Set(indices.slice(0, nTrain));
    const valIdx = new Set(indices.slice(nTrain));

    const trainFeatures: number[] = [];
    const trainLabels: number[] = [];
    const valFeatures: number[] = [];
    const valLabels: number[] = [];

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

    // ---- 计算标准化统计量（只用训练集！） ----
    const nT = trainFeatures.length;
    this._xMean = trainFeatures.reduce((s, v) => s + v, 0) / nT;
    const variance =
      trainFeatures.reduce((s, v) => s + (v - this._xMean) ** 2, 0) / nT;
    this._xStd = Math.sqrt(variance) || 1; // 防止 std=0

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

  /** 训练是否已完成（early stopping 触发） */
  isDone(): boolean {
    return this._stopped;
  }

  // ===== 一步训练 =====

  step(): V4EpochEvent {
    if (this._stopped) {
      // 已经 early stop，返回最后一个事件
      const last = this.history[this.history.length - 1];
      return { ...last, stopped: true };
    }

    const batch = sampleBatch(this._trainSet, this._batchSize);

    // ---- 清零梯度 ----
    this._hidden.zeroGrad();
    this._output.zeroGrad();

    let totalLoss = 0;

    // ---- Mini-batch: forward + backward ----
    for (let k = 0; k < batch.length; k++) {
      const xRaw = batch[k].feature;
      const y = batch[k].label;
      const x = this._standardize(xRaw);

      // 前向传播
      const h = this._hidden.forward([x]);
      const yPred = this._output.forward(Array.from(h));

      // MSE loss 累加
      const diff = yPred[0] - y;
      totalLoss += diff * diff;

      // 反向传播
      const gradOut = new Float64Array([(2 * diff) / this._batchSize]);
      const gradH = this._output.backward(gradOut);
      this._hidden.backward(gradH);
    }

    const trainLoss = totalLoss / this._batchSize;

    // ---- Adam 更新参数 ----
    this._t++;

    // 隐藏层 W
    for (let i = 0; i < this._hidden.w.length; i++) {
      const [mNew, vNew] = this._adamUpdate(
        this._hidden.gradW[i],
        this._mHiddenW[i],
        this._vHiddenW[i],
      );
      this._mHiddenW[i] = mNew;
      this._vHiddenW[i] = vNew;
      this._hidden.w[i] -= this._adamStep(mNew, vNew);
    }
    // 隐藏层 B
    for (let i = 0; i < this._hidden.b.length; i++) {
      const [mNew, vNew] = this._adamUpdate(
        this._hidden.gradB[i],
        this._mHiddenB[i],
        this._vHiddenB[i],
      );
      this._mHiddenB[i] = mNew;
      this._vHiddenB[i] = vNew;
      this._hidden.b[i] -= this._adamStep(mNew, vNew);
    }
    // 输出层 W
    for (let i = 0; i < this._output.w.length; i++) {
      const [mNew, vNew] = this._adamUpdate(
        this._output.gradW[i],
        this._mOutputW[i],
        this._vOutputW[i],
      );
      this._mOutputW[i] = mNew;
      this._vOutputW[i] = vNew;
      this._output.w[i] -= this._adamStep(mNew, vNew);
    }
    // 输出层 B
    {
      const [mNew, vNew] = this._adamUpdate(
        this._output.gradB[0],
        this._mOutputB,
        this._vOutputB,
      );
      this._mOutputB = mNew;
      this._vOutputB = vNew;
      this._output.b[0] -= this._adamStep(mNew, vNew);
    }

    // ---- 验证集评估 ----
    const valLoss = this._evaluate();

    // ---- Early stopping ----
    let isBest = false;
    if (valLoss < this._bestValLoss) {
      this._bestValLoss = valLoss;
      this._patienceCounter = 0;
      isBest = true;
      // 保存最佳参数
      this._bestParams = this._snapshotParams();
    } else {
      this._patienceCounter++;
    }

    const stopped = this._patienceCounter >= PATIENCE;
    if (stopped) {
      this._stopped = true;
      // 恢复最佳参数
      if (this._bestParams) this._restoreParams(this._bestParams);
    }

    // ---- 同步 params ----
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

  // ===== 预测 =====

  predict(xRaw: number): number {
    const x = this._standardize(xRaw);
    const h = this._hidden.forward([x]);
    return this._output.forward(Array.from(h))[0];
  }

  getNeurons(): { w: number; b: number }[] {
    const out: { w: number; b: number }[] = [];
    for (let i = 0; i < this._numNeurons; i++) {
      out.push({ w: this._hidden.w[i], b: this._hidden.b[i] });
    }
    return out;
  }

  // ===== 内部：标准化 =====

  private _standardize(x: number): number {
    return (x - this._xMean) / this._xStd;
  }

  // ===== 内部：Adam 更新 =====

  /** 计算更新后的 m, v */
  private _adamUpdate(
    grad: number,
    m: number,
    v: number,
  ): [number, number] {
    const mNew = BETA1 * m + (1 - BETA1) * grad;
    const vNew = BETA2 * v + (1 - BETA2) * grad * grad;
    return [mNew, vNew];
  }

  /** 计算实际步长：lr * m̂ / √(v̂ + ε) */
  private _adamStep(m: number, v: number): number {
    const mHat = m / (1 - Math.pow(BETA1, this._t));
    const vHat = v / (1 - Math.pow(BETA2, this._t));
    return this._lr * mHat / (Math.sqrt(vHat) + EPS);
  }

  // ===== 内部：验证集评估 =====

  private _evaluate(): number {
    let totalLoss = 0;
    const n = this._valSet.features.length;
    for (let k = 0; k < n; k++) {
      const x = this._standardize(this._valSet.features[k]);
      const y = this._valSet.labels[k];
      const h = this._hidden.forward([x]);
      const yPred = this._output.forward(Array.from(h))[0];
      const diff = yPred - y;
      totalLoss += diff * diff;
    }
    return totalLoss / n;
  }

  // ===== 内部：参数快照 =====

  private _snapshotParams(): V4Params {
    return {
      numNeurons: this._numNeurons,
      hiddenW: Array.from(this._hidden.w),
      hiddenB: Array.from(this._hidden.b),
      outputW: Array.from(this._output.w),
      outputB: this._output.b[0],
    };
  }

  private _restoreParams(p: V4Params): void {
    for (let i = 0; i < p.hiddenW.length; i++) this._hidden.w[i] = p.hiddenW[i];
    for (let i = 0; i < p.hiddenB.length; i++) this._hidden.b[i] = p.hiddenB[i];
    for (let i = 0; i < p.outputW.length; i++) this._output.w[i] = p.outputW[i];
    this._output.b[0] = p.outputB;
    this._syncParams();
  }

  // ===== 内部 =====

  private _init(): void {
    this._hidden = new Layer(1, this._numNeurons, 'relu');
    this._output = new Layer(this._numNeurons, 1, 'linear');

    // 初始化 Adam 状态
    this._mHiddenW = new Float64Array(this._numNeurons);
    this._vHiddenW = new Float64Array(this._numNeurons);
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

  private _syncParams(): void {
    this.params = {
      numNeurons: this._numNeurons,
      hiddenW: Array.from(this._hidden.w),
      hiddenB: Array.from(this._hidden.b),
      outputW: Array.from(this._output.w),
      outputB: this._output.b[0],
    };
  }
}

// ===== 终端运行 =====
//   pnpm exec tsx train.ts

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { features, labels, trueFn } = sinData(60);
  const t = new Trainer({ features, labels }, 16);

  const MAX_EPOCHS = 3000;
  for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    const ev = t.step();
    if (epoch % 100 === 0 || epoch === MAX_EPOCHS - 1 || ev.stopped) {
      const bestMark = ev.isBest ? ' ★' : '';
      console.log(
        `epoch ${epoch + 1}: trainLoss=${ev.trainLoss.toFixed(6)}  valLoss=${ev.valLoss.toFixed(6)}${bestMark}`,
      );
    }
    if (ev.stopped) {
      console.log(`\nEarly stopping at epoch ${epoch + 1}!`);
      console.log(`Best val loss: ${t['_bestValLoss'].toFixed(6)}`);
      break;
    }
  }

  // 验证: 预测几个点
  console.log('\n最终模型预测 vs 真实值:');
  for (const xNorm of [0, 0.25, 0.5, 0.75]) {
    console.log(
      `  x̂=${xNorm.toFixed(2)}  predict=${t.predict(xNorm).toFixed(4)}  true=${trueFn(xNorm).toFixed(4)}`,
    );
  }
}
