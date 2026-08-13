import { Mat } from "@handfirst/utils";

/** 数据集 — feature + label 打包在一起 */
export interface Dataset {
  features: number[][];
  labels: number[];
}

/** 2D 曲面数据 — f(x₁,x₂) = sin(√(x₁²+x₂²) · 2π) + noise */
export function surfaceData(size = 200) {
  const scale = 2 * Math.PI;
  const trueFn = (x1: number, x2: number) => Math.sin(Math.sqrt(x1 * x1 + x2 * x2) * scale);
  const features: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < size; i++) {
    const x1 = 2 * Math.random() - 1;
    const x2 = 2 * Math.random() - 1;
    features.push([x1, x2]);
    labels.push(trueFn(x1, x2) + (Math.random() - 0.5) * 0.3);
  }
  return { features, labels, trueFn, scale };
}

export class DataLoader {
  private _dim: number;

  constructor(
    readonly dataset: Dataset,
    private _batchSize: number,
  ) {
    this._dim = dataset.features[0].length;
  }

  /** 随机抽 batchSize 个样本，堆成 [dim × B] 输入矩阵和 [1 × B] 标签矩阵 */
  generate(): { X: Mat; Y: Mat } {
    const n = this.dataset.features.length;
    const m = Math.min(this._batchSize, n);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, m);

    const X = new Mat(this._dim, m);
    const Y = new Mat(1, m);
    for (let b = 0; b < m; b++) {
      const f = this.dataset.features[indices[b]];
      for (let i = 0; i < this._dim; i++) X.data[i * m + b] = f[i];
      Y.data[b] = this.dataset.labels[indices[b]];
    }
    return { X, Y };
  }
}

function trainValSplit<T>(data: T[]) {
  const n = data.length;
  const nTrain = Math.floor(n * 0.8);
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
  const trainIdx = new Set(indices.slice(0, nTrain));
  const train: T[] = [], val: T[] = [];
  for (let i = 0; i < n; i++) {
    (trainIdx.has(i) ? train : val).push(data[i]);
  }
  return { train, val };
}

function standardizeMulti(data: number[][], dim: number) {
  const n = data.length;
  const means = Array(dim).fill(0), stds = Array(dim).fill(1);
  for (let d = 0; d < dim; d++) {
    means[d] = data.reduce((s, f) => s + f[d], 0) / n;
    const v = data.reduce((s, f) => s + (f[d] - means[d]) ** 2, 0) / n;
    stds[d] = Math.sqrt(v) || 1;
  }
  return { means, stds };
}

/** 完整数据准备：split → 标准化 → 返回训练集和验证集两个 Dataset */
export function prepare(
  features: number[][], labels: number[],
): { train: Dataset; val: Dataset } {
  const dim = features[0].length;
  const fi = trainValSplit(features.map((f, i) => ({ f, l: labels[i] })));
  let trainF = fi.train.map(d => d.f), trainL = fi.train.map(d => d.l);
  let valF = fi.val.map(d => d.f), valL = fi.val.map(d => d.l);

  const { means, stds } = standardizeMulti(trainF, dim);
  trainF = trainF.map(f => f.map((v, i) => (v - means[i]) / stds[i]));
  valF = valF.map(f => f.map((v, i) => (v - means[i]) / stds[i]));

  return {
    train: { features: trainF, labels: trainL },
    val:   { features: valF,   labels: valL },
  };
}
