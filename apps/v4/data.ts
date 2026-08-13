/** 数据集 — feature + label 打包在一起 */
export interface Dataset {
  features: number[];
  labels: number[];
}

/** 非线性数据 — y = sin(x) + noise */
export function sinData(size = 60) {
  const xScale = 2 * Math.PI;
  const trueFn = (x: number) => Math.sin(x * xScale);
  const features = Array.from({ length: size }, (_, i) => i / (size - 1 || 1));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 0.3);
  return { features, labels, trueFn, xScale };
}

export class DataLoader {
  constructor(
    readonly dataset: Dataset,
    private _batchSize: number,
  ) {}

  /** 随机抽 batchSize 个样本（无放回） */
  generate(): { feature: number; label: number }[] {
    const n = this.dataset.features.length;
    const m = Math.min(this._batchSize, n);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, m);
    return indices.map((i) => ({ feature: this.dataset.features[i], label: this.dataset.labels[i] }));
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

/** 完整数据准备：split → 标准化 → 返回训练集和验证集两个 Dataset */
export function prepare(features: number[], labels: number[]): { train: Dataset; val: Dataset } {
  const fi = trainValSplit(features.map((f, i) => ({ f, l: labels[i] })));
  let trainF = fi.train.map(d => d.f), trainL = fi.train.map(d => d.l);
  let valF = fi.val.map(d => d.f), valL = fi.val.map(d => d.l);

  const mean = trainF.reduce((s, v) => s + v, 0) / trainF.length;
  const v = trainF.reduce((s, x) => s + (x - mean) ** 2, 0) / trainF.length;
  const std = Math.sqrt(v) || 1;

  return {
    train: { features: trainF.map(x => (x - mean) / std), labels: trainL },
    val:   { features: valF.map(x => (x - mean) / std),   labels: valL },
  };
}
