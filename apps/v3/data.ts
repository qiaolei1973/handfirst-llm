/** 数据集 — feature + label 打包在一起 */
export interface Dataset {
  features: number[];
  labels: number[];
}

/** 非线性数据 — y = sin(x) + noise, x ∈ [0,1] 对应 [0, 2π] */
export function sinData(size = 60) {
  const xScale = 2 * Math.PI;
  const trueFn = (x: number) => Math.sin(x * xScale);
  const features = Array.from({ length: size }, (_, i) => i / (size - 1 || 1));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 0.3);
  return { features, labels, trueFn, xScale };
}

/** 数据加载器：随机抽一批数据（mini-batch） */
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
