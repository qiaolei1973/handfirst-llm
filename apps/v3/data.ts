/** 非线性数据 — y = sin(x) + noise, x ∈ [0,1] 对应 [0, 2π] */
export function sinData(size = 60) {
  const xScale = 2 * Math.PI;
  const trueFn = (x: number) => Math.sin(x * xScale);
  const features = Array.from({ length: size }, (_, i) => i / (size - 1 || 1));
  const labels = features.map((x) => trueFn(x) + (Math.random() - 0.5) * 0.3);
  return { features, labels, trueFn, xScale };
}

/** 数据加载器：随机洗牌 + 按 batchSize 取数 */
export class DataLoader {
  constructor(
    readonly features: number[],
    readonly labels: number[],
    private _batchSize: number,
  ) {}

  nextBatch(): { feature: number; label: number }[] {
    const n = this.features.length;
    const m = Math.min(this._batchSize, n);
    const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, m);
    return indices.map((i) => ({ feature: this.features[i], label: this.labels[i] }));
  }
}
