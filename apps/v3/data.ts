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

/** 数据加载器：shuffle 后按 batchSize 顺序遍历（一个 epoch 遍历一遍） */
export class DataLoader {
  private _indices: number[] = [];
  private _pos = 0;

  constructor(
    readonly dataset: Dataset,
    private _batchSize: number,
  ) {
    this._shuffle();
  }

  /** 取下一个 batch，遍历完一个 epoch 返回 null 并重新 shuffle */
  next(): { feature: number; label: number }[] | null {
    if (this._pos >= this._indices.length) {
      this._shuffle();
      return null;
    }
    const batch = this._indices.slice(this._pos, this._pos + this._batchSize);
    this._pos += this._batchSize;
    return batch.map((i) => ({ feature: this.dataset.features[i], label: this.dataset.labels[i] }));
  }

  private _shuffle(): void {
    const n = this.dataset.features.length;
    this._indices = [...Array(n).keys()].sort(() => Math.random() - 0.5);
    this._pos = 0;
  }
}
