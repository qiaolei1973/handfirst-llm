/**
 * SGD 优化器 — 朴素梯度下降。
 *
 *   参数 = 参数 - lr × 梯度
 */

export class SGD {
  private _params: Array<{ data: Float64Array; grad: Float64Array }>;

  constructor(
    params: Array<{ data: Float64Array; grad: Float64Array }>,
    private _lr: number = 0.02,
  ) {
    this._params = params;
  }

  step(): void {
    for (const p of this._params) {
      for (let i = 0; i < p.data.length; i++) {
        p.data[i] -= this._lr * p.grad[i];
      }
    }
  }
}
